"""Enrich WordNet entries with real dictionary-style definitions via local LLM.

Process
-------
1. Load ``word_supersense_data.v1.jsonl`` entries (word + supersense + POS).
2. For each entry, call a local GGUF model to generate a real definition.
3. Validate that genus extraction from the new definition produces a useful
   head noun (not a repeat of the word itself).
4. Save enriched entries to a new JSONL file.
5. Optionally re-ingest into the lexicon store (``--reingest``).

Usage
-----
::

    # Full run on all active supersenses (default)
    python scripts/enrich_wordnet_definitions.py

    # Single supersense, limited sample
    python scripts/enrich_wordnet_definitions.py --supersense noun.artifact --limit 50

    # Re-ingest into the store after enrichment
    python scripts/enrich_wordnet_definitions.py --reingest

    # Use a specific GGUF model
    python scripts/enrich_wordnet_definitions.py --model models/qwen2.5-1.5b-instruct-q4_k_m.gguf

    # Continue from a checkpoint
    python scripts/enrich_wordnet_definitions.py --resume enriched_output.jsonl

Output format (JSONL)
---------------------
Each output line adds an ``enriched_definition`` and ``genus_lemma`` field::

    {
      "pos": "noun",
      "supersense": "noun.artifact",
      "word": "vase",
      "definition": "wordnet supersense noun.artifact: vase",
      "enriched_definition": "a container, typically made of glass or ceramic, used for displaying flowers",
      "genus_lemma": "container",
      "genus_status": "resolved_or_self"
    }
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACTS_DIR = REPO_ROOT / "melm" / "contracts"
WORDNET_JSONL = CONTRACTS_DIR / "word_supersense_data.v1.jsonl"
_15B = REPO_ROOT / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
DEFAULT_MODEL = _15B if _15B.exists() else REPO_ROOT / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

# Default: skip Tiny noun.Tops (meta-categories), include the 10 active
# supersenses from _ACTIVE_SUPERSENSES plus dormant entries if explicitly asked.
ACTIVE_SUPERSENSES: set[str] = {
    "noun.artifact", "noun.object", "noun.food", "noun.substance",
    "noun.body", "noun.animal", "noun.plant", "noun.location",
    "noun.person", "noun.attribute",
}

# Purpose/relative clause markers — same as assistant_lexicon
_PURPOSE_MARKERS: frozenset[str] = frozenset({"used", "which", "that", "who", "whom"})
_GENUS_SKIP: frozenset[str] = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with",
    "by", "from", "into", "through", "during", "without", "or",
})
_GENUS_PP_MARKERS: frozenset[str] = frozenset({
    "of", "in", "for", "with", "by", "on", "at", "from", "into",
})


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _normalize_term(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _extract_genus_lemma(rest: str) -> str:
    """Extract head noun from a definition (assistant_lexicon's algorithm)."""
    raw = _normalize_term(rest)
    words = raw.split()
    for idx, w in enumerate(words):
        if w in _PURPOSE_MARKERS and idx >= 1:
            words = words[:idx]
            break
    i = len(words) - 1
    while i >= 0:
        word = words[i]
        if word not in _GENUS_SKIP:
            pp_boundary = False
            for j in range(i - 1, -1, -1):
                if words[j] in _GENUS_PP_MARKERS:
                    pp_boundary = True
                    i = j - 1
                    break
            if pp_boundary:
                continue
            return word
        i -= 1
    return words[-1] if words else ""


def _is_useful_genus(genus: str, word: str) -> str:
    """Classify genus quality.

    Returns
    -------
    ``"resolved"``        — genus is different from the word (likely useful).
    ``"self"``            — genus == word (uninformative, synthetic def).
    ``"empty"``           — no genus extracted.
    """
    if not genus:
        return "empty"
    if genus == word.lower().strip():
        return "self"
    return "resolved"


def _build_definition_prompt(word: str, supersense: str, pos: str) -> str:
    """Build a short prompt that asks for a dictionary-style definition.

    Avoids including the supersense label (e.g. ``artifact``) to prevent
    the model from defaulting to a generic hypernym.  Instead asks for a
    concrete category in plain language.
    """
    article = "an" if word.lower()[0] in "aeiou" else "a"
    return (
        f"Define \"{word}\". Start with \"{article} [concrete category]\". "
        f"Be specific. Do not use \"artifact\", \"object\", \"thing\", "
        f"or \"entity\" as the category."
    )


# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------

_LLM_CACHE = None


def _load_llm(model_path: str | Path) -> object:
    global _LLM_CACHE
    if _LLM_CACHE is not None:
        return _LLM_CACHE
    from llama_cpp import Llama
    print(f"  [LOAD] Loading model {model_path} ...")
    t0 = time.time()
    _LLM_CACHE = Llama(model_path=str(model_path), n_ctx=1024, verbose=False)
    elapsed = time.time() - t0
    print(f"  [LOAD] Model loaded in {elapsed:.1f}s")
    return _LLM_CACHE


def generate_definition(llm: object, word: str, supersense: str, pos: str) -> str | None:
    """Call the LLM and return the generated definition, or None on failure."""
    prompt = _build_definition_prompt(word, supersense, pos)
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a dictionary. Output only the definition, nothing else."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=96,
        )
        text = response["choices"][0]["message"]["content"].strip()
        if not text or len(text) < 5:
            return None

        # Clean up: remove leading/trailing quotes
        text = text.strip('"').strip("'").strip()
        # Remove "Definition:" prefix if present
        for prefix in ["Definition: ", "definition: ", "Define: ", "define: "]:
            if text.startswith(prefix):
                text = text[len(prefix):]

        return text
    except Exception as e:
        print(f"    [WARN] LLM error for {word}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Core enrichment
# ---------------------------------------------------------------------------

def load_wordnet_entries(
    supersense_filter: str | None = None,
    limit: int | None = None,
    skip_tops: bool = True,
) -> list[dict]:
    """Load entries from the WordNet JSONL file.

    Parameters
    ----------
    supersense_filter:
        If set, only return entries with this supersense (e.g. ``"noun.artifact"``).
    limit:
        Max entries to return.
    skip_tops:
        If True, skip ``noun.Tops`` (meta-categories like "entity", "thing").
    """
    if not WORDNET_JSONL.exists():
        print(f"[ERROR] WordNet data not found: {WORDNET_JSONL}", file=sys.stderr)
        sys.exit(1)

    entries: list[dict] = []
    with open(WORDNET_JSONL, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue

            ss = entry.get("supersense", "")
            if skip_tops and ss == "noun.Tops":
                continue
            if supersense_filter and ss != supersense_filter:
                continue
            entries.append(entry)
            if limit and len(entries) >= limit:
                break

    return entries


def enrich_entries(
    entries: list[dict],
    model_path: str | Path = DEFAULT_MODEL,
    output_path: str | Path | None = None,
    save_every: int = 100,
    resume_from: str | int = 0,
) -> list[dict]:
    """Generate real definitions for a list of WordNet entries.

    Parameters
    ----------
    entries: WordNet entry dicts (each has ``word``, ``supersense``, ``pos``).
    model_path: Path to a GGUF model file.
    checkpoint_path: If set, save progress to this JSONL file every ``save_every`` entries.
    save_every: How many entries between checkpoint saves.
    resume_from: Resume processing from this index (0 = start from beginning).
        Can be an int index or "auto" (auto-detect from checkpoint file).
    """
    llm = _load_llm(model_path)

    # Determine resume: load existing enriched entries from checkpoint
    start_index = 0
    enriched: list[dict] = []
    resume_path = resume_from if isinstance(resume_from, (str, Path)) else None
    if isinstance(resume_path, str) and resume_path == "auto" and output_path:
        resume_path = output_path
    if isinstance(resume_path, (str, Path)):
        rp = Path(resume_path)
        if rp.exists():
            with open(rp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            enriched.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            # Only resume if existing entries actually have enriched_definition
            already_done = sum(1 for e in enriched if bool(e.get("enriched_definition")))
            if already_done == len(enriched) and already_done > 0:
                start_index = len(enriched)
                print(f"  [RESUME] All {already_done} entries already enriched, skipping.")
            else:
                enriched.clear()
                print(f"  [RESUME] Checkpoint has {len(enriched)} entries, but only {already_done} have definitions. Re-processing from scratch.")
    elif isinstance(resume_from, int):
        start_index = resume_from

    total = len(entries)
    stats = Counter()

    for idx, entry in enumerate(entries):
        if idx < start_index:
            continue  # Already in enriched list from checkpoint

        word = str(entry.get("word", "")).strip()
        supersense = str(entry.get("supersense", "")).strip()
        pos = str(entry.get("pos", "noun")).strip().lower()

        if not word or not supersense:
            enriched.append(entry)
            stats["skipped_empty"] += 1
            continue

        # Generate definition
        definition = generate_definition(llm, word, supersense, pos)

        # Post-process genus
        if definition:
            genus = _extract_genus_lemma(definition)
            genus_status = _is_useful_genus(genus, word)
        else:
            genus = ""
            genus_status = "empty"

        enriched_entry = dict(entry)
        enriched_entry["enriched_definition"] = definition or ""
        enriched_entry["genus_lemma"] = genus
        enriched_entry["genus_status"] = genus_status
        enriched.append(enriched_entry)

        stats["total"] += 1
        if definition:
            stats["generated"] += 1
            if genus_status == "resolved":
                stats["genus_resolved"] += 1
        else:
            stats["failed"] += 1

        # Progress
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            el = f"(of {total})" if total > 0 else ""
            perc = f"{100.0 * (idx + 1) / total:.0f}%" if total > 0 else ""
            safe_def = (definition or "FAILED")[:60].encode("utf-8", errors="replace").decode("utf-8", errors="replace")
            print(
                f"  [{idx + 1}/{total} {perc}] "
                f"word={word:16s} supersense={supersense:20s} "
                f"genus={genus:12s} status={genus_status:10s} "
                f"def={safe_def}...",
            )

        # Periodic save
        if output_path and (idx + 1) % save_every == 0:
            _save_checkpoint(output_path, enriched, stats)
            stats["checkpoint_saved"] += 1

    # Final save
    if output_path:
        _save_checkpoint(output_path, enriched, stats)

    return enriched


def _save_checkpoint(path: str | Path, enriched: list[dict], stats: Counter) -> None:
    """Overwrite the checkpoint file with all enriched entries so far."""
    cp = Path(path)
    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "w", encoding="utf-8") as f:
        for entry in enriched:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  [CHECKPOINT] Saved {len(enriched)} entries -> {cp}")
    print(f"  [STATS] {dict(stats)}")


# ---------------------------------------------------------------------------
# Re-ingestion
# ---------------------------------------------------------------------------

def reingest_enriched(
    enriched_path: str | Path,
    db_dir: str | Path | None = None,
) -> dict:
    """Re-ingest enriched WordNet entries into the lexicon store.

    Reads the enriched JSONL, re-extracts genus from the real definition,
    maps to semantic classes, and calls ``lexicon_ingest()`` for each entry.
    """
    from melm.appliance.assistant_os_store import AssistantOSStore
    from melm.appliance.assistant_lexicon import (
        lexicon_ingest,
        _extract_genus_lemma as extract_genus,
        _compute_class_candidates,
    )
    from melm.appliance.assistant_lexicon_bulk import (
        _make_wn_definition,
        _wn_supersense_map,
        _class_ids,
    )
    from melm.appliance.assistant_lexicon_ingestion_gate import _candidate

    enriched = []
    with open(enriched_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                enriched.append(json.loads(line))

    print(f"[REINGEST] Loading {len(enriched)} enriched entries ...")

    mapping = _wn_supersense_map()
    known = _class_ids()

    if db_dir is None:
        import tempfile
        db_dir = Path(tempfile.mkdtemp()) / "db"

    store = AssistantOSStore(db_dir=str(db_dir))
    store.setup()

    applied = 0
    skipped = 0
    errors = 0

    for entry in enriched:
        word = str(entry.get("word", "")).strip().lower()
        supersense = str(entry.get("supersense", "")).strip().lower()
        pos = str(entry.get("pos", "noun")).strip().lower()
        enriched_def = entry.get("enriched_definition", "")
        original_def = entry.get("definition", "")

        if not word or not supersense:
            skipped += 1
            continue

        melm_class = mapping.get(supersense)
        if melm_class is None or melm_class not in known:
            skipped += 1
            continue

        # Use enriched definition if available, else fall back to original
        definition = enriched_def if enriched_def else original_def
        genus = extract_genus(definition) if enriched_def else word

        # Build candidate with real definition + extracted genus
        candidate = _candidate(
            lemma=word,
            pos=pos,
            class_id=melm_class,
            definition=definition,
            source_ref=f"wordnet:supersense:{supersense}:{word}",
            provenance="wordnet",
        )
        # Override genus_lemma so the ingestion gate uses our extracted genus
        candidate["genus_lemma"] = genus

        # Compute class candidates from genus if it's different from the word
        if genus != word:
            class_candidates = _compute_class_candidates(store, genus, pos)
            if class_candidates:
                candidate["semantic_class_candidates"] = class_candidates

        try:
            lexicon_ingest(store, candidate, expected_provenance="wordnet")
            applied += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [ERROR] {word}: {e}")

    print(f"[REINGEST] Applied={applied}, Skipped={skipped}, Errors={errors}")
    return {"applied": applied, "skipped": skipped, "errors": errors}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_model_path(model_arg: str) -> Path:
    """Resolve model path from user argument."""
    path = Path(model_arg)
    if path.exists():
        return path
    # Try resolving relative to REPO_ROOT / models
    alt = REPO_ROOT / "models" / model_arg
    if alt.exists():
        return alt
    print(f"[ERROR] Model not found: {model_arg} (tried {path} and {alt})", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich WordNet entries with real definitions via local LLM.",
    )
    parser.add_argument(
        "--model", type=str, default=str(DEFAULT_MODEL),
        help="Path to GGUF model file (default: qwen2.5-0.5b-instruct-q4_k_m.gguf)",
    )
    parser.add_argument(
        "--supersense", type=str, default=None,
        help="Only process this supersense (e.g. noun.artifact). Default: all active supersenses.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max entries to process.",
    )
    parser.add_argument(
        "--output", type=str, default="enriched_wordnet.jsonl",
        help="Output JSONL path (default: enriched_wordnet.jsonl).",
    )
    parser.add_argument(
        "--save-every", type=int, default=100,
        help="Save checkpoint every N entries (default: 100).",
    )
    parser.add_argument(
        "--resume", type=str, default=None, nargs="?",
        const="auto",
        help="Resume from checkpoint file. Specify path or 'auto' to use --output.",
    )
    parser.add_argument(
        "--reingest", action="store_true",
        help="After enrichment, re-ingest into the lexicon store.",
    )
    parser.add_argument(
        "--db-dir", type=str, default=None,
        help="Store directory for re-ingestion (default: tmpdir).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print first 5 entries without generating.",
    )

    args = parser.parse_args()

    # Load entries
    entries = load_wordnet_entries(
        supersense_filter=args.supersense,
        limit=args.limit,
    )

    if not entries:
        print("[ERROR] No WordNet entries loaded", file=sys.stderr)
        sys.exit(1)

    # Count supersenses
    ss_counts = Counter(e.get("supersense", "?") for e in entries)
    print(f"[INFO] Loaded {len(entries)} entries from {len(ss_counts)} supersenses:")
    for ss, cnt in ss_counts.most_common(10):
        print(f"       {ss}: {cnt}")

    if args.dry_run:
        print("[DRY-RUN] First 5 entries:")
        for entry in entries[:5]:
            print(f"         {json.dumps(entry)}")
        return

    # Run enrichment
    print(f"[ENRICH] Starting enrichment (model={args.model}) ...")
    enriched = enrich_entries(
        entries,
        model_path=_resolve_model_path(args.model),
        output_path=args.output,
        save_every=args.save_every,
        resume_from=args.resume or 0,
    )

    print(f"[ENRICH] Done. {len(enriched)} entries processed.")
    print(f"[ENRICH] Output saved to {args.output}")

    # Re-ingestion
    if args.reingest:
        print("[REINGEST] Starting re-ingestion into lexicon store ...")
        result = reingest_enriched(args.output, db_dir=args.db_dir)
        print(f"[REINGEST] Done: {result}")


if __name__ == "__main__":
    main()
