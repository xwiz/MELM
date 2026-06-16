"""Download real WordNet + VerbNet data and generate JSONL for bulk ingestion.

Usage:
    python scripts/build_lexicon_seed.py [--cache-dir DIR] [--output-dir DIR]

Downloads:
    - WordNet 3.1 dict files from Princeton (~8 MB)
    - VerbNet 3.4 from GitHub (colorless-energy/verbnet)

Outputs (in melm/contracts/ by default):
    - word_supersense_data.v1.jsonl  — word -> WordNet supersense pairs
    - verb_data.v1.jsonl             — verb -> VerbNet class pairs
"""

import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


# ── WordNet helpers ──────────────────────────────────────────────────────────

_WORDNET_URL = "https://wordnetcode.princeton.edu/wn3.1.dict.tar.gz"
_VERBNET_URL = (
    "https://github.com/cu-clear/verbnet/archive/refs/heads/master.zip"
)
_SS_TYPE_TO_FILE = {"n": "noun", "v": "verb", "a": "adj", "r": "adv"}
_SS_TYPE_TO_FULL_POS = {"n": "noun", "v": "verb", "a": "adjective", "r": "adverb"}


def _lex_filenum_to_supersense() -> dict[int, str]:
    """Map WordNet lexicographer file number (0-44) to supersense tag."""
    return {
        0: "adj.all", 1: "adj.pert", 2: "adv.all",
        3: "noun.Tops", 4: "noun.act", 5: "noun.animal",
        6: "noun.artifact", 7: "noun.attribute", 8: "noun.body",
        9: "noun.cognition", 10: "noun.communication", 11: "noun.event",
        12: "noun.feeling", 13: "noun.food", 14: "noun.group",
        15: "noun.location", 16: "noun.motive", 17: "noun.object",
        18: "noun.person", 19: "noun.phenomenon", 20: "noun.plant",
        21: "noun.possession", 22: "noun.process", 23: "noun.quantity",
        24: "noun.relation", 25: "noun.shape", 26: "noun.state",
        27: "noun.substance", 28: "noun.time",
        29: "verb.body", 30: "verb.change", 31: "verb.cognition",
        32: "verb.communication", 33: "verb.competition",
        34: "verb.consumption", 35: "verb.contact",
        36: "verb.creation", 37: "verb.emotion",
        38: "verb.motion", 39: "verb.perception",
        40: "verb.possession", 41: "verb.social",
        42: "verb.stative", 43: "verb.weather", 44: "adj.ppl",
    }


def parse_synset_words(line: str, ss_type: str) -> list[str]:
    """Extract lemma words from a WordNet data file synset line.

    Format: offset lex_filenum ss_type w_cnt word1 hex1 word2 hex2 ... | gloss
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("  "):
        return []
    parts = stripped.split()
    if len(parts) < 4 or parts[2] != ss_type:
        return []
    try:
        w_cnt = int(parts[3], 16)
    except (ValueError, IndexError):
        return []
    words: list[str] = []
    idx = 4
    for _ in range(w_cnt):
        if idx >= len(parts):
            break
        word = parts[idx].replace("_", " ")
        words.append(word)
        idx += 2
    return words


def download_wordnet(cache_dir: Path) -> Path:
    """Download and extract WordNet 3.1 dict files.

    Returns path to the ``dict`` directory containing ``data.noun`` etc.
    """
    dict_dir = cache_dir / "wordnet-dict"
    if dict_dir.is_dir():
        print(f"Using cached WordNet dict at {dict_dir}")
        return dict_dir
    tarball_path = cache_dir / "wn3.1.dict.tar.gz"
    if not tarball_path.is_file():
        print(f"Downloading WordNet 3.1 from {_WORDNET_URL} ...")
        urllib.request.urlretrieve(_WORDNET_URL, tarball_path)
        print("Download complete.")
    if not dict_dir.is_dir():
        # The tarball root folder is named "dict"
        raw_extract = cache_dir / "dict"
        if raw_extract.is_dir():
            # Already extracted but under the wrong name
            raw_extract.rename(dict_dir)
        else:
            print("Extracting WordNet dict files ...")
            with tarfile.open(tarball_path, "r:gz") as tar:
                tar.extractall(path=cache_dir, filter="data")
            if raw_extract.is_dir():
                raw_extract.rename(dict_dir)
    return dict_dir


def extract_wordnet_supersenses(dict_dir: Path) -> list[dict[str, str]]:
    """Read WordNet dict files and extract word->supersense entries.

    Returns deduplicated list of dicts with keys ``word``, ``supersense``, ``pos``.
    """
    lex_map = _lex_filenum_to_supersense()
    entries: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for ss_type, prefix in _SS_TYPE_TO_FILE.items():
        fpath = dict_dir / f"data.{prefix}"
        if not fpath.is_file():
            continue
        text = fpath.read_text(encoding="latin-1")
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("  "):
                continue
            parts = stripped.split()
            if len(parts) < 4 or parts[2] != ss_type:
                continue
            try:
                lex_num = int(parts[1])
            except (ValueError, IndexError):
                continue
            supersense = lex_map.get(lex_num)
            if supersense is None:
                continue
            words = parse_synset_words(line, ss_type)
            for w in words:
                key = (w, supersense)
                if key not in seen:
                    seen.add(key)
                    entries.append({
                        "word": w,
                        "supersense": supersense,
                        "pos": _SS_TYPE_TO_FULL_POS.get(ss_type, ss_type),
                    })
    return entries


# ── VerbNet helpers ──────────────────────────────────────────────────────────


def parse_verbnet_members(xml_content: str) -> list[str]:
    """Extract all MEMBER verb names from a VerbNet VNCLASS XML, recursively.

    Searches all descendant ``<MEMBER>`` elements for their ``name`` attribute.
    """
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_content)
    members: list[str] = []
    for member in root.findall(".//MEMBER"):
        name = member.get("name")
        if name:
            members.append(name)
    return members


def download_verbnet(cache_dir: Path) -> Path:
    """Download VerbNet 3.4 and return path to vnclass XML directory."""
    vn_dir = cache_dir / "verbnet"
    if vn_dir.is_dir() and any(vn_dir.iterdir()):
        print(f"Using cached VerbNet at {vn_dir}")
        return vn_dir
    zip_path = cache_dir / "verbnet-master.zip"
    if not zip_path.is_file():
        print(f"Downloading VerbNet from {_VERBNET_URL} ...")
        urllib.request.urlretrieve(_VERBNET_URL, zip_path)
        print("Download complete.")
    print("Extracting VerbNet ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=cache_dir)
    extracted = cache_dir / "verbnet-master" / "vn3.4"
    if extracted.is_dir():
        extracted.rename(vn_dir)
    elif (cache_dir / "verbnet-master" / "verbnet").is_dir():
        (cache_dir / "verbnet-master" / "verbnet").rename(vn_dir)
    elif (cache_dir / "verbnet-master" / "verbnet3.4").is_dir():
        (cache_dir / "verbnet-master" / "verbnet3.4").rename(vn_dir)
    return vn_dir


def extract_verbnet_verbs(vn_dir: Path) -> list[dict[str, str]]:
    """Read VerbNet vnclass XML files and extract verb->verbnet-class entries.

    Only includes classes that have a mapping in ``verbnet_map.v1.json``
    (12 classes currently).  Returns deduplicated list of dicts with keys
    ``verb``, ``verbnet_class``, ``pos``.
    """
    vn_map = _load_verbnet_map()
    entries: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for fpath in sorted(vn_dir.rglob("*.xml")):
        vn_class = fpath.stem  # filename stem is the VerbNet class ID
        if vn_class not in vn_map:
            continue
        xml_text = fpath.read_text(encoding="utf-8")
        verbs = parse_verbnet_members(xml_text)
        for verb in verbs:
            key = (verb, vn_class)
            if key not in seen:
                seen.add(key)
                entries.append({
                    "verb": verb,
                    "verbnet_class": vn_class,
                    "pos": "verb",
                })
    return entries


# ── Map loaders ──────────────────────────────────────────────────────────────

_RESOURCES = Path(__file__).resolve().parent.parent / "melm" / "contracts"


def _load_wn_supersense_map() -> set[str]:
    """Load valid supersense tags from ``wn_supersense_map.v1.json``."""
    return _load_map_keys(_RESOURCES / "wn_supersense_map.v1.json")


def _load_verbnet_map() -> set[str]:
    """Load valid VerbNet class IDs from ``verbnet_map.v1.json``."""
    return _load_map_keys(_RESOURCES / "verbnet_map.v1.json")


def _load_map_keys(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data.get("mappings", {}).keys())


# ── Output writers ───────────────────────────────────────────────────────────


def write_supersense_jsonl(
    entries: list[dict[str, str]],
    output_path: Path,
) -> int:
    """Write word->supersense entries to JSONL, filtering by valid supersenses.

    Returns the number of entries written.
    """
    valid = _load_wn_supersense_map()
    seen: set[tuple[str, str]] = set()
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            if entry["supersense"] not in valid:
                continue
            key = (entry["word"], entry["supersense"])
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_verb_jsonl(
    entries: list[dict[str, str]],
    output_path: Path,
) -> int:
    """Write verb->verbnet-class entries to JSONL, filtering by valid classes.

    Returns the number of entries written.
    """
    valid = _load_verbnet_map()
    seen: set[tuple[str, str]] = set()
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            if entry["verbnet_class"] not in valid:
                continue
            key = (entry["verb"], entry["verbnet_class"])
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


# ── Main orchestrator ────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download WordNet + VerbNet and generate bulk lexicon JSONL.",
    )
    parser.add_argument(
        "--cache-dir",
        default=Path.home() / ".cache" / "melm-lexicon-seed",
        type=Path,
        help="Cache directory for downloaded archives",
    )
    parser.add_argument(
        "--output-dir",
        default=_RESOURCES,
        type=Path,
        help="Output directory for JSONL files (default: melm/contracts)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, use cached files only",
    )
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: WordNet
    print("=" * 60)
    print("Phase 1: WordNet supersense extraction")
    print("=" * 60)
    dict_dir = download_wordnet(args.cache_dir)
    wn_entries = extract_wordnet_supersenses(dict_dir)
    wn_path = args.output_dir / "word_supersense_data.v1.jsonl"
    wn_count = write_supersense_jsonl(wn_entries, wn_path)
    print(f"  WordNet: {wn_count:,} unique word->supersense entries")
    print(f"  Output:  {wn_path}")

    # Phase 2: VerbNet
    print()
    print("=" * 60)
    print("Phase 2: VerbNet verb extraction")
    print("=" * 60)
    vn_dir = download_verbnet(args.cache_dir)
    vn_entries = extract_verbnet_verbs(vn_dir)
    vn_path = args.output_dir / "verb_data.v1.jsonl"
    vn_count = write_verb_jsonl(vn_entries, vn_path)
    print(f"  VerbNet: {vn_count:,} unique verb->class entries")
    print(f"  Output:  {vn_path}")

    print()
    print(f"Total: {wn_count + vn_count:,} entries across both sources.")
    print()
    print("Next: run ``seed_bulk_lexicon(store)`` to ingest into the entity store.")
    print()


if __name__ == "__main__":
    main()
