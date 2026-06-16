# Bulk Lexicon Seed Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) to implement this plan task-by-task.

**Goal:** Build `scripts/build_lexicon_seed.py` that downloads real WordNet 3.1 (+ VerbNet 3.4) data and generates the JSONL files that feed the existing ingestion pipeline — closing the 80x scale gap from 1,759 hand-curated entries to ~100k+ word-supersense pairs.

**Architecture:** Download-only Python script (no NLTK dependency, no NLTK, uses `urllib` + `tarfile` for WordNet, `zipfile` + XML parser for VerbNet). Outputs the same JSONL format (`word_supersense_data.v1.jsonl`, `verb_data.v1.jsonl`) that the existing `seed_wordnet_supersenses()` / `seed_verbnet_classes()` ingestors consume. The existing map contracts (`wn_supersense_map.v1.json`, `verbnet_map.v1.json`) remain unchanged — the script reads them at generation time to validate supersense coverage.

**Tech Stack:** Python 3.13 stdlib only (`urllib`, `tarfile`, `zipfile`, `xml.etree.ElementTree`, `json`, `pathlib`), existing contract infrastructure (`melm/contracts/`) for maps.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `scripts/build_lexicon_seed.py` | Create | Main orchestrator: download, parse, generate JSONL |
| `scripts/build_lexicon_seed.py` | (single file) | All pipeline stages live in one script |
| `melm/contracts/word_supersense_data.v1.jsonl` | Regenerate | WordNet-derived word→supersense pairs |
| `melm/contracts/verb_data.v1.jsonl` | Regenerate | VerbNet-derived verb→class pairs |
| `tests/test_build_lexicon_seed_mvp.py` | Create | Tests for parse functions offline |

---

## WordNet data format reference

Each synset in `data.noun` / `data.verb` (Princeton WordNet 3.1 dict format):
```
offset lex_filenum ss_type w_cnt word1 hex1 word2 hex2 ... | gloss
```
- `ss_type`: `n`=noun, `v`=verb, `a`=adj, `r`=adv
- `lex_filenum`: 0–44 maps to supersense tag name via the standard mapping
- Words are separated by spaces, each followed by a hex sense number
- Underscores in words represent spaces

Supersense tag naming convention: `{pos}.{name}` where `pos` is `noun`/`verb`/`adj`/`adv`

## VerbNet XML format reference

```xml
<VNCLASS class="class_name" ...>
  <MEMBERS>
    <MEMBER name="verb" .../>
    <MEMBER name="verb" .../>
  </MEMBERS>
  <SUBCLASSES>
    <VNCLASS class="subclass" ...>...</VNCLASS>
  </SUBCLASSES>
</VNCLASS>
```

---

### Task 1: WordNet download and parse helpers

**Files:**
- Create: `scripts/build_lexicon_seed.py` (until line ~120)
- Test: `tests/test_build_lexicon_seed_mvp.py`

**Design:**
- `download_wordnet(cache_dir: Path) -> Path` — downloads `wn3.1.dict.tar.gz` from Princeton, extracts to cache_dir, returns path to dict directory
- `parse_lex_filenum_to_supersense() -> dict[int, str]` — hardcoded mapping of lex_filenum (0-44) to supersense tag (e.g., 4 → "noun.act", 29 → "verb.body")
- `parse_synset_words(line: str, ss_type: str) -> list[str]` — extracts lemmas from a synset data line
- `extract_wordnet_supersenses(dict_dir: Path) -> list[dict]` — reads data.noun, data.verb, data.adj, data.adv; for each synset, extracts lemmas and produces `{"word", "supersense", "pos"}` entries

- [ ] **Step 1: Write the failing test — parse_synset_words extracts lemmas from data.noun line**

`tests/test_build_lexicon_seed_mvp.py`:
```python
import json
import unittest
from pathlib import Path

from scripts.build_lexicon_seed import parse_synset_words


class TestParseSynsetWords(unittest.TestCase):
    def test_parse_noun_line(self) -> None:
        line = "00001740 00 n 01 entity 0 001 @ 00001740 n 0000 | something that exists"
        words = parse_synset_words(line, "n")
        self.assertEqual(words, ["entity"])

    def test_parse_verb_line(self) -> None:
        line = "01623404 38 v 02 walk 0 stroll 0 003 @ 01622543 v 0000 | walk slowly"
        words = parse_synset_words(line, "v")
        self.assertEqual(words, ["walk", "stroll"])

    def test_parse_multi_word(self) -> None:
        line = "07794744 13 n 01 pasta_salad 0 001 ~ 07793993 n 0000 | a salad with pasta"
        words = parse_synset_words(line, "n")
        self.assertEqual(words, ["pasta_salad"])

    def test_empty_line_returns_empty(self) -> None:
        self.assertEqual(parse_synset_words("", "n"), [])

    def test_comment_line_returns_empty(self) -> None:
        self.assertEqual(parse_synset_words("  # comment", "n"), [])
```

- [ ] **Step 2: Run test — should fail (module doesn't exist)**

Run: `python -m pytest tests/test_build_lexicon_seed_mvp.py -v --tb=short`
Expected: `ModuleNotFoundError: No module named 'scripts.build_lexicon_seed'`

- [ ] **Step 3: Write parser functions in scripts/build_lexicon_seed.py**

```python
"""Download real WordNet + VerbNet data and generate JSONL for bulk ingestion."""

import json
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any


# ── WordNet helpers ──────────────────────────────────────────────────────────

_WORDNET_URL = "https://wordnetcode.princeton.edu/wn3.1.dict.tar.gz"


def _lex_filenum_to_supersense() -> dict[int, str]:
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


# Map ss_type → dict file prefix
_SS_TYPE_TO_FILE = {"n": "noun", "v": "verb", "a": "adj", "r": "adv"}


def parse_synset_words(line: str, ss_type: str) -> list[str]:
    """Extract lemma words from a WordNet data file synset line.

    Format: offset lex_filenum ss_type w_cnt word1 hex1 word2 hex2 ... | gloss
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("  "):
        return []
    parts = stripped.split()
    # parts[0]=offset, parts[1]=lex_filenum, parts[2]=ss_type
    if len(parts) < 4 or parts[2] != ss_type:
        return []
    try:
        w_cnt = int(parts[3], 16)  # w_cnt is hex
    except (ValueError, IndexError):
        return []
    words: list[str] = []
    idx = 4
    for _ in range(w_cnt):
        if idx >= len(parts):
            break
        words.append(parts[idx].replace("_", " "))
        idx += 2  # skip hex sense number
    return words


def download_wordnet(cache_dir: Path) -> Path:
    """Download and extract WordNet 3.1 dict files.

    Returns path to the ``dict`` directory containing data.noun, data.verb, etc.
    """
    dict_dir = cache_dir / "wordnet-dict"
    if dict_dir.is_dir():
        return dict_dir
    tarball_path = cache_dir / "wn3.1.dict.tar.gz"
    if not tarball_path.is_file():
        print(f"Downloading WordNet 3.1 from {_WORDNET_URL} ...")
        urllib.request.urlretrieve(_WORDNET_URL, tarball_path)
    print("Extracting WordNet dict files ...")
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(path=cache_dir)
    return dict_dir
```

- [ ] **Step 4: Run tests — should pass**

Run: `python -m pytest tests/test_build_lexicon_seed_mvp.py -v --tb=short`
Expected: 5 passed

- [ ] **Step 5: Write failing test for extract_wordnet_supersenses**

Add to `test_build_lexicon_seed_mvp.py`:
```python
from scripts.build_lexicon_seed import (
    extract_wordnet_supersenses,
    _lex_filenum_to_supersense,
)


class TestExtractWordnetSupersenses(unittest.TestCase):
    def test_extract_known_verb_supersense(self) -> None:
        """walk is in verb.motion (lex_filenum 38)."""
        # We pass a mini data.noun and data.verb to test against
        line = "01623404 38 v 02 walk 0 stroll 0 003 @ 01622543 v 0000 | walk slowly"
        entries = extract_wordnet_supersenses.__wrapped__(
            {"v": [line]},  # type: ignore
            {"v": "verb"},
        )
        self.assertIn(("walk", "verb.motion", "verb"), entries)
        self.assertIn(("stroll", "verb.motion", "verb"), entries)
```

- [ ] **Step 6: Run test — should fail**

- [ ] **Step 7: Implement extract_wordnet_supersenses**

Add to `build_lexicon_seed.py`:
```python
def extract_wordnet_supersenses(dict_dir: Path) -> list[dict[str, str]]:
    """Read WordNet dict files and yield word→supersense entries."""
    lex_map = _lex_filenum_to_supersense()
    entries: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for ss_type, prefix in _SS_TYPE_TO_FILE.items():
        filepath = dict_dir / f"data.{prefix}"
        if not filepath.is_file():
            continue
        text = filepath.read_text(encoding="latin-1")
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
                    entries.append({"word": w, "supersense": supersense, "pos": ss_type})
    return entries
```

- [ ] **Step 8: Run tests — should pass**

---

### Task 2: VerbNet download and parse helpers

**Files:**
- Modify: `scripts/build_lexicon_seed.py` (add ~50 lines)
- Modify: `tests/test_build_lexicon_seed_mvp.py` (add tests)

- [ ] **Step 1: Write failing tests for VerbNet XML parsing**

```python
from scripts.build_lexicon_seed import parse_verbnet_members


class TestParseVerbnetMembers(unittest.TestCase):
    def test_parse_member_from_xml(self) -> None:
        xml = '''<?xml version="1.0" encoding="UTF-8"?>
<VNCLASS class="motion-51.3.2" ...>
  <MEMBERS>
    <MEMBER name="walk" />
    <MEMBER name="run" />
  </MEMBERS>
</VNCLASS>'''
        members = parse_verbnet_members(xml)
        self.assertEqual(members, ["walk", "run"])

    def test_parse_recursive_subclasses(self) -> None:
        xml = '''<?xml version="1.0" encoding="UTF-8"?>
<VNCLASS class="motion-51.3.2">
  <MEMBERS><MEMBER name="walk" /></MEMBERS>
  <SUBCLASSES>
    <VNCLASS class="motion-51.3.2-1">
      <MEMBERS><MEMBER name="stroll" /></MEMBERS>
    </VNCLASS>
  </SUBCLASSES>
</VNCLASS>'''
        members = parse_verbnet_members(xml)
        self.assertEqual(set(members), {"walk", "stroll"})

    def test_empty_members_returns_empty(self) -> None:
        xml = '<?xml version="1.0" encoding="UTF-8"?><VNCLASS class="empty"><MEMBERS></MEMBERS></VNCLASS>'
        self.assertEqual(parse_verbnet_members(xml), [])
```

- [ ] **Step 2: Run tests — should fail**

- [ ] **Step 3: Implement parse_verbnet_members and download_verbnet**

```python
_VERBNET_URL = (
    "https://github.com/colorless-energy/verbnet/archive/refs/heads/master.zip"
)


def parse_verbnet_members(xml_content: str) -> list[str]:
    """Extract all MEMBER verb names from a VerbNet VNCLASS XML, recursively."""
    root = ET.fromstring(xml_content)
    members: list[str] = []
    _collect_verbnet_members(root, members)
    return members


def _collect_verbnet_members(element: ET.Element, members: list[str]) -> None:
    for member in element.findall(".//MEMBER"):
        name = member.get("name")
        if name:
            members.append(name)


def download_verbnet(cache_dir: Path) -> Path:
    """Download VerbNet 3.4 and return path to vnclass XML files."""
    vn_dir = cache_dir / "verbnet"
    if vn_dir.is_dir() and any(vn_dir.iterdir()):
        return vn_dir
    zip_path = cache_dir / "verbnet-master.zip"
    if not zip_path.is_file():
        print(f"Downloading VerbNet from {_VERBNET_URL} ...")
        urllib.request.urlretrieve(_VERBNET_URL, zip_path)
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=cache_dir)
    # The source tree has the class files in verbnet-master/verbnet/
    extracted = cache_dir / "verbnet-master" / "verbnet"
    if extracted.is_dir():
        # Rename to cache_dir / verbnet for consistency
        extracted.rename(vn_dir)
    return vn_dir


def extract_verbnet_verbs(vn_dir: Path) -> list[dict[str, str]]:
    """Read VerbNet vnclass XML files and yield verb→verbnet-class entries.

    Only includes classes that have a mapping in verbnet_map.v1.json.
    """
    import melm.contracts.validation as v
    vn_map = v._load_json_file(v._contract_path("verbnet_map.v1.json"))
    mappings: dict[str, str] = vn_map.get("mappings", {})
    entries: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for fpath in sorted(vn_dir.rglob("*.xml")):
        vn_class = fpath.stem  # filename without .xml is the class name
        if vn_class not in mappings:
            continue
        xml_text = fpath.read_text(encoding="utf-8")
        verbs = parse_verbnet_members(xml_text)
        for verb in verbs:
            key = (verb, vn_class)
            if key not in seen:
                seen.add(key)
                entries.append({"verb": verb, "verbnet_class": vn_class, "pos": "verb"})
    return entries
```

- [ ] **Step 4: Run tests — should pass**

---

### Task 3: Main pipeline orchestrator

**Files:**
- Modify: `scripts/build_lexicon_seed.py` (add main() + output writers)

- [ ] **Step 1: Write main orchestrator**

```python
# ── Output writers ───────────────────────────────────────────────────────────

_CONTRACTS_DIR = Path(__file__).resolve().parent.parent / "melm" / "contracts"


def _load_map(path: Path) -> set[str]:
    """Load valid supersense tags from a map contract."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data.get("mappings", {}).keys())


def write_supersense_jsonl(
    entries: list[dict[str, str]],
    output_path: Path,
) -> int:
    """Write word→supersense entries to JSONL, deduped."""
    valid_supersenses = _load_map(
        _CONTRACTS_DIR / "wn_supersense_map.v1.json"
    )
    seen: set[tuple[str, str]] = set()
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            if entry["supersense"] not in valid_supersenses:
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
    """Write verb→verbnet-class entries to JSONL, deduped."""
    valid_classes = _load_map(
        _CONTRACTS_DIR / "verbnet_map.v1.json"
    )
    seen: set[tuple[str, str]] = set()
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            if entry["verbnet_class"] not in valid_classes:
                continue
            key = (entry["verb"], entry["verbnet_class"])
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count
```

- [ ] **Step 2: Write main()**

```python
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Download WordNet + VerbNet and generate bulk lexicon JSONL."
    )
    parser.add_argument(
        "--cache-dir",
        default=Path.home() / ".cache" / "melm-lexicon-seed",
        type=Path,
        help="Cache directory for downloaded archives (default: ~/.cache/melm-lexicon-seed)",
    )
    parser.add_argument(
        "--output-dir",
        default=_CONTRACTS_DIR,
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
    dict_dir = download_wordnet(args.cache_dir)
    wn_entries = extract_wordnet_supersenses(dict_dir)
    wn_path = args.output_dir / "word_supersense_data.v1.jsonl"
    wn_count = write_supersense_jsonl(wn_entries, wn_path)
    print(f"WordNet: {wn_count} entries → {wn_path}")

    # Phase 2: VerbNet
    vn_dir = download_verbnet(args.cache_dir)
    vn_entries = extract_verbnet_verbs(vn_dir)
    vn_path = args.output_dir / "verb_data.v1.jsonl"
    vn_count = write_verb_jsonl(vn_entries, vn_path)
    print(f"VerbNet: {vn_count} entries → {vn_path}")

    print(f"Total: {wn_count + vn_count} entries")
    print("Run `seed_bulk_lexicon(store)` to ingest.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write tests for output writers**

```python
class TestWriteSupersenseJsonl(unittest.TestCase):
    def test_filters_unmapped_supersenses(self) -> None:
        entries = [
            {"word": "foo", "supersense": "noun.NONEXISTENT", "pos": "noun"},
            {"word": "walk", "supersense": "verb.motion", "pos": "verb"},
        ]
        output_path = Path(self._test_dir.name) / "test.jsonl"
        count = write_supersense_jsonl(entries, output_path)
        self.assertEqual(count, 1)  # only walk survives
        lines = output_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn("walk", lines[0])

    def setUp(self) -> None:
        import tempfile
        self._test_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._test_dir.cleanup()
```

---

### Task 4: Run the pipeline end-to-end

- [ ] **Step 1: Run the script**

Run: `python scripts/build_lexicon_seed.py`
Expected:
- Downloads WordNet 3.1 dict (~8 MB)
- Downloads VerbNet master zip
- Generates word_supersense_data.v1.jsonl with ~100k+ entries
- Generates verb_data.v1.jsonl (entries for 12 mapped VerbNet classes)

- [ ] **Step 2: Check output stats**

Run:
```bash
python -c "
from pathlib import Path
wn = Path('melm/contracts/word_supersense_data.v1.jsonl')
vb = Path('melm/contracts/verb_data.v1.jsonl')
print(f'WordNet: {len(wn.read_text().splitlines())} lines')
print(f'VerbNet: {len(vb.read_text().splitlines())} lines')
"```
Expected: WordNet ~100k+, VerbNet ~few hundred (only 12 mapped classes)

- [ ] **Step 3: Run the ingestion pipeline**

Run (smaller timeout for ingestion test):
```python
python -c "
from melm.appliance.assistant_os_store import AssistantOSStore, seed_class_schemas
from melm.appliance.assistant_lexicon_bulk import seed_bulk_lexicon
store = AssistantOSStore(':memory:')
seed_class_schemas(store)
counts = seed_bulk_lexicon(store)
print(f'Ingested: {counts}')
from melm.appliance.assistant_lexicon import build_in_memory_lexicon
in_mem = build_in_memory_lexicon(store)
print(f'In-memory lexicon size after bulk seed: {len(in_mem)}')
"```

- [ ] **Step 4: Run all tests to check for regressions**

Run: `python -m pytest tests/ -q --tb=line`
Expected: all pass

---

### Task 5: Clean up and verify

- [ ] **Step 1: Commit**

```bash
git add scripts/build_lexicon_seed.py tests/test_build_lexicon_seed_mvp.py melm/contracts/word_supersense_data.v1.jsonl melm/contracts/verb_data.v1.jsonl
git commit -m "feat: build_lexicon_seed.py — download WordNet + VerbNet and generate bulk JSONL"
```

- [ ] **Step 2: Run final verification**

```bash
python -m pytest tests/ -q --tb=line
```
Expected: All tests pass
