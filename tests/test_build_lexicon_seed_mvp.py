"""Tests for bulk lexicon seed pipeline."""

import unittest

from scripts.build_lexicon_seed import parse_synset_words


class TestParseSynsetWords(unittest.TestCase):
    """parse_synset_words extracts lemmas from WordNet data file lines."""

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
        self.assertEqual(words, ["pasta salad"])

    def test_empty_line_returns_empty(self) -> None:
        self.assertEqual(parse_synset_words("", "n"), [])

    def test_comment_line_returns_empty(self) -> None:
        self.assertEqual(parse_synset_words("  # comment", "n"), [])


class TestLexFilenumToSupersense(unittest.TestCase):
    """_lex_filenum_to_supersense maps WordNet lexicographer file numbers."""

    def _get_map(self) -> dict[int, str]:
        from scripts.build_lexicon_seed import _lex_filenum_to_supersense
        return _lex_filenum_to_supersense()

    def test_has_entries(self) -> None:
        m = self._get_map()
        self.assertGreater(len(m), 0)

    def test_noun_act_is_4(self) -> None:
        m = self._get_map()
        self.assertEqual(m[4], "noun.act")

    def test_verb_motion_is_38(self) -> None:
        m = self._get_map()
        self.assertEqual(m[38], "verb.motion")

    def test_no_gaps(self) -> None:
        m = self._get_map()
        for i in range(45):
            self.assertIn(i, m, f"lex_filenum {i} is missing")


class TestWriteSupersenseJsonl(unittest.TestCase):
    """write_supersense_jsonl filters by valid supersenses from the map contract."""

    def setUp(self) -> None:
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self._tmpdir = self._tmp.name

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_and_count(self, entries: list[dict]) -> int:
        from pathlib import Path
        from scripts.build_lexicon_seed import write_supersense_jsonl
        out = Path(self._tmpdir) / "test.jsonl"
        return write_supersense_jsonl(entries, out)

    def test_filters_unmapped_supersenses(self) -> None:
        entries = [
            {"word": "foo", "supersense": "noun.NONEXISTENT", "pos": "noun"},
            {"word": "walk", "supersense": "verb.motion", "pos": "verb"},
        ]
        count = self._write_and_count(entries)
        self.assertEqual(count, 1)

    def test_empty_entries_writes_zero(self) -> None:
        self.assertEqual(self._write_and_count([]), 0)

    def test_deduplicates(self) -> None:
        entries = [
            {"word": "walk", "supersense": "verb.motion", "pos": "verb"},
            {"word": "walk", "supersense": "verb.motion", "pos": "verb"},
        ]
        count = self._write_and_count(entries)
        self.assertEqual(count, 1)
