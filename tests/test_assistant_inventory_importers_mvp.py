import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from melm.appliance import (
    InternetArchiveSearchMetadataImporter,
    LocalAssistantProfile,
    LocalMediaInventoryAdapter,
    ProjectGutenbergCatalogImporter,
    media_items_to_inventory_rows,
    story_items_to_inventory_rows,
)


CLI = Path("scripts/local_assistant_os_cli.py")
GUTENBERG_SAMPLE = Path("benchmarks/sample_gutenberg_catalog.csv")
IA_SAMPLE = Path("benchmarks/sample_internet_archive_search.json")
MEDIA_MANIFEST = Path("benchmarks/local_media_manifest.json")


class AssistantInventoryImportersMvpTests(unittest.TestCase):
    def test_gutenberg_csv_importer_parses_public_domain_story_metadata(self) -> None:
        profile = _profile()

        result = ProjectGutenbergCatalogImporter().import_csv_path(
            GUTENBERG_SAMPLE,
            profile,
            limit=3,
        )

        self.assertFalse(result.network_used)
        self.assertEqual(result.source_count, 4)
        self.assertEqual(result.selected_count, 2)
        self.assertEqual(result.rejected_count, 2)
        self.assertEqual(result.items[0].item_id, "pg_1001")
        self.assertEqual(result.items[0].source, "project_gutenberg_catalog_metadata")
        self.assertEqual(result.items[0].source_url, "https://www.gutenberg.org/ebooks/1001")
        self.assertEqual(result.items[0].license, "public_domain_catalog_metadata")
        self.assertIn("folktale", result.items[0].topics)
        self.assertIn("Yoruba", result.items[0].cultures)

        rows = story_items_to_inventory_rows(result.items, profile=profile)
        self.assertIn("topics", rows[0]["payload"])
        self.assertIn("cultures", rows[0]["payload"])
        self.assertGreater(rows[0]["payload"]["quality_score"], 0)
        self.assertIn("local_fit_score", rows[0]["payload"])
        self.assertGreater(rows[0]["payload"]["metadata_quality"], 0)

    def test_gutenberg_live_import_retries_transient_fetch_failure(self) -> None:
        profile = _profile()
        csv_text = (
            "Text#,Title,Language,Type,Authors,Subjects,Bookshelves\n"
            "2001,Yoruba Fairy Tales for Children,en,Text,Author,"
            "Folklore -- Nigeria; Children -- Folklore,Children\n"
        )

        with patch(
            "melm.appliance.assistant_inventory.urlopen",
            side_effect=[TimeoutError("temporary"), _FakeResponse(csv_text.encode("utf-8"))],
        ) as opener:
            result = ProjectGutenbergCatalogImporter("https://example.test/catalog.csv").import_metadata(
                profile,
                limit=2,
                timeout=0.01,
                backoff_seconds=0.0,
            )

        self.assertTrue(result.network_used)
        self.assertEqual(opener.call_count, 2)
        self.assertEqual(result.selected_count, 1)
        self.assertEqual(result.items[0].item_id, "pg_2001")
        self.assertEqual(result.observability["fetch_attempts"], 2)
        self.assertEqual(result.observability["max_attempts"], 2)

    def test_story_import_ranking_dedupes_duplicate_titles_before_selection(self) -> None:
        profile = _profile()
        csv_text = (
            "Text#,Title,Language,Type,Authors,Subjects,Bookshelves\n"
            "2001,Yoruba Fairy Tales for Children,en,Text,Author,"
            "Folklore -- Nigeria; Children -- Folklore,Children\n"
            "2002,The Yoruba Fairy Tales for Children,en,Text,Author,"
            "Folklore -- Nigeria; Children -- Folklore,Children\n"
        )

        result = ProjectGutenbergCatalogImporter().import_csv_text(csv_text, profile, limit=3)

        self.assertEqual(result.source_count, 2)
        self.assertEqual(result.selected_count, 1)
        self.assertEqual(result.items[0].title, "Yoruba Fairy Tales for Children")
        self.assertEqual(result.observability["duplicate_rejected_count"], 1)
        self.assertEqual(result.observability["quality_floor"], 0.5)

    def test_internet_archive_importer_parses_scrape_metadata(self) -> None:
        profile = _profile()

        result = InternetArchiveSearchMetadataImporter().import_json_path(
            IA_SAMPLE,
            profile,
            limit=3,
        )

        self.assertFalse(result.network_used)
        self.assertEqual(result.source_count, 3)
        self.assertEqual(result.selected_count, 2)
        self.assertEqual(result.rejected_count, 1)
        self.assertEqual(result.items[0].item_id, "ia_rainmapbedtimestory00test")
        self.assertEqual(result.items[0].source, "internet_archive_item_search_and_metadata")
        self.assertEqual(result.items[0].source_url, "https://archive.org/details/rainmapbedtimestory00test")
        self.assertIn("bedtime", result.items[0].topics)
        self.assertEqual(result.observability["candidate_count"], 2)
        self.assertEqual(result.observability["quality_rejected_count"], 0)

    def test_internet_archive_live_import_walks_cursor_pages_with_rate_limit_observability(self) -> None:
        profile = _profile()
        page_one = {
            "items": [
                {
                    "identifier": "moonstart01",
                    "title": "Moon Start Fairy Tales",
                    "subject": ["Children's stories", "Fairy tales"],
                    "language": "eng",
                    "collection": ["gutenberg"],
                }
            ],
            "cursor": "page-two",
        }
        page_two = {
            "items": [
                {
                    "identifier": "yorubarain02",
                    "title": "Yoruba Rain Bedtime Tales",
                    "subject": ["Folklore -- Nigeria", "Yoruba", "Bedtime stories"],
                    "language": "eng",
                    "collection": ["gutenberg"],
                },
                {
                    "identifier": "riveradventure03",
                    "title": "River Adventure Stories",
                    "subject": ["Adventure stories", "Children"],
                    "language": "eng",
                    "collection": ["gutenberg"],
                },
            ],
            "cursor": "",
        }

        with (
            patch(
                "melm.appliance.assistant_inventory.urlopen",
                side_effect=[
                    _FakeResponse(json.dumps(page_one).encode("utf-8")),
                    _FakeResponse(json.dumps(page_two).encode("utf-8")),
                ],
            ) as opener,
            patch("melm.appliance.assistant_inventory.sleep") as sleeper,
        ):
            result = InternetArchiveSearchMetadataImporter(
                endpoint="https://archive.example.test/scrape"
            ).import_metadata(
                profile,
                limit=3,
                max_source_bytes=20_000,
                timeout=0.01,
                backoff_seconds=0.0,
                page_size=2,
                max_pages=3,
                rate_limit_delay_seconds=2.0,
            )

        first_url = opener.call_args_list[0].args[0].full_url
        second_url = opener.call_args_list[1].args[0].full_url
        self.assertEqual(opener.call_count, 2)
        sleeper.assert_called_once_with(2.0)
        self.assertIn("count=100", first_url)
        self.assertNotIn("cursor=", first_url)
        self.assertIn("cursor=page-two", second_url)
        self.assertTrue(result.network_used)
        self.assertEqual(result.source_count, 3)
        self.assertEqual(result.selected_count, 3)
        self.assertEqual(result.observability["page_count"], 2)
        self.assertEqual(result.observability["page_size"], 100)
        self.assertEqual(result.observability["max_pages"], 3)
        self.assertEqual(result.observability["page_item_counts"], [1, 2])
        self.assertEqual(result.observability["cursors_seen"], ["", "page-two"])
        self.assertEqual(result.observability["fetch_attempts_total"], 2)
        self.assertEqual(result.observability["rate_limit_sleep_count"], 1)
        self.assertEqual(result.observability["rate_limit_delay_total_seconds"], 2.0)
        self.assertEqual(result.observability["next_cursor"], "")
        self.assertFalse(result.observability["byte_budget_exhausted"])
        self.assertEqual(result.items[0].item_id, "ia_yorubarain02")

    def test_cli_import_stories_writes_sample_metadata_to_sqlite_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            imported = _run_cli(
                "import-stories",
                "--db",
                str(db),
                "--cold-start",
                "--source",
                "both",
                "--gutenberg-csv",
                str(GUTENBERG_SAMPLE),
                "--internet-archive-json",
                str(IA_SAMPLE),
                "--limit",
                "3",
                "--json",
            )
            story = _run_cli("ask", "--db", str(db), "--utterance", "Tell me a story.", "--json")

            self.assertEqual(imported["imported_items"], 4)
            self.assertEqual(imported["counts"]["inventories"], 4)
            self.assertEqual(imported["results"][0]["source"], "project_gutenberg_catalog_csv")
            self.assertEqual(imported["results"][1]["source"], "internet_archive_search_metadata")
            self.assertEqual(imported["results"][0]["observability"]["candidate_count"], 2)
            self.assertEqual(imported["results"][1]["observability"]["candidate_count"], 2)
            self.assertEqual(story["route"], "local_answer")
            self.assertTrue(story["synthesis"]["applied"])
            self.assertTrue(story["synthesis"]["citations"][0].startswith("story_models."))

    def test_local_media_manifest_importer_parses_metadata_rows(self) -> None:
        result = LocalMediaInventoryAdapter(MEDIA_MANIFEST).import_manifest(
            _profile(),
            limit=2,
        )

        self.assertFalse(result.network_used)
        self.assertEqual(result.source, "local_media_manifest")
        self.assertEqual(result.source_count, 3)
        self.assertEqual(result.selected_count, 2)
        by_id = {item.item_id: item for item in result.items}
        self.assertIn("calm piano", by_id)
        self.assertEqual(by_id["calm piano"].kind, "audio")
        self.assertIn("piano", by_id["calm piano"].tags)
        self.assertFalse(by_id["calm piano"].path_exists)
        self.assertEqual(result.observability["missing_file_count"], 3)

        rows = media_items_to_inventory_rows(result.items)
        self.assertEqual(rows[0]["kind"], "media")
        self.assertIn(rows[0]["payload"]["title"], {"Calm Piano", "Rain Sounds"})
        self.assertEqual(rows[0]["source"], "local_media_manifest")
        self.assertEqual(rows[0]["license"], "local_device")
        self.assertIn("resolved_path", rows[0]["payload"]["metadata"])

    def test_local_media_directory_importer_scans_supported_device_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Focus Piano.mp3").write_bytes(b"audio")
            (root / "notes.txt").write_text("not media", encoding="utf-8")

            result = LocalMediaInventoryAdapter().import_directory(root, _profile(), limit=4)

            self.assertEqual(result.source, "local_media_directory")
            self.assertEqual(result.source_count, 1)
            self.assertEqual(result.selected_count, 1)
            self.assertEqual(result.items[0].item_id, "focus piano")
            self.assertEqual(result.items[0].path, "Focus Piano.mp3")
            self.assertTrue(result.items[0].path_exists)
            self.assertEqual(result.items[0].metadata["resolved_path"], str(root / "Focus Piano.mp3"))

    def test_cli_import_media_writes_manifest_to_sqlite_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "assistant.sqlite"
            imported = _run_cli(
                "import-media",
                "--db",
                str(db),
                "--cold-start",
                "--manifest",
                str(MEDIA_MANIFEST),
                "--limit",
                "2",
                "--json",
            )
            media = _run_cli(
                "ask",
                "--db",
                str(db),
                "--utterance",
                "Play calm piano.",
                "--cold-start",
                "--json",
            )

            self.assertEqual(imported["imported_items"], 2)
            self.assertEqual(imported["counts"]["inventories"], 2)
            self.assertEqual(imported["results"][0]["source"], "local_media_manifest")
            self.assertEqual(media["route"], "device_action")
            self.assertEqual(media["reason"], "local_media_action")


def _profile() -> LocalAssistantProfile:
    return LocalAssistantProfile(
        age=7,
        location="Lagos",
        culture="Yoruba",
        story_models={},
        preferences={"story_theme": "folktale bedtime rain adventure"},
    )


def _run_cli(*args: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(CLI), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        return self.payload


if __name__ == "__main__":
    unittest.main()
