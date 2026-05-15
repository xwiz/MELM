import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from melm.training import (
    preflight_small_model_stage,
    small_model_spec_from_mapping,
)


class SmallModelPreflightTests(unittest.TestCase):
    def _spec(self, root: Path):
        (root / "reports").mkdir()
        (root / "reports" / "manifest.json").write_text("{}", encoding="utf-8")
        (root / "reports" / "gate.json").write_text("{}", encoding="utf-8")
        (root / "reports" / "proxy.json").write_text("{}", encoding="utf-8")
        return small_model_spec_from_mapping(
            {
                "name": "stage",
                "manifest": "reports/manifest.json",
                "source_gate": "reports/gate.json",
                "source_proxy_decision": "reports/proxy.json",
                "tokenizers": ["hf_bpe", "tiered_morph_unigram"],
                "candidate": "tiered_morph_unigram",
                "seeds": [13],
                "max_train_bytes": 1000,
                "max_validation_bytes": 500,
                "steps": 2,
                "sequence_length": 8,
                "embedding_dim": 16,
                "layers": 1,
                "heads": 4,
                "batch_size": 2,
                "max_vocab_size": 64,
                "tokenizer_vocab_size": 64,
                "learning_rate": 0.001,
                "checkpoint_seed": 13,
                "artifact_root": "artifacts/stage",
                "report_prefix": "reports/stage",
            }
        )

    def test_preflight_passes_with_files_disk_and_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = preflight_small_model_stage(
                self._spec(root),
                root=root,
                disk_usage=lambda _path: SimpleNamespace(free=10_000_000_000),
                device_probe=lambda: {
                    "torch_available": True,
                    "cuda_available": True,
                    "devices": [{"index": 0, "name": "test gpu", "total_memory": 8_000_000_000}],
                },
            )

        self.assertEqual(report.status, "pass")
        self.assertTrue(report.cuda_available)

    def test_preflight_fails_on_low_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = preflight_small_model_stage(
                self._spec(root),
                root=root,
                min_free_disk_bytes=2_000_000_000,
                disk_usage=lambda _path: SimpleNamespace(free=1),
                device_probe=lambda: {
                    "torch_available": True,
                    "cuda_available": True,
                    "devices": [{"index": 0, "name": "test gpu", "total_memory": 8_000_000_000}],
                },
            )

        self.assertEqual(report.status, "fail")
        self.assertIn("disk", {check.name for check in report.checks if check.status == "fail"})

    def test_preflight_warns_when_auto_device_has_no_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = preflight_small_model_stage(
                self._spec(root),
                root=root,
                disk_usage=lambda _path: SimpleNamespace(free=10_000_000_000),
                device_probe=lambda: {
                    "torch_available": True,
                    "cuda_available": False,
                    "devices": [],
                },
            )

        self.assertEqual(report.status, "warn")


if __name__ == "__main__":
    unittest.main()
