"""Tests for Hook #1 (secrets.env loader) and Hook #4 (GGUF auto-detection).

See melm/appliance/provisioning.py and docs/human-friendly-NLG-pipeline.md.
"""

import os
import tempfile
import unittest
from pathlib import Path


class LoadSecretsEnvTests(unittest.TestCase):
    def setUp(self):
        self._cleanup: list[str] = []

    def tearDown(self):
        for key in self._cleanup:
            os.environ.pop(key, None)

    def _tmpfile(self, content: str) -> Path:
        fd, name = tempfile.mkstemp(suffix=".env")
        try:
            os.write(fd, content.encode())
        finally:
            os.close(fd)
        self.addCleanup(lambda: Path(name).unlink(missing_ok=True))
        return Path(name)

    def test_loads_key_value_pairs(self):
        from melm.appliance.provisioning import load_secrets_env

        path = self._tmpfile("MELM_TEST_FOO=bar\nMELM_TEST_BAZ=qux\n")
        self._cleanup += ["MELM_TEST_FOO", "MELM_TEST_BAZ"]
        count = load_secrets_env(path)
        self.assertEqual(count, 2)
        self.assertEqual(os.environ["MELM_TEST_FOO"], "bar")
        self.assertEqual(os.environ["MELM_TEST_BAZ"], "qux")

    def test_skips_existing_keys(self):
        from melm.appliance.provisioning import load_secrets_env

        os.environ["MELM_TEST_EXISTING"] = "original"
        self._cleanup.append("MELM_TEST_EXISTING")
        path = self._tmpfile("MELM_TEST_EXISTING=overridden\n")
        count = load_secrets_env(path)
        self.assertEqual(count, 0)
        self.assertEqual(os.environ["MELM_TEST_EXISTING"], "original")

    def test_skips_comment_lines(self):
        from melm.appliance.provisioning import load_secrets_env

        path = self._tmpfile("# MELM_TEST_COMMENT=should_not_load\n")
        self._cleanup.append("MELM_TEST_COMMENT")
        count = load_secrets_env(path)
        self.assertEqual(count, 0)
        self.assertNotIn("MELM_TEST_COMMENT", os.environ)

    def test_skips_blank_lines(self):
        from melm.appliance.provisioning import load_secrets_env

        path = self._tmpfile("\n\n   \nMELM_TEST_BLANK=ok\n\n")
        self._cleanup.append("MELM_TEST_BLANK")
        count = load_secrets_env(path)
        self.assertEqual(count, 1)
        self.assertEqual(os.environ["MELM_TEST_BLANK"], "ok")

    def test_absent_file_returns_zero_no_crash(self):
        from melm.appliance.provisioning import load_secrets_env

        missing = Path(tempfile.gettempdir()) / "__melm_no_such_secrets__.env"
        count = load_secrets_env(missing)
        self.assertEqual(count, 0)

    def test_value_with_equals_sign_preserved(self):
        from melm.appliance.provisioning import load_secrets_env

        path = self._tmpfile("MELM_TEST_EQ=abc=def=ghi\n")
        self._cleanup.append("MELM_TEST_EQ")
        load_secrets_env(path)
        self.assertEqual(os.environ["MELM_TEST_EQ"], "abc=def=ghi")


class ResolveGgufModelPathTests(unittest.TestCase):
    def setUp(self):
        self._orig_env = os.environ.pop("MELM_MODELS_DIR", None)

    def tearDown(self):
        if self._orig_env is not None:
            os.environ["MELM_MODELS_DIR"] = self._orig_env
        else:
            os.environ.pop("MELM_MODELS_DIR", None)

    def test_returns_none_when_file_absent(self):
        from melm.appliance.provisioning import resolve_gguf_model_path

        with tempfile.TemporaryDirectory() as d:
            os.environ["MELM_MODELS_DIR"] = d
            result = resolve_gguf_model_path()
        self.assertIsNone(result)

    def test_returns_path_when_gguf_present(self):
        from melm.appliance.provisioning import _QWEN_FILENAME, resolve_gguf_model_path

        with tempfile.TemporaryDirectory() as d:
            gguf = Path(d) / _QWEN_FILENAME
            gguf.write_bytes(b"GGUF_FAKE")
            os.environ["MELM_MODELS_DIR"] = d
            result = resolve_gguf_model_path()

        self.assertIsNotNone(result)
        self.assertEqual(result.name, _QWEN_FILENAME)

    def test_env_override_takes_priority(self):
        from melm.appliance.provisioning import _QWEN_FILENAME, resolve_gguf_model_path

        with tempfile.TemporaryDirectory() as d:
            gguf = Path(d) / _QWEN_FILENAME
            gguf.write_bytes(b"GGUF_FAKE")
            os.environ["MELM_MODELS_DIR"] = d
            result = resolve_gguf_model_path()
            self.assertIsNotNone(result)
            self.assertTrue(str(result).startswith(d))

    def test_returns_none_when_models_dir_missing(self):
        from melm.appliance.provisioning import resolve_gguf_model_path

        os.environ["MELM_MODELS_DIR"] = "/nonexistent/melm/models"
        result = resolve_gguf_model_path()
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
