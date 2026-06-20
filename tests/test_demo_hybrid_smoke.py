"""Smoke test for scripts/demo_hybrid_dispatch.py.

Runs the demo's main() with the template backend (no model present) and
asserts it completes without raising and that the weather turn used the
template decoder.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = ROOT / "scripts" / "demo_hybrid_dispatch.py"


def _load_demo():
    spec = importlib.util.spec_from_file_location("demo_hybrid_dispatch", DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_demo_runs_with_template_backend():
    demo = _load_demo()

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        demo.main()  # must not raise (template backend, no model)

    output = buffer.getvalue()

    # Demo ran end to end.
    assert "MELM Hybrid Dispatch Demo" in output
    assert "Demo complete." in output

    # Weather turn stayed on the template decoder.
    assert "[Turn 1] Weather" in output
    assert "[template]" in output

    # Output is ASCII-safe (no crash-prone non-ASCII like the arrow / degree sign).
    output.encode("ascii")
