"""Smoke test for the ADTC 7-turn demo's strict-mode assertions.

Runs ``scripts/demo_adtc_7turn.py`` as a subprocess with the template backend in
strict mode and asserts it exits 0 with a ``DEMO_RESULT: pass`` line. This guards
the demo against regressing into silently passing when memory is wrong.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "scripts" / "demo_adtc_7turn.py"


def test_demo_strict_template_backend_passes() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["MELM_BULK_MAX_ENTRIES"] = "200"

    result = subprocess.run(
        [sys.executable, str(DEMO)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
        env=env,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"demo exited {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "DEMO_RESULT: pass" in result.stdout, (
        f"missing DEMO_RESULT: pass\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
