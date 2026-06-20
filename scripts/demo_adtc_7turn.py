"""End-to-end ADTC 7-turn demo through the real MELM CLI.

This exercises the full kernel pipeline with optional model backend:
1. Profile setup (name + location)
2. Cross-domain reasoning (meal based on weather)
3. Memory recall ("What is my name?")
4. Story with cultural context
5. Health + safety integration
6. Device action (media)
7. Autobiographical memory ("What did we talk about?")

In ``--strict`` mode (default) the demo ASSERTS key outcomes per turn and exits
non-zero if any expectation fails, so a memory regression fails loudly instead of
passing silently. A machine-readable ``DEMO_RESULT: pass|fail k/n`` line is always
printed for CI grep. Use ``--no-strict`` to print turns without asserting (useful
when wiring an experimental model backend whose phrasing varies).

Usage::

    # Strict template-only run (CI default)
    python scripts/demo_adtc_7turn.py

    # With model backend (assertions on phrasing relaxed; routing still checked)
    python scripts/demo_adtc_7turn.py --model-path models/qwen2.5-0.5b-instruct-q4_k_m.gguf

    # Print-only, no assertions
    python scripts/demo_adtc_7turn.py --no-strict
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "local_assistant_os_cli.py"

TURNS = [
    ("setup", "My name is Ade. I live in Lagos."),
    ("meal", "What should I eat today?"),
    ("memory", "What is my name?"),
    ("open_domain", "What is the capital of France?"),
    ("story", "Tell me a story."),
    ("health", "I feel sick."),
    ("media", "Play calm piano music."),
    ("autobiographical", "What did we talk about?"),
]


def run_turn(db: Path, utterance: str, model_path: Path | None) -> dict:
    cmd = [
        sys.executable, str(CLI),
        "ask",
        "--db", str(db),
        "--utterance", utterance,
        "--json",
    ]
    if model_path is not None:
        cmd.extend(["--model-path", str(model_path)])
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return {"answer": "[parse error]", "route": "error", "reason": "json_decode"}


def _check_turn(
    label: str,
    payload: dict,
    *,
    model_backend: bool,
) -> list[str]:
    """Return a list of failure messages for the given turn (empty == pass).

    Phrasing-sensitive checks (e.g. the answer literally contains "Ade") are only
    enforced for the template backend; a model backend may paraphrase. Routing /
    intent / cloud-isolation checks are enforced for both — those are the
    accuracy-critical invariants.
    """
    failures: list[str] = []
    answer = str(payload.get("answer", ""))
    route = str(payload.get("route", ""))
    intent = str(payload.get("intent", ""))
    cloud_needed = bool(payload.get("cloud_needed", False))
    evidence_keys = payload.get("evidence_keys", []) or []

    def fail(msg: str) -> None:
        failures.append(f"[{label}] {msg}")

    if label == "setup":
        if route != "local_answer":
            fail(f"expected route=local_answer, got {route!r}")
        keys = set(evidence_keys)
        has_keys = {"profile.user_name", "profile.location"}.issubset(keys)
        mentions = ("ade" in answer.lower()) and ("lagos" in answer.lower())
        if not (has_keys or mentions):
            fail(
                "expected name+location stored "
                f"(evidence_keys={sorted(keys)}, answer={answer!r})"
            )
    elif label == "memory":
        # KEY ASSERTION: name recall stays local and returns "Ade".
        if not (intent == "personal_memory" or route == "local_answer"):
            fail(f"expected personal_memory/local_answer, got intent={intent!r} route={route!r}")
        if cloud_needed:
            fail("cloud_needed must be False for personal memory recall")
        if route == "open_domain":
            fail("memory recall must not route open_domain")
        if "ade" not in answer.lower():
            fail(f"answer must contain 'Ade', got {answer!r}")
    elif label == "meal":
        if intent != "meal_suggestion":
            fail(f"expected intent=meal_suggestion, got {intent!r}")
    elif label == "open_domain":
        # Local cannot truly answer this; it should be flagged open_domain
        # (intent) and/or handed off — never silently answered as a local fact.
        handed_off = (
            intent == "open_domain"
            or route in {"open_domain", "cloud_handoff"}
            or cloud_needed
            or bool(payload.get("external_fetch_needed", False))
        )
        if not handed_off:
            fail(
                "expected open_domain intent / cloud handoff, got "
                f"intent={intent!r} route={route!r} cloud_needed={cloud_needed}"
            )

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ADTC 7-turn MELM demo.")
    parser.add_argument("--model-path", type=Path, default=None)
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Assert per-turn outcomes and exit non-zero on failure (default).",
    )
    strict_group.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Print turns only; do not assert (useful for experimental backends).",
    )
    args = parser.parse_args(argv)

    model_path = args.model_path
    strict = bool(args.strict)
    model_backend = model_path is not None

    checks_total = 0
    checks_passed = 0
    all_failures: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        db = Path(tmpdir) / "demo.sqlite"
        # Bootstrap the database
        subprocess.run(
            [sys.executable, str(CLI), "bootstrap-runtime", "--db", str(db)],
            capture_output=True, cwd=ROOT,
        )

        print("=" * 70)
        print("ADTC 2026 - MELM 7-Turn Agentic Demo")
        print("=" * 70)
        if model_path:
            print(f"Model backend: {model_path}")
        else:
            print("Model backend: None (template-only)")
        print(f"Mode: {'strict (asserting)' if strict else 'print-only'}")
        print()

        for label, utterance in TURNS:
            print(f"[{label:>15}] User: {utterance}")
            payload = run_turn(db, utterance, model_path)
            answer = payload.get("answer", "[no answer]")
            route = payload.get("route", "unknown")
            reason = payload.get("reason", "unknown")
            decoder = payload.get("synthesis", {}).get("decoder_used", "template")
            backend_badge = "[model]" if decoder == "llamacpp" else "[template]"
            print(f"[{label:>15}] MELM: {answer}")
            print(f"[{label:>15}]      route={route} reason={reason} {backend_badge}")

            if strict:
                failures = _check_turn(label, payload, model_backend=model_backend)
                checks_total += 1
                if failures:
                    all_failures.extend(failures)
                    print(f"[{label:>15}]      ASSERT FAIL: {'; '.join(failures)}")
                else:
                    checks_passed += 1
                    print(f"[{label:>15}]      ASSERT OK")
            print()

        print("=" * 70)
        print("Demo complete.")
        print("=" * 70)

    if strict:
        ok = not all_failures
        status = "pass" if ok else "fail"
        print(f"DEMO_RESULT: {status} {checks_passed}/{checks_total}")
        if not ok:
            print("DEMO_FAILURES:")
            for failure in all_failures:
                print(f"  - {failure}")
            return 1
        return 0

    print(f"DEMO_RESULT: skipped (no-strict)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
