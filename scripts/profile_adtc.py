"""ADTC speed/efficiency profiler harness (Issue 9).

Drives a representative ADTC turn set through the local-assistant kernel in-process
and records per-turn timing + routing + accuracy plus a process-level peak-RSS
figure. Runs as a smoke on the template backend with no model required; an optional
``--model-path`` attaches a GGUF/llama.cpp backend and records tokens/sec when the
backend exposes timing.

This produces the *harness*. The ADTC speed(30%)/efficiency(20%) evidence is only
real once this script is run on an 8 GB ARM/laptop-class target device with the GGUF
model loaded. See ``reports/README.md``.

Usage:
    PYTHONPATH=. MELM_BULK_MAX_ENTRIES=200 python scripts/profile_adtc.py
    PYTHONPATH=. python scripts/profile_adtc.py --model-path models/qwen2.5-0.5b-instruct-q4_k_m.gguf
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Allow `python scripts/profile_adtc.py` without PYTHONPATH=. set.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from melm.appliance.assistant_os_kernel import AssistantOSKernel  # noqa: E402
from melm.appliance.assistant_os_store import (  # noqa: E402
    AssistantOSStore,
    seed_class_schemas,
)
from melm.appliance.local_assistant_router import LocalAssistantProfile  # noqa: E402

REPORTS_DIR = _REPO_ROOT / "reports"


@dataclass(frozen=True)
class TurnSpec:
    """One representative ADTC turn and its accuracy expectation."""

    label: str
    utterance: str
    # Case-insensitive substring that MUST appear in the answer for the turn to
    # count as accurate. None = no accuracy assertion (timing-only turn).
    expect_substring: str | None = None


# Representative ~7-turn ADTC set. Defined inline on purpose: another task owns
# scripts/demo_adtc_7turn.py, so we do not import or share its turn list.
TURNS: tuple[TurnSpec, ...] = (
    TurnSpec("profile_setup", "My name is Ade. I live in Lagos.", "Ade"),
    TurnSpec("memory_recall", "What is my name?", "Ade"),
    TurnSpec("meal", "What should I eat today?"),
    TurnSpec("reasoning", "How many r's in strawberry?", "3"),
    TurnSpec("story", "Tell me a short story about a brave goat."),
    TurnSpec(
        "geo",
        "The car wash is 50m away, should I drive or walk?",
        "walk",
    ),
    TurnSpec("open_domain", "What is the capital of France?"),
)


@dataclass
class TurnResult:
    label: str
    utterance: str
    wall_ms: float
    route: str
    intent: str
    cloud_needed: bool
    decoder_used: str
    expected: str | None
    accurate: bool | None
    answer_preview: str
    tokens_per_sec: float | None = None


@dataclass
class PeakRss:
    bytes: int | None
    method: str  # "ru_maxrss", "tracemalloc", "psutil", or "unavailable"
    note: str = ""


@dataclass
class ProfileReport:
    hostname: str
    timestamp: str
    platform: str
    python_version: str
    model_path: str | None
    model_timing_note: str
    turns: list[TurnResult] = field(default_factory=list)
    peak_rss: PeakRss | None = None


# ---------------------------------------------------------------------------
# Peak RSS measurement (graceful degradation, never crashes)
# ---------------------------------------------------------------------------
def measure_peak_rss() -> PeakRss:
    """Best-effort process peak resident-set size.

    POSIX: ``resource.getrusage(RUSAGE_SELF).ru_maxrss``. Note the unit differs by
    OS: Linux reports kilobytes, macOS/BSD report bytes. We normalise to bytes and
    record the assumption in ``note``.

    Non-POSIX (Windows): try ``psutil`` if importable, else fall back to the
    ``tracemalloc`` peak (Python-allocations only, an under-estimate), else report
    "unavailable".
    """
    # POSIX path.
    try:
        import resource  # noqa: PLC0415  (POSIX-only module)

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        system = platform.system()
        if system == "Darwin":  # macOS/BSD report bytes
            return PeakRss(bytes=int(ru), method="ru_maxrss", note="ru_maxrss in bytes (Darwin)")
        # Linux and most other POSIX report kilobytes.
        return PeakRss(bytes=int(ru) * 1024, method="ru_maxrss", note="ru_maxrss*1024 (KB->bytes, Linux)")
    except (ImportError, AttributeError, ValueError):
        pass

    # psutil path (works on Windows, gives true process RSS).
    try:
        import psutil  # noqa: PLC0415

        rss = psutil.Process().memory_info().rss
        return PeakRss(bytes=int(rss), method="psutil", note="psutil RSS (current, not lifetime peak)")
    except Exception:  # ImportError or runtime failure  # noqa: BLE001
        pass

    # tracemalloc fallback (Python allocations only; under-estimate).
    try:
        if tracemalloc.is_tracing():
            _current, peak = tracemalloc.get_traced_memory()
            return PeakRss(
                bytes=int(peak),
                method="tracemalloc",
                note="tracemalloc peak: Python allocations only, under-estimates true RSS",
            )
    except Exception:  # noqa: BLE001
        pass

    return PeakRss(bytes=None, method="unavailable", note="no RSS source available on this platform")


# ---------------------------------------------------------------------------
# Kernel construction
# ---------------------------------------------------------------------------
def build_kernel(model_path: str | None) -> tuple[AssistantOSKernel, str, str, bool]:
    """Construct an in-process kernel.

    Returns (kernel, db_path, model_timing_note, model_attached).
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    store = AssistantOSStore(tmp.name)
    seed_class_schemas(store)

    decoder = None
    model_attached = False
    model_timing_note = "no model (template backend)"
    if model_path:
        if not Path(model_path).exists():
            model_timing_note = f"model path not found: {model_path}; using template backend"
        else:
            try:
                from melm.appliance.assistant_decoder import ConstrainedDecoder  # noqa: PLC0415

                decoder = ConstrainedDecoder(preferred="llamacpp", model_path=model_path)
                model_attached = True
                model_timing_note = "model attached; per-turn tokens/sec is best-effort word-rate"
            except Exception as exc:  # noqa: BLE001
                model_timing_note = f"failed to attach model ({exc!r}); using template backend"

    kernel = AssistantOSKernel(
        profile=LocalAssistantProfile(),
        store=store,
        decoder=decoder,
    )
    return kernel, tmp.name, model_timing_note, model_attached


# ---------------------------------------------------------------------------
# Per-turn driver
# ---------------------------------------------------------------------------
def run_turn(kernel: AssistantOSKernel, spec: TurnSpec, model_attached: bool) -> TurnResult:
    start = time.perf_counter()
    decision = kernel.handle(spec.utterance)
    wall_ms = (time.perf_counter() - start) * 1000.0

    answer = decision.answer or ""
    decoder_used = ""
    tokens_per_sec: float | None = None
    synthesis = getattr(kernel, "last_synthesis", None)
    if synthesis is not None:
        decoder_used = getattr(synthesis, "decoder_used", "") or ""

    if model_attached and decoder_used == "llamacpp" and wall_ms > 0:
        # Best-effort: word-count over wall-clock. The backend does not expose
        # native prompt/gen token timing, so this is an approximate gen rate.
        word_count = len(answer.split())
        if word_count:
            tokens_per_sec = round(word_count / (wall_ms / 1000.0), 2)

    accurate: bool | None = None
    if spec.expect_substring is not None:
        accurate = spec.expect_substring.lower() in answer.lower()

    preview = answer.replace("\n", " ").strip()
    if len(preview) > 160:
        preview = preview[:157] + "..."

    return TurnResult(
        label=spec.label,
        utterance=spec.utterance,
        wall_ms=round(wall_ms, 3),
        route=str(getattr(decision, "route", "")),
        intent=str(getattr(decision, "intent", "")),
        cloud_needed=bool(getattr(decision, "cloud_needed", False)),
        decoder_used=decoder_used or "(none)",
        expected=spec.expect_substring,
        accurate=accurate,
        answer_preview=preview,
        tokens_per_sec=tokens_per_sec,
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return round(ordered[low] + (ordered[high] - ordered[low]) * frac, 3)


def _accuracy_kn(turns: list[TurnResult]) -> tuple[int, int]:
    scored = [t for t in turns if t.accurate is not None]
    k = sum(1 for t in scored if t.accurate)
    return k, len(scored)


def _fmt_rss(rss: PeakRss | None) -> str:
    if rss is None or rss.bytes is None:
        return "unavailable"
    mib = rss.bytes / (1024 * 1024)
    return f"{mib:.1f} MiB ({rss.method})"


# ---------------------------------------------------------------------------
# Report emission
# ---------------------------------------------------------------------------
def to_json_dict(report: ProfileReport) -> dict:
    k, n = _accuracy_kn(report.turns)
    wall_values = [t.wall_ms for t in report.turns]
    return {
        "hostname": report.hostname,
        "timestamp": report.timestamp,
        "platform": report.platform,
        "python_version": report.python_version,
        "model_path": report.model_path,
        "model_timing_note": report.model_timing_note,
        "aggregate": {
            "turn_count": len(report.turns),
            "accuracy_k": k,
            "accuracy_n": n,
            "wall_ms_p50": _percentile(wall_values, 50),
            "wall_ms_p95": _percentile(wall_values, 95),
            "wall_ms_total": round(sum(wall_values), 3),
            "peak_rss_bytes": report.peak_rss.bytes if report.peak_rss else None,
            "peak_rss_method": report.peak_rss.method if report.peak_rss else "unavailable",
            "peak_rss_note": report.peak_rss.note if report.peak_rss else "",
        },
        "turns": [
            {
                "label": t.label,
                "utterance": t.utterance,
                "wall_ms": t.wall_ms,
                "route": t.route,
                "intent": t.intent,
                "cloud_needed": t.cloud_needed,
                "decoder_used": t.decoder_used,
                "expected": t.expected,
                "accurate": t.accurate,
                "tokens_per_sec": t.tokens_per_sec,
                "answer_preview": t.answer_preview,
            }
            for t in report.turns
        ],
    }


def to_markdown(report: ProfileReport) -> str:
    k, n = _accuracy_kn(report.turns)
    wall_values = [t.wall_ms for t in report.turns]
    p50 = _percentile(wall_values, 50)
    p95 = _percentile(wall_values, 95)
    total = round(sum(wall_values), 3)

    lines: list[str] = []
    lines.append("# ADTC Profile")
    lines.append("")
    lines.append(f"- Host: `{report.hostname}`")
    lines.append(f"- Timestamp: {report.timestamp}")
    lines.append(f"- Platform: {report.platform}")
    lines.append(f"- Python: {report.python_version}")
    lines.append(f"- Model: `{report.model_path or '(none — template backend)'}`")
    lines.append(f"- Model timing: {report.model_timing_note}")
    lines.append("")
    lines.append("> NOTE: A run on this developer machine is a SMOKE only. The binding")
    lines.append("> ADTC speed/efficiency gate is a run on an 8 GB ARM/laptop-class device")
    lines.append("> with the GGUF model. See `reports/README.md`.")
    lines.append("")
    lines.append("## Per-turn")
    lines.append("")
    lines.append("| Turn | Wall (ms) | Route | Intent | Cloud | Decoder | tok/s | Accurate |")
    lines.append("| --- | ---: | --- | --- | --- | --- | ---: | --- |")
    for t in report.turns:
        if t.accurate is None:
            acc = "—"
        else:
            acc = "PASS" if t.accurate else "FAIL"
        tps = f"{t.tokens_per_sec}" if t.tokens_per_sec is not None else "—"
        lines.append(
            f"| {t.label} | {t.wall_ms:.1f} | {t.route} | {t.intent} | "
            f"{'yes' if t.cloud_needed else 'no'} | {t.decoder_used} | {tps} | {acc} |"
        )
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- Turns: {len(report.turns)}")
    lines.append(f"- Accuracy: {k}/{n}")
    lines.append(f"- Wall p50: {p50 if p50 is not None else 'n/a'} ms")
    lines.append(f"- Wall p95: {p95 if p95 is not None else 'n/a'} ms")
    lines.append(f"- Wall total: {total} ms")
    lines.append(f"- Peak RSS: {_fmt_rss(report.peak_rss)}")
    if report.peak_rss and report.peak_rss.note:
        lines.append(f"- RSS caveat: {report.peak_rss.note}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ADTC speed/efficiency profiler harness.")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional GGUF/llama.cpp model path. Without it, runs the template backend smoke.",
    )
    args = parser.parse_args(argv)

    tracemalloc.start()

    kernel, db_path, model_timing_note, model_attached = build_kernel(args.model_path)

    report = ProfileReport(
        hostname=platform.node() or "unknown-host",
        timestamp=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        model_path=args.model_path,
        model_timing_note=model_timing_note,
    )

    try:
        for spec in TURNS:
            report.turns.append(run_turn(kernel, spec, model_attached))
    finally:
        try:
            if kernel.store is not None:
                kernel.store.connection.close()
        except Exception:  # noqa: BLE001
            pass
        Path(db_path).unlink(missing_ok=True)

    report.peak_rss = measure_peak_rss()
    tracemalloc.stop()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / f"adtc_profile_{report.hostname}.json"
    md_path = REPORTS_DIR / "adtc_profile.md"
    json_path.write_text(json.dumps(to_json_dict(report), indent=2), encoding="utf-8")
    md_text = to_markdown(report)
    md_path.write_text(md_text, encoding="utf-8")

    # Human summary to stdout.
    k, n = _accuracy_kn(report.turns)
    print("ADTC profiler (smoke)" if not model_attached else "ADTC profiler (model)")
    print(f"  host={report.hostname}  python={report.python_version}")
    print(f"  model: {report.model_timing_note}")
    print("")
    for t in report.turns:
        acc = "" if t.accurate is None else ("  [PASS]" if t.accurate else "  [FAIL]")
        print(
            f"  {t.label:<14} {t.wall_ms:8.1f} ms  route={t.route:<22} "
            f"cloud={'Y' if t.cloud_needed else 'N'} decoder={t.decoder_used}{acc}"
        )
    print("")
    wall_values = [t.wall_ms for t in report.turns]
    print(
        f"  p50={_percentile(wall_values, 50)} ms  p95={_percentile(wall_values, 95)} ms  "
        f"total={round(sum(wall_values), 3)} ms"
    )
    print(f"  wrote {json_path}")
    print(f"  wrote {md_path}")
    print("")
    print(
        f"PROFILE_RESULT: {len(report.turns)} turns, accuracy {k}/{n}, "
        f"peak_rss={_fmt_rss(report.peak_rss)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
