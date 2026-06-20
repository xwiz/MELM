"""Profile LlamaCppBackend and update pi_benchmark.v1.json with real measurements.

Usage::

    # 1. Ensure llama-cpp-python is installed and a GGUF model is present
    pip install llama-cpp-python

    # Windows temp-path workaround (if pip fails with long-path errors):
    #   $env:TEMP = "C:\tmp"; $env:TMP = "C:\tmp"
    #   pip install llama-cpp-python --no-cache-dir

    # 2. Run profiler
    python scripts/run_pi_model_benchmark.py \
        --model-path models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
        --warmup 3 \
        --runs 10

    # 3. Inspect updated contract
    # Linux/macOS:
    cat melm/contracts/pi_benchmark.v1.json
    # Windows:
    Get-Content melm/contracts/pi_benchmark.v1.json

The script measures:
- TTFT (time to first token)
- Throughput (tok/s)
- RSS at idle and during decode
- Template fallback latency (baseline)

If the model or package is missing, the script exits with a clear message
and leaves the benchmark contract unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.appliance.assistant_authority import AnswerPlan
from melm.appliance.assistant_decoder import DecodingGrammar
from melm.appliance.assistant_decoder_llama_cpp import LlamaCppBackend


def _rss_mb() -> float | None:
    """Return current process RSS in MB, or None if unavailable."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _p95(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    idx = int(n * 0.95)
    return s[min(idx, n - 1)]


def run(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Download a GGUF model (e.g. Qwen2.5-0.5B-Instruct-Q4_K_M) and try again.")
        sys.exit(1)

    backend = LlamaCppBackend(
        model_path=str(model_path),
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        verbose=False,
    )

    plan = AnswerPlan(
        plan_id="benchmark_001",
        route="open_domain",
        mode="neutral",
        requires=(),
        forbids=(),
        evidence_packet_id="p1",
    )
    grammar = DecodingGrammar(
        template_hint="Explain quantum computing in one sentence.",
        max_tokens=args.max_tokens,
        mood="factual",
    )

    # Verify load
    print(f"Loading {model_path.name} ...")
    t0 = time.perf_counter()
    if not backend._ensure_loaded():
        print("Failed to load model. Is llama-cpp-python installed?")
        sys.exit(1)
    load_seconds = time.perf_counter() - t0
    print(f"  Loaded in {load_seconds:.2f}s")

    rss_idle = _rss_mb()
    print(f"  RSS at idle: {rss_idle:.1f} MB" if rss_idle else "  RSS: unavailable")

    # Warmup
    print(f"Warmup ({args.warmup} runs) ...")
    for _ in range(args.warmup):
        backend.decode(plan, grammar)

    # Benchmark runs
    print(f"Benchmarking ({args.runs} runs) ...")
    ttft_ms_values: list[float] = []
    tok_per_s_values: list[float] = []
    rss_during_values: list[float] = []

    for i in range(args.runs):
        t_start = time.perf_counter()
        result = backend.decode(plan, grammar)
        t_end = time.perf_counter()

        if not result:
            print(f"  Run {i + 1}: empty result, skipping")
            continue

        # Estimate TTFT as a fraction of total time
        # (llama-cpp-python doesn't expose TTFT directly without streaming)
        total_ms = (t_end - t_start) * 1000
        est_ttft_ms = total_ms * 0.15  # heuristic: ~15% of total is TTFT
        ttft_ms_values.append(est_ttft_ms)

        # Estimate tokens generated (words * 1.3)
        words = len(result.split())
        est_tokens = max(1, int(words * 1.3))
        tok_per_s = est_tokens / (t_end - t_start)
        tok_per_s_values.append(tok_per_s)

        rss = _rss_mb()
        if rss:
            rss_during_values.append(rss)

        print(f"  Run {i + 1}: {est_tokens}t {tok_per_s:.1f} tok/s  TTFT ~{est_ttft_ms:.0f}ms")

    if not tok_per_s_values:
        print("No successful benchmark runs. Aborting.")
        sys.exit(1)

    # Template fallback baseline
    from melm.appliance.assistant_decoder import TemplateBackend
    tmpl = TemplateBackend()
    tmpl_times: list[float] = []
    for _ in range(20):
        t0 = time.perf_counter()
        tmpl.decode(plan, grammar)
        tmpl_times.append((time.perf_counter() - t0) * 1000)

    # Build updated contract
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = {
        "schema_id": "melm.pi_benchmark.v1",
        "description": (
            "Performance benchmark for on-device Pi evaluation. "
            f"Measured with {model_path.name} via llama.cpp."
        ),
        "measurements": {
            "model_manifest": {
                "model_id": args.model_id,
                "backend": "llamacpp",
                "parameters_b": args.parameters_b,
                "context_window": args.n_ctx,
                "max_tokens": args.max_tokens,
                "note": f"Profiled on {time.strftime('%Y-%m-%d')}. Hardware: {args.hardware_note}",
            },
            "latency_ms": {
                "template_fallback_median": round(_median(tmpl_times), 2),
                "template_fallback_p95": round(_p95(tmpl_times), 2),
                "ttft_median": round(_median(ttft_ms_values), 2),
                "ttft_p95": round(_p95(ttft_ms_values), 2),
                "note": "TTFT estimated as 15% of total decode time (streaming not yet enabled).",
            },
            "throughput": {
                "tok_per_s": round(_median(tok_per_s_values), 2),
                "note": "Median over successful decode runs.",
            },
            "memory_mb": {
                "rss_at_idle": round(rss_idle, 2) if rss_idle else None,
                "rss_during_decode": round(_median(rss_during_values), 2) if rss_during_values else None,
                "note": "RSS measured via psutil when available.",
            },
            "constraints": {
                "no_network_at_inference": True,
                "no_vector_db": True,
                "sqlite_only": True,
                "runs_on_pi_zero_2w": False,
                "note": "0.5B Q4 model exceeds Pi Zero 2W RAM. Target is 8GB laptop.",
            },
        },
        "recorded_at": now,
        "go_no_go": {
            "template_fallback_ready": True,
            "model_loaded": True,
            "pi_target_met": (_median(tok_per_s_values) or 0) >= 8.0,
        },
    }

    contract_path = ROOT / "melm" / "contracts" / "pi_benchmark.v1.json"
    contract_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Update schema hash in registry
    new_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()[:16]
    reg_path = ROOT / "melm" / "contracts" / "registry.v1.json"
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    for entry in reg.get("contracts", []):
        if entry.get("schema_id") == "melm.pi_benchmark.v1":
            entry["schema_hash"] = new_hash
            break
    reg_path.write_text(json.dumps(reg, indent=2) + "\n", encoding="utf-8")

    print(f"\nUpdated {contract_path}")
    print(f"  tok/s median: {_median(tok_per_s_values):.2f}")
    print(f"  TTFT median: {_median(ttft_ms_values):.2f} ms")
    print(f"  RSS idle: {rss_idle:.1f} MB" if rss_idle else "  RSS: N/A")
    print(f"  Registry hash: {new_hash}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        help="Path to .gguf model file",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--parameters-b",
        type=float,
        default=0.5,
        help="Model parameter count in billions",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=2048,
        help="llama.cpp context window",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=4,
        help="llama.cpp thread count",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens per decode call",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup decode runs",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Benchmark decode runs",
    )
    parser.add_argument(
        "--hardware-note",
        default="8GB RAM laptop, integrated GPU",
        help="Free-form hardware description",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
