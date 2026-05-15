"""Local MELM SLM Appliance CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.appliance import MelmAppliance, MemoryRecord
from melm.benchmarks import LOCOMO_URL, load_locomo_public_memory_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-locomo", help="Build appliance memory JSONL from LoCoMo.")
    build.add_argument("--dataset", default="local_data/locomo10.json")
    build.add_argument("--download", action="store_true")
    build.add_argument("--out", default="artifacts/melm_appliance/locomo_memory.jsonl")

    ask = subparsers.add_parser("ask", help="Ask the local appliance a question.")
    ask.add_argument("--memory", required=True)
    ask.add_argument("--question", required=True)
    ask.add_argument("--k", type=int, default=5)
    ask.add_argument("--token-budget", type=int, default=1200)
    ask.add_argument("--json", action="store_true")

    serve = subparsers.add_parser("serve", help="Serve a local JSON API.")
    serve.add_argument("--memory", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--k", type=int, default=5)
    serve.add_argument("--token-budget", type=int, default=1200)

    args = parser.parse_args()
    if args.command == "build-locomo":
        _build_locomo(args)
    elif args.command == "ask":
        _ask(args)
    elif args.command == "serve":
        _serve(args)


def _build_locomo(args) -> None:
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        if not args.download:
            raise SystemExit(f"{dataset_path} does not exist. Rerun with --download.")
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(LOCOMO_URL, dataset_path)

    benchmark = load_locomo_public_memory_benchmark(dataset_path)
    records: list[MemoryRecord] = []
    for document in benchmark.documents:
        records.append(
            MemoryRecord(
                memory_id=document.doc_id,
                text=document.raw_text,
                kind="locomo_session",
                created_at=document.date_time,
                metadata={
                    "sample_id": document.sample_id,
                    "session_id": document.session_id,
                    "observation": document.observation,
                    "summary": document.session_summary,
                    "event_summary": document.event_summary,
                },
            )
        )
    appliance = MelmAppliance(records)
    appliance.save_jsonl(args.out)
    print("MELM appliance LoCoMo memory store")
    print(f"- source={dataset_path}")
    print(f"- records={len(records)}")
    print(f"- out={args.out}")


def _ask(args) -> None:
    appliance = MelmAppliance.load_jsonl(args.memory)
    answer = appliance.answer(
        args.question,
        k=args.k,
        token_budget=args.token_budget,
    )
    payload = _answer_payload(answer)
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    print(answer.answer)
    print(f"confidence={answer.confidence:.2f}")
    print(f"citations={', '.join(answer.citations)}")
    print(f"context_tokens={answer.context_tokens}")


def _serve(args) -> None:
    appliance = MelmAppliance.load_jsonl(args.memory)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self.send_error(404)
                return
            _send_json(self, {"ok": True, "records": len(appliance.records)})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/query":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            try:
                request = json.loads(body.decode("utf-8"))
                question = str(request["question"])
            except (KeyError, json.JSONDecodeError):
                self.send_error(400, "POST JSON must include question")
                return
            answer = appliance.answer(
                question,
                k=int(request.get("k", args.k)),
                token_budget=int(request.get("token_budget", args.token_budget)),
            )
            _send_json(self, _answer_payload(answer))

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"MELM appliance serving on http://{args.host}:{args.port}")
    print("- GET /health")
    print("- POST /query {\"question\": \"...\"}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _answer_payload(answer) -> dict:
    return {
        "question": answer.question,
        "answer": answer.answer,
        "confidence": answer.confidence,
        "citations": list(answer.citations),
        "retrieved_ids": list(answer.retrieved_ids),
        "context_tokens": answer.context_tokens,
    }


def _send_json(handler: BaseHTTPRequestHandler, payload: dict) -> None:
    encoded = json.dumps(payload).encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


if __name__ == "__main__":
    main()
