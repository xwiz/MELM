"""Throwaway perf harness for the sentience/mood + routing hot path.

Answers two questions:
  1. Does per-turn routing cost stay flat as a conversation grows? (O(1)/turn?)
  2. Do the store repetition-count queries scale with history? (the O(L^2) risk)
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from melm.appliance.local_assistant_router import OnDeviceAssistantRouter, LocalAssistantProfile
from melm.appliance.assistant_os_store import AssistantOSStore


def _t() -> float:
    return time.perf_counter()


def bench_router_hotpath() -> None:
    print("\n=== 1. Router hot-path: per-turn latency as conversation grows ===")
    router = OnDeviceAssistantRouter(LocalAssistantProfile())
    utterances = [
        "Who are you?", "What's the weather?", "Tell me a story.",
        "I'm so happy today!", "you are useless", "I smell smoke",
        "call mom", "what should I eat", "I'm sorry", "hello",
    ]
    # warm up (first call pays import/contract-load cost)
    router.handle("hello")

    buckets = [10, 50, 100, 200, 400]
    timings: list[float] = []
    turn = 0
    measured: dict[int, float] = {}
    target = max(buckets)
    while turn < target:
        u = utterances[turn % len(utterances)]
        s = _t()
        router.handle(u)
        timings.append((_t() - s) * 1000.0)
        turn += 1
        if turn in buckets:
            window = timings[-min(50, len(timings)):]
            measured[turn] = sum(window) / len(window)

    base = measured[buckets[0]]
    print(f"{'turn#':>8} {'ms/turn(avg last50)':>22} {'x vs turn10':>12}")
    for b in buckets:
        print(f"{b:>8} {measured[b]:>22.3f} {measured[b]/base:>11.2f}x")
    print("  -> flat ratio (~1x) = O(1) per turn = O(L) per conversation. Good.")


def _seed_events(store: AssistantOSStore, n: int, session_id: str, intent: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for i in range(n):
        rows.append((
            f"ev_{uuid.uuid4().hex[:12]}", session_id, "", "",
            f"utterance number {i}", intent, "local_answer", "reason",
            "answer", 0, 0, 0, 1, "[]", now,
        ))
    store.connection.executemany(
        """INSERT INTO events
           (event_id, session_id, previous_event_id, next_event_id, utterance,
            intent, route, reason, answer, cloud_needed, external_fetch_needed,
            device_action, local_memory_used, evidence_keys_json, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    store.connection.commit()


def bench_store_count_queries() -> None:
    print("\n=== 2. Store repetition-count query: latency vs session history size ===")
    sizes = [100, 1000, 5000, 10000]
    sid = "bench_session"
    intent = "social_greeting"
    print(f"{'events':>8} {'count_intent ms':>18} {'rapid_window ms':>18}")
    base_intent = None
    for n in sizes:
        store = AssistantOSStore(":memory:")
        _seed_events(store, n, sid, intent)
        reps = 200
        s = _t()
        for _ in range(reps):
            store.count_intent_occurrences_in_session(intent, sid)
        ci = (_t() - s) / reps * 1000.0
        s = _t()
        for _ in range(reps):
            store.count_intents_rapid_window(intent, sid, 30)
        rw = (_t() - s) / reps * 1000.0
        if base_intent is None:
            base_intent = ci
        print(f"{n:>8} {ci:>18.4f} {rw:>18.4f}")
        store.connection.close()
    print("  -> if count_intent grows ~linearly with events, each turn is O(history)")
    print("     => whole conversation is O(L^2). That's the sparse-attention target.")


def bench_parse_breakdown() -> None:
    print("\n=== 3. Where does a turn's time go? (parse vs full handle) ===")
    router = OnDeviceAssistantRouter(LocalAssistantProfile())
    router.handle("hello")  # warm
    reps = 200
    s = _t()
    for _ in range(reps):
        router._build_parse_bundle("what should I eat tomorrow")
    parse_ms = (_t() - s) / reps * 1000.0
    s = _t()
    for _ in range(reps):
        router.handle("what should I eat tomorrow")
    full_ms = (_t() - s) / reps * 1000.0
    print(f"  parse bundle (adapter+atomize): {parse_ms:.3f} ms")
    print(f"  full handle (parse+route+mood): {full_ms:.3f} ms")
    print(f"  routing+mood overhead:          {full_ms - parse_ms:.3f} ms")


if __name__ == "__main__":
    bench_router_hotpath()
    bench_store_count_queries()
    bench_parse_breakdown()
