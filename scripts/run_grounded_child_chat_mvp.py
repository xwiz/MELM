"""Run the bounded grounded child-chat MVP transcript."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from melm.appliance import (
    FiveYearOldGroundedChat,
    compare_assistant_mvp_directions,
    run_assistant_kernel_learning_probe,
    run_child_capability_probe,
    run_child_context_budget_probe,
    run_realistic_assistant_lifecycle_probe,
)


def main() -> None:
    chat = FiveYearOldGroundedChat()
    transcript = [
        "Maya opened the blue box.",
        "Maya put the red block in the blue box.",
        "Leo moved the red block to the green basket.",
        "Where is the red block?",
        "Did Maya move the red block to the green basket?",
        "Open the blue box.",
        "Maya put the silver train in the blue box.",
    ]

    for text in transcript:
        response = chat.handle(text)
        print(f"USER: {text}")
        print(f"STATUS: {response.status}")
        print(f"ROUTE: {response.frame.route}")
        print(f"SCORE: parse={response.parse_score}, complexity={response.complexity_score}")
        if response.frame.rejection_code:
            print(f"REJECT: {response.frame.rejection_code}")
        if response.evidence_event_ids:
            print(f"EVIDENCE: {', '.join(response.evidence_event_ids)}")
        if response.slm_input is not None:
            print(f"MODEL: {response.slm_input.model_family}")
            print(f"SSM: {response.slm_input.ssm_state.state_hash}")
            print(f"ATTENDS: {', '.join(response.slm_input.attended_evidence_event_ids) or '(none)'}")
        if response.trace:
            print("TRACE: " + " -> ".join(f"{stage.stage}:{stage.status}" for stage in response.trace))
        print(f"BOT: {response.answer}")
        print()

    report = run_child_capability_probe()
    print("CAPABILITY PROBE")
    print(f"CASES: {report.cases}")
    print(
        "STATUS: "
        f"accepted={report.accepted}, answered={report.answered}, "
        f"abstained={report.abstained}, rejected={report.rejected}"
    )
    print(f"AVG_PARSE_SCORE: {report.average_parse_score}")
    print(f"AVG_COMPLEXITY: {report.average_complexity}")
    print(f"MAX_COMPLEXITY: {report.max_complexity}")
    print("REASONS:")
    for reason, count in report.by_reason.items():
        print(f"  {reason}: {count}")
    print("EDGE EXAMPLES:")
    for wanted_status in ("abstained", "rejected"):
        result = next(
            item for item in report.results
            if item.status == wanted_status
        )
        print(
            f"  {result.status}/{result.reason}: {result.utterance} "
            f"(parse={result.parse_score}, complexity={result.complexity_score})"
        )
    hardest = max(report.results, key=lambda item: item.complexity_score)
    print(
        f"  max_complexity/{hardest.status}: {hardest.utterance} "
        f"(parse={hardest.parse_score}, complexity={hardest.complexity_score})"
    )

    budget_report = run_child_context_budget_probe(move_counts=(32, 128), evidence_top_k=1)
    print()
    print("CONTEXT BUDGET PROBE")
    print("query_kind moves raw unbudgeted budgeted ratio_before ratio_after evidence_before evidence_after matches")
    for result in budget_report.results:
        print(
            f"{result.query_kind} {result.move_count} "
            f"{result.raw_transcript_chars} {result.unbudgeted_payload_chars} "
            f"{result.budgeted_payload_chars} "
            f"{result.unbudgeted_compression_ratio} "
            f"{result.budgeted_compression_ratio} "
            f"{result.unbudgeted_evidence_count} "
            f"{result.budgeted_evidence_count} "
            f"{result.matching_evidence_count}"
        )

    print()
    print("ASSISTANT DIRECTION COMPARISON")
    print("strategy local_or_device cloud fetch clarify privacy memory rate")
    for report in compare_assistant_mvp_directions():
        print(
            f"{report.strategy} "
            f"{report.local_or_device_resolved}/{report.cases} "
            f"{report.cloud_handoffs} "
            f"{report.external_fetches} "
            f"{report.clarifications} "
            f"{report.privacy_exposures} "
            f"{report.memory_uses} "
            f"{report.local_resolution_rate}"
        )
    print("MEMORY-CENTRIC ROUTES:")
    memory_report = next(
        report for report in compare_assistant_mvp_directions()
        if report.strategy == "memory_centric_local_triage"
    )
    for decision in memory_report.decisions:
        print(
            f"  {decision.intent}: {decision.route} "
            f"({decision.reason})"
        )

    kernel_report = run_assistant_kernel_learning_probe()
    print()
    print("ASSISTANT OS KERNEL LEARNING PROBE")
    print(f"BEFORE_ROUTE: {kernel_report.before_route}")
    print(f"AFTER_ROUTE: {kernel_report.after_route}")
    print(f"CLOUD_HANDOFFS_BEFORE: {kernel_report.cloud_handoffs_before}")
    print(f"CLOUD_HANDOFFS_AFTER: {kernel_report.cloud_handoffs_after}")
    print(f"EXECUTED_JOBS: {', '.join(kernel_report.executed_jobs)}")
    print(f"REMEMBERED_EVENTS: {kernel_report.remembered_events}")
    print(f"STORY_INVENTORY_COUNT: {kernel_report.story_inventory_count}")
    for opportunity in kernel_report.opportunities:
        print(
            f"OPPORTUNITY: {opportunity.kind} priority={opportunity.priority} "
            f"expected_cloud_reduction={opportunity.expected_cloud_reduction}"
        )

    lifecycle_report = run_realistic_assistant_lifecycle_probe()
    print()
    print("REALISTIC ASSISTANT LIFECYCLE PROBE")
    print(f"STEPS: {lifecycle_report.steps}")
    print(f"LOCAL_OR_DEVICE: {lifecycle_report.local_or_device_resolved}")
    print(f"LOCAL_RATE: {lifecycle_report.local_resolution_rate}")
    print(f"CLOUD_HANDOFFS: {lifecycle_report.cloud_handoffs}")
    print(f"EXTERNAL_FETCHES: {lifecycle_report.external_fetches}")
    print(f"BLOCKED_OFFLINE: {lifecycle_report.blocked_offline}")
    print(f"CONFIRMATIONS_REQUIRED: {lifecycle_report.confirmations_required}")
    print(f"ACTIONS_EXECUTED: {lifecycle_report.actions_executed}")
    print(f"STORY_BEFORE: {lifecycle_report.story_route_before_inventory}")
    print(f"STORY_AFTER: {lifecycle_report.story_route_after_inventory}")
    print(f"CLOUD_STORY_HANDOFFS_BEFORE: {lifecycle_report.cloud_story_handoffs_before_inventory}")
    print(f"JOBS_EXECUTED: {', '.join(lifecycle_report.jobs_executed)}")
    print(f"INVENTORY: stories={lifecycle_report.story_inventory_count}, weather_days={lifecycle_report.weather_cache_days}, contacts={lifecycle_report.contact_count}")
    print("LIFECYCLE ROUTES:")
    for result in lifecycle_report.results:
        flags = []
        if result.executed_jobs:
            flags.append("jobs=" + ",".join(result.executed_jobs))
        if result.confirmation_required:
            flags.append("needs_confirm")
        if result.action_executed:
            flags.append("action_executed")
        if result.blocked_offline:
            flags.append("blocked_offline")
        suffix = f" [{' '.join(flags)}]" if flags else ""
        print(f"  day{result.day}: {result.route}/{result.reason}: {result.utterance}{suffix}")


if __name__ == "__main__":
    main()
