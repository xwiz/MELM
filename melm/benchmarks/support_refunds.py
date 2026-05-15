"""Support/refunds fixtures for MELM Guard and Memory OS validation."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Literal

from melm.guard import ActionProposal, Condition, Fact, GuardStatus, Rule
from melm.memory import Event


MemoryCaseKind = Literal["event", "state"]


@dataclass(frozen=True)
class GuardBenchmarkCase:
    proposal: ActionProposal
    expected_status: GuardStatus
    category: str
    current_time: int | None = None


@dataclass(frozen=True)
class SupportMemoryCase:
    query: str
    kind: MemoryCaseKind
    category: str
    order_id: str
    expected_event_id: str | None = None
    predicate: str | None = None
    expected_value: str | None = None


@dataclass(frozen=True)
class SupportRefundFixture:
    events: list[Event]
    facts: list[Fact]
    rules: list[Rule]
    guard_cases: list[GuardBenchmarkCase]
    memory_cases: list[SupportMemoryCase]
    current_time: int


def support_refund_fixture() -> SupportRefundFixture:
    """Return a deterministic support/refunds validation fixture."""

    events: list[Event] = []
    facts: list[Fact] = []

    def add_fact(
        event_id: str,
        source_span: str,
        time_index: int,
        subject: str,
        predicate: str,
        value,
        *,
        order_id: str = "",
        customer_id: str = "",
        action: str = "recorded",
    ) -> None:
        metadata = {
            "fact_subject": subject,
            "fact_predicate": predicate,
            "fact_value": str(value),
        }
        if order_id:
            metadata["order_id"] = order_id
        if customer_id:
            metadata["customer_id"] = customer_id
        events.append(
            Event(
                event_id=event_id,
                source_span=source_span,
                time_index=time_index,
                actors=("support",),
                action_or_state=action,
                objects=tuple(item for item in (order_id, customer_id, predicate) if item),
                location="support_queue",
                metadata=metadata,
            )
        )
        facts.append(
            Fact(
                fact_id=f"fact_{event_id}",
                subject=subject,
                predicate=predicate,
                value=value,
                time_index=time_index,
                source_event_id=event_id,
                metadata={"fixture": "support_refunds"},
            )
        )

    add_fact(
        "policy_e1",
        "Policy allows refunds up to 100 without manager approval.",
        0,
        "policy:support",
        "max_refund_without_approval",
        100,
        action="policy",
    )
    add_fact(
        "policy_e2",
        "Manager approvals expire after five support ticks.",
        0,
        "policy:support",
        "manager_approval_fresh_steps",
        5,
        action="policy",
    )
    add_fact(
        "policy_e3",
        "Refunds are allowed only within thirty days of delivery.",
        0,
        "policy:support",
        "return_window_days",
        30,
        action="policy",
    )

    _add_order(events, facts, "o1001", "c1", amount=80, status="delivered", days=10, verified=True, fraud=False, time_base=1)
    _add_order(events, facts, "o1002", "c2", amount=70, status="delivered", days=8, verified=False, fraud=False, time_base=10)
    _add_order(events, facts, "o1003", "c3", amount=180, status="delivered", days=9, verified=True, fraud=False, time_base=20)
    _add_order(events, facts, "o1004", "c4", amount=180, status="delivered", days=12, verified=True, fraud=False, time_base=30)
    add_fact(
        "o1004_e_approval_stale",
        "Manager approved a high-value refund for order o1004 earlier in the week.",
        31,
        "order:o1004",
        "manager_approval",
        True,
        order_id="o1004",
        customer_id="c4",
        action="approved",
    )
    _add_order(events, facts, "o1005", "c5", amount=60, status="delivered", days=7, verified=True, fraud=True, time_base=40)
    _add_order(events, facts, "o1006", "c6", amount=55, status="shipped", days=2, verified=True, fraud=False, time_base=50)
    _add_order(events, facts, "o1007", "c7", amount=40, status="delivered", days=6, verified=True, fraud=False, time_base=60)
    add_fact(
        "o1007_e_refunded",
        "Order o1007 was already refunded after a prior support chat.",
        64,
        "order:o1007",
        "refund_status",
        "refunded",
        order_id="o1007",
        customer_id="c7",
        action="refunded",
    )
    _add_order(events, facts, "o1008", "c8", amount=35, status="delivered", days=45, verified=True, fraud=False, time_base=70)
    _add_order(events, facts, "o1009", "c9", amount=160, status="delivered", days=5, verified=True, fraud=False, time_base=80)
    add_fact(
        "o1009_e_approval_fresh",
        "Manager approved a high-value refund for order o1009 during this session.",
        98,
        "order:o1009",
        "manager_approval",
        True,
        order_id="o1009",
        customer_id="c9",
        action="approved",
    )
    _add_order(events, facts, "o1010", "c10", amount=90, status="delivered", days=3, verified=True, fraud=False, time_base=90)
    add_fact(
        "o1010_e_status_refunded",
        "Support approved and recorded the refund for order o1010.",
        96,
        "order:o1010",
        "refund_status",
        "refunded",
        order_id="o1010",
        customer_id="c10",
        action="refunded",
    )

    return SupportRefundFixture(
        events=events,
        facts=facts,
        rules=_support_refund_rules(),
        guard_cases=_support_refund_guard_cases(),
        memory_cases=_support_refund_memory_cases(),
        current_time=100,
    )


def generate_support_refund_benchmark(
    *,
    scenario_repeats: int = 20,
    seed: int = 17,
    start_order_index: int = 2000,
) -> SupportRefundFixture:
    """Generate a larger support/refunds benchmark with balanced categories.

    The generator is deterministic and uses an independent oracle policy to
    label actions. It intentionally varies names, amounts, times, and query
    surfaces while preserving a balanced set of safety-critical cases.
    """

    rng = random.Random(seed)
    events, facts = _policy_events_and_facts()
    guard_cases: list[GuardBenchmarkCase] = []
    memory_cases: list[SupportMemoryCase] = [
        SupportMemoryCase(
            "Which policy says refunds above the no-approval limit need review?",
            "event",
            "policy_recall",
            "",
            expected_event_id="policy_e1",
        )
    ]
    scenario_names = (
        "valid_low_value",
        "identity_missing_or_false",
        "approval_required",
        "stale_approval",
        "fraud_flag",
        "not_delivered",
        "duplicate_refund",
        "outside_return_window",
        "valid_high_value",
        "missing_order",
        "malformed_action",
        "stale_state_trap",
    )
    order_counter = start_order_index
    current_times: list[int] = []

    for repeat in range(scenario_repeats):
        for scenario in scenario_names:
            order_id = f"o{order_counter}"
            customer_id = f"c{order_counter}"
            order_counter += 1
            time_base = (order_counter - start_order_index) * 10
            current_time = time_base + 9
            current_times.append(current_time)
            amount = _amount_for_scenario(scenario, rng)
            status = "delivered"
            days = rng.randint(2, 25)
            verified = True
            fraud = False
            refund_status: str | None = None
            approval_time: int | None = None

            if scenario == "identity_missing_or_false":
                verified = False
            elif scenario == "approval_required":
                amount = rng.randint(130, 240)
            elif scenario == "stale_approval":
                amount = rng.randint(130, 240)
                approval_time = time_base + 1
            elif scenario == "fraud_flag":
                fraud = True
            elif scenario == "not_delivered":
                status = rng.choice(("shipped", "processing", "cancelled"))
            elif scenario == "duplicate_refund":
                refund_status = "refunded"
            elif scenario == "outside_return_window":
                days = rng.randint(31, 90)
            elif scenario == "valid_high_value":
                amount = rng.randint(130, 240)
                approval_time = current_time - rng.randint(0, 4)
            elif scenario == "stale_state_trap":
                refund_status = "refunded"
            elif scenario == "missing_order":
                guard_cases.append(
                    GuardBenchmarkCase(
                        _proposal(f"{scenario}_{repeat}", order_id, customer_id, amount),
                        "deny",
                        scenario,
                        current_time=current_time,
                    )
                )
                memory_cases.append(
                    SupportMemoryCase(
                        f"What is the latest status for order {order_id}?",
                        "state",
                        "unknown_order",
                        order_id,
                        predicate="status",
                        expected_value=None,
                    )
                )
                continue

            _add_order(
                events,
                facts,
                order_id,
                customer_id,
                amount=amount,
                status=status,
                days=days,
                verified=verified,
                fraud=fraud,
                time_base=time_base,
            )

            if approval_time is not None:
                _add_fact_record(
                    events,
                    facts,
                    f"{order_id}_e_approval_{repeat}",
                    f"Manager approved refund review for order {order_id}.",
                    approval_time,
                    f"order:{order_id}",
                    "manager_approval",
                    True,
                    order_id=order_id,
                    customer_id=customer_id,
                    action="approved",
                )
                memory_cases.append(
                    SupportMemoryCase(
                        f"Which event records manager approval for order {order_id}?",
                        "event",
                        "approval_recall",
                        order_id,
                        expected_event_id=f"{order_id}_e_approval_{repeat}",
                    )
                )
            if refund_status is not None:
                _add_fact_record(
                    events,
                    facts,
                    f"{order_id}_e_refunded_{repeat}",
                    f"Order {order_id} was already refunded before this request.",
                    time_base + 6,
                    f"order:{order_id}",
                    "refund_status",
                    refund_status,
                    order_id=order_id,
                    customer_id=customer_id,
                    action="refunded",
                )
                memory_cases.append(
                    SupportMemoryCase(
                        f"Is order {order_id} already refunded?",
                        "state",
                        "refund_state",
                        order_id,
                        predicate="refund_status",
                        expected_value=refund_status,
                    )
                )
            if fraud:
                memory_cases.append(
                    SupportMemoryCase(
                        f"Which event says order {order_id} has a fraud flag?",
                        "event",
                        "risk_recall",
                        order_id,
                        expected_event_id=f"{order_id}_e_fraud",
                    )
                )

            proposal_amount = None if scenario == "malformed_action" else amount
            case = GuardBenchmarkCase(
                _proposal(
                    f"{scenario}_{repeat}",
                    order_id,
                    customer_id,
                    proposal_amount,
                    malformed=scenario == "malformed_action",
                ),
                _oracle_status(
                    amount=amount,
                    status=status,
                    days=days,
                    verified=verified,
                    fraud=fraud,
                    refund_status=refund_status,
                    approval_time=approval_time,
                    current_time=current_time,
                    malformed=scenario == "malformed_action",
                ),
                scenario,
                current_time=current_time,
            )
            guard_cases.append(case)
            memory_cases.append(
                SupportMemoryCase(
                    _status_query(order_id, rng),
                    "state",
                    "current_state",
                    order_id,
                    predicate="status",
                    expected_value=status,
                )
            )
            if repeat % 4 == 0:
                unknown_id = f"oX{repeat:03d}{scenario[:2]}"
                memory_cases.append(
                    SupportMemoryCase(
                        f"What is the latest status for order {unknown_id}?",
                        "state",
                        "unknown_order",
                        unknown_id,
                        predicate="status",
                        expected_value=None,
                    )
                )

    return SupportRefundFixture(
        events=events,
        facts=facts,
        rules=_support_refund_rules(),
        guard_cases=guard_cases,
        memory_cases=memory_cases,
        current_time=max(current_times) if current_times else 0,
    )


def _add_order(
    events: list[Event],
    facts: list[Fact],
    order_id: str,
    customer_id: str,
    *,
    amount: int,
    status: str,
    days: int,
    verified: bool,
    fraud: bool,
    time_base: int,
) -> None:
    def add(event_id: str, source_span: str, offset: int, predicate: str, value, action: str) -> None:
        subject = f"order:{order_id}" if predicate not in {"identity_verified"} else f"customer:{customer_id}"
        metadata = {
            "fact_subject": subject,
            "fact_predicate": predicate,
            "fact_value": str(value),
            "order_id": order_id,
            "customer_id": customer_id,
        }
        events.append(
            Event(
                event_id=event_id,
                source_span=source_span,
                time_index=time_base + offset,
                actors=("support", customer_id),
                action_or_state=action,
                objects=(order_id, customer_id, predicate),
                location="support_queue",
                metadata=metadata,
            )
        )
        facts.append(
            Fact(
                fact_id=f"fact_{event_id}",
                subject=subject,
                predicate=predicate,
                value=value,
                time_index=time_base + offset,
                source_event_id=event_id,
                metadata={"fixture": "support_refunds"},
            )
        )

    add(f"{order_id}_e_amount", f"Order {order_id} has a refund amount of {amount}.", 0, "amount", amount, "recorded")
    add(f"{order_id}_e_status", f"Order {order_id} is currently {status}.", 1, "status", status, "status")
    add(
        f"{order_id}_e_days",
        f"Order {order_id} was delivered {days} days ago for return-window calculation.",
        2,
        "days_since_delivery",
        days,
        "recorded",
    )
    add(
        f"{order_id}_e_identity",
        f"Customer {customer_id} identity verification is {verified}.",
        3,
        "identity_verified",
        verified,
        "verified",
    )
    add(f"{order_id}_e_fraud", f"Order {order_id} fraud flag is {fraud}.", 4, "fraud_flag", fraud, "flagged")


def _policy_events_and_facts() -> tuple[list[Event], list[Fact]]:
    events: list[Event] = []
    facts: list[Fact] = []
    _add_fact_record(
        events,
        facts,
        "policy_e1",
        "Policy allows refunds up to 100 without manager approval.",
        0,
        "policy:support",
        "max_refund_without_approval",
        100,
        action="policy",
    )
    _add_fact_record(
        events,
        facts,
        "policy_e2",
        "Manager approvals expire after five support ticks.",
        0,
        "policy:support",
        "manager_approval_fresh_steps",
        5,
        action="policy",
    )
    _add_fact_record(
        events,
        facts,
        "policy_e3",
        "Refunds are allowed only within thirty days of delivery.",
        0,
        "policy:support",
        "return_window_days",
        30,
        action="policy",
    )
    return events, facts


def _add_fact_record(
    events: list[Event],
    facts: list[Fact],
    event_id: str,
    source_span: str,
    time_index: int,
    subject: str,
    predicate: str,
    value,
    *,
    order_id: str = "",
    customer_id: str = "",
    action: str = "recorded",
) -> None:
    metadata = {
        "fact_subject": subject,
        "fact_predicate": predicate,
        "fact_value": str(value),
    }
    if order_id:
        metadata["order_id"] = order_id
    if customer_id:
        metadata["customer_id"] = customer_id
    events.append(
        Event(
            event_id=event_id,
            source_span=source_span,
            time_index=time_index,
            actors=("support",),
            action_or_state=action,
            objects=tuple(item for item in (order_id, customer_id, predicate) if item),
            location="support_queue",
            metadata=metadata,
        )
    )
    facts.append(
        Fact(
            fact_id=f"fact_{event_id}",
            subject=subject,
            predicate=predicate,
            value=value,
            time_index=time_index,
            source_event_id=event_id,
            metadata={"fixture": "support_refunds"},
        )
    )


def _proposal(
    action_id: str,
    order_id: str,
    customer_id: str,
    amount: int | None,
    *,
    malformed: bool = False,
) -> ActionProposal:
    parameters = {"order_id": order_id, "customer_id": customer_id}
    if amount is not None:
        parameters["amount"] = amount
    return ActionProposal(
        action_id=action_id,
        action_type="approve_refund",
        parameters=parameters,
        source_query=f"Please approve refund for order {order_id}.",
        malformed=malformed,
    )


def _oracle_status(
    *,
    amount: int,
    status: str,
    days: int,
    verified: bool,
    fraud: bool,
    refund_status: str | None,
    approval_time: int | None,
    current_time: int,
    malformed: bool = False,
) -> GuardStatus:
    if malformed:
        return "deny"
    if not verified or fraud or status != "delivered" or refund_status == "refunded" or days > 30:
        return "deny"
    if amount > 100 and approval_time is None:
        return "abstain"
    if amount > 100 and approval_time is not None and current_time - approval_time > 5:
        return "warn"
    return "allow"


def _amount_for_scenario(scenario: str, rng: random.Random) -> int:
    if scenario in {"approval_required", "stale_approval", "valid_high_value"}:
        return rng.randint(130, 240)
    return rng.randint(20, 95)


def _status_query(order_id: str, rng: random.Random) -> str:
    templates = (
        "What is the latest status for order {order_id}?",
        "Where does order {order_id} stand right now?",
        "What is the current order state for {order_id}?",
        "Has the status for order {order_id} changed?",
    )
    return rng.choice(templates).format(order_id=order_id)


def _support_refund_rules() -> list[Rule]:
    return [
        Rule(
            "refund_amount_required",
            "Refund approval requires an amount.",
            (Condition("action", "amount", "missing"),),
            "deny",
            provenance="support_policy.v1",
        ),
        Rule(
            "order_status_required",
            "Refund approval requires known order status.",
            (Condition("order:{order_id}", "status", "missing"),),
            "deny",
            provenance="support_policy.v1",
        ),
        Rule(
            "identity_required",
            "Customer identity must be verified before refund.",
            (Condition("customer:{customer_id}", "identity_verified", "missing"),),
            "deny",
            provenance="support_policy.v1",
        ),
        Rule(
            "identity_must_be_true",
            "Customer identity must be verified before refund.",
            (Condition("customer:{customer_id}", "identity_verified", "ne", True),),
            "deny",
            provenance="support_policy.v1",
        ),
        Rule(
            "fraud_blocks_refund",
            "Fraud-flagged orders cannot be refunded automatically.",
            (Condition("order:{order_id}", "fraud_flag", "eq", True),),
            "deny",
            provenance="support_policy.v1",
        ),
        Rule(
            "order_must_be_delivered",
            "Refunds require delivered order status.",
            (Condition("order:{order_id}", "status", "ne", "delivered"),),
            "deny",
            provenance="support_policy.v1",
        ),
        Rule(
            "duplicate_refund_block",
            "Already-refunded orders cannot be refunded again.",
            (Condition("order:{order_id}", "refund_status", "eq", "refunded"),),
            "deny",
            provenance="support_policy.v1",
        ),
        Rule(
            "return_window_block",
            "Refund request is outside the return window.",
            (
                Condition(
                    "order:{order_id}",
                    "days_since_delivery",
                    "gt",
                    value_source="fact:policy:support:return_window_days",
                ),
            ),
            "deny",
            provenance="support_policy.v1",
        ),
        Rule(
            "manager_approval_required",
            "High-value refunds require manager approval.",
            (
                Condition(
                    "action",
                    "amount",
                    "gt",
                    value_source="fact:policy:support:max_refund_without_approval",
                ),
                Condition("order:{order_id}", "manager_approval", "missing"),
            ),
            "require_approval",
            provenance="support_policy.v1",
        ),
        Rule(
            "manager_approval_stale",
            "High-value refund approval is stale and should be refreshed.",
            (
                Condition(
                    "action",
                    "amount",
                    "gt",
                    value_source="fact:policy:support:max_refund_without_approval",
                ),
                Condition(
                    "order:{order_id}",
                    "manager_approval",
                    "stale_after",
                    value_source="fact:policy:support:manager_approval_fresh_steps",
                ),
            ),
            "warn",
            provenance="support_policy.v1",
        ),
    ]


def _support_refund_guard_cases() -> list[GuardBenchmarkCase]:
    def proposal(action_id: str, order_id: str, customer_id: str, amount: int | None, *, malformed: bool = False) -> ActionProposal:
        parameters = {"order_id": order_id, "customer_id": customer_id}
        if amount is not None:
            parameters["amount"] = amount
        return ActionProposal(
            action_id=action_id,
            action_type="approve_refund",
            parameters=parameters,
            source_query=f"Please approve refund for order {order_id}.",
            malformed=malformed,
        )

    return [
        GuardBenchmarkCase(proposal("g1", "o1001", "c1", 80), "allow", "valid_low_value"),
        GuardBenchmarkCase(proposal("g2", "o1002", "c2", 70), "deny", "identity_missing_or_false"),
        GuardBenchmarkCase(proposal("g3", "o1003", "c3", 180), "abstain", "approval_required"),
        GuardBenchmarkCase(proposal("g4", "o1004", "c4", 180), "warn", "stale_approval"),
        GuardBenchmarkCase(proposal("g5", "o1005", "c5", 60), "deny", "fraud_flag"),
        GuardBenchmarkCase(proposal("g6", "o1006", "c6", 55), "deny", "not_delivered"),
        GuardBenchmarkCase(proposal("g7", "o1007", "c7", 40), "deny", "duplicate_refund"),
        GuardBenchmarkCase(proposal("g8", "o1008", "c8", 35), "deny", "outside_return_window"),
        GuardBenchmarkCase(proposal("g9", "o1001", "c1", None, malformed=True), "deny", "malformed_action"),
        GuardBenchmarkCase(proposal("g10", "o1009", "c9", 160), "allow", "valid_high_value"),
        GuardBenchmarkCase(proposal("g11", "o9999", "c9999", 25), "deny", "missing_order"),
        GuardBenchmarkCase(proposal("g12", "o1010", "c10", 90), "deny", "stale_state_trap"),
    ]


def _support_refund_memory_cases() -> list[SupportMemoryCase]:
    return [
        SupportMemoryCase(
            "What is the latest status for order o1006?",
            "state",
            "current_state",
            "o1006",
            predicate="status",
            expected_value="shipped",
        ),
        SupportMemoryCase(
            "Is order o1007 already refunded?",
            "state",
            "current_state",
            "o1007",
            predicate="refund_status",
            expected_value="refunded",
        ),
        SupportMemoryCase(
            "What is the current refund status for order o1010?",
            "state",
            "stale_state_update",
            "o1010",
            predicate="refund_status",
            expected_value="refunded",
        ),
        SupportMemoryCase(
            "Who approved the high-value refund for order o1009?",
            "event",
            "approval_recall",
            "o1009",
            expected_event_id="o1009_e_approval_fresh",
        ),
        SupportMemoryCase(
            "Which event says order o1005 has a fraud flag?",
            "event",
            "risk_recall",
            "o1005",
            expected_event_id="o1005_e_fraud",
        ),
        SupportMemoryCase(
            "What policy sets the refund limit without approval?",
            "event",
            "policy_recall",
            "",
            expected_event_id="policy_e1",
        ),
        SupportMemoryCase(
            "What is the latest status for order o9999?",
            "state",
            "unknown_order",
            "o9999",
            predicate="status",
            expected_value=None,
        ),
        SupportMemoryCase(
            "Is order o2000 already refunded?",
            "state",
            "unknown_order",
            "o2000",
            predicate="refund_status",
            expected_value=None,
        ),
    ]
