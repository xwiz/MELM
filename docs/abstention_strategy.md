# MELM Abstention Strategy

Status: validation axis retained; current selector not solved.

## Decision

Abstention is the right validation axis for MELM event memory, but it should be framed as evidence-set admission, not final answer refusal.

At this stage MELM retrieves event evidence for a later generator. Therefore the first question is:

- Should this retrieved event set be passed to the answerer, or should the system ask for clarification / say there is not enough evidence?

It is too early to claim final answer abstention because there is not yet a generator, citation checker, or answer verifier.

## What Changed

The initial sweep was useful but too optimistic because it selected the best threshold on the same benchmark it reported. The revised path now includes:

- same-set sweeps for diagnostics only;
- story-level calibration/evaluation split;
- held-out threshold reporting;
- a raw top-score selector baseline;
- an experimental score-plus-evidence-veto selector;
- positive recall and negative abstention reported together.
- optional access-history bias in event memory, so frequently and recently
  retrieved events can be promoted without overwriting semantic/entity scores.

The current held-out result shows the tradeoff clearly:

- top-score calibration reaches the target no-answer rejection rate, but sacrifices too much positive recall;
- the score-plus-evidence-veto selector now passes the synthetic held-out gate: 75.0% positive recall and 85.3% negative abstention at threshold 1.25;
- the pass is still provisional because the no-answer set is synthetic and pattern-generated.

## Research Anchors

This direction matches recent RAG and selective-answering work:

- UAEval4RAG argues that RAG evaluation must include unanswerable requests specific to the knowledge base, not only answerable QA accuracy: https://aclanthology.org/2025.acl-long.415.pdf
- RetrievalQA reports that calibration-based adaptive retrieval can depend heavily on threshold tuning: https://arxiv.org/abs/2402.16457
- COIN frames selective QA as threshold calibration on a calibration set under an explicit risk target: https://arxiv.org/abs/2506.20178
- Recent temporal-QA abstention work warns that calibration alone can be unreliable for complex temporal reasoning and may need learned abstention behavior later: https://papers.cool/arxiv/2602.04755v1

## Next Implementation Step

Do not tune thresholds directly on the final benchmark. Instead:

1. Keep raw top-score as the abstention baseline.
2. Improve evidence sufficiency separately with actor-action-object and temporal-anchor checks.
3. Add non-synthetic no-answer cases before changing the gate.
4. Evaluate access-history retrieval separately from answerability calibration,
   because recency can improve recall while also increasing false positives.
5. Later, once a generator exists, add answer-level verification with citation support and contradiction checks.
