# MELM Memory OS Support/Refunds Benchmark

Source: `support_refund_fixture`
Events: `57`
k: `2`

- Gate passed: `True`
- Cases: `8`
- Vector RAG accuracy: `75.00%`
- Temporal/entity RAG accuracy: `75.00%`
- Memory OS accuracy: `100.00%`
- Memory OS gain vs vector: `25.00%`
- Positive recall: `100.00%`
- Negative abstention: `100.00%`

| Query | Category | Expected | Vector | Temporal/entity | Memory OS | Evidence |
|---|---|---:|---:|---:|---:|---|
| What is the latest status for order o1006? | current_state | shipped | True | True | True | o1006_e_status |
| Is order o1007 already refunded? | current_state | refunded | True | True | True | o1007_e_refunded |
| What is the current refund status for order o1010? | stale_state_update | refunded | True | True | True | o1010_e_status_refunded |
| Who approved the high-value refund for order o1009? | approval_recall | o1009_e_approval_fresh | True | True | True | o1009_e_approval_fresh |
| Which event says order o1005 has a fraud flag? | risk_recall | o1005_e_fraud | True | True | True | o1005_e_fraud |
| What policy sets the refund limit without approval? | policy_recall | policy_e1 | True | True | True | policy_e1 |
| What is the latest status for order o9999? | unknown_order | ABSTAIN | False | False | True |  |
| Is order o2000 already refunded? | unknown_order | ABSTAIN | False | False | True |  |

Interpretation: Memory OS must improve over vector RAG by resolving latest state from event history and abstaining on unseen order facts.
