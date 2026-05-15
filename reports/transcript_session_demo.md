# Transcript Session Demo

Source: `annotated_transcript_persistent_session`
Annotations: `benchmarks\sample_transcript_annotations.jsonl`
Session path: `reports\sample_transcript_session_events.jsonl`
Seeded: `True`
Turns: `8`
Events after reload: `11`
Distractor events: `4`
Threshold: `1.25`
k: `2`

## Dialogue Evidence

- Accuracy: `100.00%`
- Positive recall: `100.00%`
- Negative abstention: `100.00%`
- Answer rate: `60.00%`

## Noisy Dialogue Evidence

- Accuracy: `100.00%`
- Positive recall: `100.00%`
- Negative abstention: `100.00%`
- Answer rate: `60.00%`

## Memory Retrieval

- RAG recall@k: `66.67%`
- Event-memory recall@k: `100.00%`
- Absolute gain: `33.33%`
- Event-memory MRR@k: `83.33%`

## State Resolution

- Accuracy: `100.00%`
- Answer rate: `75.00%`
- False positive rate: `0.00%`

| Query | Status | Confidence | Evidence | Answer |
|---|---|---:|---|---|
| Where did Lila put the rainbow bead? | answered | 1.689 | sample_e1, sample_e3 | I remember: Lila put the rainbow bead in the yellow box before music. Lila checked the craft tray and did not find the rainbow bead. |
| What happened right after Lila put the rainbow bead away? | answered | 1.861 | sample_e2, sample_e1 | I remember: Ben slid the yellow box under the puzzle shelf. Lila put the rainbow bead in the yellow box before music. |
| Who saw where the yellow box went? | answered | 1.412 | sample_e2, sample_e4 | I remember: Ben slid the yellow box under the puzzle shelf. Rafi saw Ben slide the yellow box under the puzzle shelf. |
| What earlier event explains why Rafi knew where the yellow box was? | answered | 1.330 | sample_e4, sample_e2 | I remember: Rafi saw Ben slide the yellow box under the puzzle shelf. Ben slid the yellow box under the puzzle shelf. |
| Where did Mina stack the star blocks first? | answered | 1.344 | sample_e5, sample_e7 | I remember: Mina stacked the star blocks beside the window. Mina rebuilt the star blocks on the carpet. |
| What happened right before Mina rebuilt the star blocks? | answered | 1.977 | sample_e6, sample_e7 | I remember: Taro knocked the star blocks while grabbing the drum. Mina rebuilt the star blocks on the carpet. |
| Where did Lila hide the silver button? | abstained | 0.000 |  | I do not have enough evidence for that yet. |
| Where was the rainbow bead after Rafi moved it? | abstained | 0.000 |  | I do not have enough evidence for that yet. |
| Where did Mina put the thing? | abstained | 0.000 |  | I do not have enough evidence for that yet. |
| Who explained the missing robot? | abstained | 0.000 |  | I do not have enough evidence for that yet. |
