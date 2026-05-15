# MELM Public Memory Benchmark: LoCoMo

Dataset: `local_data\locomo10.json`
Dataset URL: https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
k: `5`
Documents/questions: `272` / `1978`
LoCoMo event summaries enabled for MELM: `True`
Gate passed: `True`
Statistical gate passed: `False`
Context budget: `1200` tokens
Context-budget answer gate passed: `True`
Load/eval seconds: `0.031` / `106.224`

## Architecture-Family Results

| Architecture | Mean recall@k | Hit@k | Full evidence@k | Scope |
|---|---:|---:|---:|---|
| `vector_rag` | 50.65% | 55.71% | 46.61% | bag-of-words cosine over raw session text |
| `mem0_additive_arch` | 86.00% | 90.90% | 81.55% | ADD-only memory proxy: raw session memories scored by BM25 plus entity boosts |
| `memgpt_tiered_arch` | 87.74% | 92.62% | 83.27% | tiered-memory proxy: raw session text plus session summaries |
| `zep_temporal_graph_arch` | 87.21% | 91.76% | 82.91% | temporal-graph proxy: extracted observations plus graph/session neighbor expansion |
| `melm_memory_os` | 88.23% | 92.62% | 84.18% | MELM Memory OS: raw turns plus extracted observations, session summaries, event summaries, temporal query routing, and question-guided raw snippets |

## MELM Event-Summary Ablation

This ablation disables LoCoMo event-summary memory and leaves MELM with raw turns, observations, session summaries, entity boosts, and temporal routing.

- MELM recall without event summaries: `87.54%`
- Difference vs MemGPT-style proxy: `-0.20%`
- Difference vs Zep-style proxy: `0.32%`

## Bounded Context Support

This metric packs retrieved memories into a fixed token budget and asks whether gold evidence sessions and gold answer tokens survive in the context. It is closer to the MemGPT question than retrieval@k alone.

| Architecture | Evidence recall in budget | Answer support | Mean answer-token recall |
|---|---:|---:|---:|
| `vector_rag` | 29.83% | 29.79% | 52.41% |
| `mem0_additive_arch` | 69.55% | 55.41% | 70.14% |
| `memgpt_tiered_arch` | 70.58% | 56.78% | 70.80% |
| `zep_temporal_graph_arch` | 87.21% | 54.50% | 71.10% |
| `melm_memory_os` | 77.86% | 61.41% | 74.37% |

| Context-budget estimate | Mean | 95% CI |
|---|---:|---:|
| MELM - Mem0-style answer support | 6.00% | [3.78%, 8.02%] |
| MELM - MemGPT-style answer support | 4.63% | [3.00%, 6.71%] |
| MELM - Zep-style answer support | 6.91% | [4.50%, 9.06%] |


## Paired Bootstrap Intervals

| Estimate | Mean | 95% CI |
|---|---:|---:|
| MELM recall | 88.23% | [86.94%, 89.37%] |
| MELM - vector RAG recall | 37.58% | [35.67%, 39.58%] |
| MELM - Mem0-style recall | 2.24% | [1.27%, 3.23%] |
| MELM - MemGPT-style recall | 0.49% | [-0.32%, 1.34%] |
| MELM - Zep-style recall | 1.02% | [0.30%, 1.72%] |

## Category Recall

| Architecture | Multi-hop | Single-hop | Temporal | Open-domain | Adversarial |
|---|---:|---:|---:|---:|---:|
| `vector_rag` | 31.52% | 47.51% | 31.27% | 56.60% | 57.62% |
| `mem0_additive_arch` | 52.63% | 85.88% | 57.79% | 95.72% | 94.39% |
| `memgpt_tiered_arch` | 55.16% | 89.10% | 57.79% | 97.03% | 95.74% |
| `zep_temporal_graph_arch` | 55.35% | 84.42% | 60.18% | 96.91% | 96.41% |
| `melm_memory_os` | 58.55% | 87.33% | 56.57% | 97.15% | 97.09% |

## Scope

- Claim: local architecture-family comparison on public LoCoMo evidence retrieval.
- Not claimed: official Mem0, Zep, Letta, or MemGPT vendor benchmark numbers.
- Reason: The proxies implement documented memory architecture ideas over the same public records without LLM APIs, so the run is reproducible and isolates memory representation/retrieval choices.

## References

- LoCoMo public dataset: https://github.com/snap-research/locomo
- Mem0 benchmark repository: https://github.com/mem0ai/memory-benchmarks
- MemGPT paper: https://arxiv.org/abs/2310.08560
- Zep/Graphiti temporal knowledge graph paper: https://arxiv.org/abs/2501.13956
