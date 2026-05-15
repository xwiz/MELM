# nameless_vector Parts Donor Notes

Source project: `C:\dev\nameless_vector`  
Use policy: design reference only until build/test health is verified.

## Observed Useful Concepts

- Semantic frame memory over verb/noun state files.
- State algebra for preconditions, effects, and contradictions.
- Temporal relation graph for sequence validation and causal reasoning.
- Grounding layer for validating LLM output against structured frames.
- Query routing and validation outcomes.
- Hallucination failure-mode taxonomy.
- Benchmark scaffolding for physical, logical, temporal, and precondition failures.

## Do Not Blindly Reuse

- Do not make `nameless_vector` the primary MELM codebase.
- Do not port Rust modules before confirming tests and simplifying interfaces.
- Do not treat its frame data as validated linguistic or cognitive ground truth.
- Do not use it as evidence that MELM's event memory works; it is a source of implementation ideas only.

## Known Health Note

During planning, `cargo test --no-default-features` timed out after about two minutes. This may be dependency compilation, a hanging test, or local environment friction. Before porting code, run a dedicated health pass:

```powershell
cargo test --no-default-features
cargo test --no-default-features --lib
cargo test --no-default-features --examples
```

Record exact failures before copying any implementation.

## Concepts To Port First In Python

1. Minimal event schema with actors, action/state, objects, time index, embedding, and temporal links.
2. State contradiction checks for simple mutually exclusive states.
3. Temporal-neighbor retrieval baseline.
4. Hallucination/grounding benchmark categories.

## Rust Sidecar Decision

Consider a Rust sidecar only if:

- the Python event-memory prototype beats RAG by the required 15%;
- retrieval or validation latency becomes a bottleneck;
- the `nameless_vector` test suite is healthy or the needed modules are isolated and covered.
