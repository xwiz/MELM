"""Human-input repair tiers (pre-UOL normalization cascade).

See docs/human-friendly-NLG-pipeline.md. Tier 0 (surface expansion) lives in the
language adapter; this package holds Tier 1 (lexicon-backed SymSpell typo
correction) and Tier 1.5b (deterministic proper-noun / number protection mask).
All tiers are deterministic and degrade to no-ops when an optional dependency
(symspellpy) is absent.
"""
