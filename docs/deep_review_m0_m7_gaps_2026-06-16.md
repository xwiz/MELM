# Deep Review: M0–M7 Gaps and Regressions (2026-06-16)

**Scope:** Re-review of the Local Assistant OS MVP Plan v2 after the 2026-06-15/16 fix session. All claims are reproducible against the current working tree.

---

## 1. Executive Summary

| Milestone | Claimed Status | Actual Status | Blocking Gap |
|---|---|---|---|
| **M0** | "Complete" (commit `b19579f`) | **Incomplete** | 61 files uncommitted; working tree is ahead of HEAD |
| **M1** | "Contract kernel built" | **Partial** | `check_compatibility()` does not check version mismatch; no 60-case normative UOL set; no perf harness |
| **M2** | "9/9 classifiers migrated" | **Partial** | 5 bridge functions still pre-filter before frame linker; sealed-dictionary routing test bypasses full router |
| **M3** | "Sealed dictionary ≥60 words, ≥80% routing" | **Meets threshold** | 66 words, 4/4 tests pass; BUT routing test uses `_classify_from_frame_linker` not end-to-end router |
| **M4** | "Decoder scaffold + template fallback" | **Partial** | Decoder wired to kernel (fixed); no Pi tok/s/TTFT/RSS; no model loaded; no go/no-go record |
| **M5** | "E3 reranker, top-3 ≥95%, precision ≥98%" | **Partial** | Top-3 ≥95% passes; precision threshold is 0.95 not 0.98; bridge functions still gate 5 intents |
| **M6** | "Not started" | **Not started** | No E2 shadow mode; no learned UOL families |
| **M7** | "Not reached" | **Not reached** | Integration gates not assessable |

**Critical new finding:** The `_REQUIRES_MAP` in `assistant_authority.py` had only 3 entries (weather, health_advice, meal_suggestion) in the committed tree. This meant any random valid utterance for story, personal_memory, media_playback, social_contact, autobiographical_memory, common_sense_safety, assistant_identity, or assistant_status would fail authority verification with `missing_required_evidence`. This was hidden because all existing hardcoded tests used only the 3 mapped intents. The random conversation generator exposed it; we fixed it by expanding to 14 entries.

---

## 2. M0 — Recover Truth

### Exit Gates (from plan)
1. `pytest`, `pi-smoke`, and `shortcut-audit` green in CI
2. v1 superseded
3. Changing a debug label does not fail a behavioral gate
4. **Commit the tree**

### Current State
- **Tests:** 852 collected, 1 deselected (known bundle failure). Core subset: **240 passed, 272 subtests passed** in 14.35s.
- **shortcut-audit:** Passed. Reports bridge-eliminated intents correctly.
- **pi-smoke:** Exit code 0. Contains expected safe failures (`soak_passed: false` for empty-source cases).
- **CI:** `pyproject.toml` deselects only `test_cli_pi_bundle_builds_portable_self_checked_bundle`.

### Gaps
- **CRITICAL: Tree not committed.** `git status` shows **29 modified, 32 untracked** files on top of `b19579f`. This includes new skill modules (`assistant_skill_*.py`), new contracts (18 total), frame ranker, experience writer, and the random test generator. M0 explicitly requires committing the tree.
- The `_REQUIRES_MAP` deficiency was a **latent regression** in the committed tree. It only surfaced when random utterances hit unmapped intents. The hardcoded test suite was accidentally colluding with the bug by never testing those intent/evidence combinations.

---

## 3. M1 — Contract Kernel

### Exit Gates
1. Current regression suite passes through validators
2. Initial 60-case normative UOL set passes
3. Incompatible versions fail closed
4. Dev + Pi benchmark JSON produced

### Current State
- **Contract registry:** `registry.v1.json` has 18 registered contracts with schemas, validators, and `compatible_predecessors`.
- **Kernel validation:** `AssistantOSKernel.__init__` calls `_validate_contract_compatibility()` which raises `ContractValidationError` on errors.
- **Validators:** `validation.py` has `validate_frame_templates`, `validate_contract_registry`, `validate_semantic_class_registry`, etc.

### Gaps
- **`check_compatibility()` is structurally weak.** It only verifies that `compatible_predecessors` strings exist as keys in the registry. It does **not** compare semantic versions, check schema hash mismatches, or verify that a v2 contract artifact is compatible with v1-consuming code.
  ```python
  # melm/contracts/validation.py:126-134
  def check_compatibility(self) -> list[str]:
      for schema_id, entry in self.contracts.items():
          for pred in entry.get("compatible_predecessors", []):
              if pred not in self.contracts:  # Only checks key existence
                  errors.append(...)
  ```
- **No 60-case normative UOL set.** There is a sealed dictionary of 66 made-up words for M3 routing, but no normative set testing UOL parse correctness (slot roles, predicate identification, speech act classification) against human-annotated gold data.
- **No dev/Pi benchmark JSON.** No performance harness exists. The `pi-smoke` CLI checks constraints (no network, no vector DB, SQLite indexes) but does not produce tok/s/TTFT/RSS measurements.

---

## 4. M2 — Meaning Substrate

### Exit Gates
1. 100% route agreement on existing regression cases for migrated families
2. Deleting a seed row changes behavior predictably
3. No user/atlas write can enable a capability or action

### Current State
- **Frame templates:** 11 templates in `frame_templates.v1.json`.
- **Migrated intents:** `weather`, `story`, `media_playback`, `autobiographical_memory`, `meal_suggestion`, `common_sense_safety`, `social_contact`, `health_advice`, `personal_memory` (9 total).
- **Capability manifest:** `_get_capability_manifest()` gates dispatch; uninstalled families route to `open_domain`.

### Gaps
- **5 bridge functions still exist** and are called **before** the frame linker in `_classify_intent_from_uol_slots`:
  1. `_is_common_sense_safety_request` (`local_assistant_router.py:1736`)
  2. `_is_social_contact_request` (`local_assistant_router.py:1794`)
  3. `_is_personal_memory_frame` (`local_assistant_router.py:1886`)
  4. `_is_autobiographical_debug_request` (`local_assistant_router.py:1983`)
  5. `_has_urgent_health_frame` (`local_assistant_router.py:4222`)

  These functions contain hardcoded token sets (`{"call", "phone", "ring", "reach"}`, `{"without", "clothes"}`, `{"who", "am", "i"}`, etc.) that pre-filter utterances before the linker ever sees them. This means:
  - The reranker (which now returns `rerank_score` for threshold checks) does not have full control over these 5 intents.
  - The frame linker thresholds and context gates for these intents are partially bypassed.
  - The M2 exit gate "100% route agreement on migrated families" is **not truly met** for these 5 families because the migration is hybrid, not pure.

- **No test for "deleting a seed row changes behavior predictably."** The existing tests verify that adding/promoting words works, but there is no regression test that deletes a row from `semantic_classes.v1.json` or `frame_templates.v1.json` and asserts a specific behavioral change.

---

## 5. M3 — Learning Vertical Slice

### Exit Gates
1. Sealed ≥60-word dictionary set
2. ≥80% correct next-turn use and retention
3. Zero reserved-namespace promotions
4. Zero capability grants
5. Correction/rollback trace queryable end to end

### Current State
- **Sealed dictionary:** 66 words (`test_assistant_lexicon_mvp.py:1694`).
- **Tests (all pass):**
  - `test_sealed_dictionary_count` — 66 ≥ 60
  - `test_sealed_dictionary_all_ingest_and_promote` — ≥80% ingest rate
  - `test_sealed_dictionary_routing_agreement` — ≥80% route correctly after ingest
  - `test_sealed_dictionary_retention` — ≥80% retain after restart

### Gaps
- **Routing agreement test bypasses the full router.** `_run_routing_agreement` calls `_classify_from_frame_linker` directly (`test_assistant_lexicon_mvp.py:1908`), not `OnDeviceAssistantRouter.handle()`. This means:
  - Bridge functions are not exercised.
  - The capability manifest is not checked.
  - Slot state resolution is not tested.
  - The 80% figure measures **frame linker accuracy**, not **end-to-end routing accuracy**.
- **No "next-turn use" test.** The exit gate says "correct next-turn use" — i.e., after the assistant learns a word, the *next* user utterance using that word should route correctly in a live conversation. The current tests ingest all 66 words at once and then test routing. They do not simulate a turn-by-turn conversation where word N is learned in turn T and used in turn T+1.

---

## 6. M4 — Bounded Generation

### Exit Gates
1. On Pi: report tok/s/TTFT/RSS
2. 0 unsafe applied outputs
3. 100% fallback on verifier failure
4. Model accepted on ≥70% of eligible rendering cases
5. Go/no-go recorded before training spend

### Current State
- **Authority module:** `assistant_authority.py` with `AnswerPlan`, `AuthorityEvidencePacket`, `verify_answer`.
- **Decoder scaffold:** `assistant_decoder.py` with `ConstrainedDecoder`, `TemplateBackend`, `DecodingGrammar`. `assistant_decoder_llguidance.py` with `LlguidanceBackend`.
- **Template fallback:** Verified in `test_assistant_decoder_mvp.py`.
- **Kernel wiring:** Fixed in this session — `AssistantOSKernel` now accepts `decoder` and passes it to `BoundedLocalSynthesizer`.

### Gaps
- **No Pi measurements.** `pi-smoke` checks environment constraints but does not run the decoder or measure tok/s, TTFT, or RSS.
- **No model loaded.** The `LlguidanceBackend` lazy-loads a HuggingFace model on first `decode()`, but in production the decoder defaults to `TemplateBackend` because no model is configured. The M4 exit gate "model accepted on ≥70% of eligible rendering cases" is **untestable** without a model.
- **No go/no-go record.** There is no artifact or database table recording a go/no-go decision before training spend.
- **No adversarial output safety test.** The exit gate says "0 unsafe applied outputs." There is no test that feeds adversarial prompts to the decoder and asserts that the verifier blocks them.

---

## 7. M5 — Learned Frame Linking

### Exit Gates
1. On sealed set: top-3 frame recall ≥95%
2. Accepted-route precision ≥98%
3. Zero false-local safety cases
4. No regression on supported minimal pairs

### Current State
- **Top-3 recall:** `test_top3_recall_meets_threshold` asserts ≥95%. **Passes.**
- **Precision@1:** `test_precision_top1_meets_threshold` asserts ≥95% (all) and ≥95% (precision-target subset). **Passes.**
- **False-local safety:** `test_precision_no_false_local_safety` asserts zero safety frames above expected. **Passes.**
- **Minimal pairs:** 50 pairs in `frame_minimal_pairs.v1.json`. All pass.

### Gaps
- **Precision threshold is 0.95, not 0.98.** The plan explicitly says "accepted-route precision ≥98%." The test asserts 0.95. This is a **documented gap** — the test comment even says "Target ≥95% overall, ≥95% on precision-target cases toward M5 exit criterion of ≥98%." The 0.98 target is aspirational, not enforced.
- **Reranker score now gates routing.** We changed `ScoredCandidate.score` from `rule_score` to `rerank_score`. The minimal pair tests pass, but this changes the semantics of `_classify_intent_from_uol_slots` at lines 1457-1458:
  ```python
  return top.intent if top.score > top.threshold else "unknown"
  ```
  Since `rerank_score` can be higher or lower than `rule_score`, borderline cases that previously failed the rule gate may now pass (or vice versa). The 50 minimal pairs happen to be robust, but there is **no calibrated safety margin** for this change. The frame templates' thresholds were calibrated against `rule_score`, not `rerank_score`.
- **Bridge functions still override the reranker for 5 intents.** As noted in M2, `_is_common_sense_safety_request`, `_is_social_contact_request`, `_is_personal_memory_frame`, `_is_autobiographical_debug_request`, and `_has_urgent_health_frame` all return before the linker/reranker ever runs. This means the reranker does not actually control accepted-route precision for ~45% of the frame templates.

---

## 8. M6 — Learned UOL Families

### Exit Gates
1. E2 enters shadow mode
2. Per promoted family on ≥50 sealed examples: slot/role F1 beats rule owner
3. No safety regression
4. p95 parse latency within budget
5. Ownership flip and rollback are data-only

### Current State
- **NOT STARTED.** No E2 implementation found in the codebase.
- No shadow-mode infrastructure.
- No learned UOL family models.

---

## 9. M7 — v0.2 Integration

### Exit Gates
1. All hard safety invariants green
2. Dictionary and external NLU bars reported
3. p95 TTFT <1.5 s and RSS <1.2 GB
4. If E4 ships, >30 tok/s, otherwise local-generation claims explicitly reduced

### Current State
- **Not reached.** All gates depend on M4-M6 completion.
- No integration test suite exists that runs the full pipeline end-to-end with timing.

---

## 10. Cross-Cutting Concerns

### A. `_REQUIRES_MAP` Collusion with Hardcoded Tests
The committed `assistant_authority.py` had:
```python
_REQUIRES_MAP: dict[str, tuple[str, ...]] = {
    "weather": ("weather",),
    "health_advice": ("health_goal",),
    "meal_suggestion": ("food_inventory",),
}
```
Every hardcoded test in the suite used one of these 3 intents for synthesis + authority. No test exercised `personal_memory`, `story`, `social_contact`, etc. through the full kernel → synthesizer → authority pipeline. This is a **test-coverage blind spot** that allowed a fundamental routing bug to persist.

**Fix applied:** Expanded to 14 entries covering all synthesizable intents.

### B. `_tokenize` Lowercase Fix Already in Tree
`local_assistant_router.py:2139` already has `text.lower()`. The fix was in the working tree before this session. We added a regression test to protect it.

### C. `device_action` Route Missing from Test Invariants
The random generator discovered `route="device_action"` for social_contact utterances that trigger phone calls. This route was missing from the `known_routes` set in the invariant test. Added.

### D. `_classify_from_frame_linker` vs. Full Router Divergence
Multiple tests (`test_sealed_dictionary_routing_agreement`, some frame ranker tests) call `_classify_from_frame_linker` directly instead of `OnDeviceAssistantRouter.handle()`. This means they test the frame linker in isolation, not the actual routing pipeline. As the bridge functions show, the two can diverge significantly.

---

## 11. Prioritized Recommendations

### P0 — Before Any New Features
1. **Commit the tree.** 61 files are uncommitted. This blocks M0 exit.
2. **Add an end-to-end routing test for the sealed dictionary.** Replace `_classify_from_frame_linker` with `OnDeviceAssistantRouter.handle()` in `test_sealed_dictionary_routing_agreement` so the test exercises bridges, capability manifest, and slot states.
3. **Calibrate reranker thresholds.** Since `ScoredCandidate.score` now returns `rerank_score`, either recalibrate all frame template thresholds against `rerank_score`, or add a reranker-specific margin. Currently the thresholds were set for `rule_score`.

### P1 — M1/M2 Hardening
4. **Strengthen `check_compatibility()`** to compare semantic versions or schema hashes, not just predecessor key existence.
5. **Create the 60-case normative UOL set.** This is a M1 exit gate. It should be a contract with human-verified gold parses for slot roles, predicates, and speech acts.
6. **Eliminate remaining 5 bridge functions.** Each bridge function contains hardcoded token sets that duplicate frame template logic. The path is: extend frame templates with the missing structural/context gates, add lexicon entries for bridge-only tokens, then delete the bridge function and let the linker own the intent.

### P2 — M3/M4/M5 Validation
7. **Add "next-turn use" test for M3.** Simulate a conversation where a word is taught in turn T and the next utterance uses it.
8. **Add adversarial safety test for M4.** Feed forbidden-term prompts to the synthesizer and assert `verification.passed == False`.
9. **Raise precision threshold to 0.98** or formally document why 0.95 is acceptable for the v0.2 milestone.
10. **Produce a Pi benchmark JSON** even if it records "no model available, template fallback only." The exit gate requires the artifact to exist.

### P3 — M6/M7 Foundation
11. **Design E2 shadow-mode infrastructure.** This is the critical path to M6. It needs: a UOL parse comparator, a sealed evaluation set with slot/role gold labels, and a data-only ownership flip mechanism.
12. **Add end-to-end integration test with timing.** Measure `handle()` latency for a 10-turn conversation and assert p95 < 1.5s.

---

## Appendix: Reproducibility Commands

```bash
# Full test suite (excluding slow lexicon tests for speed)
python -m pytest tests/ -q --tb=line \
  --ignore=tests/test_assistant_lexicon_mvp.py \
  --ignore=tests/test_entity_architecture_mvp.py \
  --ignore=tests/test_assistant_os_store_mvp.py

# Sealed dictionary
python -m pytest tests/test_assistant_lexicon_mvp.py::SealedDictionaryMvpTests -v

# Frame ranker
python -m pytest tests/test_assistant_frame_ranker_mvp.py -v

# Random stress-test
python -m pytest tests/test_random_conversation_sequences.py -v

# CLI gates
python scripts/local_assistant_os_cli.py shortcut-audit --json
python scripts/local_assistant_os_cli.py pi-smoke --reset --json

# Git status
git status --short
```

---

*Review conducted on 2026-06-16 against working tree at HEAD `b19579f` + 61 uncommitted files.*
