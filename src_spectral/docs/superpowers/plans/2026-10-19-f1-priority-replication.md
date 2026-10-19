# F1 priority replication implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Execute inline under the user's continuing authorization.

**Goal:** Check whether InfoNCE's F1 improvement persists across cross-stage seeds 0, 1, 2, with F1 as the primary validation criterion.

**Architecture:** Add a separate driver using unchanged model, data, loss, and evaluator modules. Reuse immutable seed-0 evidence and fixed amplitude L5/R5; train signed and InfoNCE once for each new seed. Collect nearest-neighbor errors and threshold statistics through the existing tester's F1 hook.

**Tech Stack:** Python 3.10, PyTorch, NumPy, FAISS, unittest.

**Spec:** User steering on 2026-10-19: prioritize F1; continue error analysis and paired random-seed replication already announced.

## Global constraints

- Preserve original src, historical code, checkpoints, locks and reports.
- Both global/model seed and sampling seed are 1 or 2; unary checkpoints remain seed 0. This is cross-stage replication only.
- Each new run has 1 cross epoch, 2071 updates, phase auxiliary .1, fusion supervision 0, raw amplitude, Adam 5e-5, 5 hard + 5 random negatives, deterministic execution.
- Select loss by mean Sejong F1 across seeds 0/1/2, then mean Sejong R1, then signed. Do not select an individual lucky seed.
- Lock selection after all validation, before new-seed DCC/Riverside evaluations. Earlier benchmark results have been seen; this is exploratory.
- Preserve the evaluator's legacy maximum-F1 definition. A frozen validation threshold is a separately labeled diagnostic, never substituted for official F1.

## Review focus

- Model seed must change, not merely sampler seed.
- Missing, nonfinite, duplicated or incomplete seed results must not select a loss.
- Checkpoint and metric hashes must bind cached results during restart.
- F1 numerator/denominator and strict threshold comparison must match the existing evaluator, including empty-positive queries.
- Freeze selection and source before benchmark evaluation; do not use test metrics to choose.

### Task 1: Diagnostic and replication helpers

Files: create `scripts/f1_priority_replication.py`, `tests/test_f1_priority_replication.py`.

- [ ] Write failing tests for F1-first aggregation, paired seeds, seed propagation, checkpoint audit, legacy PR statistics and fixed-threshold counts.
- [ ] Run tests and observe missing-feature failures.
- [ ] Implement selection, seed-aware training, checkpoint auditing and nearest-neighbor diagnostic extraction using the existing evaluator.
- [ ] Run focused and complete regression suites.

### Task 2: Frozen experiment execution

- [ ] Implement orchestration with immutable source/protocol/checkpoint hashes and restart-safe result bindings; test lock tampering and missing results.
- [ ] Obtain fresh read-only review, fix material issues, commit new files.
- [ ] Launch four new training trials, validate all, lock F1-first loss selection, evaluate all arms and seeds.
- [ ] Compare per-query errors for seed 0, including original released model; show legacy max F1 and validation-frozen threshold separately.
- [ ] Audit teacher freeze, update counts, paired sampling streams/mined negatives, source hashes and all results; report per-seed and mean/sample SD with caveats.

## Ledger

- Ruling: no renewed design approval is required; the user repeatedly authorized autonomous execution and explicitly changed the primary metric to F1.
- Ruling: do not retune temperature or train duration in this replication, so the objective comparison stays interpretable.
