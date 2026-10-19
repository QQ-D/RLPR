# Word-complete Spectral-RLPR Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans. Execute in the existing linked worktree with tests first.

**Goal:** Implement the remaining prose requirements of the Word proposal.
**Architecture:** A separate spectral_word variant uses pure phase and the original Transformer architecture inside the frequency path, paired training cross-attention, and coherence alignment.
**Tech Stack:** Existing Python 3.10, PyTorch 2.1, torchvision and SciPy; no new dependencies.
**Spec:** `docs/superpowers/specs/2026-10-14-word-complete-design.md`

## Global Constraints

- Preserve original src, all prior model modes, data, weights and result directories.
- No extra backup, no GitHub upload, no full real-data training by the agent.
- FP32 FFT; 512-D independent descriptors; radar teacher frozen including buffers.
- New teachers must be trained using the new architecture, not silently supplied by old weights.

## Review Focus

Zero energy and nearly zero energy; coherence requires multiple windows;
geographic positives cannot become negatives; paired contexts cannot leak into
evaluation; all teacher modules/buffers and old checkpoints must remain safe.

## Task 1: Phase-only frequency path and phase Transformer

- [x] Write `tests/test_word_spectral.py` for scaling invariance, zero input, finite mask/Transformer gradients, and Q/K phase independence from amplitude; observe missing-module failures.
- [x] Implement `modules/word_spectral.py` and `modules/phase_transformer.py` with unit-phase normalization, adaptive phase-conditioned low-pass weights, real-spectrum projection and inherited Transformer attention/FFN/residual architecture.
- [x] Run the tests, expect all pass.

## Task 2: Coherence loss and complete model integration

- [x] Write `tests/test_word_models.py` and `tests/test_coherence.py`: SciPy reference agreement, teacher detachment, separation of positives/negatives, both geometries, old weight rejection, independent evaluation and trainer loss without InfoNCE.
- [x] Implement `modules/coherence.py`, `modules/word_models.py`; connect variant selection, trainer dispatch and strict checkpoint validation.
- [x] Run new and existing unit tests, expect all pass.

## Task 3: CLI, launcher and verification

- [x] Add spectral_word to all three CLI entrypoints; default paired fusion on for this variant.
- [x] Add `scripts/sejong_word_train.sh` with existing indices/BEV, new three-stage teachers, status metadata and fresh outputs.
- [x] Verify CLI help, bash syntax, all unit tests, GPU two-step update/freeze and checkpoint roundtrip, and a real-BEV forward/backward.
- [x] Document exact user train/test commands and Word coverage, request independent review, fix material findings, and commit locally.

## Execution ledger

Pre-flight: Fourier auxiliary spectra feed paired fusion; model_variant routes both trainer and checkpoint validation; all six registry classes must map consistently.
Ruling: User's request to implement the previously listed gaps authorizes this design and inline execution; no repeated approval gate.
Task 1 complete: missing-module RED -> four phase-layer/Transformer tests GREEN.
Task 2 complete: missing-module/function RED -> coherence/model tests GREEN; full suite 40/40.
Ruling: Word leaves the coherence estimator unspecified. Use both direct 3x3 complex-field MSC and Welch descriptor MSC with geographic negatives; cosine consistency resolves constant phase offsets. Wrong choice could reduce recognition accuracy, which must be measured by real training/evaluation.
GPU finding: checkpoint tensors were bitwise identical, but restoring radar trainability changed CUDA inference by up to 8.2e-6. Restoring frozen radar made outputs bitwise identical. Added failing evaluation freeze test and fixed Word checkpoint evaluation to restore frozen teacher.
Final review: independent review found no Critical/Important issues; CPU 40/40 plus Cartesian branches independently verified.
Final minor (deferred): malformed numeric launcher environment strings can abort before failure-status metadata is written; normal validated values run correctly and the terminal traceback identifies the invalid string.

Task 3 complete: all three CLI modes, exclusive three-stage launcher, shell syntax and diff checks passed; final CPU suite 41/41. GPU synthetic updates, real BEV backward, both geometries and exact frozen-teacher roundtrip passed. Implementation commit 0ba65cb. No merge or push.
