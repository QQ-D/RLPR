# Amplitude retention implementation and experiment plan

> **For agentic workers:** Use superpowers:executing-plans inline, with TDD and one fresh final code review.

**Goal:** Test whether retaining measured Fourier magnitude improves the Word model, with equal additional training in a pure-phase control.

**Architecture:** `RLPR_WORD_AMPLITUDE=1` retains raw FFT magnitude after phase filtering, the phase Transformer and training-only fusion. Attention inputs remain phase based. Default 0 preserves existing computation and checkpoint configuration exactly.

**Tech Stack:** Existing PyTorch 2.1 / CUDA 12.1 / Python 3.10 rlpr environment.

**Spec:** Word `版本1-1-饶.docx`, whose sample retains amplitude despite contrary prose; user-authorized incremental performance experiments. This is an explicit ablation, not a claim that either interpretation must work better.

## Global Constraints

- Work only in existing `src_spectral`; preserve original `src`, all old weights and results. No backup or push.
- Historical L5/R5 initialize both arms with fresh Adam, LR 5e-5, gamma .8, batch 8, seed/sampler seed 0, strict CUDA determinism and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Each arm fine-tunes L and R for exactly 3 additional epochs; historical five epochs are not relabeled as new training.
- Select each teacher by Sejong unimodal Recall@1, then earliest epoch. Cross-modal train 2 epochs, 5 hard + 5 random negatives, signed coherency v2, phase .1, fusion .5; select by Sejong R1, then F1, then earliest epoch.
- Lock all choices before DCC/Riverside tests. These benchmarks were inspected before; results are exploratory, single seed.
- Cache batching 32 for both arms, sequential dedicated generator, no validation RNG consumed in training processes.
- Fail on incompatible configurations or incomplete checkpoints. Initialization model-only is distinct from resume and from loading a frozen teacher.

## Review Focus

- Default checkpoints and output remain compatible; amplitude checkpoints cannot silently evaluate as pure phase.
- Magnitude is restored after both Transformer and fusion, including real-spectrum boundaries and zero input.
- Explicit warm start rejects cross-modal/wrong branch/incomplete weights and unrelated configuration changes.
- Teacher selection and validation never use benchmark metrics or train held-out frames.
- Parent failures leave recoverable checkpoints, immutable protocol and logs; RNG resume and source identity are checked.

### Task 1: Optional magnitude representation and strict unary warm start

**Files:** modules/word_spectral.py, modules/word_models.py, utils/pretrain.py, trainer/unimodal.py; tests/test_word_amplitude.py.

**Interfaces:** `WordSpectralFilterLayer(channels, retain_amplitude=False)`; amplitude aux includes `phase_content`, `source_amplitude`; `warmstart_word_unimodal(model, checkpoint)` loads complete model only with explicitly allowed pure-phase to amplitude config delta. Trainer consumes `params.initial_checkpoint_path` only when not resuming.

- [x] Write failing physical tests: amplitude scaling, phase-only encoder input, measured magnitude survives both attention stages, finite zero gradients, default configuration and strict variant rejection.
- [x] Add warm-start tests for complete loading, wrong branch/cross weights, unrelated config change and existing optimizer preservation.
- [x] Run new tests and observe missing-interface failures, then implement minimal optional behavior.
- [x] Run full unittest suite. Expected: all pass. Commit implementation and tests.

### Task 2: Paired experiment driver and verification

**Files:** scripts/amplitude_ablation.py, tests/test_amplitude_protocol.py, docs/word-complete-usage.md.

**Interfaces:** Driver parent freezes protocol/source/checkpoint hashes before launch; child `train`/`evaluate` commands operate one stage/epoch with explicit environment, atomic JSON and per-process logs. Store permanent evidence in a new runs/mulran_word_amplitude_* directory.

- [x] Write failing tests for validation-only ranking, equal budgets and mode-specific environment; implement driver with source identity, train/validation index checks, checkpoint finite/settings/frozen-teacher audits.
- [x] Run tests and full suite; obtain fresh readonly code review, fix important findings with failing tests first.
- [ ] Commit reviewed code, freeze protocol, run both arms through unary and cross stages, lock choices then benchmark.
- [ ] Publish numerical comparison, distinguish additional training from magnitude effect, preserve failed runs if any. Update documentation and verification JSON.

## Progress

Implementation and fresh readonly review completed; 77 tests pass. GPU experiment running. Baseline commit: 2cd8f483a9b6d5e4fbaa854a01fe80e858c0521f.
