# Longer MulRan alignment implementation plan

**Goal:** Establish whether cross-modal training is limited by its one-epoch budget before changing the learned representation.

**Execution:** Native implementation in the existing `src_spectral` worktree, authorized by the user's instruction to proceed step by step. Preserve `src`, existing experiments and all pretrained weights.

## Fixed experiment

- Reuse Word LiDAR/Radar epoch5 weights from `mulran_word_full_2026-10-14_232504`.
- Full Sejong training regions + KAIST01/02; batch8; seed0; separate worker/sampler generators seeded0.
- Signed coherency v2, phase auxiliary0.1, cross-fusion0.5, Adam5e-5, StepLR gamma0.8, frozen radar.
- Train total epochs1 through5, validating each on Sejong's geographic held-out regions.
- Select highest validation Recall@1, then F1, then earliest epoch. Lock selection before DCC/Riverside evaluation.
- Compare selected and final checkpoints with the epoch1 control; do not choose epochs from benchmark scores.
- Checkpoints at epoch boundaries include Python/NumPy/Torch/CUDA and loader/sampler random state.

## Tasks

- [x] 1. RED tests: epoch resolution (CLI > environment > dataset default), invalid values, resumed epoch ceiling.
- [x] 2. Implement explicit `--epochs` in both training CLIs and fix environment precedence without changing default epochs.
- [x] 3. RED tests: uninterrupted vs resumed toy optimization with shuffled data and stochastic operations; teacher/loss resume compatibility.
- [x] 4. Implement and integrate epoch-boundary random-state capture/restore, deterministic optional sampler seeding, and no-op resume rejection.
- [x] 5. Run the complete CPU suite; review changes; resolve substantive findings before training.
- [x] 6. Freeze run protocol and source hashes; train epochs1–5 with fresh per-epoch subprocesses, restore states, validate each epoch.
- [x] 7. Verify finite weights, unchanged radar, actual update counts, learning-rate schedule, and the strict within-run epoch1 control; record historical numerical differences separately.
- [x] 8. Lock selection; test the declared checkpoints on DCC/Riverside; save CSV/JSON and interpret the validation curve.

## Next stage

After inspecting this training curve, define a separate controlled hard-negative experiment. Preserve the longer-training baseline; change only the sampling policy. Multi-positive objectives and amplitude/spatial representations require subsequent independent experiments rather than changes to this run.

## Verification focus

- An explicit epoch default must not swallow an environment override.
- Resuming to the already completed epoch must fail clearly instead of silently performing no updates.
- Restoring model/optimizer/scheduler alone is insufficient for reproducible shuffled training.
- Evaluation subprocesses must not consume the training process's random streams.
- Training-state restoration must handle checkpoint tensors loaded onto CPU and must not initialize CUDA for CPU-only tests.

## Execution record

- 53 regression tests passed on 2026-10-15, including exact stochastic resume with workers0 and1.
- Independent review found legacy top-level loss metadata was skipped by the compatibility check. Added failing regression cases, implemented fallback, and reran all53 tests successfully.
- Independent experiment-driver review found no blocking issues; confirmed the fixed Sejong database/query split has no held-out frame overlap.
- First attempt `/media/rwj/ZX1 2TB/RLPR/runs/mulran_word_longer_2026-10-15_232554` stopped after epoch1: historical GPU weights were not bitwise reproduced despite identical sampling generators and first two batch losses. Full diagnosis and short probes are preserved there; no epoch selection or benchmark tests were performed.
- The old policy only fixed cuDNN. Strict PyTorch checks raised an error without the CuBLAS workspace setting. Two independent strict three-batch probes with `:4096:8` produced bitwise-identical weights. Historical allocator/kernel behavior remains unreconstructed; use the new run's own epoch1 for causal comparisons.
- Added `RLPR_DETERMINISTIC=1`, requiring `CUBLAS_WORKSPACE_CONFIG=:4096:8` (or `:16:8`) before launch. Train and evaluation subprocesses now share this policy. Runtime reproducibility settings are saved and checked on resume. 54 regression tests passed after a RED/GREEN policy test.
- Active strict run: `/media/rwj/ZX1 2TB/RLPR/runs/mulran_word_longer_2026-10-15_233921`.
- Source Python files are hashed in `protocol.json`; the driver checks them before each training/evaluation subprocess. Existing source is frozen until this experiment completes.

## Completed result (2026-10-16 00:33)

- Strict run completed all5 epochs and the declared benchmark tests. Checkpoint,
  protocol and selection hashes verified afterwards. Radar state unchanged;
  optimizer step counts2071,4142,6213,8284,10355; LR follows the frozen schedule.
- Validation Recall@1:72.75,72.75,72.46,73.04,72.39%. Epoch4 selected before tests.
- Epoch1 DCC/Riverside Recall@1:53.72/55.86%; selected epoch4:53.19/52.96%;
  final epoch5:52.66/53.07%. More epochs did not improve these benchmark scores.
- Do not extend this configuration to10 epochs. Proceed with the separate
  default-off cross-modal hard-negative experiment. Original `src` is clean.
- Full results: `docs/verification/mulran-word-longer-2026-10-16.json` and run CSV.
