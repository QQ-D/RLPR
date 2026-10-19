# Official-code epoch budget implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline and TDD. User explicitly requested continuing with the paper's epoch count.

**Goal:** Run the pure-phase/amplitude comparison using the author's MulRan code budget: 5 LiDAR + 5 radar + 1 cross-modal epoch.

**Architecture:** Add an explicit `--schedule official_code` to the existing experiment driver. Both arms start with the same seed and torchvision ResNet18 initialization, train their unary branches for five total epochs, then use epoch5 checkpoints for one cross-modal epoch. Preserve the old continuation schedule and its evidence.

**Spec:** User request on 2026-10-18. Local paper implementation-details paragraph states Adam LR5e-5 and epoch-wise .8 decay, but no MulRan epoch count. Original `src/scripts/registry.py` uses UNIMODAL_EPOCHS=5 and CROSS_MODAL['mulran']['epochs']=1. This changes the epoch budget only; modified Word modules/loss/mining policies remain the experiment's methods.

**Tech Stack:** Existing rlpr environment, PyTorch, CUDA, MulRan processed training/validation/benchmarks.

## Constraints

- Original src and all previous checkpoints/results stay intact. Work in existing spectral-rlpr worktree.
- Official-code schedule means total5/5/1, not five additional warm-start epochs. Both arms use epoch5 unary weights, not a best earlier epoch.
- Same dataset exclusions, batch8, seed0, strict determinism, LR5e-5/gamma.8, cache32 and existing alignment/mining settings in both arms.
- Store a new immutable protocol and source hashes before launch; benchmarks follow both completed arms. Preserve interrupted old experiment as a separate superseded record.
- Make status JSON publication durable (file and directory fsync), since run.json was empty after the host restart; do not claim the precise cause of the empty file is proven.

## Task 1: Epoch policy and durable records

Files: scripts/amplitude_ablation.py; tests/test_amplitude_protocol.py.

Interfaces: `experiment_settings(schedule='continuation')`; `stage_epoch_budget(protocol, stage)`; `choose_stage_epoch(rows, protocol, stage)`; trainer child reads the frozen protocol before deciding fresh versus warm-start initialization.

- [x] Add tests for official5/5/1, final unary checkpoint selection even when an earlier validation score is higher, old continuation behavior, and flush failure preserving an existing JSON record.
- [x] Observe RED, implement the schedule and durable write behavior, then verify all tests and CLI help.
- [x] Fresh readonly review; fix important findings using RED/GREEN. Commit verified changes.

## Task 2: Launch and monitor

- [x] Freeze a new official-code protocol with backbone-initialization hash and verified data isolation.
- [x] Launch the complete driver, verify fresh initialization and first optimization steps. Record old run as superseded while preserving its data.
- [x] On completion verify checkpoints, frozen teacher, sampling streams and results; report DCC/Riverside against official/pre-existing results. If user pauses, freeze processes and state explicitly.

## Review Focus

- Fresh schedule must never silently load historical L5/R5 and add five more epochs.
- Cross stage must load each arm's actual epoch5 teacher/student checkpoints.
- Continuation runs retain their frozen protocol and defaults.
- Resume status must become RUNNING before work and retain optimizer/RNG exactness.
- A file-flush failure before replacement must leave the prior report readable. A directory-flush failure after replacement raises and retains the complete new JSON; no unconditional rollback guarantee.

## Review outcome

79 tests passed. Fresh readonly review found no critical/important blockers. Deferred minor: eager fallback lookup in stage_epoch_budget requires the legacy additional-epochs key even when a total-epochs key exists; all generated/frozen protocols include that key. Directory-sync failure semantics clarified above. GPU execution/convergence and actual power-loss durability remain runtime observations, not claims established by CPU tests.

## Active run

`runs/mulran_word_amplitude_official_2026-10-18_193110`: protocol official_code, budgets5/5/1, backbone-only initialization hash. First child log confirms Training From Scratch and optimization has begun. Old continuation run retained as SUPERSEDED. Final training/evaluation results remain pending.

## Completion (2026-10-19)

Run finished at03:49:42+08:00 after resuming from the saved first LiDAR epoch following another host restart. All22 checkpoints were re-audited: hashes, finite model/optimizer states,2071 optimizer steps per epoch, paired sampler/worker streams, frozen cross-stage teachers, source/protocol/selection/initialization hashes verified. Fixed selections L5/R5/cross1 in both arms.

Pure-phase DCC/Riverside R@1:51.06/53.89%; retained-amplitude:55.32/62.82%. Matched gains:+4.26/+8.93percentage points. Retained amplitude still trails released official Riverside65.26% by2.44points. Single-seed exploratory results, no claim of superiority over RLPR. Exports: comparison.csv, curves.csv, results_summary.json, completion_verification.json in the run root. Final Chinese PDF: reports/2026-10-19/RLPR修改版最终实验结果汇报_2026-10-19.pdf. No further training launched.
