# Controlled cross-modal hard-negative experiment

## Purpose and fixed design

The first experiment changes only the cross-modal epoch budget. Its first three
validation Recall@1 values are72.75,72.75,72.46%; better loss alone has not yet
improved the first retrieval result. Finish the declared five epochs first.

The next experiment changes the training sampling policy while retaining the
Word architecture and signed alignment loss. Each 12-frame tuple contains the
query, one geographic positive, five hard negatives and five random negatives.
Both kinds of negative must be in the existing training-only candidate set
(geographic distance strictly greater than12m). Random examples exclude the
five selected hard examples and are distinct when sufficient candidates exist.

Mine the same retrieval direction as evaluation: frozen radar query descriptors
against current LiDAR database descriptors. Extract independent descriptors in
evaluation mode without cross-fusion, using the training database followed by
training queries. Refresh once before each training epoch. Cache frame order
must equal the triplet dataset order, including offsets between info files.
Reject an incomplete/shuffled cache or an invalid geographic negative.

Use the same L5/R5 weights, batch8, seed0, strict deterministic runtime, signed
coherency_v2, phase auxiliary0.1, fusion0.5, Adam5e-5 and StepLR0.8. Run the same
five-epoch search budget and Sejong validation rule as the longer-training
baseline. Declare the protocol before training; lock selection before DCC and
Riverside. Compare same-epoch curves as well as each method's validation-selected
checkpoint. Benchmarks have been examined before and remain exploratory.

This is an engineering sampling ablation beyond the supplied Word design.
Amplitude, spatial branches and objective changes are excluded from this
experiment. Original `src` and all existing weights are preserved.

## Implementation and verification

- [x] RED/GREEN: allowed-candidate ranking, stable ties, finite descriptors,
  capacity checks, mixed sampling and preservation of later positive draws.
  Six helper tests pass.
- [x] RED/GREEN: training database/query cache offsets, unchanged training RNG,
  model-mode restoration, frame-order mismatch and geographic exclusion.
  Three cache helper tests pass.
- [x] RED/GREEN: configuration flag, actual MulRan tuple sampling integration,
  and rejection of changing an old random-negative checkpoint into hard mining.
- [x] After the baseline finishes, integrate default-off
  `RLPR_CROSS_HARD_NEGATIVES` (0..10; experiment5) into parameters, checkpoint
  settings, the cross-modal trainer and MulRan dataset. Keep the old random and
  unimodal hard-mining paths unchanged when this feature is disabled.
- [x] Run the complete regression suite and an independent read-only review.
  Resolve substantive findings before GPU training.
- [x] Freeze protocol/source hashes; train/evaluate epochs1..5 using exact
  epoch-boundary resume and independent cache/worker/sampling random streams.
- [x] Check all hard candidates, unchanged frozen radar, finite losses/weights,
  optimizer update counts, LR schedule and actual mixed tuple sampling.
- [x] Lock Sejong selection, evaluate declared checkpoints, save comparison
  JSON/CSV, and retain the stronger validated configuration.

## Runtime isolation during preparation

Helpers/tests were added as new modules while the longer-training experiment
was running. They are not imported by its frozen production code. Do not edit
the existing dataset, parameters or trainer until that run has completed.

## Execution record

- The baseline completed at00:33 on2026-10-16. It did not improve benchmark
  Recall@1, so no extension to10 epochs was made.
- All66 regression tests passed after integration. Independent helper and
  integration reviews found no blocking issues. Baseline initialization hashes,
  architecture/loss/evaluation hashes and fixed hyperparameters matched.
- The new protocol also checks the completed baseline protocol/hash and its
  fixed model/configuration files before training.
- Active run: `/media/rwj/ZX1 2TB/RLPR/runs/mulran_word_hardneg_2026-10-16_003626`.
- Existing Python sources are now frozen for this experiment. Source changes to
  amplitude, spatial branches or losses must wait for completion.

## Completed result

- All5 epochs completed; losses1.6580013,1.4696172,1.3842536,1.3343748,1.2985636.
- Validation Recall@1:72.83,73.48,72.68,72.83,73.48%. Epoch2 selected using F1
  to break the tie with epoch5, before benchmark evaluation.
- Selected DCC/Riverside Recall@1:55.32/56.09%; F1:0.73929961/0.73103448.
- A matched-epoch analysis was declared during epoch4, before final selection.
  Additional baseline epoch2 tests gave54.26/55.45% Recall@1. Thus the sampling
  change adds1.06/0.64 percentage points at the same training duration. DCC F1
  decreased slightly (0.74015748→0.73929961); not every metric improves.
- Epoch5 hard-negative DCC/Riverside Recall@1:53.72/53.77%; do not select it
  merely because it is the last checkpoint.
- Protocol, selection, all checkpoint/cache hashes verified. Frozen radar
  unchanged; sampler/worker states match the baseline at each epoch. All mined
  negatives are>12m away. All66 tests passed again; log stored in the run folder.
- Current results remain below official Riverside performance. These are
  single-seed exploratory comparisons, with no claim of stable paper gains.
- Original completed baseline report remains immutable; matched evaluation
  outputs are stored separately as `matched_epoch_results.json` and CSV.
- Read-only teacher diagnostics completed after tests: Riverside official vs
  selected Word R2R Recall@1=96.06/90.02%, L2L=84.22/77.78%. DCC R2R is close,
  L2L identical. Representation quality deserves targeted investigation before
  assuming a longer alignment stage will resolve the remaining Riverside gap.
  This is not proof of a particular module's causal effect; amplitude/spatial
  changes need independent training and ablation, including a trained radar
  teacher. Full diagnostic output is preserved in the verification directory.
