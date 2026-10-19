# Spectral-RLPR implementation (2026-10-14)

Specification: `../版本1-1-饶.docx`, approved for implementation in this conversation.
Baseline: `ad6478a5499bab227d7f063cb77d4843dc6f9b4a`; backup under
`../backups/rlpr_code_2026-10-14_201621`. Worktree: `../src_spectral`.

## Design

Replace the two PCE modules in the spectral variant. Each branch independently
processes BEV -> ResNet shallow stem -> real FFT -> input-conditioned low-pass
amplitude mask -> phase-driven self-attention -> inverse FFT -> remaining
ResNet blocks -> existing Transformer/local pooling/NetVLAD. Keep 512-D
descriptors and the existing four-output trainer/test interface.

The spectral auxiliary outputs include circular phase, detached reliability,
and the existing 256-D local/global descriptors. Stage two freezes the entire
radar branch, including new modules and BatchNorm statistics, and trains LiDAR
with local InfoNCE + global InfoNCE + lambda * weighted phase consistency.
Default lambda is 0.1. Phase comparison uses synchronized same-frame modalities;
it does not guarantee rotation, sparsity, or translation invariance. Low-energy
frequencies and DC are excluded/downweighted. Retain amplitude information.

Use runtime FFT shapes, FP32 FFT and real-spectrum symmetry constraints. The
amplitude mask receives retrieval gradients; phase gradients train the shallow
features and phase attention. A positive real amplitude mask does not directly
change phase, but can affect it indirectly through the subsequent attention.
Attention is within each modality, so LiDAR descriptors remain precomputable.

Original `rlpr` remains the default model. Add `--model spectral` and matching
environment selection to all CLI entrypoints. Spectral stage-one checkpoints
are required for stage two; original weights cannot initialize new radar
modules as a trained/frozen teacher. Retain independent result directories.

## Implementation tasks and verification

1. Write tests for dynamic FFT shapes, finite gradients, zero-energy spectra,
   circular phase loss, and paired confidence weighting; observe missing-feature
   failures before implementing spectral layers and loss.
2. Add spectral polar/100x100 model variants retaining checkpoint key naming;
   verify 512-D descriptors, independent modalities, and stage-one weight loading.
3. Add variant selection, phase loss and loss component logs to trainers; check
   frozen radar parameters/statistics and checkpoint configuration validation.
4. Add a Sejong spectral short-run wrapper using existing processed data, plus
   training/evaluation commands and limitations documentation.
5. Run focused tests, GPU forward/backward and checkpoint roundtrip with synthetic
   inputs; run real BEV inference. Dataset training/evaluation is run by the user.

Initial model implementation excluded evaluation metric changes; model execution
verification is not a performance claim.

## Evaluation follow-up (2026-10-14)

After the user completed real Sejong short training/evaluation and reported nan
F1, fix zero/zero PR points and independent recall cutoff counting. Retain the
author's query-count-based 1percent cutoff (Top-13 for Sejong) and explicitly
print it; do not silently change the evaluation protocol. Preserve AR@1/5/10/20,
PR threshold sampling, and original nearest-candidate TP/FP/FN/TN convention.
Tests reproduce the arithmetic defects; the user reruns real evaluation using
the existing checkpoint, with no retraining required.
