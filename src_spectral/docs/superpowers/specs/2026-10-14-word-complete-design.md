# Complete Word proposal implementation

Authority: `../版本1-1-饶.docx` and the user's instruction to implement all remaining
ideas after the three gaps were explicitly listed. Implement in the existing
`src_spectral` linked worktree; preserve `../src`, existing checkpoints and runs.

## Resolution of the Word's conflicting requirements

The prose takes priority over its illustrative code: discard measured Fourier
amplitude and replace stage-two InfoNCE. Keep adaptive low-pass frequency weights,
but apply them to unit complex phase, not measured amplitude. Condition the mask
on circular phase statistics. Zero-energy bins stay zero. This guarantees positive
amplitude scaling invariance of the Fourier layer while bins remain above its
1e-7 phase-validity threshold, not invariance of an entire CNN
to arbitrary point sparsity. No unproved theoretical claims are made.

Move the original Transformer encoder architecture into the Fourier path. Its
real/imaginary sequence width is 128 (64 complex channels), with eight heads and
a 512-wide feed-forward block. Q/K derive only from circular phase; V is the
phase-derived content. The encoder runs before inverse FFT and NetVLAD. Keep the
existing local/global 256-D heads and 512-D independent retrieval descriptors.

Use the existing asymmetric paired radar-to-LiDAR phase cross-attention in stage
two. Radar parameters and BatchNorm buffers stay frozen. Test-time descriptors
remain independent because retrieval has no known corresponding other-modality
scan. Normalize the fused spectrum back to unit phase times the learned mask.

## Coherence alignment definition

Word does not define a statistical estimator or a negative-pair objective. Use
Welch-style magnitude-squared coherence: average cross/auto spectra over Hann
windows BEFORE dividing `abs(Pxy)^2/(Pxx*Pyy)`. A single unsmoothed product would
degenerate to one. For each 256-D local/global embedding, use 32-component windows
with stride 16; this is an estimator over learned descriptor components, not a
sensor time-series coherence claim. Alignment maximizes corresponding scan scores
and q/positive scores. A margin 0.2 ranking term separates the tuple's ten mined
geographic negatives; no softmax/InfoNCE is used. Detach the radar teacher.

Also estimate MSC directly on the complex spatial Fourier fields by smoothing
cross/auto spectra over 3x3 frequency neighborhoods before division. This term
aligns the actual phase topology, while a cosine phase term resolves constant
phase offsets (MSC alone cannot distinguish them).

Total stage-two loss: local coherence alignment + global coherence alignment +
0.1 * (complex-field MSC loss + circular phase consistency). Mix independent and fused LiDAR supervision at
the existing configurable weight 0.5. Stage-one geographic triplet training is
retained; Word only changes TACMA alignment in stage two.

## Interfaces and execution

Expose `--model spectral_word`; leave old `spectral` models loadable for comparison.
New checkpoint variant/config validation rejects old unimodal teachers. All six
model geometries/branches use the same new Fourier and Transformer path. Provide
a three-stage Sejong launcher, default two epochs per stage, reusing the existing
training indices and BEV when available. Always use a fresh output directory.

## Verification and completion

Tests cover positive amplitude scaling, zero spectra, finite gradients through
mask/encoder, true coherence against SciPy, positive/negative discrimination,
frozen teacher parameters/buffers, both image geometries, independent test
descriptors, checkpoint/config mismatch and exact roundtrip. Run existing tests
too. Run GPU synthetic forward/backward and actual processed-BEV execution checks;
the user runs real training/evaluation. Record commands and clearly distinguish
execution checks from measured recognition improvements.

Reference for the coherence definition:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html
