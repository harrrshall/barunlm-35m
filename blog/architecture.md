# StrataLM-35M design rationale

StrataLM is a decoder-only model organized into three four-layer strata. It is designed for a single 24GB GPU, useful 2K-token training contexts, and eventual streaming inference. The selected configuration has exactly 35,072,768 physical parameters. It is `configs/strata_no_mtp_35m.json`; `configs/strata_no_mtp_no_selector_35m.json` is the faster, slightly lower-quality variant.

## Decisions

**Backbone.** Twelve PreNorm decoder blocks at width 448. This is deep enough to build hierarchical features while leaving approximately 20% of parameters for a 16,384-token embedding table. Dense blocks activate all capacity on every token; small-scale MoE would strand parameters behind undertrained routes.

**Sequence mechanism.** Seven query heads share one KV head (head dimension 64). Layers 1–3 in each stratum use a causal 256-token window; layer 4 uses full causal attention. Local layers model morphology, syntax, and nearby code dependencies at bounded cache size. The anchor layer provides exact global retrieval, preventing compressed/local-only information loss. FlashAttention's native window mode is used when installed; otherwise a compiled PyTorch FlexAttention block mask realizes the sparse window on CUDA, with masked SDPA retained as the portable correctness fallback.

**Position.** RoPE is applied to 32 of each head's 64 channels, with base 10,000. The rotated half expresses relative position while the unrotated half remains a pure content-matching subspace. Learned absolute embeddings were rejected because they add parameters and extrapolate poorly; ALiBi was rejected because a fixed monotone distance bias is restrictive for code and long-range references.

**Attention controls.** Shared per-head-dimension RMS normalization is applied to Q and K immediately before RoPE. A sigmoid projection gates attention output channel by channel. QK norm prevents logit scale drift; the output gate lets a layer decline an irrelevant attention update. The gate adds 2,408,448 parameters across the model. A parameter-matched removal was 2.3% faster but 17.6% worse in perplexity, so the gate is retained.

**Feed-forward.** A dense 1,228-wide SwiGLU uses one fused gate/up projection and one down projection. Gate values are capped above 10 and the linear branch is clamped to `[-10, 10]`. This preserves SwiGLU's multiplicative expressivity while preventing isolated BF16 outliers. Routed experts were rejected at this scale because all 35M parameters should receive every token rather than being fragmented behind undertrained routes.

**Depth flow.** At the end of every four-layer stratum, the model RMS-normalizes the stratum input and output, scores both, softmaxes across the two candidates, and takes their convex combination. This checkpoint residual selector can bypass an unhelpful stratum and bounds its mixer. It costs only 1,344 parameters rather than mHC's multiple persistent streams or K3's growing depth bank. Under Muon, removing it improved throughput by 5.3% but worsened perplexity by 1.2%; it is retained in the quality reference and disabled in the efficient variant.

**Normalization.** Learnable RMSNorm before attention/FFN plus a final RMSNorm. RMSNorm omits mean subtraction, works well in mixed precision, and is cheaper than LayerNorm. Learnable scales are retained because there is no deep weight-shared recurrence that would motivate HRM's parameterless form.

**Tokenizer and vocabulary.** A 16,384-entry byte-level BPE is trained on the exact pretraining mixture. Byte coverage prevents unknown-token failures in code, Unicode, and formulas. The vocabulary is much smaller than K3/V4 because, at 448 width, a 128K vocabulary would consume more parameters than the entire intended network. Input and output embeddings are tied, saving 7.3M parameters and regularizing rare tokens.

**Optimizer.** The quality reference uses Muon for ordinary 2D linear matrices and AdamW for tied embeddings, normalization scales, and other non-matrix parameters. AdamW uses `(beta1,beta2)=(0.9,0.95)` and weight decay 0.1; Muon uses momentum 0.95, Newton–Schulz orthogonalization, and the same decay. At the same peak learning rate, Muon improved perplexity by only 0.6% and reduced throughput by 5.0% versus full AdamW. Muon is therefore the quality default, not an unconditional efficiency win; AdamW is the sensible faster option.

**Learning-rate schedule.** Five-percent linear warmup, an 80% constant plateau, then 15% cosine decay to 10% of peak. Warmup protects early BF16 training, the plateau allows Muon's token-efficiency advantage to appear, and late decay consolidates rather than spending the whole short pilot continuously changing step size.

**Objectives.** The selected pretraining objective is ordinary causal next-token cross entropy. The initial candidate added weight 0.2 on a shared-head prediction two positions ahead, but removing it lowered perplexity by 3.0%, increased throughput by 12.8%, and reduced peak memory by 11.2%; MTP is therefore rejected at this scale and budget. The implemented completion stage can mask user/system labels and optionally use a PrefixLM prompt mask, but no post-training quality claim is made without a matched SFT experiment.

**Data mixture.** 50% deduplicated FineWeb-Edu, 20% Cosmopedia v2, 20% FineMath-4+, and 10% documented Python functions from CodeSearchNet. Broad educational text dominates; coherent synthetic exposition repairs web fragmentation; high-quality math supplies explicit reasoning chains; code plus docstrings supplies formal structure without overwhelming a small model. Details and licenses are in `docs/datasets.md`.

**Regularization.** Deduplication/quality filtering, document-level held-out hashing, tied embeddings, weight decay 0.1, gradient clipping at 1.0, and bounded activations. Dropout is initially zero because the token stream is much larger than the model and dropout confounds throughput; it remains configurable if longer runs show a train/validation gap.

**Inference.** GQA reduces KV cache by 7× relative to seven independent KV heads. Nine of twelve layers retain at most 256 KV positions; only three anchor layers keep full history. Partial RoPE and gating add no cache. The implemented decode path maintains a separate cache per logical layer, truncates local caches after prefill and each token, and preserves absolute RoPE offsets. At an 8,128-token prompt, the measured cache is 6.83 MB versus 24.97 MB for the dense GQA baseline. Median-of-three L4 timing does not show a speed win: Strata prefill is 6.9% slower and decode is 32.8% slower. Weights are BF16 during research; weight-only INT8/INT4 export is deferred until accuracy is established.

## Expected trade-offs

The hybrid pattern demonstrably reduces long-context cache memory and improves pilot validation quality, but it does not currently improve wall-clock speed on the tested L4. The compiled FlexAttention dispatch, three selector operations, and full-rank output gate retain fixed overhead, especially during one-token decode. The output gate and selector earn their quality costs; MTP does not. The matched 16.384M-token runs remain architecture pilots; the later 700M-token run establishes healthy validation scaling for the selected design but still does not establish downstream capability or long-context retrieval accuracy.
