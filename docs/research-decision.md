# BarunAction-35M research decision

Status: measurement and trial designs frozen before post-training on 2026-08-03; development
evidence is current through the frozen four-trial rescue selection and candidate-v2 ARM64 int8
retention result on 2026-08-03. This document separates release evidence, project hypotheses, and
go/no-go decisions. A usable narrow research package does not convert reused-development results
into a blind breakthrough.

## Decision

Build a local personal-action compiler that turns a short request, a compact candidate-tool schema,
and optional conversational/structured context into typed action(s) or a safe abstention,
clarification, or confirmation request. The first
release is English-centric and covers reminders, calendar, contacts, notes/lists, maps, messages,
media, and device settings.

The terminal deliverable is a runnable post-trained model: checkpoint/tokenizer, stable prompt and
action contract, Python API and CLI, deterministic safety/validation layer, sandbox demo, model and
data documentation, quantized artifact or an explicit runtime blocker, and reproducible evidence.
The documents and experiments in this repository serve that product; they are not the product.

Canonical model outputs are deliberately short Action IR v1 JSON objects: `CALL` and `CONFIRM`
carry typed tool calls, `CLARIFY` carries the sorted missing-field list, and `ABSTAIN` has no
payload. The exact serialization is frozen in `docs/evaluation-protocol.md`. Real side effects,
authorization, confirmation, and policy enforcement stay outside the model.

### Current checkpoint decision — 2026-08-03

The frozen outer rescue selector evaluated exactly four reused-development trials—continual
recovery C and interpolation trials I25, I50, and I75—and found **no eligible joint checkpoint**.
The selection receipt is
`experiments/runs/20260803-2203-parallel-rescue-selection-s17/selection-receipt.json`, SHA-256
`e021eff97cf4816ac299dc55cf6d2c356b07754639fe136540acef5987897582`. It authorizes no
additional alpha, recovery-recipe retry, or cross-metric compensation.

Candidate-v2 remains the current BarunAction-35M product/release-development checkpoint and Mobile
incumbent. Its model SHA-256 is
`fdb95ccf58a095e0d321be998924318b35ee59a334f6dd97d8726d2cf80021d3`. This is a conservative
fallback decision, not evidence that candidate-v2 passes the joint Mobile-plus-PRESTO gate or that
the broader research target passed.

The candidate-v2 Darwin ARM64 dynamic-int8 artifact subsequently passed a separate frozen
post-selection retention gate at 607/756, versus the preregistered 602/756 H200 BF16 reference and
a 603/756 same-host FP32 control. It produced all 756 rows with no failures or truncations. This
authorizes packaging that exact runtime-bound artifact; it does not reopen candidate selection or
establish latency, energy, safety, official-test, or hidden-suite performance.

The independently verified and locally imported Qwen matched-baseline run changes the larger-model
decision. The 494,032,768-parameter (0.494B, not 500B)
BF16 checkpoint contains 290 tensors and scored 663/756 (87.70%) strict AST exact, versus
candidate-v2's 602/756 (79.63%). The paired outcomes are 80 Qwen-only wins, 19 Barun-only wins, and
657 ties; Qwen produced 755 parse-valid and 754 schema-valid outputs. BarunAction is 14.09 times
smaller and retains 90.80% of Qwen's exact-match accuracy, but trails by 8.07 percentage
points. The 961 official Mobile rows remained untouched. The hypothesis that candidate-v2 beats a
strong larger matched baseline is therefore **rejected for this development comparison**. The
earlier SmolLM2 0/756 result is model-and-recipe-specific and does not override this counterexample.
Run `20260803-2122-mobile-qwen05b-matched-s17` and its 63-file selected import are hash-checked
locally. The import excludes the comparison checkpoint, full Jarvis inventory, and remote log;
`import-receipt.json` records the exact boundary.

Continual-recovery trial C is the closest result and remains **research-only**. Its checkpoint
SHA-256 is `744b640828122ea9805131671c9792af9acfbd748e3199bacc513dcd8799c4c9`:

- Mobile Actions derived-development AST exact was 642/756 (84.92%), passing its frozen gate.
- PRESTO derived Action IR AST exact was 9,862/14,288 (69.02%), 140 correct rows below the frozen
  10,002-row threshold.
- Every other frozen hard check passed: PRESTO schema validity was 14,254/14,288, abstention F1 was
  0.97435, false calls were 102/10,407, both revision/disfluency gaps were below 0.10, all four
  revision raw-label families were present, and Mobile recorded zero catastrophic unauthorized
  actions and 0/0 eligible false actions.

Checkpoint interpolation showed a steep task tradeoff rather than a joint solution. I25 retained
Mobile at 598/756 but scored 0/14,288 PRESTO exact; I50 scored 452/756 and 961/14,288; I75 reached
9,447/14,288 PRESTO while collapsing Mobile to 8/756. None is eligible or promoted.

These are one-seed results selected on reused public development populations. PRESTO is scored as
derived Action IR, not native semantic parsing. The runs performed zero official-test evaluations:
the 961 Mobile official rows and 194,118 PRESTO official-test rows remained opaque and unread.
Nothing here establishes a blind breakthrough, general larger-model superiority, hidden-suite
safety, cross-seed robustness, or quantized target-device performance. The release decision is
narrower: package candidate-v2 as a reproducible proposal-only research artifact while preserving
these failed broader gates as first-class evidence.

### Headline hypothesis — rejected for candidate-v2

> With identical eligible inputs and scoring, a 35M causal model can exceed fairly evaluated
> 270M--600M models on exact personal-action execution involving context, corrections,
> disfluencies, irrelevant requests, ambiguous requests, distractor tools, and renamed or unseen
> schemas.

This is falsifiable. Mobile Actions alone cannot establish it, valid JSON cannot establish it, and a
comparison that gives Barun fewer candidate tools or stronger decoding constraints cannot establish
it. Parameter count, post-training tokens, pretraining tokens, training compute, latency, and memory
must all be disclosed.

Candidate-v2 does not satisfy the larger-baseline part of this hypothesis: the imported Qwen
matched result is 663/756 versus 602/756. The broader hidden-suite and safety portions remain
unmeasured, not rescued by the model's parameter-efficiency result.

## Why this task

The task is useful without requiring factual recall, browsing, or unsafe autonomous behavior. It is
small enough for a 2,048-token context, produces objectively executable structures, and can run
privately on low-power hardware. Errors can be classified by tool choice, argument grounding,
schema validity, abstention, and simulated execution rather than by a subjective judge alone.

It is also a credible but nontrivial tiny-model target. Google's 270M
[FunctionGemma](https://deepmind.google/models/gemma/functiongemma/) is explicitly designed for
local function calling and rises from 58% to 85% on Mobile Actions after task fine-tuning. Cactus
Compute's [Needle repository](https://github.com/cactus-compute/needle) reports a 26M specialized
model outperforming several 270M--600M models on single-shot personal tool calls. Needle is useful
feasibility evidence and a mandatory peer baseline, but its headline comparison is not a substitute
for our evaluation: it is not a peer-reviewed result, its public repository does not currently expose
the full sample-level benchmark evidence, and it reports much more pre/post-training data than
BarunLM has seen.

The novelty bar is therefore robust, contextual, leakage-controlled action compilation—not the fact
that a tiny model can emit a function call.

## Alternatives considered

1. **Local PII/secret redaction** was second. It has exact span metrics and high privacy value, but
   decoder-only copying is a poor inductive fit and a single false negative can be high-stakes.
2. **Short grammar/spelling correction** has abundant synthetic pairs, but multiple valid outputs
   complicate exact evaluation and edit-tagging architectures have a strong structural advantage.
3. **Inverse text normalization** is exactly scoreable, but mature WFST/hybrid systems are already
   strong and public training material is limited.

The action compiler offers the best combination of everyday utility, deterministic verification,
licensable data, and architectural fit. If the small SFT gate fails, PII redaction is the planned pivot;
RL is not a rescue mechanism for a model that has not learned the basic task.

## Public evidence and benchmark roles

### PRESTO: primary public task

[PRESTO](https://github.com/google-research-datasets/presto) is CC BY 4.0 and contains over 550,000
contextual conversations in six languages, including prior turns, contacts, lists, notes,
disfluencies, revisions, and code switching. English has 95,671 examples. The
[EMNLP 2023 paper](https://aclanthology.org/2023.emnlp-main.667/) reports exact intent-plus-all-
arguments match and supplies a direct 580M mT5-Base anchor. With all PRESTO training data, its
English exact-match results are 86.18% without a marked phenomenon, 85.36% on revisions, 86.20% on
disfluencies, and 75.99% on code switching.

Use the official English train/development/test assignment. Preserve official phenomenon buckets.
Because the paper warns that language-ID filtering may leak code-switched material, code switching
is secondary rather than the sole headline. A separately derived context-critical subset must use a
frozen rule and manual audit; PRESTO's original study found much structured context to be ignorable.

Pinned repository revision: `fa47167477453afebe698a287409514df5a7dadf`.

### Mobile Actions: smoke test, not headline

[Mobile Actions](https://huggingface.co/datasets/google/mobile-actions) is CC BY 4.0 and has 9,654
records carrying train/evaluation metadata, seven Android-style tools, timestamps, natural-language
requests, and exact calls. It is valuable for a fast end-to-end pilot and for reproducing the
FunctionGemma setting, but the small fixed schema and templated distribution make it insufficient
for a breakthrough claim.

Carve a grouped development set only from Google's training records; the official FunctionGemma
notebook uses the 961 evaluation records as trainer validation/model-selection data, which is not a
clean final-test protocol. Keep those 961 evaluation records untouched until one final scoring run.
Report the official set separately from derived hard
variants such as identifier renaming, distractor schemas, missing required information, and
irrelevant requests. Derived variants are diagnostics, never silently merged into the official score.

Pinned dataset revision: `e920309bc2acbc2e99a5e3201cf37df2b9fd9151`.

### MASSIVE: auxiliary fixed-ontology supervision

[MASSIVE](https://github.com/alexa/massive) is CC BY 4.0 and contains more than one million
utterances across 52 languages, 60 intents, and 55 slot types. It is localized from SLURP, so the two
must not be treated as independent evidence without an overlap audit. Use English MASSIVE only as
auxiliary intent/slot supervision or robustness data, not as the principal tool-schema result.

Pinned repository revision: `f966f21846043aabef9b0f974fa7970027f43738`.

### Schema/tool data and external diagnostics

[APIGen](https://arxiv.org/abs/2406.18518) created xLAM Function Calling 60K by checking format,
actual execution, and semantic consistency across 3,673 APIs. The verification pattern is more
important here than its larger-model result. xLAM material may be used for format acquisition only
after its current revision, file hashes, access conditions, and CC BY 4.0 attribution are captured.

[BFCL](https://proceedings.mlr.press/v267/patil25a.html) evaluates AST correctness, serial and
parallel calls, abstention, and stateful cases. Barun's 2,048-token limit makes the whole benchmark
ineligible. A preregistered subset may be reported as an explicitly adapted external diagnostic if
every compared model gets the same schema compression and context budget. It must not be called a
full BFCL score.

## Post-training evidence

### Closest-scale support

- [Baby Llama](https://aclanthology.org/2023.conll-babylm.24/) showed useful distillation gains in a
  58M decoder-only model. Its `T=2`, equal CE/KL weighting is an anchor for a test, not a validated
  action-calling optimum.
- [Generalized Knowledge Distillation](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5be69a584901a26c521c2b51e40a4c20-Abstract-Conference.html)
  improved a 77M T5-small student by teaching on student-generated contexts. It is the closest-scale
  primary evidence for on-policy distillation, but it used an encoder-decoder and non-tool tasks.
- [Scaling Laws for Forgetting](https://arxiv.org/abs/2502.06042) includes a 41M GPT-style model and
  found that a small amount of pretraining-data replay could prevent forgetting in its domain-
  adaptation setting. We therefore ablate 0%, 1%, 5%, and 10% replay instead of assuming a mixture.
- [TinyStories](https://arxiv.org/abs/2305.07759) supports the broader premise that very small causal
  models can learn coherent behavior when the target distribution is intentionally constrained.

These sources support narrow outputs, SFT, distillation, and replay. None directly proves a 35M
personal-action result.

### Domain methods that require scale extrapolation

- [Hammer](https://openreview.net/pdf?id=yVQcr4qjD6) masks function/parameter names on a fraction of
  training samples and adds irrelevant candidates, encouraging description-based selection. Its
  reported models are much larger, so a 0.33 masking-ratio arm is an ablation rather than a default.
- [ToolACE](https://proceedings.iclr.cc/paper_files/paper/2025/hash/663865ea167425c6c562cb0b6bcf76c7-Abstract-Conference.html)
  combines diverse API synthesis with rule- and model-based verification. Transfer the verification
  discipline, not the large-model performance claim.
- [BalanceSFT](https://aclanthology.org/2026.findings-acl.900/) identifies scarce hard examples and
  the risk that long reasoning tokens dominate short calls. BarunAction therefore omits long CoT by
  default and mines hard errors. Its 7B evidence does not validate a learnable loss balancer at 35M.
- [API-aware constrained decoding](https://arxiv.org/abs/2305.15338) shows that formal validity and
  semantic accuracy are different. Retrieval, constrained decoding, and model quality must be
  ablated independently.

### Methods not selected as defaults

[DPO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html)
and verifier RL/GRPO results are overwhelmingly at billion scale. Tool-call preference pairs also
often differ by only one name or value; research on
[likelihood displacement](https://arxiv.org/abs/2410.08847) warns that preference optimization can
lower the likelihood of both chosen and rejected responses. Preference training must beat an
equal-compute correction-SFT control and retain absolute chosen likelihood.

GRPO is attempted only after the model has useful pass@K and heterogeneous verifier rewards. An
all-wrong 35M policy provides no productive group-relative signal. A rising learned reward without
locked exact-execution gains is a failed run.

## Selected experiment ladder

### 0. Freeze the measurement

Implement the parser, canonicalizer, simulator, schema validator, OOS/ambiguity policy, sample-level
artifact format, and paired statistics first. Freeze public splits, hard-set derivations, prompts,
decoding rules, context limits, model revisions, and compute reporting. Evaluate the untouched base
model and larger baselines before changing training data.

### 1. Full-parameter response-only SFT

Start with a 2K--10K Mobile Actions/PRESTO probe. Train only on answer tokens; do not let repeated
schema/prompt tokens dominate the objective. Screen learning rates `1e-5`, `3e-5`, and `1e-4`, with
one aggressive `3e-4` arm only if gradients remain stable. Evaluate quarter-epoch checkpoints and
stop on held-out regression. Compare shuffled sampling with a cumulative easy/medium/hard curriculum
at equal examples, tokens, and steps.

Promote only if schema validity reaches 95% and exact match improves at least 25 points over base.

### 2. Full public-data SFT and replay

Train the best stable setup on official training data, balancing simple, revision, disfluency,
contextual, OOS, and ambiguous buckets. Ablate 0%, 1%, 5%, and 10% licensed general/pretraining-like
replay. Promotion targets are 80% Mobile Actions exact, 70% PRESTO overall exact, OOS F1 at least
0.80, and no more than a ten-point simple-to-revision/disfluency gap.

### 3. Verifier-filtered distillation and hard-data feedback

For black-box/different-tokenizer teachers, sample 4--16 concise candidates, canonicalize and
execute them, and accept only semantically verified outputs. Mix accepted teacher/gold targets with
student-error repairs. Generate deterministic minimal negatives: wrong nearby tool, missing/extra/
swapped argument, wrong type/entity/time, false call on OOS, and unsafe or ambiguous calls.

Barun's tokenizer differs from likely teachers, so ordinary token-logit KL is not currently valid.
True logit/on-policy GKD is deferred unless a compatible-tokenization teacher or a validated cross-
tokenizer mapping is available. Sequence-level teacher repair plus deterministic verification is the
default. Keep a distilled tranche only for at least a two-point semantic gain beyond format validity.

### 4. Optional preference ablation

Only after correction SFT plateaus, compare more correction SFT with DPO/ORPO-style training on
verified, length-controlled pairs. Start with low learning rates and short duration. Reject the
method if chosen likelihood, exact execution, OOS behavior, calibration, or output entropy worsens.

### 5. Optional verifier iteration, then RL

Try expert iteration first: sample 8--32 student candidates, mechanically select valid canonical
successes, and train on them. GRPO is eligible only when roughly 20--80% of hard prompts have mixed
pass/fail candidates in a group. Limit the first pilot to 100--500 updates, use exact semantic and
execution rewards, retain a KL anchor, and validate every 25--50 updates. Keep it only for a
replicated 1.5--2 point gain over the best non-RL checkpoint across three seeds without safety
regression or reward hacking.

## Claim and stop rules

The release comparison has two separate regimes:

- **Product:** off-the-shelf models with their official templates and best reasonable prompting.
- **Scientific:** each model trained on the same eligible examples and canonical target, with
  comparable hyperparameter effort and the same retrieval, schema, decoding, and verifier access.

Core comparison models are FunctionGemma-270M, SmolLM2-360M-Instruct, Qwen2.5-0.5B-Instruct or
Qwen3-0.6B, a current 350M-class action model when license/access allows, mT5-Base's published
PRESTO anchor, and Needle-26M. Exact revisions will be frozen in the evaluation manifest.

A headline win requires a post-release, human-authored hidden suite with at least 2,500 independent
efficacy clusters (target 3,000) plus at least 1,500 independent safety/abstention/ambiguity/
confirmation clusters; no teacher or model may generate its test items. It also requires
sample-level predictions, at least three seeds (five preferred) for promoted training recipes,
simultaneous paired confidence intervals with all lower bounds above zero across the four named
larger baselines, and at least a three-point practical point-estimate margin on the sealed hard set.
One critical unauthorized action blocks release pending diagnosis.
If Barun wins only the official Mobile Actions split but loses PRESTO revisions/disfluencies, OOS,
schema holdout, or the sealed hard set, the breakthrough hypothesis is rejected or narrowed.

## Compute decision

Evaluator/unit work and source edits are cheap. Model/data execution belongs on newly created
JarvisLabs `barun-*` instances under the exact-ID lifecycle in `infra/jarvis/README.md`. One L4 is
the default for 35M SFT probes; promote to A100-80GB only after profiling shows a real throughput or
memory advantage, or for a larger baseline/teacher that cannot fit. No GPU is provisioned until the
base-model correctness tests and evaluation manifest pass.

JarvisLabs latency is not on-device evidence. A deployment claim must additionally pin and test an
actual target phone/edge runtime, including quantized accuracy, cold load, end-to-end latency
percentiles, peak memory, package size, and energy where measurable.
