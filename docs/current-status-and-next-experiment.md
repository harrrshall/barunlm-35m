# BarunAction-35M current status and next experiment

Status: authoritative handoff as of 2026-08-04. Read this before changing weights, launching a
JarvisLabs resource, publishing artifacts, or interpreting the larger-model comparisons.

## Decision in one paragraph

BarunAction-35M candidate-v2 is now a public, proposal-only compact research release; state
plainly that the present larger-model-outperformance hypothesis failed. Candidate-v2 is
14.09 times smaller than the audited Qwen2.5-0.5B-Instruct checkpoint and retains 90.80% of its
strict exact-match rate, but Qwen scored 663/756 and BarunAction scored 602/756, an 8.07-point
Qwen lead. Preserve candidate-v2 and its int8 derivative; do not tune this result away, relabel the
SmolLM2 failure as a general win, or inspect the sealed 961-row Mobile evaluation population.

## Execution status: v1 stopped before science; v2 is frozen

The first Month-Boundary Counterfactual launch created fresh H200 **463786** and failed during
JarvisLabs dependency resolution, before the experiment runner started. The PyTorch template
provided CPython 3.10.20, but frozen `numpy==2.4.6` requires Python 3.11 or newer. Safe-run
pause-verified only that exact ID. Raw lifecycle SHA-256 is
`ae2739f5b8598a3382bf45908ee16c376f166940f0f181c84d7fa51adeac9982`; remote-log SHA-256 is
`54e5aaeaa6a321c53dd4c5c489e9af9738798a9c3651132f1860adef59431add`; and bound attempt SHA-256
is `3168fdeffc039bdceae855c34bdf616c71f54ece0aa2b0a79a8bc3642cc64f90`.

This is a blocked infrastructure attempt, not a failed model hypothesis. The runner never began,
so model and CUDA access were false; construction, selection, confirmation, reused-756, and
official-961 rows read were all zero; no prediction, loss, or score was produced. Because no remote
`execution/essential` tree existed, artifact retrieval truthfully failed. No synthetic artifact
or ordinal-2 authorization was created.

A fresh L4 environment check then created exact ID **463788** with the Axolotl template. Its
provider preamble selected CPython 3.11.10; lifecycle and remote-log SHA-256 are
`c2727275ef97d4e3bb64d4fec67edbe245a04ac2b9f2df55057c5c6c1c88a828` and
`9e8d0e1c559d8378f44f3dae919b020942a22069e2a11a888ffdde0374e293a0`.
The probe script itself did not run because the deliberately minimal directory had no project
metadata, so this proves only the provider runtime preamble. ID 463788 is pause-verified and
protected.

The current run is the new preregistration
`20260804-0210-mobile-temporal-counterfactual-axolotl-s17`, not a v1 retry. Its frozen config is
`configs/mobile_temporal_counterfactual_v2.json`, SHA-256
`893a2e0ff59ccb98767af653d0f80745e32c05abc4620d3dfae3f277033f20b8`; its scientific
projection is `d077effff172a4870ec1d8af2ee4e6b3b3fcf80a8b860eaf65eb5d0781942186`.
The run binds Axolotl and CPython 3.11.10 before create, re-attests both on the fresh exact H200
before upload, and permits only the active non-symlink repository-root `.venv` created by the
provider. The provider environment stays outside the scientific tree. Hypothesis, data,
materializations, arms, seeds, optimization, decoding, thresholds, gates, requirements, and the
official-evaluation firewall are unchanged.

## Current release candidate

| Item | Frozen result or identity |
| --- | --- |
| Base name | BarunLM-35M, 35,072,768 parameters |
| Product name | BarunAction-35M candidate-v2 |
| Float weights SHA-256 | `fdb95ccf58a095e0d321be998924318b35ee59a334f6dd97d8726d2cf80021d3` |
| Float Mobile development result | 602/756 strict AST exact (79.63%) |
| ARM64 int8 weights SHA-256 | `18102649bb4f8507ee754ae0c298e20064580833f05ca515faf5e04eac6f4488` |
| ARM64 int8 development retention | 607/756 (80.29%), gate passed |
| Int8 size reduction | 58.56%; no latency claim |
| Official Mobile evaluation | 0/961 rows read or materialized |
| Product boundary | Proposes and validates Action IR; never executes a real tool |

Candidate-v2 remains the release checkpoint even though recovery checkpoint C reached 642/756 on
Mobile: C missed its frozen PRESTO exact gate by 140 rows and was never promoted. The public API,
CLI, fail-closed validation, and in-memory simulator load only hash-bound artifacts.

## Larger-model evidence

The Qwen run is `20260803-2122-mobile-qwen05b-matched-s17`, using revision
`7ae557604adf67be50417f59c2c2f167def9a775`. Its 1,008,838,966-byte logical source tree contained
85 manifested payload files plus the self-excluded checksum manifest; all 85 entries passed. A
safe 63-file evidence subset was imported under that run directory. The 988,097,824-byte model,
the rest of its checkpoint/tokenizer subtree, full JarvisLabs inventory, remote log, and duplicate
source copies were deliberately excluded. The immutable original import receipt contains a
clerical `files_checked: 80` error; `import-receipt-correction-v1.json` and
`full-handoff-sha256-check.log` preserve the correction without rewriting released evidence. The
selected boundary remains `selected-artifact-sha256.txt`, SHA-256
`0ca5c9458e8e2b3e2d2930f8d4ec21e29b0f4744e6c8057a2e1e76b9f5a424e0`.

| Measure | Qwen2.5-0.5B-Instruct | BarunAction candidate-v2 |
| --- | ---: | ---: |
| Exact parameters | 494,032,768 across 290 BF16 tensors | 35,072,768 |
| Strict AST exact | 663/756 (87.70%) | 602/756 (79.63%) |
| Parse valid | 755/756 | 756/756 |
| Schema valid | 754/756 | 755/756 |
| Paired-only wins | 80 | 19 |

There were 657 ties: 583 rows both correct and 74 rows both wrong. Semantic adaptation examples,
targets, one-epoch presentation, effective batch, optimizer-step count, seed, and decoding were
matched. Learning rate, tokenizer/native rendering, upstream pretraining, and total model-selection
effort were not; BarunAction had prior Mobile tuning while Qwen received one frozen trial. This is
strong negative public-development evidence, not a universal model-family ranking or a hidden-test
result. The exact recipe SHA-256 is
`6855eaf52c34089d802b8584c566b95619f9ff2780d4f130f333126bc33357e1`; the final Qwen weights are
`21f3f234ce9bfc2fa387cf396807fa35f5eb72e311f23a89cd98c98224c78793`; and verifier source is
`355018916f839fdb0b6bfe4cc23b41381200cfa1bde59ed040bba513abdde064`. The earlier
361,821,120-parameter SmolLM2 run at 0/756 is specific to that checkpoint and recipe.

## What the paired errors say

The frozen, post-hoc analyses are recorded in
`experiments/runs/20260803-2122-mobile-qwen05b-matched-s17/posthoc-paired-error-analysis.json` and
`posthoc-temporal-error-analysis-v1.json` beside it. They recompute categories from the immutable
development manifest plus Qwen and candidate sample scores without changing any prediction or
metric.

| Scenario | Rows | BarunAction | Qwen | Qwen lead |
| --- | ---: | ---: | ---: | ---: |
| Calendar | 178 | 114 (64.04%) | 143 (80.34%) | 16.29 points |
| Contacts | 128 | 104 (81.25%) | 118 (92.19%) | 10.94 points |
| Flashlight on | 82 | 71 (86.59%) | 78 (95.12%) | 8.54 points |
| Flashlight off | 74 | 67 (90.54%) | 72 (97.30%) | 6.76 points |
| Wi-Fi settings | 117 | 111 (94.87%) | 114 (97.44%) | 2.56 points |
| Maps | 92 | 58 (63.04%) | 60 (65.22%) | 2.17 points |
| Email | 85 | 77 (90.59%) | 78 (91.76%) | 1.18 points |

Of the 80 Qwen-only wins, 76 BarunAction outputs have a pure argument-value error. Across all 154
BarunAction failures, 149 are value-only; 98 of 164 mismatched argument occurrences are datetimes.
All 98 datetime errors preserve the correct time but use the wrong date, and all 98 take their
year-month from `NOW`. On the 59 calendar calls whose gold month differs from `NOW`,
BarunAction misses 43 (72.9%) while Qwen misses seven (11.9%); all 43 BarunAction misses retain
`NOW`'s month. That clean 36-call difference accounts for about 59% of Qwen's 61-row lead. This is
strong shortcut evidence, not causal proof. Map copying is lower priority: total query mismatches
are nearly tied, 33 for BarunAction and 34 for Qwen.

## Current experiment: Month-Boundary Counterfactual SFT v2

Do not launch generic hard-example oversampling, weighted-token loss, DPO, RL, distillation, or a
learning-rate sweep. The previous calendar/map/multi-call hard mix fell to 566/756, while the new
paired analysis isolates missing temporal counterevidence. The current experiment asks:

> Does teacher-free SFT that decorrelates `NOW`'s month from the requested calendar month remove
> the month-copy shortcut, beyond the effect of merely repeating the same source rows?

All screening arms start from canonical BarunLM-35M revision
`ef3e483a9fd7d906ecf2a7929babeffaf82d1d16`, not candidate-v2: candidate-v2 has already seen all
7,937 source rows and would contaminate the new shadow holdouts. Hash each existing connected
component into 20 folds with `barun-mobile-temporal-shadow-split-v1`; folds 0--13 are construction,
14--16 selection, and 17--19 confirmation. The frozen memberships are:

| Role | Rows | Components | Calendar | Same month | Cross month | Membership SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Construction | 5,745 | 3,722 | 2,070 | 1,686 | 384 | `b70f6dfe79cfe49792b6e6f5d414fd509c889d18101f75cd8df6d4e4f146c588` |
| Selection | 1,024 | 752 | 360 | 294 | 66 | `b4191540878ec28c6819e08e0b390fbfd57eab836d3361727ad89862a46cf927` |
| Confirmation | 1,168 | 831 | 450 | 377 | 73 | `519c22724873ee579d0f59be886cfc297088113bd1a297521b9ca280a79c35b9` |

Components are indivisible, so 72.4/12.9/14.7 is the deterministic result rather than a cosmetic
70/15/15 row ratio. Generate at most one variant from each eligible same-month construction row:
for explicit-only requests, move only `NOW` to the last day of the preceding month while leaving
the request and target byte-identical; for pure-relative requests, shift `NOW` and the sole
calendar datetime by the same deterministic nonzero multiple of seven days. Reject mixed,
ambiguous, numeric-date, same-day, multi-calendar, invalid, or overlength cases. No teacher output,
paraphrase, outside text, reused 756 prompt, or official-evaluation material is allowed.

The executable preregistration is `configs/mobile_temporal_counterfactual_v2.json`, full-file
SHA-256 `893a2e0ff59ccb98767af653d0f80745e32c05abc4620d3dfae3f277033f20b8`; its separately checked
scientific projection is `d077effff172a4870ec1d8af2ee4e6b3b3fcf80a8b860eaf65eb5d0781942186`.
The frozen provider requirements remain byte-identical at
`a037db0943d563ea04bcc45b963b5a27931e6f745a7f472c079bb9f9fa6df853`.
Two independent local materializations were byte-identical. Construction produced 1,093 safe
pairs (748 explicit and 345 relative), so B and C each contain 6,838 rows and 109 batch-63
optimizer steps. C contains
1,477/3,163 cross-month calendar presentations (46.696%). The conditional full refit was also
frozen in advance: 1,546 variants, 9,483 rows, 151 steps, and train SHA-256
`0877a7db29989a2e824de907300aad63137c511b2c8fdaaa5597afe6f046b6fd`. Materialized data remain
ignored local/remote experiment artifacts rather than source-release content.

Run exactly three arms at seeds 17, 29, and 43:

- A, `standard`: the 5,745 construction rows once;
- B, `repeat`: A plus unchanged repeats of every safely transformable source row; and
- C, `mbcf`: A plus the paired month-boundary counterfactuals.

B and C must have identical source-ID sets, row counts, ordering policy, and optimizer steps. C
versus B isolates new temporal information; C versus A measures practical gain. Freeze all
materialized manifests, accepted/rejected receipts, tokenizer lengths, and hashes before CUDA or
model loading. Preflight requires at least 1,000 safe variants and at least 45% cross-month calendar
presentations. Use BF16, batch 63, one epoch, response-only cross-entropy, AdamW `1e-4`, betas
0.9/0.95, epsilon `1e-8`, weight decay 0.1, clip 1.0, 12 warmup steps, cosine floor 0.1, maximum
length 2,048, unconstrained greedy decoding, and final checkpoints only.

Selection is a hard conjunction. C must beat both A and B by at least 15 absolute points in mean
cross-month calendar-datetime exact and three points in mean overall strict AST exact; at least two
of three seed-matched comparisons must improve; same-month calendar and non-calendar exact may each
lose at most two points; parse validity must be at least 99.5%, schema validity at least 99%, and
failures, truncations, and catastrophic actions must be zero. Only then score confirmation once.
Confirmation requires at least +10 cross-month points and +2 overall points versus both controls,
the same regression/safety gates, and a one-sided 10,000-resample cluster-bootstrap fifth
percentile above zero for the targeted endpoint.

If confirmation passes, run one full-7,937-row C refit at predeclared seed 17. Score it once on the
reused 756 only as a compatibility veto: at least 602 exact, 756 parse-valid, 755 schema-valid, zero
failures/truncations/catastrophic actions, at least 25/59 cross-month datetimes correct versus the
incumbent's 16/59, and no scenario more than two points below its candidate-v2 floor. No repair,
retry, threshold change, seed choice, or arm switch may follow that score. Maximum budget is nine
screening fits plus one final fit, with at most two byte-identical infrastructure retries and a
30-GPU-minute job cap. If any gate fails, retain candidate-v2 and stop this recipe.

The 756-row result generated this hypothesis and cannot establish the new claim. The sealed 961
rows remain unopened until weights, tokenizer, inference code, evaluator, and claims are frozen.
If opened, it is a one-time final evaluation: no later training, threshold revision, arm switch, or
reselection is allowed.

## Operational order from here

1. W&B publication is complete: immutable float, Darwin ARM64 int8, and evidence `v0` artifacts,
   upload receipt, fresh 310-file redownload, and anonymous float-weight verification all passed.
   Finish the secret-scanned public source branch/release; do not include workspace data or weights.
2. The Axolotl-bound v2 generator, duplicate firewall, evaluator, materialized hashes,
   three-arm/three-seed budget, retry firewall, dependency-only environment, and exact-ID
   lifecycle are frozen. The full repository suite passes 423 tests. The launch builder requires a
   clean canonical Git HEAD, validates the complete staged preflight, permits only the six
   screening and two full-refit JSONLs plus the pinned reused-756 manifest and identical root
   requirements copy, and creates the source snapshot and attempt template without bytecode.
3. Run the compute-heavy training and evaluation on one newly created, exact-ID-controlled,
   non-spot JarvisLabs H200 using template `axolotl`. Safe-run must attest exact template and
   CPython 3.11.10 before inventory binding or upload. Do not use or mutate any pre-existing
   instance. Download the recursive `execution/essential` tree before pausing and verifying the
   exact created ID.
4. Preserve every outcome. Promote only through the frozen gate; otherwise keep candidate-v2.

## Resource and publication invariants

Kroda 463058 is running and protected. Independently owned 463719 remains protected regardless of
its observed lifecycle state. Qwen 463689, failed H200 evidence ID 463786, and L4 runtime probe ID
463788 are paused and protected. Previously observed unrecognized ID 463697 remains protected even
when absent from the latest listing. Do not access, resume, stop, rename, or delete any of them.
Read the exact-ID lifecycle in `AGENTS.md` before creating a new resource. The
current local orchestrator uses an open-lid `/usr/bin/caffeinate -i` assertion; it does not survive
lid close, shutdown, power loss, or network loss.

No credential belongs in the repository, artifacts, logs, documentation, or command arguments.
The latest user direction authorizes a public BarunAction model/evidence release. Keep the score
claim narrow, include Apache-2.0 plus Mobile Actions/PRESTO attribution, publish immutable artifact
versions rather than `latest`, and verify every downloaded byte before calling the release usable.
