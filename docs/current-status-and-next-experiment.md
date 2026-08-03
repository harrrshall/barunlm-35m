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
The valid temporal v3 experiment strongly corrected its targeted month-copy shortcut but failed
the complete frozen selection gate, so it produced no promotable checkpoint. That recipe is
closed. There is no active training run until a distinct Grounded PlanIR hypothesis, fresh
populations, implementation, and immutable preregistration pass pre-model review.

## Execution status: v1 and v2 stopped before science; v3 completed and was rejected

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

Axolotl-bound v2 run `20260804-0210-mobile-temporal-counterfactual-axolotl-s17` then created fresh
H200 **463793**. Exact H200, IN2, non-spot, Axolotl, and CPython 3.11.10 attestation passed before
upload. The CPU-only validation stopped with 146 tests passed, one skipped, and two failed: both
tests implicitly expected the private workstation denylist, which was intentionally absent from
the clean source stage. Parent Torch was not imported; model/CUDA access, reused-756 reads/scores,
and official-961 reads were zero; no held-out artifact exists. Managed run `r_2186968e` exited 1,
and only exact ID 463793 was pause-verified at `2026-08-03T21:15:20.511416+00:00`.

This is an inconclusive infrastructure failure with no scientific result. Raw lifecycle, remote
log, bound attempt, source snapshot, validation log, launch receipt, and watchdog records are
preserved in an ignored private subdirectory; the public run directory contains a redacted
classification, pause proof, failure, validation receipt, and artifact manifest. The two affected
tests now inject an explicit temporary denylist, while a new regression proves production still
fails closed when the private denylist is absent. A clean no-private-denylist simulation passes
150 remote tests with one intentional skip. Because the corrected test file changes the complete
frozen source tree, v2 ordinal 2 is forbidden even though v2 had zero signal.

Hermetic v3 run `20260804-0255-mobile-temporal-counterfactual-axolotl-hermetic-s17`, attempt
ordinal 1, completed on fresh JarvisLabs H200 **463802**. Its immutable config remains
`configs/mobile_temporal_counterfactual_v3.json`, SHA-256
`e17f9ef0d755735ce14d66029706534a24145d51e89786aa92d3cd334760b22f`; its scientific projection
is `f1705c38c064bef646a5814ee9f6cc6c652e9db3c6714e76ff65049fe65ed280`. Managed run
`r_fe346e61` exited zero after all nine A/B/C screening fits. The recursive essential bundle and
remote log were collected before exact ID 463802 was pause-verified at
`2026-08-03T22:04:56.393070+00:00`. The controller observed one H200, IN2, non-spot, Axolotl,
CPython 3.11.10, BF16, PyTorch 2.13.0+cu130, and CUDA 13.0 exactly as frozen.

The scientific selection gate failed. Across three seeds, C achieved 2,526/3,072 overall AST
exact, compared with 2,477 for standard A and 2,451 for equal-budget repeat B. Cross-month
calendar-datetime exact was 123/198 for C, 60/198 for A, and 21/198 for B: C passed the targeted
minimum with +31.82 and +51.52 percentage-point gains. The full conjunction rejected C because
its overall gains were only +1.60 and +2.44 points versus the required +3.00; its same-month
calendar loss versus B was 2.49 points, above the 2-point ceiling; schema validity was
3,041/3,072, one valid output short of 99%; and two outputs truncated when zero were allowed.
Parse validity passed at 3,058/3,072. Missing predictions, generation failures, and catastrophic
unauthorized actions were all zero.

Independent verification rehashed all 145 essential payloads and 17,231,542 bytes, replayed all
9,216 predictions through the frozen scorer, reproduced every sample score, aggregate, temporal
subset, comparison, and the failed gate, and verified all 930 optimizer steps. The result is a
valid scientific rejection, not an infrastructure failure. Confirmation, conditional full refit,
reused-756 compatibility scoring, and official-961 evaluation were not reached. The evidence
bundle contains zero optimizer, screening-weight, or promoted-weight files. Candidate-v2 remains
the release checkpoint. Never retry or rescue v3, change its thresholds, choose a favorable seed,
modify its frozen config, resume 463802, or reuse that machine.

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

## Completed experiment: Month-Boundary Counterfactual SFT v3

V3 answered its causal development question: repeating transformed temporal information, rather
than merely repeating the same rows, produced a large and replicated cross-month improvement. It
did not answer the product question because the gain was too localized and the hard conjunction
failed.

| Frozen selection measure | A: standard | B: repeat | C: MBCF | C difference |
| --- | ---: | ---: | ---: | ---: |
| Overall AST exact | 2,477/3,072 | 2,451/3,072 | 2,526/3,072 | +1.60 points vs A; +2.44 vs B |
| Cross-month datetime exact | 60/198 | 21/198 | 123/198 | +31.82 points vs A; +51.52 vs B |
| Same-month datetime exact | 651/882 | 666/882 | 644/882 | -0.79 points vs A; -2.49 vs B |
| Non-calendar AST exact | 1,822/1,992 | 1,816/1,992 | 1,816/1,992 | -0.30 points vs A; tied with B |

C had the best pooled schema validity of the three arms, so the 99% miss does not show that MBCF
uniquely damaged formatting. Twenty-six of C's 31 schema-invalid outputs were non-calendar. The
two truncations were rare greedy runaways rather than evidence that ordinary gold outputs exceeded
the 256-token generation allowance: frozen gold targets topped out at 195 tokens. These are useful
diagnostics, but they do not change the rejection.

The v3 selection and confirmation memberships are retired from future model selection. Selection
was observed; confirmation was not, but preserving it for another related variant would invite
optional stopping. Any later analysis may describe v3; it may not authorize another v3 arm,
threshold, seed, mixture, or checkpoint.

## Proposed next experiment: Grounded PlanIR

There is no active training experiment yet. The next recommended hypothesis changes the model's
output decomposition rather than adjusting the failed temporal data mixture:

> A 35M model will compile personal actions more accurately when it predicts a compact typed
> semantic plan—tool, policy decision, input/context references, and symbolic temporal operators—
> and a frozen deterministic compiler grounds that plan into canonical Action IR.

The compiler must be fail-closed. It may use only the prompt, supplied context and tool schemas,
reference timestamp/timezone, and the emitted plan. It may never inspect a gold label at runtime,
guess an omitted semantic value, silently repair an invalid plan, or execute a real tool. The
public API continues to return final Action IR plus validation evidence; PlanIR is an internal,
versioned model/compiler contract.

Before any model or CUDA access:

1. Keep every old v3 row out of new selection and confirmation. Former v3 construction rows may
   be used only as explicitly labeled training data; old selection, old confirmation, reused-756,
   and official-961 populations remain forbidden during development. Acquire and freeze a genuinely
   new post-v3 human-authored or independently sourced batch for component-disjoint selection and a
   separate once-only confirmation population. If source provenance and licensing cannot support
   that boundary, do not launch the experiment.
2. Freeze a PlanIR schema, grounding/reference-table format, compiler, evaluator, and duplicate
   firewall. Require zero overlap by example, connected component, family, prompt-target pair,
   entity source, and temporal construction.
3. Require at least 95% of rows to be unambiguously representable and exact PlanIR-to-Action-IR
   round trips for every included gold label. Exclusions are recorded, not coerced.
4. Freeze three matched arms from canonical BarunLM-35M: A emits direct Action IR; B emits direct
   Action IR with the same grounding table exposed to C; C emits typed PlanIR and uses the frozen
   compiler. Use identical source IDs, presentations, seed set, optimization budget, and
   unconstrained deterministic decoding.

The proposed selection conjunction is at least +3 points compiled AST exact versus A and B, at
least +5 points row-level argument-value exact, positive seed-matched gains in at least two of
three seeds, no action family or policy class more than two points worse, raw PlanIR parse validity
at least 99.5%, emitted Action IR schema validity 100%, and zero truncations, missing outputs,
generation failures, or catastrophic actions. Only a selection pass may read a once-only fresh
confirmation, where the overall gain must remain at least +2 points versus both controls and the
10,000-resample one-sided cluster-bootstrap fifth percentile must exceed zero.

Grounded PlanIR remains a proposal until the new source, derivation code, round-trip audit, splits,
exact budgets, thresholds, and immutable config are implemented, tested, and hash-frozen. A failed
pre-model coverage/round-trip gate ends it without GPU spend. A failed model gate ends it without
rescue. If it passes, compare it with a separately matched Qwen2.5-0.5B-Instruct PlanIR lane before
any larger-model claim.

## Operational order from here

1. Keep candidate-v2 and its public W&B `v0` artifacts as the usable release. Import the verified
   v3 negative result into public source evidence without predictions, private inventory, logs, or
   credentials.
2. Select and provenance-audit a genuinely new post-v3 source for selection and confirmation;
   old v3 rows remain training-only. In parallel, implement and test only the Grounded PlanIR
   representation, compiler, split firewall, oracle round-trip audit, and matched-arm evaluator.
   Do not train while source independence, coverage, or exact round-trip correctness is unproven.
3. Run the lightweight materialization and oracle audit on remote JarvisLabs compute after a clean
   local source review. If the pre-model gate passes, preregister a new immutable run/config and
   independently audit it before creating a fresh exact-ID-controlled instance.
4. Use remote compute for model training and evaluation. Download evidence before pause-verifying
   only the newly created ID. Preserve every outcome and promote only through the new frozen gate.

## Resource and publication invariants

Kroda 463058 is running and protected. Independently owned 463719 remains protected regardless of
its observed lifecycle state. Qwen 463689, failed H200 evidence IDs 463786 and 463793, completed
v3 H200 evidence ID 463802, and L4 runtime probe ID 463788 are paused and protected. Previously
observed unrecognized ID 463697 remains protected even when absent from the latest listing. Do not
access, resume, stop, rename, or delete any of them.
Read the exact-ID lifecycle in `AGENTS.md` before creating a new resource. The
current local orchestrator uses an open-lid `/usr/bin/caffeinate -i` assertion; it does not survive
lid close, shutdown, power loss, or network loss.

No credential belongs in the repository, artifacts, logs, documentation, or command arguments.
The latest user direction authorizes a public BarunAction model/evidence release. Keep the score
claim narrow, include Apache-2.0 plus Mobile Actions/PRESTO attribution, publish immutable artifact
versions rather than `latest`, and verify every downloaded byte before calling the release usable.
