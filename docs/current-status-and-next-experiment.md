# BarunAction-35M current status and next experiment

Status: authoritative handoff as of 2026-08-05. Read this before changing weights, launching a
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
closed. A distinct narrow Grounded PlanIR hypothesis then passed pre-model review, but its sole
remote attempt failed closed during launch provenance before model or row access; its run and
placeholder-v2 recipe are closed without a scientific result. GVS-v1 was then closed before
population collection or model access because its requested effective-component counts were
incompatible with its honest lineage rule. Generate-Correct SFT v1 was then closed before
population or model access: its frozen hash-bit rule produced 8,076 fault and 8,308 clean rows
instead of 8,192/8,192, and its parsed draft object could not carry arbitrary raw model output.
No BarunAction training experiment is active. Candidate-v2 therefore remains the usable checkpoint
while the versioned v2 correction contract is built and audited behind a fresh population boundary.

The matched-adaptation mobile scale sweep
(`20260805-1554-mobile-scale-sweep-s17`) is at attempt-5 CPU prefreeze and is
**blocked pending a fresh independent prelaunch audit of v5**. Attempt 4 attested axolotl /
CPython 3.11.10 on H200 lineage **465183→465186**, completed the in-run candidate-v2 reference
evaluation at **590/725**, then aborted on the first challenger because `jl run` created
`uv venv --system-site-packages` and Transformers imported the image `flash_attn_2_cuda`
ABI-mismatched against venv torch 2.13.0. Failure receipt SHA-256
`adad51dad1e3216272496dd83072c6d0514d6585da16fe66d56d4944184fa5e0`; spent attempt-4 go SHA-256
`9bf60db79cc543e35eb5e664d1a79e761337720c0dbd663da64a8b62c6f4dac2`. v5 freezes isolated project
venv (`include-system-site-packages=false`) with a fail-closed flash_attn preflight; config
SHA-256 `57dcfe573c17759404545c272f1fc945aabb82b2893f697615eb7fe428d1f2d7`. No GPU may be created
until a new independent v5 audit returns go. Never reuse 465183, 465186, or 465155.

## Public showcase release

GitHub release
[`barunaction-v1.1.0`](https://github.com/harrrshall/barunlm-35m/releases/tag/barunaction-v1.1.0)
is public and immutable by project policy. Its annotated tag peels to audited commit
`ea26b83eba1a1819ab0088c611dad977265669f7`; the release contains the tested `barunlm` 1.1.0
wheel, Python sdist, and `SHA256SUMS`. All three assets were downloaded without authorization and
reverified. The exact wheel passed the network-denied, weight-free safety demo after clean install
on CPython 3.10.20 and 3.11.15, and two independent clean-clone builds produced the same wheel
SHA-256. The sdist did not reproduce byte-for-byte, so the complete research/process bundle is the
GitHub tag/source archive rather than the Python sdist. The post-publication receipt, asset hashes,
test counts, firewall state, and immutability rules are in
`experiments/runs/20260805-0347-barunaction-github-release-v1/result.json`.

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

## Closed experiment: construction-internal Grounded PlanIR screen

The next authorized training experiment changed the model's output decomposition rather than
adjusting the failed temporal data mixture:

> A 35M model will compile personal actions more accurately when it preserves canonical Action IR
> but predicts compact typed references and symbolic operators for calendar datetimes, which a
> frozen deterministic compiler grounds into final values.

The compiler must be fail-closed. It may use only the prompt, supplied context and tool schemas,
reference timestamp/timezone, and the emitted plan. It may never inspect a gold label at runtime,
guess an omitted semantic value, silently repair an invalid plan, or execute a real tool. The
public API continues to return final Action IR plus validation evidence; PlanIR is an internal,
versioned model/compiler contract. Compiler success proves schema-valid lowering only; the
separate BarunAction policy assessment still requires authorization/confirmation and never permits
real execution.

Its one allowed remote attempt is now over without a model result. On clean commit `d2f434a`,
JarvisLabs' managed dependency preamble copied `mobile-planir-screen.txt` into the uploaded stage
root. The first stdlib launch-provenance check rejected that unexpected file before config decode,
the CPU gate, Torch/CUDA/model access, any construction-row read, or any output. Machine `463843`
was pause-verified and protected after the fail-closed bundle was downloaded. The bound attempt is
`dd69923ce26b49567daa687805588a6b79438ea9f80e357c5598d88ae6d24eff`; the remote log is
`80c28afb9c8cb5b717653207e9bfc330c81cd545f1a1630c9750f46855a58461`. The frozen contract
makes every operational failure terminal, so this run and placeholder-v2 recipe are closed with no
retry, rescue, or relaunch. This is operational evidence, not evidence for or against PlanIR model
quality.

The implemented placeholder-v2 compiler is a narrow seven-Mobile-tool, empty-context,
timezone-naive calendar mechanism. It is not yet the generalized context/timezone/unseen-schema
system required by the mission. Its sole authorized use was a construction-internal A/B/C
mechanism screen. That use ended at launch provenance before model access, so no further
placeholder-v2 use is authorized. Under the frozen counterfactual, even a pass could not have been
reported as fresh evidence, a Qwen win, or a breakthrough; it would only have licensed
generalized-compiler development.

A construction-only schema audit now shows that this representation is implementable: a frozen
four-operation date grammar plus prompt-grounded clocks represents 2,069/2,070 calendar calls and
should byte-exactly round-trip 5,744/5,745 complete construction labels (99.98%). The sole reject
is `mobile-actions-04894-6bd7643b1bc95697`, whose “next Tuesday” request is labeled as Wednesday;
preserve it as `temporal_reference_mismatch` evidence. The first quote-based encoding was caught
before training as a poor primary treatment: it used 735,301 target tokens versus 514,097 for
direct Action IR and lengthened every accepted target. The selected placeholder-v2 encoding uses
506,062 target tokens, shortens every representable calendar target, and leaves every non-calendar
target byte-identical. B and C will receive the same target-independent reference table. These are
feasibility and token facts only, not a model result or permission to treat old rows as new
evaluation data. The exact schema, compiler, source-audit, and human-collection contract is in
`docs/grounded-planir-and-human-evaluation.md`.

The primary-source audit found no public dataset that can honestly supply the later decisive fresh
selection and confirmation populations. xLAM-60K is conditionally suitable for licensed,
verifier-audited training only after its gated terms and bytes are checked. BFCL, MASSIVE, TOPv2,
and PRESTO remain public compatibility diagnostics: they are old/public, contamination is unknown,
or their released metadata cannot enforce the required author/source/batch boundary. Do not call
one of these populations fresh. A separately commissioned, model-independent human collection is
therefore a pre-model requirement for the breakthrough claim.

The construction-internal mechanism screen is frozen under these conditions before model or CUDA
access:

1. Use only former v3 construction rows for a clearly labeled internal train/screen split. Old
   selection, old confirmation, reused-756, and official-961 populations remain forbidden. The
   split and all scores are mechanism evidence only and may not be relabeled as fresh.
2. Freeze the `PLAN_IR_V2` placeholder schema, target-independent grounding/reference-table format,
   compiler, evaluator, full-prompt renderer and prediction-receipt hashes, and duplicate
   firewall. Require zero overlap by example, the connected closure of the available source
   `cluster_id` and `family_id`, and prompt-target content. This old manifest lacks independently
   auditable author, entity-source, generator-template, and temporal-construction provenance; that
   limitation is another reason the screen is internal-only and can never be called fresh.
3. Require at least 95% of rows to be unambiguously representable and exact PlanIR-to-Action-IR
   round trips for every included gold label. Exclusions are recorded, not coerced.
4. Freeze three matched arms from canonical BarunLM-35M: A emits direct Action IR; B emits direct
   Action IR with the same grounding table exposed to C; C emits typed placeholder-v2 PlanIR and
   uses the frozen compiler. Use identical source IDs, presentations, seed set, optimization budget, and
   unconstrained deterministic decoding.

The proposed internal screen conjunction is at least +3 points compiled AST exact versus A and B, at
least +5 points row-level argument-value exact, positive seed-matched gains in at least two of
three seeds, no action family or policy class more than two points worse, raw PlanIR parse validity
at least 99.5%, emitted Action IR schema validity 100%, and zero truncations, missing outputs,
generation failures, or catastrophic actions. Passing licenses a generalized
context/timezone/arbitrary-schema PlanIR implementation and fresh preregistration; it does not
unlock human confirmation.

The narrow derivation/compiler layer has now passed an independent 258-test review, including the
pinned 5,745-row oracle and tokenizer receipts; Ruff and format checks also pass. The source split
is frozen before model output: 4,596 training rows and 1,149 screen rows across disjoint
`cluster_id` components and nested families. The sole known oracle reject is a singleton screen
component and is excluded symmetrically from A/B/C, so the scored population is 1,148 identical
IDs per arm. The final independent audit authorized exactly one attempt of run
`20260804-0545-mobile-planir-construction-screen-s17`. The frozen config SHA-256 is
`cca598e6f5a76a5e848a8b9a869cad779bbcbb17999217d2488111b22e2aa34c`; the requirements
SHA-256 is `6db8f37c0c21aea4a4ad93d7193b82217e73d3090db6b8947a9a9321aefb21c9`. The exact remote
gate passes 343 tests; the full repository passes 799 tests; Ruff and formatting pass. The runner
enforces nine base-reset completion-only step-73 fits, frozen token totals, checkpoint-before-screen
and raw-before-score barriers, an exact pinned scorer, export-time evidence hashes, one fresh
H200/Axolotl/CPython 3.11.10/Torch 2.13.0/CUDA 13.0 environment, and no retry. The attempt then
failed operationally at the first provenance check, so the preregistered terminal rule closed the
run and recipe. The conditional generalized and matched-Qwen lanes were never unlocked.

## Operational order from here

1. Keep candidate-v2 and its Darwin ARM64 int8 derivative as the usable public release. Preserve
   the Qwen loss, the v3 rejection, and the terminal PlanIR failure without rescue or relabeling.
2. Preserve immutable W&B evidence `v0` as the authoritative scientific mirror and keep its
   corrected privacy statement visible. The proposed redacted `v1` publication is closed without
   upload after an adversarial prelaunch no-go; never retry that run or invoke its upload path.
3. Preserve GVS-v1 as closed CPU research evidence. Do not collect its populations, run candidate
   support, train its rankers, deploy its custody rehearsal, or create compute for that contract.
4. Preserve Generate-Correct SFT v1 as closed feasibility evidence. Develop only the versioned v2
   CPU-prefreeze successor; neither v1 nor v2 currently authorizes population rendering, tokenizer
   or model access, CUDA, JarvisLabs, or training.
5. Freeze and independently audit exact raw-draft transport, the balanced T/P/D roster, real source
   semantics, duplicate closure, token/runtime/scorer gates, phase firewall, and provider transform.
   A later model-run config must bind the completed receipt and issue a separate go decision.
6. If and only if that later config authorizes it, create one fresh exact-ID-controlled H200. Train
   the reusable full-budget C seed-17 fit first and open only the disconnected 1,024-row futility
   probe. Train the remaining eight fits only after its immutable conjunction passes. Download all
   evidence before pausing the exact ID. Never touch Kroda, Kimi, Kriti, or a protected instance.
7. Apply the frozen hierarchy without seed selection, threshold edits, rescue data, or a tenth fit.
   Candidate-v2 remains the public incumbent unless a later checkpoint satisfies every applicable
   quality, safety, efficiency, compatibility, packaging, and fresh-evaluation requirement.

As of 2026-08-05 05:00 Asia/Kolkata, only the narrow exact-string runtime item has passed a final
independent audit. Its attempt-3 receipt SHA-256 is
`3d50e9eb0029fefb86301c8aefa08e679ecabea78d648ab503beff1e7b18a6ae`; it supports the existing
T and D prompt shapes and preserves invalid, duplicate-key, suffix-bearing, truncated,
noncanonical, and Unicode `draft_raw` evidence. It explicitly does not provide P rendering,
accepted pass-assessment receipts, tokenizer caps, provenance, scoring, or launch authority.

The first phase/scoring revision failed because forged check receipts and caller validity/safety
booleans could advance the state machine or hide a raw catastrophe. Its immutable P0 no-go is
`experiments/runs/20260805-0450-action-correction-phase-scoring-audit-s17/attempt-1-no-go.json`,
SHA-256 `533d4bd41e3cd6f40cdff50edd92359e3f5915cff0ad29b4f91ae35e9a257c17`. The first duplicate
firewall revision also failed as a scientific boundary because its receipt/runtime could be forged
and its duplicate/provenance views were caller-authored rather than derived from frozen raw rows.
Its no-go SHA-256 is `e178edaba63fa33dfcdade3b4bf7c92f7ed7d7c35dfac9debeb1465f3786568c`.
Both are being replaced under new schema revisions; neither failure opened population or model
access. The balanced roster successor and real per-candidate semantic catalog remain outstanding.

## Closed CPU proposal: Generate-Verify-Select v1

GVS-v1 is closed before population collection, candidate generation, model loading, CUDA, training,
or human-label access. Its CPU implementation remains useful for simulator, candidate-decoder,
ranker, custody-rehearsal, and inferential-design research, but it is not launchable and produced no
verifier or model-quality result.

The decisive feasibility audit found that the frozen joint-lineage rule unions every shared author,
source, batch, schema, template, paraphrase, entity, temporal, content, and duplicate ancestry.
Under that rule, the proposed eight schema families cannot support 2,000 effective D components,
the identity schema cannot honestly span roles, one alias macro is not thousands of independent
schema sources, and required nonapplicable fields collapse rows. Reaching the requested counts
would require identifier laundering. The full result is
`experiments/runs/20260805-0205-gvs-v1-contract-feasibility-audit-s17/result.json`, SHA-256
`27a1f8a536b2c2193091428d13e8b6bd543587fbdb56e48d21d2e76c82105dac`.

The preceding real-roster reproduction produced eight components from 16 otherwise unique rows and
failed a floor of nine exactly; a shared batch collapsed the same population to one component.
That record is
`experiments/runs/20260805-0140-gvs-effective-component-audit-s17/result.json`, SHA-256
`f5285533105ccd867076d344955b10e9160200472bf0f41be5fe0ead127a0ba2`.
Never patch GVS-v1 identifiers, weaken closure, or reinterpret nominal rows as effective evidence.

A future verifier experiment would need a new hierarchical contamination and inference contract,
nullable inapplicable axes, role-blind family allocation, an explicit identity-compatibility lane,
annotator dependence, and reserve collection. That would be a new experiment, not a GVS-v1 retry.

## Closed CPU proposal: Generate-Correct SFT v1

The now-closed generator-side v1 hypothesis is documented in
`docs/generate-correct-sft-v1.md`. CPU-prefreeze run
`20260805-0230-action-correction-forge-screen-s17` asks whether the same
35,072,768-parameter BarunAction-35M weights can learn to inspect one draft and repair argument
bindings in one correction pass. The typed config is
`configs/action_correction_forge_screen_v1.json`, SHA-256
`3f1862a403dcf76edb82268f46688f44ce33027904dc11357b1c9cfc4b68931a`.

This direction is motivated by the observed failure shape—149 of candidate-v2's 154 reused-Mobile
misses were value-only—but it may use that result only as aggregate motivation. It may not train on
the 756 prompts, targets, paired Qwen wins, v3 selection or confirmation shadows, the PlanIR screen,
PRESTO development, GVS populations, or the official 961 rows.

### Frozen causal design

Every trained fit starts from exact candidate-v2 and preserves the 35,072,768-parameter
architecture. Arms A, B, and C use seeds 17, 29, and 43, for exactly nine fits:

- A repeats direct `prompt -> complete Action IR` supervision.
- B masks one semantic field and reconstructs the complete Action IR.
- C receives either an exact draft or one simulator-certified semantic fault, exactly 8,192 of
  each, and returns the complete corrected Action IR.

No-fit modes include canonical BarunLM-35M base B0, frozen candidate-v2 G0 and two-pass G02,
two-pass A2, and arm-C one-pass C1 versus intended two-pass C2. Pass 2 may see only the original
model-visible input, the pass-1 draft, and draft-derived parse/schema/policy/simulator diagnostics.
No gold value, fault type, oracle, label, or reference answer may be visible. A policy-conformant
proposal is still only a proposal: `execution_permitted` remains false and no real tool runs.

The fallback is deterministic: return a valid, schema-valid, policy-conformant pass 2; otherwise a
conformant pass 1; otherwise canonical `ABSTAIN`. There is no retry loop. Pass-1 and pass-2 token
caps are explicit unresolved blockers and must be frozen before any model access.

Each fit is planned for 38,512 rows and 602 optimizer updates: 5,744 shared replay rows, 16,384
shared direct synthetic rows, and 16,384 arm-specific rows. The recipe proposes one full-parameter
BF16 epoch, batch 64, AdamW at 5e-5, exact 18-step warmup, cosine decay, response-only loss, no
packing, no truncation, no early stopping, and final checkpoint only. Equal source IDs, complete
targets, supervised target-token totals, example counts, and update counts remain requirements to
verify after materialization; input tokens and FLOPs are deliberately not claimed matched.

### Data boundary and first verified primitive

`ActionCorrection-Forge-v1` plans 16,384 deterministic program-first training rows and 4,096
nominal hypothesis-fresh internal synthetic screen rows across thirteen action families plus
`ABSTAIN`, `CLARIFY`, and `CONFIRM`. The screen is not human-fresh, decisive, powered, or a
breakthrough benchmark. Its 4,096 rows are not claimed as 4,096 independent components. Role
assignment must precede rendering, and honest semantic/template/schema/entity/temporal/context
ancestry plus exact, normalized, delexicalized, and near-duplicate closure must determine the
effective grouping.

The replay boundary is now independently reproducible. The exact 5,745-row construction source,
minus the pinned singleton bad label, yields 5,744 byte-preserved direct Action IR training rows:

| Replay item | SHA-256 |
| --- | --- |
| Output manifest | `f964587bbed79f451c2d5360677c22d176ed9a96c750fff6e12852e57530d863` |
| Accepted membership | `27f9acb8905db132420ddb5d11a20bed138f2e2927aec41672f6852b3895452d` |
| Direct-target commitment | `131b7b3094fcfc3cbf919cdf2cd5fb8062ac7a492fc57a8eff45fbdcfe163389` |
| Materializer receipt | `3df25123e1761c54fa0e0b01ca501fd1913ec65abece38683c2b3706cb7a15f5` |

The audit is
`experiments/runs/20260805-0244-action-correction-replay-audit-s17/result.json`, SHA-256
`2e7c2ed74088fc3a130b6c6c0bbb9628154dd2295cc7912f2970ff0d756652fa`.
It is training-only, non-fresh, non-selection, and nonauthorizing.
Its original “without importing PlanIR ... targets” phrase was too broad at the source-module
level: the audited implementation imported the shared tool registry through `grounded_planir` and
used that run's exclusion ledger as provenance, while reading no PlanIR representation, compiler
output, target, or screen row. The append-only correction SHA-256 is
`4b06a63570bce9e746f4aef83bdb7ba9ea38495111aa0133ad96dbb2c18372ea`; the schema registry now
lives in the neutral `mobile_action_schemas.py` module.

The bounded program-first forge prototype also passed its CPU audit. It covers thirteen simulator
operations plus `ABSTAIN`, `CLARIFY`, and `CONFIRM`; fixes role-exclusive families before render;
constructs matched A/B/C views; exposes only draft-derived diagnostics; and certifies a single
semantic fault without exposing that certificate to the model view. It is hard-limited to two
fixtures per stratum and built only 16 T plus 16 D rows in tests. It cannot materialize the planned
population or claim independent components. The audit is
`experiments/runs/20260805-0230-action-correction-forge-screen-s17/prototype-audit.json`, SHA-256
`d07b8aa69594c33b52e88f3f9639fa99012130fec647c99a9490ca7d264d20db`.

### Keep rule and unresolved blockers

C2 is kept internally only if the complete frozen quality, validity, safety, and M5 Pro efficiency
conjunction passes. The principal margins are +5 execution points over G0, +3 over A, B, A2, and
G02, +3 Action IR exact over A and B, +2 for C2 over C1, +5 argument-value exact over A and B,
positive comparisons in at least two seeds and 12 of 16 strata, at least 98% clean-draft retention,
at least 40% one-fault repair, and fixes exceeding regressions by at least 3:1. Parse validity must
be at least 99.5%, schema validity at least 99%, false actions may rise by at most one point, and
catastrophic unauthorized actions, missing outputs, generation failures, and truncations must all
be zero.

The sole possible preview checkpoint is the predesignated seed-17 final checkpoint of the first arm
that passes the hierarchy. Seed 17 must independently pass every applicable quality, validity,
safety, and efficiency gate; seeds 29 and 43 are replication only. Best-seed selection, averaging,
post-screen refit, and a tenth fit are forbidden.

The config remains nonauthorizing and is now terminally closed. Feasibility run
`20260805-0406-action-correction-v1-feasibility-audit-s17` reproduced 8,076 single-fault and 8,308
exact training assignments instead of the registered 8,192/8,192. It also proved that v1's parsed
`draft` object cannot represent arbitrary exact pass-1 bytes. Its immutable result SHA-256 is
`eec225c1f0e9a88c58446db638b89fa30fb0429ed222c1265e6e6fb225b8a13e`. The audit read no
population, tokenizer, model, reused-756 row, official-961 row, private label, CUDA device, network,
or JarvisLabs resource. Never patch or retry v1. Any successor must use a new config and run ID,
pass the complete CPU population/runtime audit, and receive a separate prelaunch authorization.

## Resource and publication invariants

Kroda 463058 is running and protected. Independently owned 463719 remains protected regardless of
its observed lifecycle state. The adjacent Kimi owner has repeatedly recreated its job under the
same display name. Structured machine IDs 463912, 463936, 463964, 464346, 464367, 464377, 464378,
and 464382, plus the
endpoint-derived historical misidentification 463904, are all durably denylisted regardless of
their observed lifecycle state. The last three came from a local adjacent-controller process
observation, not a JarvisLabs query; do not inspect them to refine their state. Never infer a
lifecycle target from the stable display name, and
do not inspect, connect to, rename, pause, resume, reuse, or delete any of those resources. Qwen
463689, failed H200 evidence IDs 463786, 463793, and 463843, completed v3 H200 evidence ID 463802,
and L4 runtime probe ID 463788 are paused and protected. Previously observed unrecognized ID
463697 remains protected even when absent from the latest listing. Do not access, resume, stop,
rename, or delete any of them.
At 22:37 Asia/Kolkata on 2026-08-04, an explicit user-authorized cleanup permanently destroyed the
twelve paused historical Barun instances 463556, 463572, 463575, 463594, 463606, 463622, 463631,
463636, 463642, 463674, 463675, and 463686 after verifying that their corresponding local run
directories were present. A post-cleanup safe inventory showed all twelve absent and showed only
the six protected Barun evidence instances above remaining. Kroda and all non-Barun resources were
outside the operation and untouched. The deleted instances are irrecoverable; their local evidence
and append-only ledger records remain authoritative.
Read the exact-ID lifecycle in `AGENTS.md` before creating a new resource. The
current local orchestrator uses an open-lid `/usr/bin/caffeinate -i` assertion; it does not survive
lid close, shutdown, power loss, or network loss.

No credential belongs in the repository, artifacts, logs, documentation, or command arguments.
The latest user direction authorizes a public BarunAction model/evidence release. Keep the score
claim narrow, include Apache-2.0 plus Mobile Actions/PRESTO attribution, publish immutable artifact
versions rather than `latest`, and verify every downloaded byte before calling the release usable.
The local 296-file privacy-redacted evidence view is not published. Run
`20260805-0250-candidate-v2-redacted-evidence-v1` failed its prelaunch publication audit because
W&B version assignment cannot atomically reserve `v1`, postcommit failures can leave no local
receipt, and anonymous-read verification was absent. No upload or network operation occurred.
Closure verification SHA-256
`ddd4ac6fd6f34999b104fca01f627ac40ddd38e73050a6b7c66f75e37141d72a` proves the public and
internal executors now reject remote modes and the mutation/download implementations are removed.
