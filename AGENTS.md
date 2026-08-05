# BarunLM post-training operating contract

## Start here: workspace and immediate checkpoint

Read `../AGENTS.md` completely before acting. It maps the separate Barun product folders and is the
local cross-repository handoff. This repository is the internal research/evidence repository even
though its filesystem directory retains the former working name. The concise public product lives
in sibling `../barunaction-35m/`; private raw evidence lives in sibling
`../barun-private-evidence/`. Never copy this file, internal research notes, private evidence, or
credentials into the public product, Hugging Face, a demo stage, or a release bundle.

Reconciled 2026-08-05: branch `agent/barunaction-release-mbcf-v1`. Matched-adaptation scale-sweep
attempt 6 **completed and closed**. Spent GO
`experiments/runs/20260805-1554-mobile-scale-sweep-s17/prelaunch-audit-attempt-6-go.json`, SHA-256
`3d30a295fc358cccd5bab22d1e1f37a8d220dc22533ec527096ec652e400d9f1`, authorized exactly one
Axolotl/CPython-3.11.10 H200 with `safe_run --isolated-project-venv` and frozen v6 config SHA-256
`0885b32f14751e77539f0bf58cae6f89b7c72e1d80c1b8a6d3fe61cff9c55540`. Exact ID **465257**
(`barun-scale-sweep-a6-20260805`) ran `r_7809186b` to exit 0, downloaded `essential-attempt-6/`, and
is pause-verified/protected. Reference **591/725** (81.52%); all challenger arms falsified under the
+3.0-point rule (best: pythia 45/725, SmolLM2-135M 422/725, SmolLM2-360M 585/725). Decision:
`reject`; candidate-v2 remains the release checkpoint. Completion receipt
`experiments/runs/20260805-1554-mobile-scale-sweep-s17/attempt-6-completion.json`. Never reuse the
spent go, 465257, or protected evidence IDs; no official-961 access; no candidate-v2 retraining.

The working tree contains untracked research configs, source, tests, manifests, and experiment
evidence from action-correction and sub-100M work. Untracked does not mean unused. Preserve those
items unless an exact evidence/ownership audit explicitly retires them. Safe hygiene is limited to
verified reconstructible caches/build products and stale PID/path pointers whose targets are
proven absent. Never use broad `git clean`, bulk stage, or recursive deletion at this repository
root.

Public product checkpoint: sibling `../barunaction-35m/` is the public GitHub repository;
`harrrshall/BarunAction-35M` is the canonical Hugging Face model and `v1.0.0` is the live public
revision; `https://barunaction-d9123728.nip.io` is the browser demo. The demo runs on separately
owned protected infrastructure and must never be inspected or used for research. The public repo
has active uncommitted sub-100M benchmark work and a local-only ignored `AGENTS.md`; never package
or push that dirty tree without its public-boundary audit passing.

## Mission

Turn `harrrshall/BarunLM-35M` into **BarunAction-35M**: a compact, local personal-action
compiler for reminders, calendars, contacts, notes and lists, maps, messages, media, and
device settings.

The model must map a user's request and supplied context to canonical typed action(s), or to
`ABSTAIN`, `CLARIFY`, or `CONFIRM` when acting would be unsupported, underspecified, or
side-effecting. Actions are evaluated in a simulator; experiments must never contact real people,
edit real calendars, or control real devices.

The falsifiable research target is to outperform fairly evaluated 270M--600M baselines on
exact execution for personal actions, especially contextual revisions, disfluencies,
abstention, distractor tools, and renamed or unseen schemas. A win on a small template-style
split alone is not a breakthrough.

The project is complete only when there is a usable post-trained release, not merely a plan or an
experimental score. The release must include a downloadable checkpoint and tokenizer, a stable
input/output contract, a Python inference API and CLI, deterministic parsing/schema/policy checks,
a sandboxed demonstration, quantized artifact or a documented blocker, model/data cards,
sample-level evaluation evidence, and a reproducible path from pinned data to weights. A score gain
that cannot be run by another person is not the target outcome.

## Canonical names

- **BarunLM-35M** is the only canonical name for the 35,072,768-parameter base model. Its canonical
  checkpoint is `harrrshall/BarunLM-35M` and its source repository is `barunlm-35m`.
- **BarunAction-35M** is the canonical name for this project's post-trained personal-action model.
- **StrataLM-35M** was an earlier working name. The local workspace directory may still be named
  `stratalm-35m` for historical/tooling reasons; that path does not name the model. Do not create new
  `stratalm`, `strata-*`, or `StrataLM` checkpoint names, run names, package names, headings, claims,
  or artifact IDs.
- If historical material or a path forces a Markdown file to mention StrataLM, label it explicitly
  as the former working name and immediately give the current BarunLM name. Every new or modified
  Markdown file must follow this distinction. Do not silently rewrite historical URLs or filesystem
  paths that would stop working.

## Current handoff and experiment state

The authoritative 2026-08-05 state and experiment boundary are in
`docs/current-status-and-next-experiment.md`. Read that file before launching training or changing
release claims.

The implementation and evaluation contract for the latest closed hypothesis is in
`docs/grounded-planir-and-human-evaluation.md`. The narrow construction-internal Grounded PlanIR
screen was authorized for exactly one attempt under run
`20260804-0545-mobile-planir-construction-screen-s17`. Attempt 1 failed closed in the first remote
launch-provenance validation because the JarvisLabs dependency preamble copied
`mobile-planir-screen.txt` into the stage root outside the frozen allowlist. It failed before
config decode, CPU validation, Torch/CUDA/model access, construction-row reads, or model output.
The run and placeholder-v2 recipe are now closed; generalized or decisive Grounded PlanIR is not
launch-authorized. The config SHA-256 is
`cca598e6f5a76a5e848a8b9a869cad779bbcbb17999217d2488111b22e2aa34c` and the requirements
SHA-256 is `6db8f37c0c21aea4a4ad93d7193b82217e73d3090db6b8947a9a9321aefb21c9`. Every retry,
rescue, refit, promotion, Qwen comparison, and human-label access is forbidden. Its
construction-only schema audit predicts 5,744/5,745 exact round trips. Quote-v1 is retained only as
feasibility evidence because it is about 43% longer than direct targets; compact placeholder-v2 is
the selected treatment and is shorter than direct targets on the audited construction rows. Old
construction rows remain training/internal-ablation data and public BFCL/xLAM/MASSIVE/PRESTO data
cannot substitute for separately collected human selection and once-only confirmation
populations. Read that contract before changing the PlanIR schema, compiler, oracle, data-source
decision, or evaluation firewall.

The narrow compiler/oracle layer passed an independent 258-test review with its pinned receipts
unchanged. The frozen source split assigns 4,596 construction rows to training and 1,149 to screen
by disjoint source components/families; the singleton known bad-label component is symmetrically
excluded, leaving 1,148 identical screen IDs for A/B/C. This is retrospective internal evidence:
the full construction labels already informed representation design, and the old manifest lacks
decisive human-provenance fields. Do not describe the screen as fresh even after it runs.

The final independent prelaunch audit passed the exact remote CPU gate (343 tests), the complete
repository suite (799 tests), Ruff, formatting, config/data/token/runtime binding, phase-firewall
review, raw-freeze revalidation, and compact-bundle failure-path tests. The sole launch used clean
commit `d2f434a`, source snapshot
`ca090cdaf88555fb0255dd3e7cc5a77ae8f78a51579281805540c9fe1986b5d7`, and bound attempt
`dd69923ce26b49567daa687805588a6b79438ea9f80e357c5598d88ae6d24eff`. Its terminal failure is
documented under the run's `attempt-1-launch-provenance-failure/` directory. The no-retry clause is
now spent: never patch and relaunch this run or recipe.

Placeholder-v2 is only a narrow seven-Mobile-tool, empty-context, timezone-naive calendar
mechanism, and its one allowed construction-internal attempt is over without a model result. It may
remain as CPU-only representation/compiler evidence, but no further placeholder-v2 training,
screen, Qwen comparison, release, rescue, or breakthrough claim is authorized. A genuinely
distinct next hypothesis and fresh leakage-controlled population boundary must be preregistered
before any new model or CUDA access.

**Generate-Verify-Select v1 (GVS-v1)** was the preferred next proposal and is specified in
`docs/generate-verify-select-v1.md`. It kept candidate-v2 frozen, generated eight Action IR
candidates, and would test whether a 135,617-parameter shared-backbone verifier can recover correct
argument values from the candidate set. The complete proposed system is 35,208,385 unique
parameters, conditional on exactly 24 bias-free rank-8 q/v LoRA modules and one biased scalar head.
It is now closed at its pre-collection population/inference-contract gate without model access,
CUDA, training, or human data. It never received a frozen model run ID or scientific config. The
CPU reference remains useful nonauthorizing research code, but it may not be patched into a launch,
rescued, or described as a model-quality failure. Even
`prototype_passed=true` is serialized with `authorizes_model_or_label_access=false` and must never
unlock a model, CUDA, training, or private prompt/label read. Never use old selection, old
confirmation, reused-756, official-961, v3 shadows, or the PlanIR screen to tune or select GVS-v1.

CPU prefreeze checkpoint, 2026-08-05 00:46 Asia/Kolkata: the native K=8 decoder, bounded
state-transition simulator, certified single-fault generator, rank-hidden bridge, schema
presentation/partition, exact 135,617-parameter LoRA verifier, 449-parameter linear probe,
29,793-parameter MLP, joint-firewall primitives, component-level support/power plumbing, and
component-balanced shortcut controls passed the final patch-and-rerun cycle. The complete
repository passed 1,338 tests; the independent integration slice passed 365 overlapping tests;
Ruff, formatting, `py_compile`, and whitespace checks passed. The immutable audit record is
`experiments/runs/20260805-0046-gvs-cpu-prefreeze-audit/result.json`, SHA-256
`6d948c05a14881c761990f070a96b7caeb87a23ac83bc4d38fba1bf9e5aa0468`.
Its append-only correction
`experiments/runs/20260805-0046-gvs-cpu-prefreeze-audit/correction-20260805-0057.json`,
SHA-256 `9c374760429b488f9c756d189b6c547876b3446ecbb702e08039ef92234bb3d0`,
clarifies that 85% oracle pass@8, +10 points over greedy, and 50% greedy-failure recovery are
provisional proposal targets, not frozen gates. Never use them as authorization until the actual
pre-outcome component roster and external assumptions produce a frozen power specification.

This checkpoint remains deliberately nonauthorizing. The 24,000/2,000 T/D allocation is a
deterministic pre-authoring plan, not a materialized or authored population; the 32 rename families
prove only role-disjoint names under known semantics, not authenticated held-out chronology or
novel-schema competence. External secret custody, a single-use signer, durable atomic
compare-and-append retirement, original T/D requests, independently human-authored S/C, joint
T/D/S/C duplicate closure, and real-roster pre-outcome power thresholds still block every model,
CUDA, JarvisLabs, private-label, training, and launch action. Do not create or use a compute
resource until those gates pass and a distinct immutable experiment preregistration authorizes an
exact new ID.

Effective-component audit, 2026-08-05 01:40 Asia/Kolkata: the real-roster power bridge is now
implemented, but it rejected the current population allocation before collection. With 16 honest
rows assigned across the eight role-scoped schema-family IDs, the live firewall produced exactly
eight components of size two; a fixed floor of nine failed with `8 components from 16 eligible
rows, minimum 9`. One shared collection batch collapsed the eight components to one, and reusing
family or batch IDs across roles triggered the cross-role firewall. Thus 2,000 nominal rows over
the same eight families still have at most eight effective components. The exact record is
`experiments/runs/20260805-0140-gvs-effective-component-audit-s17/result.json`, SHA-256
`f5285533105ccd867076d344955b10e9160200472bf0f41be5fe0ead127a0ba2`. The current
schema/batch/source allocation is rejected; the GVS mechanism has not received a model test.
Redesign and independently audit honest presentation and provenance allocation before authoring,
collection, custody deployment, model access, or GPU creation. Never weaken lineage closure or
invent per-row family IDs to manufacture a denominator.

Offline-boundary audit, 2026-08-05 01:49 Asia/Kolkata: the local S/C collection packager and
SQLite retirement storage rehearsal passed independent adversarial reviews after closing
authorization-helper rebinding, identifier and expectation leakage, quota/runtime substitution,
false content claims, repository-root and hardlink bypasses, ownership/mode gaps, path-swap races,
and authenticator rebinding. The combined record is
`experiments/runs/20260805-0149-gvs-collection-retirement-cpu-audit-s17/result.json`, SHA-256
`38b411c8378a36370cec3b512be27f8e23273feb9839a2fb1f2c2a77ee8c8b76`.
These are nonauthorizing CPU scaffolds only: no human was recruited and no prompt or label was
collected. Same-process Python/UID is not a custody boundary. Honest powered allocation, the
schema-to-simulator/population bridge, real provenance/chronology, external signer and secret
custody, isolated retirement deployment, backup/anti-rollback, target-filesystem crash testing,
joint closure, and real-roster power all remain mandatory.

Final GVS-v1 feasibility audit, 2026-08-05 02:05 Asia/Kolkata: the current contract cannot honestly
produce its requested 2,000/2,000/4,000 effective D/S/C components. It unions every shared
provenance/dependence axis; the eight schema families cap components, identity schema cannot span
roles, one generator macro does not create independent schema sources, required nonempty
not-applicable lineages collapse rows, and the planner's unique slots explicitly are not
provenance. Reaching the requested counts would require identifier laundering. The exact record is
`experiments/runs/20260805-0205-gvs-v1-contract-feasibility-audit-s17/result.json`, SHA-256
`27a1f8a536b2c2193091428d13e8b6bd543587fbdb56e48d21d2e76c82105dac`. Do not collect,
deploy custody, create compute, or run candidate support for GVS-v1. A future verifier test needs a
genuinely versioned hierarchical contamination/inference contract; the active next hypothesis must
otherwise be a distinct generator-side intervention with a fresh internal boundary.

**Generate-Correct SFT v1 is closed before population materialization or model access.** Its
feasibility audit is
`experiments/runs/20260805-0406-action-correction-v1-feasibility-audit-s17/result.json`, SHA-256
`eec225c1f0e9a88c58446db638b89fa30fb0429ed222c1265e6e6fb225b8a13e`. Recomputing the frozen
low-bit assignment over the registered 16 x 1,024 training slots produced 8,076 single-fault and
8,308 exact rows, not the required 8,192/8,192, with every per-stratum count required to be
512/512. The v1 correction view also stores a parsed JSON object, so it cannot transport exact raw
model output containing invalid JSON, duplicate keys, suffix text, truncation, or noncanonical
valid JSON. Do not repair, retry, expand, materialize, or train v1. It produced no model-quality
result and authorizes no population, tokenizer, model, CUDA, JarvisLabs, or private-label access.

A successor must be a new immutable v2 CPU-prefreeze contract. It must bind exact `draft_raw`
transport; a finite pre-render roster with exact per-role/per-stratum rank allocation and no
backfill; jointly frozen T/P/D membership; role-stripped semantic lineage and transitive duplicate
closure; a real request-plan renderer and simulator; token/runtime/scorer/provider gates; and the
one-full-C17 futility ladder. Until that contract, its population, and an independent prelaunch
audit all pass under a separate model-run config, there is no active training experiment and no GPU
may be created.

Generate-Correct v2 CPU checkpoint, 2026-08-05 05:00 Asia/Kolkata: the corrected exact-string
runtime passed its third independent audit for the narrow T/D `draft_raw` transport only. The go
receipt is
`experiments/runs/20260805-0418-action-correction-runtime-audit-s17/attempt-3-go.json`, SHA-256
`3d50e9eb0029fefb86301c8aefa08e679ecabea78d648ab503beff1e7b18a6ae`; runtime source SHA-256
is `8cad30fc58a9e67ab83cf9a845b7811f5a088a60abdefed8ab4b15d901f59e86`. This does not cover P,
accepted pass-assessment receipts, tokenizer caps, population provenance, scoring, or launch.
The first phase/scoring and duplicate-firewall revisions independently failed P0 audits because
their public receipts, constructors, raw metric inputs, or runtime bindings were forgeable. Their
immutable no-go SHA-256 values are
`533d4bd41e3cd6f40cdff50edd92359e3f5915cff0ad29b4f91ae35e9a257c17` and
`e178edaba63fa33dfcdade3b4bf7c92f7ed7d7c35dfac9debeb1465f3786568c`. Correct successor
implementations and new independent audits remain mandatory. These results authorize no model,
tokenizer, population, network, CUDA, JarvisLabs, training, phase transition, or launch action.

The independently audited 5,744-row direct replay primitive remains training-only; its result
SHA-256 is `2e7c2ed74088fc3a130b6c6c0bbb9628154dd2295cc7912f2970ff0d756652fa`.
The result's phrase “without importing PlanIR ... targets” is narrowed by append-only correction
SHA-256 `4b06a63570bce9e746f4aef83bdb7ba9ea38495111aa0133ad96dbb2c18372ea`: no PlanIR
representation, compiler, target, or screen row was read, but the audited implementation imported
the shared seven-tool schema through the PlanIR module and used its exclusion ledger as
provenance. The registry now lives in neutral `mobile_action_schemas.py`; never erase the original
dependency mistake.

The bounded forge prototype is recorded at
`experiments/runs/20260805-0230-action-correction-forge-screen-s17/prototype-audit.json`, SHA-256
`d07b8aa69594c33b52e88f3f9639fa99012130fec647c99a9490ca7d264d20db`. It covers the thirteen
simulator operations plus the three control decisions and produces matched A/B/C fixtures with
draft-only diagnostics, but it is hard-capped at two rows per stratum. Its rows share declared
families and prove neither quota fulfillment nor statistical independence. PAUSE_MEDIA and
ABSTAIN lack an argument field, so their bounded fault fixture changes the decision field; this is
not evidence that argument correction works. The prototype authorizes no full materialization,
model, CUDA, JarvisLabs, training, or launch action.

The linear and small-MLP arms are high-information falsification controls, not favored treatments.
SCaTR's reported calibration results were on 1.7B--30B models, while a separate 2026 mechanistic
tool-calling study reported that its linear tool-selection circuit was absent at 270M and began
emerging around 1B. BarunLM-35M is far below both demonstrated regimes. These arms and the K=8
support gate are now archival GVS-v1 design evidence only; the feasibility rejection occurred
before candidate support could be measured. Never run them under v1 or treat the absence of that
measurement as evidence for or against ranking. A future verifier experiment requires a new
contract and preregistration; a distinct generator-side correction experiment must compare against
simple matched SFT before considering RL.

The evaluation firewall uses label-free public prompt records and separate HMAC-authenticated
private label envelopes. A pinned renderer and tokenizer recompute model-visible bytes and token
counts. A signed exact-gate receipt binds selection/confirmation memberships, candidates, compiler,
evaluator, source, and code; confirmation disclosure requires an atomic append to the global
tamper-evident retirement ledger. The repository validates this protocol but does not provide
secret storage, key custody, a single-use selection signer, or durable atomic persistence. Those
external services are mandatory before any fresh human label is collected or scored.
Validators must snapshot caller inputs once into detached exact JSON and reuse that snapshot for
hashing, membership, retirement, and returned evidence. Effective human cluster counts are
connected components over provenance/duplicate lineages, not untrusted `cluster_id` counts.
Duplicate scans precede eligibility, labels precede envelope sealing, and a passing selection
receipt may not predate the frozen confirmation envelopes it unlocks.

Candidate-v2 is the current public-release checkpoint: 602/756 float and 607/756 retained ARM64
int8 on the reused Mobile development probe. The independently verified Qwen2.5-0.5B comparison
is now imported under `experiments/runs/20260803-2122-mobile-qwen05b-matched-s17/`: Qwen scored
663/756 versus BarunAction's 602/756. BarunAction is 14.09 times smaller and retains 90.80% of
Qwen's exact-match rate, but trails by 61 rows or 8.07 percentage points. The current
larger-model-outperformance hypothesis failed; never generalize the SmolLM2 0/756 result or hide
the Qwen counterexample.

The candidate-v2 float checkpoint, Darwin ARM64 int8 derivative, and evidence bundle are published
as immutable public W&B `v0` artifacts. Their upload, fresh 310-file redownload, and anonymous
float-weight access checks passed. The scientific `v0` evidence is immutable and authoritative,
but it contains non-secret workstation paths and protected resource identifiers. Do not claim that
`v0` excluded all unrelated local names. Run
`20260805-0250-candidate-v2-redacted-evidence-v1` proposed one derived, nonauthoritative redacted
`v1` view, but an independent adversarial prelaunch audit returned no-go. W&B version assignment
cannot atomically reserve `v1`; a race can create irreversible `v2` before the client rejects it,
and anonymous public-read verification was unimplemented. The run is closed without network
access, upload, download, or artifact creation. Never invoke its upload path or retry it. The local
296-file view remains build evidence only. Any future redacted distribution needs a new
content-addressed destination, run ID, durable precommit/terminal-incident protocol, and anonymous
verification. The secret-scanned source branch is public in draft PR #2; never bulk-stage the
workspace or experiment directories.
Closure verification SHA-256
`ddd4ac6fd6f34999b104fca01f627ac40ddd38e73050a6b7c66f75e37141d72a` confirms that both public
and internal executors reject remote modes before path validation and that the upload mutation and
download implementations were removed; only local dry-run validation remains.

The public GitHub showcase release is
[`barunaction-v1.1.0`](https://github.com/harrrshall/barunlm-35m/releases/tag/barunaction-v1.1.0).
Its annotated tag object `cd1c102c04ed30db08c5694dfede047a834b8b32` peels locally and remotely
to audited commit `ea26b83eba1a1819ab0088c611dad977265669f7`. The attached wheel, Python
sdist, and `SHA256SUMS` have SHA-256 values `43c6f420f87f41e1f438217578cadaaf13de7f8797a35f28f4a43b33a3bdf970`,
`156ed7b9571129a47bd38ab55cd08e771961f78dc392c4e356aef27523d14eed`, and
`f6cdcdb23c39abffddbf054904203811abeeeb53c0936119688c2c631cf5e1f4`. Fresh anonymous downloads
matched them; the exact wheel passed the network-denied safe demo on CPython 3.10.20 and 3.11.15
and reproduced byte-for-byte across two clean clones. The sdist did not reproduce byte-for-byte,
so make no such claim: the immutable Git tag/source archive is the complete research bundle, while
the sdist is only the installable Python package. The post-publication receipt is
`experiments/runs/20260805-0347-barunaction-github-release-v1/result.json`. Never move or replace
the tag or its assets; any change requires a new versioned release.

Month-Boundary Counterfactual SFT v1 did not reach the experiment runner. Fresh H200 ID **463786**
failed during provider dependency resolution because the PyTorch template exposed CPython 3.10.20
while frozen `numpy==2.4.6` requires Python 3.11 or newer. The exact ID was pause-verified. The
runner never started: model/CUDA access, construction/selection/confirmation/terminal reads,
predictions, scores, and official-961 access were all zero. Its raw lifecycle and remote-log
hashes are `ae2739f5b8598a3382bf45908ee16c376f166940f0f181c84d7fa51adeac9982` and
`54e5aaeaa6a321c53dd4c5c489e9af9738798a9c3651132f1860adef59431add`.
Do not synthesize the missing remote essential artifact or call this a scientific rejection.

Axolotl-bound v2 run `20260804-0210-mobile-temporal-counterfactual-axolotl-s17` created fresh
H200 **463793** and passed exact hardware/runtime attestation, but its CPU-only gate stopped before
parent-runner Torch import, model/CUDA, or held-out access: two tests implicitly depended on the
intentionally private workstation denylist absent from the clean stage. The result was 146 passed,
one skipped, and two failed. Exact ID 463793 was pause-verified; reused-756 and official-961 reads
were zero. This is an inconclusive infrastructure failure, not a scientific rejection. Its raw evidence is
preserved privately and its redacted classification and pause proof are under the v2 run directory.
The corrected source cannot use v2 ordinal 2 because the frozen retry contract is byte-identical.

Hermetic v3 run `20260804-0255-mobile-temporal-counterfactual-axolotl-hermetic-s17` completed all
nine frozen screening fits on fresh H200 **463802**. The infrastructure and provenance checks
passed, managed run `r_fe346e61` exited zero, the essential bundle was downloaded, and only exact
ID 463802 was pause-verified. The scientific selection gate rejected the recipe. Arm C passed the
targeted cross-month calendar-datetime endpoint by +31.82 points versus A and +51.52 points versus
equal-budget repeat control B, but its overall gains were only +1.60 and +2.44 points versus the
required +3.00. It also lost 2.49 points of same-month calendar accuracy versus B, above the
2-point ceiling. Across C's 3,072 predictions, 3,058 were parse-valid, 3,041 were schema-valid,
and two truncated; schema validity needed one more valid output and truncations had to be zero.
There were no missing predictions, generation failures, or catastrophic unauthorized actions.

The run stopped exactly where preregistered: confirmation, conditional full refit, reused-756
compatibility scoring, and official-961 evaluation were never reached. No optimizer state,
screening checkpoint, or promoted weight is in the evidence bundle. Candidate-v2 remains the
release checkpoint and the Month-Boundary Counterfactual SFT recipe is closed. Do not retry,
rescue, alter thresholds, select a seed, or modify frozen
`configs/mobile_temporal_counterfactual_v3.json`. The later narrow PlanIR mechanism screen above
was a genuinely distinct, construction-internal hypothesis with a new immutable config and run ID;
its sole attempt also closed before model access. There is now no active training experiment.
Neither PlanIR nor any future direction reopens v3 or makes its old populations fresh. The observed
v3 selection shadow must not be used to tune a variation for the untouched confirmation shadow.

## Decision order

When choices conflict, optimize in this order:

1. Truthful, leakage-controlled measurement.
2. Safe and useful behavior.
3. Reproducibility and inspectable artifacts.
4. Model quality per parameter and per unit of inference compute.
5. Training speed and cost.

Do not add a fashionable post-training stage unless an ablation beats the simpler incumbent.
Research, evaluator construction, and cheap probes come before expensive training.

Parallel JarvisLabs experiments are encouraged when they answer independent, preregistered
decisions. Unlimited compute permission is not permission to waste compute: run cheap local/static
preflights first, use early termination gates, and do not scale a failed method. The first paid remote
run must produce useful evidence by combining environment/tests, pinned-data and token-length audit,
base evaluation, and a small SFT probe or sweep; a CUDA-only smoke test is insufficient as the sole
outcome.

## JarvisLabs standing context

Most data processing, model execution, evaluation, and training belongs on remote JarvisLabs
compute. Local work is limited to source edits, repository inspection, lightweight unit tests,
and small metadata operations when practical.

This section is the durable operating context requested by the user: "with the Jarvis Lab content somewhere in the proper place so I don't have to be right again and again."

For an unattended, open-lid macOS session with active local orchestration, keep a dedicated
`/usr/bin/caffeinate -i` process alive and verify that `pmset -g assertions` attributes a live
`PreventUserIdleSystemSleep=1` assertion to that exact PID. Do not add `-d`: the display should
remain free to sleep. End the assertion after the active goal no longer needs a local controller.
This protects against idle system sleep only; it cannot survive a closed lid, shutdown, loss of
power, or a network outage.

JarvisLabs CLI authentication is already configured outside this repository in the user's
OS application-support directory. Never copy the API key into this repository, a command,
an environment file, an experiment artifact, a log, an issue, or a model card. Never print
the credential. If authentication fails, report that fact without exposing configuration
contents.

Weights & Biases authentication is also configured outside the repository in the user's standard
credential file with mode `0600`. Never sync that credential file to JarvisLabs or copy its token
into source, config, CLI arguments, logs, or artifacts. Use W&B only where it adds durable
experiment evidence; local immutable JSON/JSONL artifacts remain the source of truth if tracking
is unavailable.

Raw `jl list --json` responses can contain signed notebook URLs or other access-bearing endpoint
fields. Treat those fields as credentials: do not quote, persist, commit, or place the raw listing
in an experiment artifact. For ordinary inventory, use
`.venv/bin/python infra/jarvis/safe_run.py inventory`; it projects only lifecycle-safe fields such
as `machine_id`, `name`, `status`, GPU type/count, region, template, and spot/reservation state.
Lifecycle targets always come from the structured `machine_id`, never from an endpoint string.

### Protected resources

Every JarvisLabs resource that predates this project session is protected. In particular,
instance **463058**, named **Kroda**, was already running and must never be stopped, paused,
restarted, deleted, renamed, or used for this project. The spelling may be dictated as
"cruda"; treat it as the same protected instance. Discovery of another pre-existing or
unrecognized resource makes it protected by default.

The independently owned eight-H200 instance **463719**, named
`kimi-k3-jl-node-a-20260803-1724`, remains protected regardless of its observed lifecycle state.
The independently owned eight-H200 resources named
`kimi-k3-jl-node-opt-20260804` and `kimi-k3-jl-node-opt128-20260804` belong to the adjacent Kimi
experiment and must not be inspected, connected to, paused, resumed, renamed, reused, or otherwise
interrupted by BarunAction work. A read-only
inventory correction on 2026-08-04 at 10:38 Asia/Kolkata resolved its actual JarvisLabs
`machine_id` as **463912**. A later safe inventory at 11:11 showed that the adjacent owner had
recreated the same named job under current `machine_id` **463936**. The earlier **463904** record
came from endpoint naming rather than a lifecycle identifier. Preserve both 463912 and 463904 as
protected historical evidence rather than erasing them. At 22:29, a later safe inventory found
463936 paused and discovered the separate paused opt128 resource under `machine_id` **463964**.
At 02:14 on 2026-08-05, another safe projected inventory showed newly recreated resources with the
same two names under structured `machine_id` values **464346** (`opt`) and **464367** (`opt128`).
At 04:31, a local process-only observation (not a JarvisLabs query) showed the independently owned
Kimi controller referencing **464377** and **464378** and running its optimization session against
active node **464382**. Treat all three as protected; do not query them to refine their state.
**464382**, **464378**, **464377**, **464367**, **464346**, **463964**, **463936**, **463912**,
and **463904** are all durably
denylisted; never infer that a stable display name means a stable machine ID or that a paused or
pausing adjacent resource is available.

The previously observed unrecognized ID **463697** remains protected even when absent from a later
live listing.

Project evidence instances **463689** (the paused Qwen comparison H200), **463786** (the v1
zero-signal H200 failure), **463788** (the paused L4 Axolotl runtime probe), **463793** (the v2
pre-CUDA zero-signal H200 failure), **463802** (the completed v3 selection-gate failure), and
**463843** (the terminal pre-model PlanIR launch-provenance failure) are now durably protected too.
Never resume, reuse, rename, stop, or delete any of them. The L4
provider preamble proved Axolotl selected CPython 3.11.10, but the probe script itself did not run
because the intentionally minimal directory had no project metadata; do not describe it as a
successful model or scientific run.

The long-lived public browser demo CPU instance **465072** is separately owned and protected; do
not inspect, connect to, restart, pause, rename, reuse, or delete it from research work. Scale-sweep
evidence IDs **465155**, **465183**, and **465186** are durably protected and must never be reused.
Exact H200 **465257** is the completed attempt-6 evidence instance (pause-verified). Retain it as
protected evidence and never resume/reuse/rename/stop/delete it.

On 2026-08-04 at 22:37 Asia/Kolkata, explicit user-authorized unused-resource cleanup permanently
destroyed the following twelve paused historical Barun project instances after their corresponding
local run directories were verified present: **463556**, **463572**, **463575**, **463594**,
**463606**, **463622**, **463631**, **463636**, **463642**, **463674**, **463675**, and **463686**.
The post-cleanup safe inventory verified that all twelve were absent. Their local evidence and
ledger history remain; the remote instances are irrecoverable and must never be treated as
available or reusable IDs. The six protected project-evidence instances above were deliberately
retained. Kroda 463058 and every Kimi, Kriti, or unrecognized resource were outside the cleanup
scope and were not mutated.

### Exact-ID lifecycle

1. Before provisioning, run a read-only inventory and record the pre-existing IDs as a
   denylist in the run record.
2. Create only clearly named `barun-*` resources. Capture the returned resource ID and append
   it immediately to the experiment record.
3. Operate only on that exact recorded ID. Never run bulk pause, stop, restart, or delete
   commands, and never pipe a general instance listing into a mutating command.
4. Use an explicit cleanup trap or watchdog for unattended jobs. Detaching from a CLI process
   is not evidence that an instance will auto-pause.
5. After artifacts and logs are safely copied, explicitly pause the exact project-created ID.
   Query that same ID and record proof that its state is paused.
6. On interruption or failure, first inspect the exact project-created ID, preserve useful
   logs, and pause it. Do not "clean up" anything whose provenance is uncertain.

No existing instance may be borrowed without a new, explicit user instruction naming its ID.
Cost does not justify weakening these rules.

## Evaluation firewall

- Freeze schemas, serialization, normalization, eligible subsets, split indices, and evaluator
  tests before optimizing on a benchmark.
- Official test labels and the sealed human set are final-evaluation-only. Never include them,
  paraphrases of them, or teacher answers derived from them in training or selection.
- The decisive hidden suite must contain at least 2,500 independent human-authored efficacy task
  clusters (target 3,000) plus at least 1,500 independent no-call, ambiguity, confirmation, unsafe,
  or adversarial safety clusters, all authored after the compared model releases. No model or
  teacher may generate its test items. Group by author, source, paraphrase family, delexicalized
  template, and collection batch.
- Hash every raw input, derived split, accepted synthetic shard, prompt template, evaluator,
  and checkpoint. Pin dataset and model revisions rather than relying on moving branches.
- Split by schema/API family, generator template, paraphrase family, entity source, and temporal
  construction where applicable; random row splits are insufficient.
- Scan for exact and near duplicates across training, synthetic, development, and evaluation
  material. Record exclusions rather than silently discarding them.
- Fix a reference timestamp and timezone for relative dates. Canonicalize JSON/AST structural
  whitespace, Unicode NFC, argument order, and scalar types in one versioned evaluator. Action IR
  v1 preserves whitespace inside string values and has no implicit schema defaults; any typed
  string normalizer or default policy must be explicit and versioned.
- The primary result uses unconstrained deterministic decoding. Report grammar-constrained
  decoding separately because syntax enforcement is not semantic competence.
- Keep product comparisons (off-the-shelf models) separate from scientific comparisons
  (baselines tuned on the same data with comparable effort).
- A "beats larger models" claim requires the matched-adaptation lane: identical train/dev example
  IDs, semantic labels, augmentations, teacher information, example budget, seed set, selection
  rule, and number of dev-only hyperparameter trials. Fine-tune the larger baselines too.
- Do not claim superiority without sample-level predictions, paired uncertainty estimates,
  exact model revisions, prompts/templates, decoding settings, and a practical effect-size
  threshold fixed in advance.

## Experiment discipline

Use immutable run IDs such as `YYYYMMDD-HHMM-short-name-sN`. Each run must record:

- hypothesis and one explicit decision it can change;
- git commit plus dirty-tree patch hash;
- command, environment/container, hardware, duration, and estimated cost;
- base checkpoint and tokenizer revisions and hashes;
- raw/processed dataset revisions, licenses, hashes, split derivation, and counts;
- random seeds, hyperparameters, precision, optimizer state, and checkpoint lineage;
- training curves, evaluator version, aggregate metrics, and sample-level predictions;
- failures, anomalies, exclusions, and manual interventions;
- exact JarvisLabs project resource IDs and final paused-state evidence;
- conclusion: `keep`, `reject`, `inconclusive`, or `blocked`, with the preregistered reason.

Append a summary to `experiments/ledger.jsonl`; never rewrite an old result to make a new run
look cleaner. Store detailed artifacts under `experiments/runs/<run-id>/`. Large datasets and
weights stay in remote/object storage and are represented here by manifests, hashes, and
retrieval instructions rather than committed binaries.

Maintain `experiments/mistakes.md` as an append-only incident and lesson log. Record wrong
assumptions, leakage discoveries, metric bugs, failed cleanup, unstable training, and negative
results promptly. Correct prior entries with dated addenda instead of erasing history.

## Training gates

The default ladder is full-parameter completion-only SFT, verifier-filtered hard-data
curricula, student-error/teacher repair, and then preference learning or RL only if warranted.
LoRA may be used for cheap screening, but the final comparison should use the best justified
method for each model.

Do not advance merely because a run finished:

1. A small SFT probe must reach at least 95% schema validity and improve exact match by at
   least 25 percentage points over the base model.
2. Full public-data SFT should target at least 80% Mobile Actions exact match, 70% PRESTO
   overall exact match, OOS F1 of at least 0.80, and no more than a 10-point gap from simple
   queries to revision/disfluency buckets.
3. A synthetic-data tranche must add at least 5 points on hard development data while losing
   no more than 2 points on official development sets.
4. Distillation is retained only for at least a 2-point semantic exact-match gain beyond JSON
   validity; DPO/SimPO requires at least 1 point plus a lower unsafe-call rate.
5. RL is attempted only when the best SFT policy has meaningful exploration (roughly 20--80%
   pass@8 on the targeted hard prompts). Keep it only for a replicated 1.5--2 point gain over
   the best non-RL checkpoint across three seeds without reward hacking.
6. The sealed headline comparison requires 100,000-resample simultaneous paired cluster-bootstrap
   confidence intervals for all named larger baselines with lower bounds above zero and at least a
   3-point practical point-estimate margin on the blind hard set. The simultaneous one-sided upper
   confidence bound for Barun-minus-baseline false-action rate must be below +1 point, with zero
   catastrophic unauthorized actions. Otherwise narrow the claim.

Thresholds may be revised only before looking at the corresponding held-out results, with a
dated rationale in the ledger.

## Code and review rules

- Support Python 3.10 or newer and keep CPU-testable components free of unnecessary GPU imports.
- Put reusable logic in `src/`; keep shell entrypoints thin and non-interactive.
- Add tests for each evaluator rule and each corrected model behavior before launching paid
  experiments.
- Prefer explicit typed configuration files and stable JSON/JSONL artifacts over implicit
  notebook state.
- Never silently fall back to a different dataset, model, split, metric, precision, or device.
- Treat NaNs, missing predictions, parse failures, truncation, and OOMs as measured failures,
  not rows to drop.
- JarvisLabs timing is training/server evidence, not on-device evidence. Any deployment claim must
  be rerun after quantization on named target hardware with the runtime, memory, latency, and energy
  protocol frozen in advance.
- Preserve unrelated user changes and do not commit secrets, caches, checkpoints, or generated
  datasets.
