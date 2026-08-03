# BarunLM post-training operating contract

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

## Current handoff and active experiment

The authoritative 2026-08-04 state and active experiment are in
`docs/current-status-and-next-experiment.md`. Read that file before launching training or changing
release claims.

Candidate-v2 is the current public-release checkpoint: 602/756 float and 607/756 retained ARM64
int8 on the reused Mobile development probe. The independently verified Qwen2.5-0.5B comparison
is now imported under `experiments/runs/20260803-2122-mobile-qwen05b-matched-s17/`: Qwen scored
663/756 versus BarunAction's 602/756. BarunAction is 14.09 times smaller and retains 90.80% of
Qwen's exact-match rate, but trails by 61 rows or 8.07 percentage points. The current
larger-model-outperformance hypothesis failed; never generalize the SmolLM2 0/756 result or hide
the Qwen counterexample.

The candidate-v2 float checkpoint, Darwin ARM64 int8 derivative, and evidence bundle are published
as immutable public W&B `v0` artifacts. Their upload, fresh 310-file redownload, and anonymous
float-weight access checks passed. The secret-scanned source branch is public in draft PR #2;
never bulk-stage the workspace or experiment directories.

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

The current preregistration is the new hermetic v3 run
`20260804-0255-mobile-temporal-counterfactual-axolotl-hermetic-s17`. Its executable config is
`configs/mobile_temporal_counterfactual_v3.json`, full-file SHA-256
`e17f9ef0d755735ce14d66029706534a24145d51e89786aa92d3cd334760b22f`, with scientific-field
projection `f1705c38c064bef646a5814ee9f6cc6c652e9db3c6714e76ff65049fe65ed280`.
V3 is the run revision; its wire/config schema intentionally remains
`barun-mobile-temporal-counterfactual-v2`.
Only `run_id` changed among the 32 scientific fields; hypothesis, data, A/B/C manifests, seeds
17/29/43, optimization, thresholds, gates, requirements hash
`a037db0943d563ea04bcc45b963b5a27931e6f745a7f472c079bb9f9fa6df853`, compute contract, and
official firewall are unchanged. Attempt ordinal is 1 on a fresh stage, output directory, and H200.
Exact template `axolotl` and CPython 3.11.10 must be attested before upload. Compute, runtime,
requirements, or attempt-contract mismatches stop before upload; source-tree or config mismatches
stop before model or CUDA access.

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

### Protected resources

Every JarvisLabs resource that predates this project session is protected. In particular,
instance **463058**, named **Kroda**, was already running and must never be stopped, paused,
restarted, deleted, renamed, or used for this project. The spelling may be dictated as
"cruda"; treat it as the same protected instance. Discovery of another pre-existing or
unrecognized resource makes it protected by default.

The independently owned eight-H200 instance **463719**, named
`kimi-k3-jl-node-a-20260803-1724`, remains protected regardless of its observed lifecycle state.
The previously observed unrecognized ID **463697** remains protected even when absent from a later
live listing.

Project evidence instances **463689** (the paused Qwen comparison H200), **463786** (the v1
zero-signal H200 failure), **463788** (the paused L4 Axolotl runtime probe), and **463793** (the v2
pre-CUDA zero-signal H200 failure) are now durably protected too. Never resume, reuse, rename, stop,
or delete any of them. The L4 provider preamble proved Axolotl selected CPython 3.11.10, but the
probe script itself did not run because the intentionally minimal directory had no project
metadata; do not describe it as a successful model or scientific run.

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
