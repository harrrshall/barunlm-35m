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

## Current handoff and next experiment

The authoritative 2026-08-03 state and recommended next experiment are in
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
float-weight access checks passed. Public source publication is still pending and must use an
explicit secret-scanned allowlist; never bulk-stage the workspace or experiment directories.

The next recommended research lane is Month-Boundary Counterfactual SFT v1: a separately
preregistered, teacher-free, three-arm causal experiment on a newly frozen component split. Compare
standard SFT, unchanged repetition, and deterministic month-boundary counterfactuals across seeds
17, 29, and 43. The reused 756 rows generated the hypothesis and are only a terminal compatibility
veto; the 961 official Mobile rows remain sealed. Freeze generator, evaluator, materialized hashes,
thresholds, and the exact JarvisLabs lifecycle record before loading a model or starting CUDA.
The frozen executable config is `configs/mobile_temporal_counterfactual_v1.json`, full-file
SHA-256 `8c31c4fee18bf4a19e8fa079eec5448c3b62118a13a3eef5a0ba924bb05d07da`; its scientific-field
projection is `964a1a8f25a53f7d11280fb9e1d2cd1c8269bd4afc63078ce63bded02a6351a5`. It binds
1,093 screening transform pairs, the A/B/C manifests, both shadow holdouts, a conditional
1,546-variant full refit, and the audited implementation/dependency hashes. If either digest
changes, stop before model or CUDA access and document a new preregistration.

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
into source, config, CLI arguments, logs, or artifacts. Use W&B only where it adds durable experiment
evidence; local immutable JSON/JSONL artifacts remain the source of truth if tracking is unavailable.

### Protected resources

Every JarvisLabs resource that predates this project session is protected. In particular,
instance **463058**, named **Kroda**, was already running and must never be stopped, paused,
restarted, deleted, renamed, or used for this project. The spelling may be dictated as
"cruda"; treat it as the same protected instance. Discovery of another pre-existing or
unrecognized resource makes it protected by default.

The latest read-only inventory also found running eight-H200 instance **463719**, named
`kimi-k3-jl-node-a-20260803-1724`. It belongs to an independent experiment and is protected. The
previously observed unrecognized ID **463697** remains protected even when absent from a later live
listing; disappearance does not transfer ownership. Never use or mutate either ID.

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
