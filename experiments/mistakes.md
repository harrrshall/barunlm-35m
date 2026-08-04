# Mistakes and lessons

This is an append-only record. Add dated corrections instead of deleting or rewriting prior
entries.

## 2026-08-03 — Reproducibility gaps found before post-training

- The release repository contains aggregate benchmark values but not the sample-level
  predictions, exclusion masks, or executable benchmark command needed to reconstruct every
  published number. Treat the existing benchmark as reported evidence, not a fully reproduced
  result, until those artifacts are recovered or regenerated.
- The native model loss originally required an objective-semantics audit before any new
  training run. A same-position label convention could train token identity instead of
  next-token prediction. Add an explicit regression test before using the trainer.
- Cached attention masks were keyed too coarsely for multi-GPU devices and could retain
  inference tensors across later gradient-enabled calls. Replace unsafe caching behavior and
  test inference-to-training transitions before paid compute.
- The initial CLI status summary and full instance listing did not present the running
  resource inventory identically. The full read-only inventory is authoritative for safety;
  all pre-existing IDs, including Kroda 463058, are protected.

## 2026-08-03 — Cross-sample argument-score cancellation caught in review

- The first Action IR evaluator draft aggregated argument facts without a sample namespace.
  A wrong value predicted on one row could therefore cancel the corresponding miss on another
  row and inflate micro-F1. Namespace facts by immutable sample ID before Counter intersection;
  the regression test swaps values across two samples and requires argument-value micro-F1 zero.

## 2026-08-03 — Fresh H200 launch raced SSH readiness

- JarvisLabs created project-owned H200 instance 463544, but the initial managed-run command tried
  SSH before the instance was ready and failed before project code executed. The controller found
  the one exact new `barun-*` name, recorded its ID, paused only that ID, and verified `Paused`;
  Kroda and all other pre-existing resources were unchanged. Retry the immutable command only by
  resuming this proven-owned ID, waiting for an explicit SSH probe, and pausing it again in `finally`.

## 2026-08-03 — Remote retry and data-firewall preflight corrections

- Resuming a JarvisLabs container can replace its machine ID. The controller must capture the
  explicit replacement ID, re-prove that it is neither protected nor pre-existing, and make that
  exact ID the only cleanup target. The local OpenSSH identity initially matched none of the
  account's registered public keys; a dedicated user-level identity is now registered outside the
  repository, and its fingerprint must be verified before provisioning paid compute.
- JarvisLabs rejects fresh-instance lifecycle flags such as `--keep` when `jl run` attaches with
  `--on`. The first attached command therefore stopped before project code ran. Remove lifecycle
  flags from attached commands and let the exact-ID controller perform the final pause; keep a
  regression test for the generated command.
- A stopped CUDA preflight proved the H200 environment and all repository tests, but it exposed two
  issues before training. Deterministic cuBLAS configuration must be set before importing Torch or
  initializing CUDA, and a public evaluation split is not "unmaterialized" if its labels are still
  parsed for audit statistics. The official 961 Mobile Actions rows must remain opaque count/hash
  evidence during train/development preparation, and near-duplicate grouping must use only the
  internal training population. Project instance 463556 was verified `Paused` after stopping this
  preflight; it produced no model checkpoint or score.

## 2026-08-03 — Real-GPU test ordering and efficient artifact collection

- Attempt `20260803-1800-mobile-blind-s17` exposed a GPU-only test-order bug: earlier tests could
  initialize CUDA before a duplicate-run test reached the CUDA preflight. The attempt exited before
  model download, data preparation, scoring, or training, and instance 463572 was pause-verified.
  Claim the run root atomically before any CUDA inspection; the retry includes a regression test
  that proves duplicate IDs fail without touching CUDA.
- Attempt `20260803-1810-mobile-blind-s17` completed successfully, but the default recursive
  download included three large optimizer states that were unnecessary for candidate inference.
  A compact essential bundle was downloaded separately and every file was matched against the
  remote artifact-tree hashes. Only after that proof was the redundant bulk transfer interrupted;
  the successful run remained exit 0 and exact project instance 463575 was pause-verified. Future
  runners should export inference/evidence bundles separately from resumable optimizer state.

## 2026-08-03 — Clean source snapshots exposed a non-hermetic integration test

- Attempt `20260803-1835-mobile-followup-s17` intentionally uploaded 5.3 MB of source instead of
  the 1.4 GB workspace containing old environments and checkpoints. Repository tests stopped before
  data preparation or model access because one candidate-identity test read a prior experiment's
  local checkpoint manifest. Re-running the exact staged snapshot locally reproduced the sole
  failure: 1 failed and 124 passed. Candidate provenance now lives in a small checked-in JSON
  manifest, so source-only and wheel/clone test environments are hermetic.
- The progressive essential bundle initially had environment evidence but not the captured test log
  because the log copy occurred only after a successful test return. Test stdout/stderr now streams
  to the managed-run log and the repository test log is copied into the progressive bundle before
  checking its exit code. Fresh project instance 463594 was pause-verified; no development trial,
  model score, training step, or official-evaluation read occurred.

## 2026-08-03 — Bottleneck oversampling was worse than uniform extra updates

- In the final frozen Mobile development sweep, uniform `batch63` training reached 602/756 strict
  exact matches, a +3.17-point gain over the 578/756 reference. The fixed-size `hardmix70` arm
  instead fell to 566/756 despite emphasizing the observed calendar/map/multicall bottlenecks; it
  repeated 817 hard slots and omitted the same number of easy-source rows. A plausible error
  taxonomy is not evidence that manual oversampling will help a 35M model. Keep the uniform arm,
  reject the hard mix, and do not spend more Mobile development trials trying to rescue it.
- The promoted 602/756 candidate clears the sweep's preregistered +1-point practical margin but is
  three correct rows short of the separate 80% full-public-data target. Report 79.63% exactly and
  move to contextual PRESTO/safety evidence; do not round it into a passed 80% gate or inspect the
  sealed 961-row Mobile evaluation set.

## 2026-08-03 — Text-only baseline inherited an incompatible vision package

- The first matched SmolLM2 launch passed its repository and frozen-data checks and downloaded the
  pinned public snapshot, but Transformers 4.48.3 discovered JarvisLabs' system `torchvision` while
  importing Llama. That system build did not match the isolated Torch 2.13 runtime, so import failed
  at the absent `torchvision::nms` operator before model construction, rendering, training, or
  development scoring. Instance 463631 was pause-verified and the attempt consumed no development
  trial. For the infrastructure-only retry, pin `torchvision==0.28.0`, the release paired by the
  resolver with `torch==2.13.0`, so the isolated environment shadows the unrelated system package.
  Keep this dependency correction separate from the frozen model/data/training recipe.

## 2026-08-03 — A literal-string taxonomy dropped the entire PRESTO revision family

- The frozen PRESTO scorer grouped a row as `revision` only when its raw label contained that
  literal word. PRESTO instead records four subtype labels for the paper-defined user-revision
  family, so the original evaluator exposed no revision bucket and the focus selector replayed
  zero revision rows. The original preregistered gate correctly remains failed. Preserve its v1
  semantics for artifact reproduction, version the complete primary-source taxonomy separately,
  and report any rescored gate only as a post-hoc diagnostic with per-subtype evidence.

## 2026-08-03 — PRESTO-only continuation catastrophically forgot Mobile Actions

- Sharing Action IR syntax did not preserve the earlier task. The PRESTO checkpoint scored 0/756
  on the frozen Mobile regression set versus candidate-v2's 602/756: all 602 formerly correct rows
  regressed, with zero fixes or retained-correct rows. The mandatory regression gate prevented a
  misleading broader-candidate promotion. Future continual stages must preregister retention
  explicitly—through checkpoint interpolation or rehearsal/recovery—and must pass both task gates
  before receiving a shared product identity. Do not infer retention from schema compatibility or
  a low post-training loss.

## 2026-08-03 — Jarvis requirements staging added an accounted root file

- Continual-recovery launch `20260803-2106-continual-recovery-s17` stopped at the staged-tree
  provenance check, before CUDA inspection, repository tests, checkpoint hashing, data preparation,
  training, or scoring. Jarvis copied `requirements/continual-recovery.txt` to remote staged-root
  `continual-recovery.txt` before invoking the runner. Virtually adding that one identical file to
  the local 125-file receipt reproduces the observed remote SHA-256
  `9eccac423a75b7aab1dde8db0b88d653ac7799ae626e5fd60834ea544da8d5ec` exactly, proving the
  mismatch rather than guessing from it. The fail-closed guard worked, evidence was downloaded,
  and exact owned H200 463675 was pause-verified. A fresh infrastructure retry may include this
  deterministic provider-created file in its preregistered source tree; it does not authorize a
  recipe change or another scientific trial.

## 2026-08-03 — Checkpoint interpolation exposed a steep task tradeoff

- The three frozen points on the candidate-v2-to-PRESTO trajectory did not contain a broadly useful
  checkpoint. Alpha 0.25 retained Mobile at 598/756 but scored 0/14,288 PRESTO exact; alpha 0.50
  fell to 452/756 Mobile and 961/14,288 PRESTO; alpha 0.75 reached 9,447/14,288 PRESTO but collapsed
  Mobile to 8/756. Shared syntax and a continuous parameter path therefore did not imply a smooth
  capability tradeoff. No arm passed the hard conjunction, so retain candidate-v2 and do not use
  the negative result to justify another alpha after seeing the scores.
- The H200 log emitted one allocator allocation-failure warning during each PRESTO pass. None raised
  a measured generation exception: all three arms recorded zero generation failures and contain
  every expected 756-row Mobile and 14,288-row PRESTO prediction and score. Preserve the warnings as
  anomalies, but do not misreport them as dropped rows or silently remove affected evidence.

## 2026-08-03 — Continual recovery was a 140-row PRESTO near-miss, not a promotion

- The one-shot train-only rehearsal recipe restored Mobile to 642/756 and reached 9,862/14,288
  PRESTO derived Action IR exact match. PRESTO required 10,002 correct rows, so 69.02% remains a
  140-row miss rather than a rounded 70% pass. Schema validity (14,254/14,288), abstention F1
  (0.97435), false calls (102/10,407), both hard-bucket gaps, all four revision subtypes, Mobile
  safety, and Mobile exact match passed their frozen gates; cross-metric strength cannot compensate
  for the one failed capability gate.
- Keep checkpoint SHA-256
  `744b640828122ea9805131671c9792af9acfbd748e3199bacc513dcd8799c4c9` as research-only evidence.
  Candidate-v2 remains the current product/release-development checkpoint, not a demonstrated joint
  passer. The outer four-trial selector found no eligible candidate and authorizes neither another
  interpolation alpha nor a recipe retry. Both official test populations remained unread, so these
  reused-development results support no blind-breakthrough, larger-model, or release claim.

## 2026-08-03 — SmolLM2 collapse was model-and-recipe-specific, not a larger-model win

- The adapted SmolLM2 checkpoint's 0/756 result looked like a dramatic parameter-efficiency win,
  but one failed baseline cannot support a claim about larger models as a class. A user-supplied,
  independently verified Qwen matched-baseline handoff reports 663/756 (87.70%) strict AST exact
  against candidate-v2's 602/756 (79.63%) on the aligned reused development set. The paired result
  is 80 Qwen-only wins, 19 Barun-only wins, and 657 ties; Qwen produced 755 parse-valid and 754
  schema-valid outputs. All 961 official Mobile Actions rows remained untouched.
- The Qwen checkpoint contains 494,032,768 BF16 parameters across 290 tensors—0.494B, not 500B.
  BarunAction is 14.09 times smaller and retains 90.80% of Qwen's exact-match rate, but trails by
  61 rows or 8.07 percentage points. Record the larger-model-outperformance hypothesis as failed,
  retain only the honest size-versus-quality tradeoff, and never generalize the SmolLM2 failure.
  The Qwen handoff remains pending local bundle import and hash verification; do not fabricate a
  run ID, hash, or checked-in provenance claim before that evidence arrives.

### Same-day addendum — Qwen handoff imported and hash-bound

- The complete 965 MB handoff subsequently arrived at the supplied path and every entry in its
  whole-bundle SHA-256 manifest passed. Source files were imported byte-identically from the
  reviewed patch. A safe 63-file evidence subset was copied into run
  `20260803-2122-mobile-qwen05b-matched-s17`; the comparison checkpoint, full Jarvis inventory,
  remote log, and duplicate source copies were excluded. `import-receipt.json` and
  `selected-artifact-sha256.txt` preserve the exact boundary. The earlier pending-status sentence
  remains above as chronology, not current state.
- Post-hoc paired inspection found that 76 of 80 Qwen-only wins include a BarunAction argument-
  value mismatch. Calendar contributes the largest net gap. In 23 Qwen-only calendar rows, the
  candidate preserved year/day/time but emitted the preceding month, usually matching the `NOW`
  month. Preserve this as a dated diagnostic and preregister any counterfactual-time curriculum on
  a fresh split; never tune repeatedly on the 756-row diagnostic population.

### Same-day correction — the Qwen whole-handoff check covered 85 files, not 80

- The immutable import receipt says `files_checked: 80`, but the preserved command output contains
  85 `OK` lines and zero failures, exactly matching all 85 entries in `artifact-sha256.txt`. The
  source tree has 86 regular files because a checksum manifest cannot list itself. The origin of
  80 is unknown; it must not be explained away merely because other subsets happen to total 80.
- The incorrect receipt was already bound into the public release manifest, so silently editing it
  would destroy that provenance chain. Preserve it and use `import-receipt-correction-v1.json`
  plus `full-handoff-sha256-check.log` as an immutable addendum. Future summaries must say the full
  tree is 1,008,838,966 logical bytes (1008.839 MB or 962.104 MiB); `du -sh` reported `965M`
  allocated size, which is not a precise logical-byte measurement.

## 2026-08-03 — Release evidence filtering must permit legal notice files explicitly

- The first candidate-v2 W&B dry run failed closed because the evidence allowlist permitted only
  selected text/JSON suffixes and rejected the required no-suffix `LICENSE` and `NOTICE` files.
  Add only those two exact names; do not weaken the extension filter generally. A regression test
  now proves that both legal files pass while an arbitrary `weights.bin` evidence payload fails.
- A second dry run passed, followed by immutable upload and a fresh 310-file redownload. The W&B
  project initially inherited private visibility, so publication was not complete merely because
  upload succeeded. It was changed to public-read/team-write (`USER_READ`) and then checked without
  credentials, including a full 140,304,464-byte float-weight stream whose SHA-256 matched. Always
  verify distribution visibility and bytes separately from the upload receipt.

## 2026-08-03 — Temporal diagnostics must distinguish rows, calls, and argument occurrences

- An intermediate summary understated one candidate detail as 97 datetime errors preserving the
  correct time. Recomputing against the frozen 756-row manifest and both immutable sample-score
  files gives 98 of 98 aligned datetime mismatches with the correct time, wrong date, and predicted
  year-month copied from `NOW`. The added temporal-analysis receipt records its source hashes and
  definitions; the original released post-hoc receipt remains unchanged.
- The targeted cross-month population is 59 `create_calendar_event` call occurrences across all
  single- and multi-call examples, not only the 21 sole-calendar rows or 43 rows whose first call
  is calendar. BarunAction gets 16/59 datetime values exact versus Qwen's 52/59; 43 versus seven are
  wrong. Evaluators for the next experiment must score the aligned calendar argument directly so
  an unrelated call error cannot be mistaken for a temporal error.

## 2026-08-04 — Pre-CUDA audit closed retry, source-install, and failure-evidence gaps

- The first Month-Boundary Counterfactual launch draft allowed JarvisLabs requirements handling to
  perform an editable project install. That could create `src/barunlm.egg-info` before the staged
  source-tree check and make provenance depend on installer side effects. The frozen requirements
  file now contains third-party pins only; the CPU test child and backend import BarunLM solely from
  the exact staged `src` directory, and the controller performs no setup command when this explicit
  requirements file is supplied.
- Early retry validation proved only a few named hashes. Unlisted source could change, prior
  evidence could collide with a scientific path, or a standalone zero-signal failure receipt could
  conceal held-out artifacts present in the downloaded bundle. Retry evidence is now confined to
  an attempt-scoped namespace, bound through the downloaded essential artifact manifest and
  lifecycle receipt, and compared against the complete prior scientific content-tree projection.
  Any held-out marker, prediction, score, or development metric forbids retry. Ordinal-two and
  tampering regressions pass.
- The outer failure wrapper could mask the original backend error or return a compact bundle that
  omitted partial scores, the promoted full-refit weights, or the compact-export diagnosis. It now
  preserves the original exception, validates an existing backend failure, and copies a bounded
  fail-closed evidence set including the only promotable `model.safetensors`. Retrieval remains the
  exact recursive `execution/essential` tree before the owned instance is paused.
- The first launch-provenance builder design trusted caller-supplied Git identity and could write a
  Python bytecode cache while importing the staged runner. It now derives identity from a clean
  canonical repository HEAD, verifies every committed file plus an exact data-addition allowlist,
  runs the real materialization preflight, rejects credential-like content, requires all critical
  implementation hashes, and disables/restores bytecode writes during staged imports. A real-runner
  regression proves no `__pycache__` is created.
- These were pre-execution control failures, not failed model experiments: no model or CUDA was
  loaded, no selection, confirmation, reused-756, or official-961 row was read or scored, and no
  JarvisLabs resource was created or mutated. After the corrections, 401 repository tests and the
  129-test frozen remote CPU gate pass. Preserve this distinction so infrastructure hardening is
  never counted as a scientific trial.

## 2026-08-04 — Template Python and provider-created environments must be attested, not assumed

- MBCF v1 attempt 1 created fresh H200 463786 but stopped in JarvisLabs dependency resolution.
  The default PyTorch template exposed CPython 3.10.20, while the frozen
  `numpy==2.4.6` pin requires Python 3.11 or newer. Safe-run paused and rechecked the exact ID.
  The experiment entrypoint never began, so model/CUDA access, all held-out reads, predictions,
  scores, and official-961 access remained zero. Record this as blocked infrastructure evidence,
  not a scientific rejection and not an excuse to alter the recipe after seeing a score.
- Recursive essential-artifact retrieval failed because the remote runner never created
  `execution/essential`. The raw lifecycle truthfully records that failure. Do not synthesize a
  remote artifact or adjudicated ordinal-2 receipt merely to satisfy the retry validator. The v1
  byte-identical retry lane therefore remains unauthorized even though the substantive failure
  contained no scientific signal.
- A cheap fresh L4 Axolotl check, exact ID 463788, established from the provider preamble that the
  template selects CPython 3.11.10. The probe script itself did not run: without an explicit
  requirements file, the provider attempted an editable project install and rejected the minimal
  directory because it had no `pyproject.toml` or `setup.py`. Preserve both facts; the Python
  observation is useful operational evidence, but the run is not a successful probe workload or
  model experiment.
- The same provider workflow creates and activates a repository-root `.venv` for directory
  targets. The v1 source validator would reject that directory even after dependency resolution
  succeeded. A local `uv run --help` had already demonstrated the same trap and correctly
  disqualified the first draft stage. Future launch validation must model provider setup
  side-effects explicitly rather than assume uploaded bytes remain the entire runtime tree.
- The replacement is a new v2 preregistration, not v1 attempt 2. It leaves the hypothesis, data,
  materializations, requirements, training recipe, gates, and official firewall unchanged, while
  binding Axolotl and CPython 3.11.10. Safe-run now validates the unbound compute contract before
  create, attests template/hardware and exact Python through the fresh ID before binding or upload,
  and fails closed on any mismatch. The runner permits only the active non-symlink root `.venv`
  and still excludes it from the scientific tree; every other cache/environment directory remains
  forbidden. The full suite passes 423 tests before the new run is staged.

## 2026-08-04 — Remote tests must not inherit a private workstation denylist implicitly

- MBCF v2 attempt 1 correctly omitted the ignored private JarvisLabs denylist from its clean source
  stage, but two tests called the production denylist loader without installing their existing
  temporary fixture. The remote CPU gate therefore reported 146 passed, one skipped, and two
  failed before parent-runner Torch import, model/CUDA access, or held-out reads. Production
  remained correctly fail-closed; adding fallback resource IDs or uploading the private account
  inventory would have weakened the safety boundary rather than fixed the tests.
- Inject the temporary denylist only in tests that exercise later inventory behavior, and keep an
  explicit regression proving the default production path raises when its private denylist is
  unavailable. Before provisioning paid compute, execute the exact remote test allowlist from a
  clean archive with the ignored private file absent; the corrected gate passes 150 tests with one
  intentional skip.
- A zero-signal failure does not by itself authorize a retry. V2 froze the complete source tree, so
  changing `tests/test_jarvis_safe_run.py` invalidates its byte-identical ordinal-2 lane. Preserve
  v2 as inconclusive infrastructure evidence and preregister v3 at ordinal 1 with a fresh config,
  source snapshot, output directory, and H200. Never smuggle an undeclared environment dependency
  into the old bytes to make a retry validator pass.

## 2026-08-04 — A strong targeted repair is not a product-gate pass

- Hermetic MBCF v3 completed all nine screening fits and independently replayed exactly, so its
  rejection cannot be attributed to infrastructure. Counterfactual arm C improved cross-month
  calendar-datetime exact by 31.82 points versus standard A and 51.52 points versus repeat B, yet
  its overall gains were only 1.60 and 2.44 points against a frozen 3-point requirement. It also
  exceeded the same-month loss ceiling versus B. Localized causal evidence is valuable, but it may
  not substitute for the preregistered whole-product conjunction.
- Exact rational values in the gate JSON are reduced fractions. The truncation rate appears there
  as `1/1536`, but the pooled C population contains 2 truncations in 3,072 predictions. Report raw
  counts from the comparison aggregate, not a reduced numerator/denominator as though it were the
  evaluated sample count. C had 3,058/3,072 parse-valid and 3,041/3,072 schema-valid outputs; the
  schema gate missed by one valid output.
- The zero-weight evidence policy worked as intended. Because selection failed, confirmation,
  full refit, reused-756, and official-961 access stayed at zero, and no optimizer, screening, or
  promoted weight file entered the essential bundle. Do not rerun the recipe to recover a favorable
  seed or reconstruct discarded checkpoints.
- Retire both v3 shadow populations from later selection, preserve candidate-v2, and require a
  genuinely distinct hypothesis with fresh leakage-controlled populations. The proposed Grounded
  PlanIR direction must first prove unambiguous coverage and exact deterministic round trips before
  any GPU training; it is not a relabeled MBCF rescue.

### Same-day addendum — a new salt does not make an old population fresh

- The first Grounded PlanIR handoff draft proposed repartitioning the 5,745 former v3 construction
  rows into new train, selection, and confirmation roles. That would move data already used to fit
  prior models into later evaluation roles and cannot supply genuinely new post-v3 evidence. It
  also lacks author, entity-source, generator-template, and collection-batch provenance needed for
  the strongest independence claims.
- Keep former v3 construction rows training-only if they are used at all. New PlanIR selection and
  once-only confirmation require a newly acquired, licensed, provenance-audited source batch.
  Failing to establish that source boundary is a pre-model stop, not permission to weaken the word
  “fresh” or spend the sealed official populations as iterative development sets.

### Same-day addendum — semantic compactness does not imply token compactness

- The first quote-based Grounded PlanIR design looked structurally simple but tokenized poorly. On
  the 5,744 accepted construction-training rows it used 735,301 target tokens versus 514,097 for
  direct Action IR, lengthening every target. This was caught before GPU training. Preserve
  quote-v1 as feasibility evidence; do not call it compact or promote it to the primary treatment.
- A target-independent input reference table plus a typed calendar placeholder is the selected
  placeholder-v2 design. It uses 506,062 target tokens, shortens all 2,069 representable calendar
  targets by 8,035 tokens total, and leaves non-calendar targets byte-identical. The matched direct
  control must receive the exact same table so a table benefit cannot be credited to PlanIR.
- Always audit the pinned tokenizer over both targets and complete prompt-plus-target sequences
  before freezing a representation. A human-readable schema can still add a generation-length and
  optimization confound; cheap token accounting should reject that confound before paid compute.

### Same-day addendum — a narrow compiler is not a decisive general-purpose lane

- Placeholder-v2 currently handles seven frozen Mobile tools, one calendar datetime argument,
  empty context, and timezone-naive reference timestamps. That is sufficient to test a calendar
  grounding mechanism, but not contextual revisions, timezone-aware behavior, renamed tools, or
  unseen schemas. Do not spend the decisive human confirmation population on it or call a narrow
  pass the project's breakthrough.
- Screen the mechanism only on a preregistered internal split of former construction-training rows.
  Preserve those scores as non-fresh mechanism evidence. A pass licenses a generalized
  context/timezone/arbitrary-schema compiler and new fresh evaluation; a failure closes the compact
  placeholder direction without further human-data or Qwen spend.
- Scope review belongs before data commissioning. A component can be correct, efficient, and
  well-tested while still being incapable of supporting the claim or population designed for the
  final system.

### Same-day addendum — validate and authorize from one immutable snapshot

- An early human-firewall implementation validated receipt membership and ledger history, then
  reused the caller's original generic `Sequence` for later membership and retirement checks. A
  custom sequence could expose one value while validating and another later. Validators also
  returned caller-owned dictionaries, leaving a checked value mutable during authorization.
- Canonically snapshot every protocol object once into detached exact JSON, reject custom sequence
  implementations, and reuse only that snapshot for hashes, membership, retirement, event
  construction, and returned evidence. Regression tests must exercise the public disclosure path,
  not merely the lower-level hash validator.
- Do not trust a declared `cluster_id` as an independent-power counter. Count connected components
  across author/source/batch, semantic/entity/temporal/schema lineage, and duplicate evidence so
  aliases cannot inflate a one-shot confirmation suite. Enforce the full chronology: duplicate
  scan, input-only eligibility, independent labels, envelope sealing, passing selection receipt,
  then disclosure.

### Same-day addendum — a minimal stage must reject pre-upload caches and bind the whole config

- An early PlanIR launch builder excluded `__pycache__`, `.pytest_cache`, `.ruff_cache`, bytecode,
  and `.DS_Store` entries from its scientific tree even before upload. Exclusion was unsafe at that
  boundary because hidden credential or held-out bytes could still be transferred. The builder now
  rejects every such entry before the snapshot; only the remote validator may ignore caches created
  after safe-run has bound the fresh instance and uploaded the already-clean stage.
- Validating selected config fields is not equivalent to freezing a preregistration. The builder
  now pins the complete config bytes at SHA-256
  `cca598e6f5a76a5e848a8b9a869cad779bbcbb17999217d2488111b22e2aa34c`, in addition to strict
  structural checks, and pins the requirements bytes at
  `6db8f37c0c21aea4a4ad93d7193b82217e73d3090db6b8947a9a9321aefb21c9`.
- A hand-written “zero-signal” receipt is insufficient authority for another paid attempt. This
  mechanism screen has exactly one attempt and every retry, resume, rescue, threshold change, and
  new ordinal is forbidden, including after an infrastructure-only failure.

### Same-day addendum — export integrity is part of the one-shot experiment

- A correct training/scoring path can still lose its scientific value if its final export silently
  omits replay data or preserves bytes that changed after preflight. The PlanIR essential exporter
  now requires the config, requirements, source snapshot, and bound attempt; rehashes the config,
  requirements, and all nine materialized manifests immediately before copying; includes all nine
  manifests; preserves raw predictions, metrics, checkpoint manifests, gates, and receipts; and
  derives a complete size/hash inventory.
- Screening weights are deliberately not downloaded or promoted from this internal mechanism
  screen, but the exporter now independently rejects any selected `.safetensors`, optimizer, or
  other weight-like file. Dedicated temporary-tree tests cover successful replay export, mutated or
  extra data, missing mandatory provenance, and accidental weight selection before a no-retry H200
  launch is allowed.

### Same-day addendum — freeze the provider transformation, not only the uploaded directory

- The sole placeholder-v2 attempt uploaded the exact validated 47-file scientific stage, but the
  JarvisLabs managed `--requirements` preamble then copied the requirements file into the remote
  target root as `mobile-planir-screen.txt`. The first stdlib provenance walk correctly rejected
  that unbound extra file. This was a terminal operational failure before the CPU gate, Torch/CUDA,
  model access, row decoding, training, or scoring.
- A clean local stage and mocked runtime-cache tests do not reproduce a provider-managed launch.
  Before authorizing a future run, capture and simulate the provider's exact target transformation:
  copied requirement basenames, generated virtual environments, launch wrapper, working directory,
  and argument ordering. Validate the post-transformation tree locally with the same runtime mode
  that will execute remotely.
- Keep scientific fail-closed behavior. The lesson is not to broadly ignore root files; it is to
  separate provider-owned runtime inputs from the immutable scientific target or bind every
  provider-created path explicitly. Tests must reject an unknown neighboring file while accepting
  only the exact provider transform that the next new contract freezes.
- Run `20260804-0545-mobile-planir-construction-screen-s17` allowed one attempt and declared every
  operational failure terminal. Machine 463843 is paused and protected, the evidence is preserved,
  and no retry, rescue, relaunch, or new ordinal is authorized. A reusable orchestration fix belongs
  to a genuinely distinct future hypothesis, never retroactive permission to rerun placeholder-v2.

### 2026-08-04 addendum — distinguish machine IDs from endpoint identifiers

- The adjacent `kimi-k3-jl-node-opt-20260804` job was initially recorded in `AGENTS.md` as instance
  463904. A fresh read-only `jl list --json` inventory at 10:38 Asia/Kolkata showed that its actual
  `machine_id` is 463912; 463904 appears inside provider endpoint naming and was mistaken for the
  lifecycle identifier.
- Lifecycle code and manual audits must read the structured `machine_id` field. Never derive a
  pause/resume/destroy target from a notebook URL, endpoint hostname, public address, display name,
  list position, or copied prose. Preserve the incorrect historical identifier rather than
  rewriting the record, and denylist both 463912 and 463904.
- The raw JSON listing can itself contain access-bearing notebook URLs. Future inventory checks
  must project only lifecycle-safe fields at the command boundary and must not preserve or quote
  raw endpoint values in logs, receipts, messages, or committed evidence.
- No connection or mutation was made. The eight-H200 adjacent job and Kroda 463058 remained
  untouched. This correction is evidence for stricter exact-ID parsing, not permission to inspect
  either resource.

At 11:11 Asia/Kolkata, a later safe projected inventory showed the same adjacent job name running
under a newly created structured `machine_id` 463936. This does not rewrite the earlier correction:
463912 is now a protected historical machine ID and 463904 remains a protected endpoint-derived
identifier. All three are denylisted. A display name is not a lifecycle identity, and an adjacent
owner may recreate a job while this work is running. No connection or lifecycle mutation was made.

At 22:29 Asia/Kolkata, a further safe projected inventory found 463936 paused and discovered a
separate paused eight-H200 resource named `kimi-k3-jl-node-opt128-20260804` with structured
`machine_id` 463964. Paused does not mean available: 463964 is an adjacent, unrecognized resource
and is protected by default. It was added to the durable denylist without connection or lifecycle
mutation. The lesson is broader than correcting one ID: every fresh inventory may reveal a new
protected resource, and only a newly created, recorded `barun-*` ID can be operated on.

### 2026-08-04 addendum — restore PyTorch's default-device mode, not only its displayed value

- A decoder test temporarily called `torch.set_default_device("meta")` and restored it with
  `torch.set_default_device("cpu")`. Although `torch.get_default_device()` then displayed `cpu`,
  PyTorch retained a process-global `DeviceContext` function mode. A later verifier test correctly
  rejected that unexpected execution mode, so a broad decoder-to-verifier run failed.
- The teardown now calls `torch.set_default_device(None)`, which restores the native CPU default
  and removes the installed mode. The exact decoder test followed by the complete verifier slice
  passes 42 tests; Ruff and formatting checks pass for the changed decoder test.
- This was test-process contamination only. It changed no prediction, metric, checkpoint, dataset,
  remote job, or release artifact. Never weaken verifier execution-mode rejection to accommodate a
  leaking test; restore process-global state at the test that changed it.

### 2026-08-05 addendum — private factory tokens are not validation

- The deterministic GVS program planner used module-private factory tokens to restrict receipt
  construction, but a caller able to import the module could still obtain those tokens and create
  a `ProgramPopulationRecord` whose `program_sha256` was false while `round_trip_passed` serialized
  as true. The corresponding plan constructor also accepted a false membership commitment.
- Constructors now live-recompute the exact program, world, authoring packet, simulator round trip,
  record, roster, membership, runtime, record-set, and artifact-body commitments. The regression
  test deliberately imports the private tokens and proves that both forged objects fail closed.
- A private name or unexported token can guide ordinary callers, but it is never an authenticity or
  integrity boundary. Any serialized scientific claim must be rederived from the bound payload at
  its public construction and load boundaries.

### 2026-08-05 addendum — independent components, not paraphrase rows, define support

- The first tool-frequency shortcut calibration counted raw D-support rows. Ten paraphrases in one
  connected component could therefore outweigh one row from an independent component by 10:1,
  despite the protocol defining effective evidence through connected components.
- Tool-frequency calibration now assigns unit mass to each independent component and divides that
  mass equally among its rows. Raw row counts remain diagnostics; exact component masses are
  serialized and revalidated. The focused audit also closed cross-role prompt/request/trace
  overlap hidden behind different metadata and rejected forged shortcut receipts.
- A separate schema audit found that deterministic role-disjoint rename maps were being described
  too strongly. They prove unseen names under known semantics only; without external custody and
  authenticated chronology they are not held-out evidence, and they never establish novel semantic
  schema competence. Those claim flags now remain false.

The final patched repository passed 1,338 tests; the independent integration slice passed 365
overlapping tests. No model, private label, CUDA device, JarvisLabs resource, reused-756 row, or
official-961 row was accessed while finding or fixing these issues.

### 2026-08-05 addendum — proposal targets are not frozen power gates

- The first CPU-prefreeze result serialized the proposed 85% oracle pass@8, +10-point greedy gap,
  and 50% greedy-failure recovery values under a `pass_requirements` key even though the research
  contract says all numerical thresholds remain provisional until a real pre-outcome component
  roster and external baseline/discordance assumptions exist. That key could overstate the gate.
- The result was already hash-bound in the append-only ledger, so it was not silently rewritten.
  `correction-20260805-0057.json` preserves the original bytes and explicitly limits those values to
  proposal targets with no authorization effect; the correction is also appended to the ledger.
- Future prefreeze artifacts must serialize unfrozen values as `provisional_targets`, never
  `pass_requirements`. Field names are part of the scientific claim boundary, not cosmetic prose.

### 2026-08-05 addendum — nominal rows are not an effective-component roster

- The preliminary GVS allocation proposed 2,000 `D-support` rows and only eight schema families
  per role. The live population firewall correctly joins every pair of rows that shares a schema
  family. A controlled reproduction with 16 otherwise independent rows therefore produced eight
  components of size two, not 16 independent components; a fixed floor of nine failed closed.
- One honest shared `collection_batch` joined all eight families transitively and reduced the same
  role to one component. Reusing a family or batch identifier across roles failed even earlier
  because the component crossed population boundaries. Adding rows does not repair either defect.
- The mistake was treating a nominal allocation table as if it implied statistical independence.
  `MAX_FAMILIES_PER_ROLE=8` cannot be copied into `provenance.schema_family` for a powered
  population, and shared batch/source fields must be audited through the complete lineage graph.
- The current allocation is rejected before authoring, collection, model loading, CUDA, or
  JarvisLabs access. The GVS mechanism itself remains untested. Any replacement must bind claimed
  components to genuinely distinct source and presentation evidence; per-row identifiers invented
  only to raise the denominator are forbidden.

### 2026-08-05 addendum — an offline collection form is not human-data custody

- The first GVS human-collection scaffold leaked private assignment identity and S/C role/stratum
  expectations into author or labeler views, allowed quota validation to be altered through a
  substituted `Counter`, and exposed a rebindable helper that could forge authorization flags.
  It also described prompt content as authenticated, used a loose license contract, and omitted
  complete independent-label/no-prediction attestations.
- The package now exports grouped identifier-free author tasks, derives blinded labeling tasks
  only after prompt intake, seals separate exact-JSON label envelopes, binds its live runtime, and
  keeps all provenance, custody, effective-cluster, schema-bridge, power, and authorization claims
  false. Twenty-six focused tests and an independent adversarial review pass.
- These checks validate local serialization only. No person was recruited and no prompt or label
  was collected or opened. Declared identity, time, consent, and license metadata do not become
  authentic merely because a validator checks their shape.

### 2026-08-05 addendum — a durable database is not an external security boundary

- The initial SQLite retirement rehearsal accepted a caller-supplied false repository root,
  existing hardlinks, non-exact ownership/modes, and a connect-time path replacement. Its
  authenticator attribute could also be rebound by ordinary same-process Python.
- The service now fixes the actual source root, requires owner-only `0700`/`0600`, rejects links,
  uses no-follow descriptor opens plus descriptor/path identity checks, seals ordinary attributes,
  and closes connections on validation failure. The focused 20-test suite and 43-test
  authorization integration pass.
- Same-UID code can still inspect or interfere with the process and filesystem. External signing,
  isolated service identity, ACL and target-filesystem validation, backup/anti-rollback anchoring,
  and crash testing remain mandatory. The storage callback therefore still authorizes no label
  access and proves no external atomicity.

### 2026-08-05 addendum — adjacent display names can be recreated repeatedly

- A safe projected inventory at 02:14 Asia/Kolkata found that the adjacent Kimi owner had recreated
  `kimi-k3-jl-node-opt-20260804` as structured `machine_id` 464346 and
  `kimi-k3-jl-node-opt128-20260804` as 464367. These are distinct from the earlier protected IDs
  463912, 463936, and 463964 even though the display names repeat.
- Both new IDs were added to the durable denylist immediately. No endpoint was retained, no
  connection was made, and no lifecycle action was taken. Kroda 463058 and all unrelated resources
  remained untouched.
- A preflight denylist is a snapshot, not a permanent name-to-ID map. Every future provisioning
  decision must take a fresh safe inventory and protect every pre-existing ID before creating one
  explicitly named BarunAction resource.

### 2026-08-05 addendum — validate the uploaded artifact, not only its staging allowlist

- The candidate-v2 release documentation said unrelated project-resource names were not
  redistributed, but the immutable public W&B evidence `v0` retained non-access-bearing local
  workstation paths and mentions of unrelated protected resources. No credential marker or signed
  endpoint was found, and predictions, scores, hashes, and scientific provenance were unaffected,
  but the privacy claim was too strong. Immutable `v0` must not be rewritten or hidden.
- The first redaction rehearsal correctly verified all 291 files in the staging manifest, yet the
  uploaded evidence artifact actually contains 296 files because the release logger adds the local
  release manifest and four hash-bound int8-retention files. Cross-checking the independent W&B
  redownload receipt caught that five-file boundary mismatch before publication.
- The corrected local-only builder now reproduces all 296 public files and verifies 135,926,068
  source bytes before changing anything. Its deterministic derived view redacts 339 occurrences in
  44 files while rejecting credentials and preserving unmatched bytes. Receipt SHA-256 is
  `141203cff6491f9e23a3bbf62e4c6e0ae9c62bc0c4503e14e9b3702db5564a4e`; derived-tree
  SHA-256 is `c01de7b30c516bfb7c44625cf46ddd9bf2034ccd171f3af0f841383a94b73827`.
- This is not yet a remote publication claim. Any immutable redacted W&B version must be uploaded
  separately, downloaded into a fresh directory, rehashed over all 296 files plus its receipt, and
  documented as a nonauthoritative distribution view whose scientific source remains `v0`.

### 2026-08-05 addendum — a version preflight is not an atomic publication reservation

- The proposed redacted W&B publisher checked that `latest` was exact authoritative `v0` before
  calling `log_artifact`, then required the committed result to be `v1`. An adversarial fake-server
  run showed that a concurrent or partial publisher could still make W&B assign `v2`; the code
  would detect the mismatch only after the irreversible remote commit and would write no local
  success receipt.
- Any failure after the remote commit but before the local receipt has the same split-brain shape.
  The preregistration also required anonymous public-read proof, but the verifier used ambient SDK
  credentials, and it did not bind a clean source commit, fixed entity, or exact code hashes.
- Run `20260805-0250-candidate-v2-redacted-evidence-v1` is therefore closed without upload,
  download, network access, model access, CUDA, or JarvisLabs access. Its local redacted view remains
  useful build evidence only. Never invoke its `--upload` path or retry the run. A future public
  redacted view needs a new content-addressed destination and a durable precommit/terminal incident
  protocol; authoritative public evidence remains immutable `v0`.

### 2026-08-05 addendum — representation isolation is not module isolation

- The direct replay audit said it reconstructed direct Action IR “without importing PlanIR ...
  targets.” It correctly read no PlanIR target, compiler output, or screen row, but the Python module
  imported the seven Mobile tool schemas from `grounded_planir` and consumed the PlanIR run's
  one-row exclusion ledger as provenance. The wording could falsely imply complete module
  independence.
- The original result and ledger entry remain immutable. A claim-correction receipt narrows the
  statement without changing any row, target, hash, firewall fact, or authorization. The frozen
  seven-tool registry now lives in neutral `mobile_action_schemas.py`; the replay imports that
  registry directly, while the closed PlanIR code re-exports the same object for compatibility.
- Future audit claims must distinguish representation/compiler access, label/population access,
  and ordinary source-module dependencies. “No PlanIR targets or compiler” was supported; “no
  PlanIR module import” was not supported by the audited bytes.
