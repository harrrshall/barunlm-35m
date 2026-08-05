# BarunAction continual-recovery stage

This documents the single preregistered scientific recovery attempt from the exact
`20260803-1920-presto-stage-s17` checkpoint. It tested whether one train-only rehearsal pass could
repair the observed Mobile Actions catastrophic forgetting without selecting against either
development set.

The frozen recipe is `configs/continual_recovery_v1.json`, SHA-256
`b45f6efb28af44e0a2772b9239aff3f321a38765b964f1925e5bfc47c24fbf0d`. It presents every one of
the 7,937 pinned Mobile train rows exactly once and interleaves 2,646 unique PRESTO train rows, a
3:1 Mobile-to-replay ratio. PRESTO replay is correctness-blind, excludes exact prompt/target
development duplicates, and uses deterministic proportional joint-stratum sampling over policy
decision, contextuality, and the corrected phenomenon family. The four raw user-revision tags are
separate replay strata so none can disappear inside the aggregate revision family.

Training is full-parameter response-only SFT for 168 optimizer steps at a frozen `5e-5` peak
learning rate, effective batch 63, 16-step warmup, and cosine decay to ten percent. There is no
development loss, intermediate evaluation, early stopping, checkpoint choice, arm, retry, or
runtime hyperparameter override. Only the final checkpoint is evaluated.

## Completed result

An initial controller attempt, `20260803-2106-continual-recovery-s17`, failed its staged content-tree
check before CUDA inspection, repository tests, checkpoint hashing, data preparation, training, or
scoring. The fixed infrastructure retry therefore consumed the frozen scientific recipe exactly
once as run `20260803-2131-continual-recovery-infra-retry-s17`.

The retry presented 10,583 examples in one epoch—7,937 Mobile rows and 2,646 train-only PRESTO
replay rows—and completed 168 optimizer steps in 58.13 seconds. The final model SHA-256 is
`744b640828122ea9805131671c9792af9acfbd748e3199bacc513dcd8799c4c9`.

- Mobile passed: **642/756 = 0.849206** strict AST exact, 40 more correct rows than candidate-v2.
- PRESTO schema validity (14,254/14,288), ABSTAIN F1 (0.974351), conservative false CALL
  (102/10,407 = 0.009801), both hard-bucket gaps, and all four revision-tag reporting checks passed.
- PRESTO derived AST exact **failed** at **9,862/14,288 = 0.690230**. The frozen 0.70 threshold
  required 10,002 correct rows, so the checkpoint was 140 rows short. It must not be rounded to a
  pass.

The conjunctive gate therefore rejected the recovery checkpoint as research-only. The independent
outer selector also found no eligible rescue candidate and authorized no recipe retry. The immutable
[`result.json`](../experiments/runs/20260803-2131-continual-recovery-infra-retry-s17/essential/result.json)
has SHA-256 `9a909cc59b2b5bf01a38f1ab4dcbbb07fc6dac12c2aabd578e2a3cf3a9fe3cb8`;
the 29-file artifact manifest has SHA-256
`6baa0ab6a57c2921f570947c23d40e09e5d00114063a91e62eb38f76aa131f08`.

The completed H200 retry used project-owned instance 463686 for 768.14 seconds, estimated INR
80.71, downloaded all evidence, and verified that exact instance `Paused`. Exit code 2 is the
expected fail-closed scientific-gate result, not missing evidence. It read zero official Mobile or
PRESTO test rows and zero sensitive PRESTO member bytes.

This one-seed result uses reused public development data and derived PRESTO Action IR. It is not a
breakthrough, official/hidden-suite result, or broad/larger-model-superiority result.

The remaining sections preserve the preregistered method and launch contract as historical
reproducibility evidence.

## Keep rule

The terminal round evaluates the final checkpoint once on each frozen development population.
Both sides must pass:

- Mobile Actions strict AST exact must be at least 587/756 under the existing immutable regression
  protocol.
- Corrected full-family PRESTO development must reach at least 0.70 derived AST exact, 0.95 schema
  validity, 0.80 ABSTAIN F1, and at most 0.01 false CALL on ABSTAIN/CONFIRM rows. The
  no-phenomenon, revision, and disfluency buckets must all be represented, and neither hard bucket
  may trail no-phenomenon by more than 0.10. Metrics for all four raw revision tags must be present.

Failure on either side rejects this recipe. It is not eligible for a tuned retry under the same
preregistration. The 961 official Mobile rows and every official PRESTO test/combined member remain
opaque and are not accepted as runner inputs.

## Frozen launch inputs

Build a clean stage containing the repository code needed by the runner, the exact PRESTO
checkpoint directory, the three pinned Mobile data artifacts, and the two candidate-v2 Mobile
reference score artifacts. Preserve the paths encoded in the preregistration or pass the optional
relocated Mobile paths.

The runner additionally requires two distinct provenance records at the root of that clean stage:

- `source-snapshot.json`, schema `barun-continual-recovery-source-snapshot-v1`;
- `attempt-preregistration.json`, schema `barun-continual-recovery-attempt-v1`.

They must be constructed in that order to avoid a hash cycle. First compute the staged content
tree while excluding the two not-yet-authoritative provenance filenames. Write the snapshot with
that tree receipt and the exact runner/config hashes. Immediately before instance creation, save
the complete read-only JarvisLabs inventory as the protected-ID denylist. After the controller
creates one new `barun-*` instance, it must late-bind that exact returned ID and the complete
pre-create denylist into the attempt, hash the completed snapshot into the same attempt, and finish
the attempt before uploading the stage or running remote code. Never edit the staged tree,
snapshot, or finalized attempt after that point.

### Strict attempt schema

The attempt object accepts exactly these top-level fields:

```json
{
  "schema_version": "barun-continual-recovery-attempt-v1",
  "run_id": "<same exact recovery run ID passed to the CLI>",
  "status": "frozen_before_remote_data_prep_training_or_scoring",
  "registered_at": "<timezone-aware ISO-8601 timestamp>",
  "created_before_remote_launch_or_data_prep": true,
  "hypothesis": "<nonempty frozen hypothesis>",
  "decision": "<nonempty frozen conjunctive keep decision>",
  "recovery_config": {
    "path": "configs/continual_recovery_v1.json",
    "sha256": "b45f6efb28af44e0a2772b9239aff3f321a38765b964f1925e5bfc47c24fbf0d"
  },
  "shared_selector": {
    "path": "configs/parallel_rescue_selection_v1.json",
    "sha256": "68bb6132a486dfbd6c9ad478db9c87ae9f2350bfc1d302193169fd7f5c290130"
  },
  "source_snapshot": {
    "path": "source-snapshot.json",
    "sha256": "<SHA-256 of the completed source-snapshot.json>"
  },
  "runner": {
    "path": "scripts/run_continual_recovery.py",
    "sha256": "<SHA-256 after all recovery-runner changes are complete>"
  },
  "input_checkpoint": "<checkpoint-chain object defined below>",
  "prelaunch_inventory": {
    "captured_before_project_instance_creation": true,
    "protected_machine_ids": "__SAFE_RUN_PREEXISTING_IDS__",
    "project_machine_id": "__SAFE_RUN_MACHINE_ID__",
    "fresh_project_instance": true
  },
  "evaluation": {
    "terminal_rounds": 1,
    "mobile_development_rows": 756,
    "presto_development_rows": 14288,
    "official_mobile_evaluation_rows_read": 0,
    "official_presto_test_rows_read": 0
  },
  "claim_limits": ["<at least one nonempty claim limit>"]
}
```

The angle-bracket values above and the `input_checkpoint` string are explanatory construction
placeholders. The two `__SAFE_RUN_*__` strings, however, are the exact controller placeholders.
Pass `--bind-attempt-inventory attempt-preregistration.json` to `safe_run.py`; after fresh creation
and before target upload, it atomically replaces `__SAFE_RUN_PREEXISTING_IDS__` with the complete
sorted pre-create ID array and `__SAFE_RUN_MACHINE_ID__` with its exact owned machine ID. It records
the before/after attempt hashes and the binding event in `jarvis.json`. The remote runner accepts
only the bound form: `protected_machine_ids` must be a nonempty, duplicate-free integer array, must
contain Kroda/“cruda” ID 463058 and every other ID observed before creation, and must not contain
the new project ID. `project_machine_id` must be the exact positive integer passed as
`--jarvis-machine-id`. A missing timing assertion, a non-fresh instance assertion, a protected CLI
ID, or an attempt/CLI ID mismatch fails before CUDA inspection, repository tests, checkpoint
hashing, or data preparation.

The attempt also binds the cross-method promotion policy at
`configs/parallel_rescue_selection_v1.json`, schema
`barun-parallel-rescue-selection-v1`, to SHA-256
`68bb6132a486dfbd6c9ad478db9c87ae9f2350bfc1d302193169fd7f5c290130`. The runner requires that
canonical staged path, verifies its bytes and schema directly, and records the identity in the
provenance receipt. The source snapshot also pins it explicitly in `frozen_files`, and the selector
is a regular staged source file, so the content tree independently covers the same bytes.

`input_checkpoint` is an object, not the placeholder string shown above. The identical object must
appear in the source snapshot:

```json
{
  "source_run_id": "20260803-1920-presto-stage-s17",
  "parent_run_id": "20260803-1845-mobile-followup-retry-s17",
  "path": "experiments/runs/20260803-1920-presto-stage-s17/essential/checkpoint",
  "checkpoint_manifest_sha256": "90e50f316456947bbee10b710b6f828f1ca8955ed0e43998b15adb0e09f46e23",
  "file_sha256": {
    "barun_config.json": "9b3a1d71baa95a198744d250f9629231738d942570b8685c44307fd83dd33565",
    "model.safetensors": "83c78952e719456574d2bb808fb486d0320519dd4f5df3c67736ae17931ee357",
    "tokenizer.json": "70ded9605fccd09c2340ca7e225361eab0ae8b4dbbb0d6e26343ab5183979db6"
  },
  "parent_file_sha256": {
    "barun_config.json": "9b3a1d71baa95a198744d250f9629231738d942570b8685c44307fd83dd33565",
    "model.safetensors": "fdb95ccf58a095e0d321be998924318b35ee59a334f6dd97d8726d2cf80021d3",
    "tokenizer.json": "70ded9605fccd09c2340ca7e225361eab0ae8b4dbbb0d6e26343ab5183979db6"
  }
}
```

The runner verifies this against the actual release checkpoint manifest, including source run,
candidate-v2 parent hashes, and selected source step 947, before it hashes or downloads any data.

### Strict snapshot schema and tree accounting

The snapshot accepts exactly `schema_version`, `run_id`,
`created_before_remote_launch_or_data_prep`, `stage_root`, `content_tree`, `frozen_files`,
`input_checkpoint`, and `staging_policy`. `stage_root` must be `.` and the snapshot itself must be
at the staged repository root. `frozen_files` contains exactly:

```json
{
  "configs/continual_recovery_v1.json": "b45f6efb28af44e0a2772b9239aff3f321a38765b964f1925e5bfc47c24fbf0d",
  "configs/parallel_rescue_selection_v1.json": "68bb6132a486dfbd6c9ad478db9c87ae9f2350bfc1d302193169fd7f5c290130",
  "scripts/run_continual_recovery.py": "<final runner SHA-256>"
}
```

`content_tree` uses schema `barun-staged-content-tree-v1`. Hash every accounted regular file in
relative POSIX path order, then hash the concatenation of UTF-8 lines:

```text
<file SHA-256><two spaces><relative POSIX path><newline>
```

Record the resulting `sha256`, `file_count`, and `content_bytes`. The hash-method string must be
exactly `SHA-256 of UTF-8 lines '<file SHA-256><two spaces><relative POSIX
path><newline>' in relative-path sort order`.

The remaining exact `content_tree` fields are:

```json
{
  "excluded_exact_paths": [
    "attempt-preregistration.json",
    "source-snapshot.json"
  ],
  "excluded_directory_names": [
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "venv",
    "wandb"
  ],
  "excluded_directory_suffixes": [".egg-info"],
  "excluded_file_names": [".DS_Store"],
  "excluded_file_suffixes": [".pyc", ".pyo"]
}
```

These exclusions keep environment setup, bytecode, and tracking caches out of tree accounting;
they cannot be expanded in a launch record. Symlinks and credential-like regular files such as
`.env*`, private keys, credential files, and secret files are rejected rather than excluded. The
snapshot's strict `staging_policy` sets all five booleans to false:
`credentials_present`, `cache_directories_present_before_launch`,
`symlinks_present_before_launch`, `official_mobile_evaluation_present`, and
`official_presto_test_present`.

The remote runner recomputes the complete tree under exactly these rules before PRESTO download,
training, or scoring. It also requires the CLI checkpoint to live inside that tree. Any added,
removed, renamed, or changed accounted file fails closed.

After creating both records, run the focused preflight before provisioning:

```bash
uv run ruff check \
  src/barunlm/recovery scripts/run_continual_recovery.py tests/test_continual_recovery.py
uv run pytest -q \
  tests/test_continual_recovery.py tests/test_mobile_regression.py tests/test_presto_stage.py
```

The reviewed remote-script arguments are:

```text
--run-id <YYYYMMDD-HHMM-continual-recovery-s17>
--attempt-preregistration attempt-preregistration.json
--source-snapshot source-snapshot.json
--input-checkpoint experiments/runs/20260803-1920-presto-stage-s17/essential/checkpoint
--mobile-train experiments/runs/20260803-1810-mobile-blind-s17/remote/data/train.jsonl
--mobile-dev experiments/runs/20260803-1810-mobile-blind-s17/remote/data/dev.jsonl
--mobile-audit experiments/runs/20260803-1810-mobile-blind-s17/remote/data/audit.json
--artifact-root /home/barun-artifacts
--hourly-cost-inr 378.27
```

The exact-ID controller must append `--jarvis-machine-id` itself. A launch template is:

```bash
uv run python infra/jarvis/safe_run.py run \
  --name barun-continual-recovery-s17 \
  --record experiments/runs/<run-id>/jarvis.json \
  --target /absolute/path/to/clean-recovery-stage \
  --script scripts/run_continual_recovery.py \
  --gpu H200 --num-gpus 1 --storage 100 --region IN2 \
  --requirements requirements/continual-recovery.txt \
  --poll-seconds 60 --max-runtime-minutes 25 \
  --artifact /home/barun-artifacts/<run-id>/export/essential \
  --artifact-dest experiments/runs/<run-id>/essential \
  --artifact-recursive \
  --bind-attempt-inventory attempt-preregistration.json \
  --append-jarvis-machine-id -- \
  --run-id <run-id> \
  --attempt-preregistration attempt-preregistration.json \
  --source-snapshot source-snapshot.json \
  --artifact-root /home/barun-artifacts \
  --hourly-cost-inr 378.27
```

This command is the frozen launch template retained for auditability; the result and lifecycle
records cited above are the evidence that the retry occurred. Inventory was captured immediately
before provisioning. Before `jl run` uploaded the clean stage, the controller
late-bind the new exact ID and the complete pre-create protected-ID list into the target-relative
`attempt-preregistration.json` via `--bind-attempt-inventory`;
`--append-jarvis-machine-id` must append that same ID to the remote CLI. The controller may create
and later pause only its exact new `barun-*` ID; Kroda (463058) and every other pre-existing or
unrecognized resource remain protected.

## Runtime and artifact record

The completed retry took 768.14 seconds and estimated INR 80.71. The preregistered budget was
11--18 minutes on one H200: roughly 2--4 minutes for pinned-data
preparation/tests, 2--3 minutes for the 168-step recovery epoch, under one minute for Mobile, and
6--8 minutes for full PRESTO generation and scoring. At the previously observed INR 378.27/hour,
that is approximately INR 69.35--113.48. The controller's 25-minute watchdog corresponds to INR
157.61 and is a ceiling, not a target.

The compact `export/essential` download contains the exact attempt preregistration, source snapshot,
provenance-validation receipt (including the shared-selector and inventory identities), final
checkpoint/tokenizer, training summary and metrics, exact view audit, environment, terminal sample
predictions/scores, result, and an artifact hash manifest. The final manifest therefore covers both
prelaunch evidence files. The optimizer state and combined reconstruction manifest stay in the
project-owned remote export and are referenced by exact SHA-256; they are intentionally excluded
from the compact download.

The failed result is one-seed public-development recovery evidence only. It is not an official-test
result, a native PRESTO semantic-parse result, a hidden safety result, a larger-model comparison, or
a usable release by itself.
