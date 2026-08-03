# BarunAction-35M interpolation rescue

This completed three-arm checkpoint interpolation tested the exact candidate-v2 checkpoint against
its direct PRESTO-trained continuation. The immutable protocol is
`configs/interpolation_rescue_v1.json`, SHA-256
`1e2f005f61bf8f8cb62c01928c23266023e2e20760321832f80bc4dbe3c9ee3c`.

The candidate-v2 endpoint model SHA-256 is
`fdb95ccf58a095e0d321be998924318b35ee59a334f6dd97d8726d2cf80021d3`. The PRESTO endpoint
model SHA-256 is
`83c78952e719456574d2bb808fb486d0320519dd4f5df3c67736ae17931ee357`. The PRESTO release
manifest records the complete candidate-v2 file-hash map as its input, so this is interpolation
along one observed training trajectory rather than an unsupported model merge.

## Completed result

Run `20260803-2102-interpolation-rescue-s17` evaluated all three frozen arms once on both reused
public-development populations. The immutable result is
[`interpolation-result.json`](../experiments/runs/20260803-2102-interpolation-rescue-s17/essential/interpolation-result.json),
SHA-256 `a98e06d9e1f92ce30576c86ec8ecf0f35bcfbb4f26a87d3d70858fd1fbb11f97`.

| Arm | Mobile AST exact | PRESTO derived AST exact | PRESTO schema valid | ABSTAIN F1 | False CALL | Joint gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| alpha 1/4 | 598/756 | 0/14,288 | 0/14,288 | 0 | 8,264/10,407 | fail |
| alpha 1/2 | 452/756 | 961/14,288 | 3,316/14,288 | 0.393845 | 7,466/10,407 | fail |
| alpha 3/4 | 8/756 | 9,447/14,288 | 14,256/14,288 | 0.965724 | 161/10,407 | fail |

There was no joint passer: interpolation was rejected, no additional alpha was authorized, and
candidate-v2 remained the fallback. The independent outer selector later reproduced the same
decision in
[`selection-receipt.json`](../experiments/runs/20260803-2203-parallel-rescue-selection-s17/selection-receipt.json),
SHA-256 `e021eff97cf4816ac299dc55cf6d2c356b07754639fe136540acef5987897582`.
All three arms generated every Mobile and PRESTO row with no recorded generation failure. The
completed H200 run used project-owned instance 463674, cost an estimated INR 161.74 over 1,539.26
seconds, downloaded hash-verified evidence, and verified that exact instance `Paused`.

These results show a steep same-trajectory capability trade-off, not a breakthrough. Both
populations are reused public development data; PRESTO uses derived Action IR rather than native
semantic-parse exact match. No official Mobile or PRESTO test, hidden human suite, or sealed safety
suite was scored, and the result establishes no broad or larger-model superiority.

The remaining sections preserve the preregistered method and launch contract as historical
reproducibility evidence.

## Frozen experiment

For alpha equal to 1/4, 1/2, and 3/4, the runner computes

    theta_alpha = ((denominator - numerator) * theta_candidate_v2
                   + numerator * theta_presto) / denominator

Every stored float32 tensor is accumulated on CPU in float64 and rounded once back to float32.
Both endpoint files must have the same 137 tensor keys, shapes, dtypes, 35,072,768 stored
parameters, and finite values. Non-floating tensors, if any, must be bitwise equal. The tied
`lm_head.weight` to `embedding.weight` safetensors alias must be identical at both endpoints,
preserved in the output metadata, and share storage after the output checkpoint is loaded.

The alpha set is intentionally small. The endpoints already show opposite behavior, while three
interior dyadic points provide a low, middle, and high movement along the same trajectory without
turning development data into an adaptive line search. There are exactly three checkpoint trials,
one Mobile generation and one PRESTO generation for every arm, no endpoint regeneration, no early
elimination, and no follow-up alpha.

Each arm must satisfy all of the following:

- Mobile Actions AST exact match at least 587/756;
- PRESTO derived Action IR AST exact at least 0.70;
- PRESTO schema validity at least 0.95;
- PRESTO ABSTAIN F1 at least 0.80;
- PRESTO conservative false-CALL rate at most 0.01;
- the no-phenomenon, revision, and disfluency buckets all represented with frozen membership;
- no-phenomenon minus revision and disfluency exact-match gaps each at most 0.10;
- all four corrected revision labels present with frozen counts: cancel-action 454,
  correct-action 230, correct-argument 389, and within-turn-correction 3,067.

Among joint passers, selection first maximizes PRESTO AST-exact numerator, then Mobile AST-exact
numerator, then chooses the lower alpha. The implementation independently recomputes every joint
gate before selection. Unlisted metrics cannot break a tie. If no arm passes, interpolation is
rejected, candidate-v2 remains the fallback, and the protocol authorizes no extra alpha.

This experiment consumes three new joint development-selection trials. The Mobile and PRESTO
official tests remain inaccessible: the runner accepts no official-test path, and PRESTO's test
and combined archive members must remain unopened. These reused development sets cannot support a
blind breakthrough claim.

## Frozen launch provenance

The runner accepts no unregistered source tree. Two evidence files are mandatory at these exact
paths in the staged repository root:

- `interpolation-source-snapshot.json` using schema
  `barun-interpolation-source-snapshot-v1`;
- `interpolation-attempt-preregistration.json` using schema
  `barun-interpolation-attempt-preregistration-v1`.

Build them without a circular hash. First create the credential-free staged tree without either
evidence file and compute its content-tree receipt. Then create the source snapshot. Hash that
complete snapshot and place its SHA-256 in the attempt preregistration. The staged attempt is a
pre-create template: set `protected_machine_ids` to the exact string
`__SAFE_RUN_PREEXISTING_IDS__` and `project_machine_id` to the exact string
`__SAFE_RUN_MACHINE_ID__`. `safe_run.py` replaces only those two placeholders atomically after it
captures the read-only inventory and creates a fresh project instance, but before it uploads the
stage. A raw `jl run` must not be used with the unbound template.

The source snapshot has exactly these top-level fields: `schema_version`, `run_id`, `created_at`,
`created_before_remote_launch_or_model_scoring`, `git_commit`, `git_dirty`, `content_tree`,
`frozen_files`, and `staging_policy`. Its `content_tree` is the object printed by:

    uv run python - <<'PY'
    import json
    from scripts.run_interpolation_rescue import compute_interpolation_content_tree
    print(json.dumps(compute_interpolation_content_tree("."), indent=2, sort_keys=True))
    PY

The content-tree algorithm hashes lines of `<file SHA-256><two spaces><relative POSIX
path><newline>` in relative-path order. Its canonical exclusions are only the two provenance files,
Git metadata, virtual environments, runtime caches, W&B output, `node_modules`, `jl-runs`, Python
bytecode and egg-info, and `.DS_Store`. The runner rejects any changed exclusion list, symlink,
credential-like included path, or recognized credential token marker. Thus newly created runtime
caches cannot perturb the source identity and cannot enter its accounting.

Both the source snapshot and attempt preregistration require this exact `frozen_files` map. The
shared selector binds the three interpolation trials to the already-frozen four-trial outer
comparison; it does not alter the interpolation protocol:

    {
      "configs/interpolation_rescue_v1.json": "1e2f005f61bf8f8cb62c01928c23266023e2e20760321832f80bc4dbe3c9ee3c",
      "configs/parallel_rescue_selection_v1.json": "68bb6132a486dfbd6c9ad478db9c87ae9f2350bfc1d302193169fd7f5c290130",
      "scripts/run_interpolation_rescue.py": "64094dbc221b0b241686e87ad39c3baccd5266d093971a2657953673787fe9e5"
    }

The source snapshot's `staging_policy` must be exactly:

    {
      "credentials_included": false,
      "caches_included": false,
      "official_test_artifacts_included": false,
      "optimizer_state_included": false
    }

The attempt preregistration has exactly these top-level fields: `schema_version`, `run_id`,
`status`, `created_at`, `created_before_remote_model_or_data_scoring`, `protocol`,
`source_snapshot`, `frozen_files`, and `prelaunch_inventory`. Its fixed values are:

    {
      "status": "frozen_before_remote_model_or_data_scoring",
      "created_before_remote_model_or_data_scoring": true,
      "protocol": {
        "path": "configs/interpolation_rescue_v1.json",
        "sha256": "1e2f005f61bf8f8cb62c01928c23266023e2e20760321832f80bc4dbe3c9ee3c"
      },
      "source_snapshot": {
        "path": "interpolation-source-snapshot.json",
        "sha256": "<exact-source-snapshot-sha256>"
      }
    }

Its `prelaunch_inventory` must state that inventory was captured before project-instance creation,
contain every pre-existing machine ID as `protected_machine_ids`, bind `project_machine_id` to the
fresh machine passed on the CLI, and set `fresh_project_instance` true. The denylist must contain
463058 and must not contain the fresh project ID.

Before it queries CUDA, runs repository tests, hashes a model checkpoint, prepares PRESTO, or
scores any example, the runner verifies both strict schemas, both run IDs, protocol identity, the
attempt-to-snapshot hash, runner, interpolation-protocol and shared-selector hashes, inventory
binding, staging policy, exclusion contract, and a fresh recomputation of file count, bytes, and
content-tree SHA-256. It copies both evidence files plus
`interpolation-provenance-verification.json` into the essential bundle, so the final artifact
manifest covers the full chain. The per-arm internal interpolation winner remains diagnostic; the
shared selector at `configs/parallel_rescue_selection_v1.json` controls final promotion across the
parallel rescue methods.

## Frozen launch inputs

Use one newly created H200 instance and run all arms sequentially. The lifecycle controller takes
the read-only inventory before creation, records every listed ID as protected, binds the attempt
template, watches the run, downloads the requested evidence, and pauses only the exact project ID
with post-pause verification. Machine 463058, Kroda (also dictated as “cruda”), is explicitly
protected and the runner rejects that ID. Do not use any existing instance.

The dynamic values are a new immutable run ID, the fresh machine ID captured by the controller,
the current H200 hourly price, and the two hash-bound provenance receipts above. All scientific
inputs are fixed. Launch from the source repository with the clean staged directory as `--target`.
The `--bind-attempt-inventory interpolation-attempt-preregistration.json` option is mandatory;
without it, the two safe-run placeholders remain deliberately invalid:

    uv run python infra/jarvis/safe_run.py run \
      --name barun-interpolation-s17 \
      --record experiments/runs/<run-id>/jarvis-safe-run.json \
      --target <clean-staged-directory> \
      --script scripts/run_interpolation_rescue.py \
      --gpu H200 \
      --storage 40 \
      --max-runtime-minutes 30 \
      --append-jarvis-machine-id \
      --bind-attempt-inventory interpolation-attempt-preregistration.json \
      --artifact /home/barun-artifacts/<run-id>/export/essential \
      --artifact-dest experiments/runs/<run-id>/downloaded-essential \
      --artifact-recursive \
      -- \
      --run-id <YYYYMMDD-HHMM-interpolation-rescue-s17> \
      --attempt-preregistration interpolation-attempt-preregistration.json \
      --source-snapshot interpolation-source-snapshot.json \
      --candidate-checkpoint-dir experiments/runs/20260803-1845-mobile-followup-retry-s17/essential/arms/batch63/checkpoint \
      --presto-checkpoint-dir experiments/runs/20260803-1920-presto-stage-s17/essential/checkpoint \
      --artifact-root /home/barun-artifacts \
      --hourly-cost <current-h200-inr-per-hour>

The immutable run ID in the command must exactly match the run ID already frozen in both staged
provenance files. `--append-jarvis-machine-id` supplies the same freshly captured ID to the runner
that `--bind-attempt-inventory` writes into the attempt. Keep the controller record and artifact
destination outside the staged directory so neither can change its content-tree identity.

The optional `--presto-archive` input may be used only for an already verified copy with SHA-256
`1fc671692cceb31fbda17e351e47f2cc52ee8779042f92dc26674cc0cca2167f`; otherwise the runner
downloads and verifies that pinned archive. Monitor the managed run by its exact run ID, download
`/home/barun-artifacts/<run-id>/export/essential`, verify the downloaded artifact manifest, and
pause only the exact newly created machine ID. Query that same ID afterward and record paused-state
proof. Never issue a bulk lifecycle command.

The result bundle contains all three interpolated checkpoints, sample-level Mobile and PRESTO
predictions and scores, per-arm joint gates, the fixed selection receipt, environment evidence,
the verified attempt and source-snapshot receipts, the staged PRESTO development manifest and audit,
and an immutable file-hash manifest. It contains no optimizer state or official-test artifact.

## Runtime and cost

The completed run measured 1,539.26 seconds and estimated INR 161.74. The following figures are
the prelaunch budget retained for auditability.

Matched H200 evidence gives 20.43 seconds for one 756-row Mobile pass and 395.43 seconds for the
successful PRESTO endpoint's 14,288-row pass. The less compatible candidate-v2 PRESTO pass took
561.67 seconds because it generated many long failures. With one-time data preparation and focused
tests, the protocol budgeted about 24 minutes expected and 30 minutes conservatively on one H200.
At the recorded reference price of INR 378.27/hour, that is about INR 151 expected and INR 189 for
the 30-minute budget. The completed result above supersedes that estimate with launch-time cost and
measured elapsed time.

The frozen local verification command before launch was:

    uv run pytest -q tests/test_interpolation_rescue.py tests/test_mobile_regression.py tests/test_presto.py
    uv run ruff check src/barunlm/evaluation/interpolation_rescue.py scripts/run_interpolation_rescue.py tests/test_interpolation_rescue.py
