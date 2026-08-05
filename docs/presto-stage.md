# BarunAction-35M PRESTO post-training stage

This stage takes an exact local BarunAction-35M checkpoint, prepares only PRESTO English official
train/development data, scores the incoming checkpoint, runs one full-parameter response-only SFT
recipe, and scores the selected checkpoint. It never executes a predicted action.

## Completed result

Run `20260803-1920-presto-stage-s17` completed the one frozen recipe. Its immutable
[`result.json`](../experiments/runs/20260803-1920-presto-stage-s17/essential/result.json) has
SHA-256 `b40d41cfb3d93cfab2af285e8780626402c6ad600e1698623dbc0f5ca6e162f2`; the complete
`artifact-sha256.json` has SHA-256
`6f7ae0b10337cf89ffe9cbecbc75ac32f69b96630a605a538cbfbf682b03bd7a`.

- One epoch selected step 947 after 947 optimizer steps. Training took 628.18 seconds and moved
  development loss from 3.10907 to 0.019323. The output model SHA-256 is
  `83c78952e719456574d2bb808fb486d0320519dd4f5df3c67736ae17931ee357`.
- On 14,288 PRESTO development rows, derived Action IR AST exact rose from 0 to
  **10,620/14,288 = 0.743281**. Schema validity was 14,285/14,288, decision accuracy was
  14,030/14,288, ABSTAIN F1 was 0.979565, and conservative false CALL was
  89/10,407 = 0.008552.
- The original preregistered gate **failed** because the v1 literal-string taxonomy represented no
  revision bucket. A later read-only, post-hoc primary-source correction classified the four raw
  revision tags and measured 3,050/4,140 = 0.736715 revision AST exact; its diagnostic gate passes,
  but it does not rewrite the original failed gate or repair the zero-row revision replay used in
  training.
- A frozen Mobile regression run then scored this PRESTO checkpoint **0/756**, regressing all 602
  candidate-v2-correct rows. The broader checkpoint was therefore rejected rather than promoted.

The H200 run used project-owned instance 463622 for 1,732.65 seconds, downloaded the evidence, and
verified that exact instance `Paused`. The test firewall recorded all 194,118 PRESTO official-test
rows as opaque and unparsed with zero sensitive-member bytes read.

This is a useful public-development capability result and an equally important forgetting result,
not a breakthrough. PRESTO development is reused, has known train-template overlap, and is scored
through derived Action IR rather than native PRESTO semantic parsing. No official test or hidden
human suite was scored, and no broad or larger-model superiority follows.

## Frozen decision

The complete recipe is [`configs/presto_stage_v1.json`](../configs/presto_stage_v1.json), SHA-256
`62c1222e8b9d348fa6b9aa9582001a5a06675f541e630a08b0d35f0efcd53cd9`. It is one development
trial, not a sweep. Every official-train row that survives the frozen exact-development-duplicate
exclusion appears once. A deterministic train-only replay adds at most 3,200 rows from each of five
useful categories: ABSTAIN, CONFIRM, revision, disfluency, and contextual. Category overlap is
intentional and audited. No development row is used to select the replay. Before focus selection,
the stage removes every official-train row whose exact
prompt-plus-target identity appears in development. This correctness-blind exclusion is the only
development access in training-view construction; it prevents the generic trainer's duplicate
firewall from failing after paid compute has started and records every excluded train ID.

The executed input was the promoted Mobile development winner:

- Directory:
  `experiments/runs/20260803-1845-mobile-followup-retry-s17/essential/arms/batch63/checkpoint`.
- `model.safetensors`:
  `fdb95ccf58a095e0d321be998924318b35ee59a334f6dd97d8726d2cf80021d3`.
- `barun_config.json`:
  `9b3a1d71baa95a198744d250f9629231738d942570b8685c44307fd83dd33565`.
- `tokenizer.json`:
  `70ded9605fccd09c2340ca7e225361eab0ae8b4dbbb0d6e26343ab5183979db6`.

The runner remains configurable: another input directory is accepted only with all three explicit
hashes. PRESTO preparation additionally requires the canonical tokenizer hash frozen in the recipe.

## Test-data firewall

The whole pinned archive is byte-hashed, and its central directory is checked. Only
`presto_train.jsonl` and `presto_dev.jsonl` are opened. `presto_test.jsonl`, the redundant combined
member, and every `test_partitions/` member remain unopened, unparsed, and unmaterialized. The stage
requires member-level runtime open and byte counts of zero and aborts before training if that proof
is absent. Its exported data hashes must contain exactly `train`, `dev`, and `audit`.

The incoming and post-SFT checkpoints are evaluated with unconstrained deterministic greedy
decoding. Outputs receive strict JSON, Action IR, policy-decision, and recursive ordered-slot
validation. Artifacts include every raw prediction, every sample score, exact raw PRESTO phenomenon
buckets, derived no-phenomenon/revision/disfluency/code-switching buckets, a prompt-derived
contextual bucket, ABSTAIN F1, CONFIRM exact match, and false CALL on safety-gated rows.
Checkpoint selection follows the trainer's frozen rule: retain the latest checkpoint whose
development response-only loss improves the previously selected loss by more than 0.001, and use
the final checkpoint only when none qualifies.
The false-CALL metric is conservative: any parse-valid top-level `CALL` on an ABSTAIN or CONFIRM
gold row counts, even when the proposed tool or recursive payload is schema-invalid and would be
blocked by the deterministic policy layer.

## Frozen remote arguments

The completed run used ID `20260803-1920-presto-stage-s17`. The source staging directory contained
the current repository source plus the exact checkpoint path above and excluded credentials, caches,
unrelated run artifacts, and optimizer states. The only launch-specific value appended by the safe
controller is the newly captured JarvisLabs machine ID.

Use the explicit staging target `/tmp/barun-presto-stage-1920-source`; do not allow the controller's
default `--target .` to upload this entire research workspace. Build that target from `src/`,
`tests/`, `scripts/`, `configs/`, `examples/`, `infra/`, `pyproject.toml`, `uv.lock`,
`barun_config.json`, `README.md`, `LICENSE`, and only the promoted checkpoint at its exact relative
path under `experiments/`. Preserve a source-snapshot hash manifest. Do not copy any other
`experiments/` content, `wandb/`, virtual environments, caches, optimizer files, `.netrc`, SSH
material, or credentials. Keep the run record and downloaded artifact destination in the original
workspace, outside this staging target.

```text
--run-id
20260803-1920-presto-stage-s17
--input-checkpoint-dir
experiments/runs/20260803-1845-mobile-followup-retry-s17/essential/arms/batch63/checkpoint
--input-model-sha256
fdb95ccf58a095e0d321be998924318b35ee59a334f6dd97d8726d2cf80021d3
--input-config-sha256
9b3a1d71baa95a198744d250f9629231738d942570b8685c44307fd83dd33565
--input-tokenizer-sha256
70ded9605fccd09c2340ca7e225361eab0ae8b4dbbb0d6e26343ab5183979db6
--artifact-root
/root/barun-artifacts
--generation-batch-size
128
--hourly-cost
378.27
```

The safe-controller launch parameters were:

```text
name: barun-presto-stage-1920-s17
target: /tmp/barun-presto-stage-1920-source
record: experiments/runs/20260803-1920-presto-stage-s17/jarvis.json
script: scripts/run_presto_stage.py
gpu: H200
num-gpus: 1
region: IN2
storage: 100
poll-seconds: 60
max-runtime-minutes: 360
append-jarvis-machine-id: true
artifact: /root/barun-artifacts/20260803-1920-presto-stage-s17/export/essential
artifact-dest: experiments/runs/20260803-1920-presto-stage-s17/essential
artifact-recursive: true
```

The controller must perform its normal read-only inventory first, create only this fresh
`barun-*` instance, operate on its exact captured ID, download the essential bundle, and pause and
verify that exact ID. It must not use Kroda or any other existing instance.

## Essential bundle and limitations

The compact bundle includes the selected inference checkpoint, checkpoint lineage and hashes,
recipe and effective preregistration, environment and exact JarvisLabs ID, progressive phase log,
repository-test log, data/firewall and focus-view audits, trainer curves and manifests, input/post
sample predictions and scores, result/gate decision, and a full bundle hash manifest. Optimizer
state is explicitly excluded.

The completed stage is research evidence, not the promoted broader candidate. PRESTO development
has known train-template overlap; the derived Action IR score is not native PRESTO semantic-parse
exact match; only one seed and one recipe ran; and PRESTO supplies no CLARIFY supervision. The
subsequent Mobile regression failed at 0/756, and the first matched larger baseline was only a
single-seed public-development diagnostic. Neither event supplies the official-test, hidden-suite,
cross-seed, or deployment evidence required for a breakthrough claim.
