# JarvisLabs runbook

This directory holds the standing remote-compute context for BarunLM-35M and its post-trained
BarunAction-35M model so it does not need to be supplied again. It deliberately contains no API
key, network address, or account identity.

Authentication was configured with `jl` 0.2.17 outside the repository. The CLI's configuration
file is owned by the user and permission-restricted. Never inspect or serialize the token as part
of an experiment. Official references: [CLI documentation](https://docs.jarvislabs.ai/cli/) and
[SDK documentation](https://docs.jarvislabs.ai/sdk/).

## Safety invariant

The private `protected-resources.json` file is the minimum denylist captured before this project
provisioned any compute. It is intentionally ignored and absent from the public source release.
Keep it beside `safe_run.py`, or set `BARUN_JARVIS_PROTECTED_RESOURCES` to an external JSON file
with a top-level `resources` list whose entries contain positive integer `machine_id` values. The
controller fails closed when this file is missing, empty, or malformed. Every ID in it is
permanently out of scope, regardless of its later name or state. A new resource that is not
recorded as created by a Barun experiment is also protected by default.

The running CPU instance Kroda, ID 463058, must remain uninterrupted. No project job may execute
on it. Do not start a paused pre-existing instance to save setup time.

## Month-Boundary Counterfactual v3 completed record

BarunAction-35M Month-Boundary Counterfactual v3 completed on fresh exact H200 ID **463802**. Its
managed run `r_fe346e61` exited zero after all nine screening fits, evidence and logs were collected,
and the exact ID was pause-verified. The scientific selection gate rejected the recipe, so no
checkpoint was promoted or preserved and no confirmation, full-refit, reused-756, or official-961
evaluation was performed. ID 463802 is permanently protected and must never be resumed or reused.
V2 exact ID 463793 remains paused and protected as a distinct zero-signal pre-CUDA failure.

The successful v3 launch used the frozen provider contract
`--template axolotl --python-implementation CPython --python-version 3.11.10` together with one
non-spot H200 in IN2. These controls remain the reference pattern for a later preregistered run,
but there is no active training launch. Any next experiment requires a new hypothesis, config,
source snapshot, output directory, and fresh instance; do not treat v3 as retryable.

When `--bind-attempt-inventory` is used, the controller validates the still-unbound attempt JSON
before the pre-creation inventory or any instance creation. Its `prelaunch_inventory` fields must
still contain the two `__SAFE_RUN_*__` placeholders, and its compute object must exactly match the
requested JarvisLabs provider, Axolotl template, CPython version, GPU, region, spot, and time
contract. A mismatch stops before any resource is created.

After a fresh `barun-*` ID is created and becomes SSH-ready, the controller first attests the
requested hardware and the exact `axolotl` template from that ID's live record. It then runs a
small, standard-library-only identity command through `jl exec <exact-id>` and requires `python3`
to report exactly `CPython` and `3.11.10`. The runtime receipt is persisted before the controller
binds the attempt's inventory placeholders or invokes the attached `jl run` upload. A missing,
malformed, or mismatched template/runtime identity fails closed: no target is uploaded, and only
the newly recorded exact ID is paused and queried for paused-state proof. Existing or protected
resources remain out of scope even when an attestation fails.

## Remote run sequence

1. Run `jl list --json` read-only and filter it to ID, name, state, GPU type, and GPU count. Compare
   every ID with this denylist and the experiment's own resource record.
2. Use `jl gpus --json` to select current capacity. The default pilot choice is one L4; use an
   A100-80GB only when measured throughput or teacher/baseline memory requires it. Parallel
   `barun-*` instances are appropriate for independent preregistered hyperparameter/data ablations,
   with early stopping; do not duplicate identical work merely because capacity is available.
3. Create one instance with an unambiguous `barun-<run-id>` name using `jl create ... --yes --json`.
   Require an explicit positive integer `machine_id` in the response and persist that ID plus the
   non-sensitive creation metadata immediately, before any SSH or run command. If creation errors
   or returns malformed JSON after creating a resource, recovery may use the inventory only when
   exactly one new, nonprotected resource has the exact requested name. A malformed successful
   response is recovered for cleanup only; it never authorizes a run. Never guess among candidates.
4. Confirm that the captured ID is absent from `protected-resources.json`, absent from the
   pre-provisioning inventory, and still named with the `barun-` prefix. Wait until that exact ID is
   running and SSH-ready before uploading or launching work.
5. Attach the managed run with `jl run ... --on <exact-id> --no-follow --yes --json`; do not use the
   inline `jl run --gpu` provisioning path. Attached runs must not use `--keep`, `--pause`, or
   `--destroy`; the controller remains responsible for explicit cleanup.
6. For a remote script that records resource provenance, pass controller option
   `--append-jarvis-machine-id`. It appends `--jarvis-machine-id <captured-id>` after the script
   arguments, including the current replacement ID after a proven resume migration. The generic
   default is off, so enable it only when the remote script accepts that argument.
   If a preregistration must also contain the not-yet-known fresh ID, put
   `__SAFE_RUN_MACHINE_ID__` and `__SAFE_RUN_PREEXISTING_IDS__` in its strict
   `prelaunch_inventory` block and pass `--bind-attempt-inventory <target-relative-json>`. The
   controller atomically replaces only those operational fields after the exact template and
   Python runtime attestations and before target upload, records both file hashes, and refuses
   denylist drift, symlinks, path escape, or compute-contract drift.
7. Poll the run and instance by their exact IDs. Download logs, sample predictions, metrics,
   manifests, and checkpoints before cleanup. Hash transferred artifacts and record transfer
   failures.
8. Pause only the exact ID created or safely recovered above with
   `jl pause <exact-id> --yes --json`. Then query it with `jl get <exact-id> --json` and require a
   paused state. Save the filtered proof in the run manifest.

The Jarvis CLI directory sync does not honor `.gitignore`; it excludes only a small built-in set.
`safe_run.py` therefore refuses targets containing `.env*`, private-key, or common credential files.
Keep authentication outside the workspace and inspect any new ignored files before a remote run.

Never use a bulk lifecycle command. Never derive mutation targets from the general listing alone.
`jl run stop` terminates a tracked process; it does not pause its instance. Loss of a local stream
or terminal is not evidence of cleanup.

The first paid run must do more than test CUDA: after local/static preflight, it should run the full
repository tests, capture the environment/GPU manifest, audit pinned dataset hashes/licenses and
rendered token lengths, score the untouched base checkpoint on the frozen development diagnostic,
and execute the smallest gated SFT probe. Official/final evaluation records remain unopened.
Separate parallel jobs may screen independent learning-rate or data-mixture arms once all share the
frozen evaluator and manifests.

## Interruption handling

For unattended runs, the local controller must install a cleanup handler and an independent
deadline/watchdog scoped to the newly created machine ID. On error or interruption, collect the
available log, pause that exact project resource, verify its state, and mark the run failed. If the
ID cannot be proved to belong to this project, do not mutate it; report the ambiguity.
