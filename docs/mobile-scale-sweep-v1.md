# Mobile scale sweep: matched-adaptation base-model size/token sweep

Run ID: `20260805-1554-mobile-scale-sweep-s17` (immutable).
Status: attempt-5 CPU prefreeze complete; **blocked pending a fresh independent prelaunch
audit of v5**. No GPU was created for this freeze. The spent attempt-4 go is not reused. H200
lineage **465183→465186** (attempt-4) and **465155** (attempt-1/3 infra failure) remain
permanently protected.

Frozen scientific config (attempt 5, current): `configs/mobile_scale_sweep_v5.json`, SHA-256
`57dcfe573c17759404545c272f1fc945aabb82b2893f697615eb7fe428d1f2d7`.

Naming note: StrataLM is only the former working name of the base model; the canonical names are
BarunLM-35M (base) and BarunAction-35M (post-trained). The pretraining evidence file
`blog/stratalm-architecture-blog.md` keeps its historical path.

## Attempt-4 go spent by flash_attn system-site abort; attempt-5 isolation correction

Attempt 4 (config `configs/mobile_scale_sweep_v4.json`, SHA-256
`c89b5c77a5531f617f1acc23e754c27336cf039830e9de7f00d44c8353e8dcb0`) passed its independent
prelaunch audit (go SHA-256
`9bf60db79cc543e35eb5e664d1a79e761337720c0dbd663da64a8b62c6f4dac2`) and created fresh H200
**465183** (`barun-scale-sweep-a4-20260805`) under `template=axolotl` with live CPython
**3.11.10**. After a local-controller death mid-upload and an owned reattach (resume migrated
**465183→465186**), managed run `r_28232019` passed the 62-test gate, completed the in-run
candidate-v2 reference evaluation at **590/725** exact (schema 723/725, 0 truncations), verified
the first challenger snapshot, then aborted in `AutoModelForCausalLM.from_pretrained`: JarvisLabs
`jl run` creates `uv venv --system-site-packages`, so Transformers imported the axolotl image
`flash_attn_2_cuda` extension ABI-mismatched against the managed venv torch 2.13.0. No LR screen
and no adoption decision. Pause-verified on **465186**. Immutable receipt:
`experiments/runs/20260805-1554-mobile-scale-sweep-s17/attempt-4-infrastructure-failure.json`,
SHA-256 `adad51dad1e3216272496dd83072c6d0514d6585da16fe66d56d4944184fa5e0`. The attempt-4 go is
**spent**; a third GPU under that go is forbidden.

Attempt 5 keeps the same immutable run directory with
`preregistration-attempt-5.json` binding the successor v5 config. Scientific bindings
(roster/snapshot/split/recipe/LR/decision/reference/decoding/measured-failure/axolotl/3.11.10)
are byte-identical to attempt 4. Infrastructure corrections only:

- **INFRA-1** — freeze project venv isolation with `include-system-site-packages=false`. Launch
  packaging must recreate `.venv` without `--system-site-packages` after jl's contaminated
  default (`safe_run --isolated-project-venv`) and reinstall frozen requirements that pin
  `torch==2.13.0` so CUDA torch does not depend on system-site inheritance.
- **INFRA-2** — runner `enforce_venv_isolation` fail-closes before Torch/CUDA import and before
  challenger `from_pretrained`: live `pyvenv.cfg` must read
  `include-system-site-packages = false`, and `import flash_attn` must fail. A CPU unit test
  synthesizes the attempt-4 contaminated case and would have rejected it.
- **INFRA-3** — add **465183** and **465186** to `compute.protected_machine_ids`; keep **465155**,
  **465072**, and the full prior denylist.
- **INFRA-4** — cite the attempt-4 infrastructure-failure and spent-go hashes; state explicitly
  that v5 does **not** reuse the spent go — a new independent audit cycle is required.
- **INFRA-5** — the 590/725 reference score is attempt-4 evidence only; the in-run hash-verified
  candidate-v2 reference evaluation remains mandatory (no CLI float).

**Isolation approach chosen:** isolated project venv without `--system-site-packages` (not a
flash_attn stub or uninstall-only shadow). A durable `pyvenv.cfg` boolean is least forgeable,
matches the temporal-lane torch pin pattern, and removes the entire system-site ABI surface from
the Transformers import path.

All authorization flags remain false.

## Attempt-3 go spent by infrastructure failure; attempt-4 runtime correction

Attempt 3 (config `configs/mobile_scale_sweep_v3.json`, SHA-256
`a67959b9b95aa72a6c9153234bb502490c800dba2c539ba9163e37cf4e539451`) passed its independent
prelaunch audit (go SHA-256
`d74d01e2078322c969db1158c54ebdc7343432635d223d78523226bae452f03b`) and created fresh H200
**465155** (`barun-scale-sweep-20260805`) under `template=pytorch`. `safe_run` aborted before
upload because live CPython was **3.10.20** while the controller requires **3.11.10** — the
same failure class as Month-Boundary Counterfactual v1. Pause-verified; zero
upload/tests/training/scoring/official-961. Immutable receipt:
`experiments/runs/20260805-1554-mobile-scale-sweep-s17/attempt-1-infrastructure-failure.json`,
SHA-256 `429ba84586a9b1ea503138e8defab9e595bdad7a08ec4fd148eee2743eee2855`. The attempt-3 go is
**spent** and must never authorize a second H200 or an unaudited retry.

Attempt 4 kept the same immutable run directory with
`preregistration-attempt-4.json` binding the successor v4 config. Scientific bindings
(roster/snapshot/split/recipe/LR/decision/reference/decoding/measured-failure) were
byte-identical to attempt 3. Infrastructure corrections only:

- **INFRA-1** — freeze `template=axolotl` and `CPython 3.11.10` (plus the matching safe_run
  fields: JarvisLabs, H200, `num_gpus=1`, `region=IN2`, `is_spot=false`, `storage_gb=100`,
  `max_gpu_job_minutes=360`), matching successful prior H200 runs (MBCF hermetic v3 /
  PlanIR / axolotl-template probe).
- **INFRA-2** — add **465155** and **465072** to `compute.protected_machine_ids`; keep the full
  prior denylist.
- **INFRA-3** — cite the infrastructure-failure and spent-go hashes; state explicitly that v4
  does **not** reuse the spent go — a new independent audit cycle is required.
- **INFRA-4** — the runner validates frozen runtime fields at config load and refuses live
  template/Python attestation that is not axolotl + CPython 3.11.10 before Torch/CUDA import,
  download, training, or scoring (`--jarvis-template` required).

## Attempt-1 no-go and attempt-2 corrections

The attempt-1 CPU prefreeze (config `configs/mobile_scale_sweep_v1.json`, SHA-256
`d3ee897f9afeefe1e01ec32fe9b2721479b7496785e742d0954ad081758953b8`, commit `e183a3f`) was
rejected by an independent adversarial prelaunch audit. The immutable receipt is
`experiments/runs/20260805-1554-mobile-scale-sweep-s17/prelaunch-audit-attempt-1-no-go.json`,
SHA-256 `9bc9d3af9e633b08b6e0d0e1c3bfeedfa660cbe7755443ad58987f47e86e99e0`. The fresh split,
all hashes, roster revision pins, salt-bias analysis, tests, and commit hygiene verified
exactly; the rejection was confined to the runner and config bindings. The v1 config, the v1
`preregistration.json`, and the no-go receipt are immutable rejected evidence: never edit or
load them.

Attempt 2 keeps the same immutable run directory (the run never reached a compute action, so no
retry clause was spent) with `preregistration-attempt-2.json` binding the successor v2 config.
Corrections, each covered by a CPU test:

- **P0-1** — the candidate-v2 reference score is now produced only by an in-run, hash-verified
  evaluation: the checkpoint is verified against the committed `src/barunaction/candidate.py`
  pin (with a signed-manifest cross-check), the `barun-16384` gold-token audit is re-verified
  field by field, and greedy generation plus scoring run on the frozen selection manifest before
  any challenger arm. The forgeable `--reference-exact-percent` CLI float is deleted; no
  unauthenticated path can supply the decision-critical number.
- **P0-2** — `enforce_machine_id` rejects any non-positive or protected `--jarvis-machine-id`
  against the frozen denylist before any work (mirroring the `mobile_qwen05b_matched` lane).
- **P1-1** — pythia-70m-deduped is rebound to its true unique trainable parameter count
  **70,426,624** (verified analytically from the pinned architecture in a CPU test); the earlier
  95,592,496 was the safetensors total including 25,165,824 persisted causal-mask buffer entries
  and 48 rotary `inv_freq` entries, and is retained only as labeled hub metadata.
- **P1-2** — `fit_outcome_counts` consumes the real scorer aggregate's `schema_valid` /
  `ast_exact_match` / `truncation` / `missing_prediction` exact `Rate` numerators (the old code
  read a nonexistent `schema_validity` key and rounded float products); tested against genuine
  `write_scores` output.
- **P2** — the per-arm gold-audit drift check now compares every frozen field
  (`max_target_tokens_with_eos`, `prompt_tokens_max`, `total_with_eos_max`) on both splits, and
  `generation_batch_size` is read from the frozen config with no CLI override.
- **P3** — dead prompt-prefix invariant removed, empty-target guard raises `ScaleSweepError`,
  and the exported config in the evidence bundle is named `config.json`, not
  `preregistration.json`.

## Attempt-2 no-go and attempt-3 corrections

The attempt-2 CPU prefreeze (config `configs/mobile_scale_sweep_v2.json`, SHA-256
`c8d57f84013198094c27d06d35851e5320f66a5106e6f1cbc6407bfd5e78f593`, commit `315070b`) was also
rejected. The immutable receipt is
`experiments/runs/20260805-1554-mobile-scale-sweep-s17/prelaunch-audit-attempt-2-no-go.json`,
SHA-256 `e693302843678bd6622b149a74320d8ca3bbbba77ea8ab4fb8a065b986ec8ef3`. Every attempt-1 fix
verified genuine and survived nine crafted attacks; the sole blocking defect was the unbound
challenger snapshot evidence. The v2 config, `preregistration-attempt-2.json`, and the receipt
are immutable rejected evidence: never edit or load them.

Attempt 3 keeps the same immutable run directory (attempt 2 again never reached a compute
action) with `preregistration-attempt-3.json` binding the successor v3 config. Corrections,
each covered by a CPU test:

- **P0-3** — every challenger arm's complete downloadable snapshot is now hash-pinned in the v3
  config (`snapshot_files` per arm): `model.safetensors` SHA-256 obtained via **read-only
  Hugging Face LFS metadata** (`files_metadata`; no weight downloaded locally), plus
  `config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, and
  `generation_config.json` where it exists at the pinned revision (pythia-70m-deduped has none;
  its absence is pinned through exact file-set equality). `verify_challenger_snapshot` enforces
  exact file-set equality plus per-file byte size and SHA-256 after `snapshot_download`, before
  any tokenizer or weight load, iterating the config's own pin table so advertisement and
  enforcement cannot drift; the previously advertised-but-unenforced
  `config_json_sha256`/`tokenizer_config_sha256` pins are cross-validated against the pin table
  at config load. Verified hashes are bound into `snapshot-verification.json`,
  `arm-result.json`, and `result.json`. Pin receipt:
  `experiments/runs/20260805-1554-mobile-scale-sweep-s17/snapshot-pins.json`.
- **P2-A** — the frozen `generation_config_overrides` are validated against the exact greedy
  contract at config load (`decoding_kwargs`) and passed explicitly to every challenger
  `model.generate` call, so no snapshot-side generation default can influence decoding; the
  generation summary records the explicitly passed overrides.
- **P2-B** — per-fit measured-failure semantics are implemented as preregistered: non-finite
  training loss (`MeasuredFitFailure`) or CUDA out-of-memory during one fit records that fit as
  a measured failure and the run continues; an arm with zero completed fits enters the
  decision's `measured_failed_arm_ids` (`build_decision`) and can never be adopted; if every arm
  fails the decision is explicit all-arms falsification. Integrity violations (hash mismatch,
  budget drift, contract violations) still abort the whole run.
- **P3** — the global torch seed is installed (`torch.manual_seed` /
  `torch.cuda.manual_seed_all`) with its exact scope documented in the config;
  `environment_versions` binds python/torch/transformers/tokenizers/huggingface_hub/safetensors
  versions into `result.json`; the residual reference-verification TOCTOU window is consciously
  accepted with rationale in the attempt-3 preregistration (in-call re-verification narrows it;
  the remainder requires concurrent host compromise).

## Hypothesis and evidence basis

BarunAction candidate-v2 (35,072,768 unique parameters) scored 602/756 on the reused Mobile
development probe; the independently verified Qwen2.5-0.5B matched run scored 663/756
(`experiments/runs/20260803-2122-mobile-qwen05b-matched-s17`). The repository cannot currently
say whether that 61-row gap comes from parameter count or from pretraining-token budget:
BarunLM-35M pretrained on about 5.7e9 tokens (`blog/stratalm-architecture-blog.md`), while
trillion-token-class small models saw 300-4,500x more. The earlier SmolLM2-360M-Instruct matched
attempt returned 0/756 with 407/756 truncations — a recipe artifact of chat-template/EOS
termination, not a model capability result (`experiments/mistakes.md`, entry "SmolLM2 collapse
was model-and-recipe-specific"). The MBCF v3 hermetic run additionally showed that narrow
targeted gains can trade off against same-month retention (ledger entry for
`20260804-0255-mobile-temporal-counterfactual-axolotl-hermetic-s17`), which motivates a
whole-split decision metric rather than a bucket-targeted one.

This sweep fine-tunes a preregistered roster of pinned off-the-shelf **base** checkpoints on the
identical frozen Mobile Actions SFT recipe used for candidate-v2 and scores them on a fresh
leakage-controlled selection split:

- **Token axis** at near-constant small parameters: `EleutherAI/pythia-70m-deduped`
  (~3e11 pretraining tokens, arXiv:2304.01373) versus `HuggingFaceTB/SmolLM2-135M`
  (~2e12 tokens, arXiv:2502.02737).
- **Parameter axis** at saturated tokens: `HuggingFaceTB/SmolLM2-135M` versus
  `HuggingFaceTB/SmolLM2-360M` (~4e12 tokens), anchored above by the existing Qwen2.5-0.5B
  result (never rerun) and below by candidate-v2 (never retrained).

Preregistered decision: which base-model scale, and which axis, the project adopts for the next
post-training phase.

## Fresh selection boundary (held out from train)

No untouched Mobile Actions rows exist outside train/756-dev/961-official: the pinned source
(9,654 rows) is exactly 8,693 eligible internal-train rows (7,937 train + 756 dev) plus the 961
sealed official rows. The preregistered fallback therefore applies: a grouped holdout from the
frozen 7,937-row train manifest, removed identically from every arm so the matched-budget
property is preserved.

- The pinned source and BarunLM-35M tokenizer reproduced the frozen train
  (`131473cc...`), dev (`988bdce5...`), and audit (`dc756f97...`) SHA-256 values byte-identically
  on CPU before derivation.
- Grouping uses the existing `cluster_id` connected components (ordered gold tool-name signature
  plus entity-delexicalized template families joined by verified near-duplicate edges), exactly
  as frozen by `barun-mobile-actions-connected-components-v1`.
- Assignment: `sha256("barun-mobile-scale-sweep-selection-v1:" + cluster_id)`, first 8 bytes
  big-endian, modulo 10; fold 0 is selection. Whole clusters land on one side; no exclusions.
- Result: **7,212 sweep-train rows (4,716 clusters) and 725 selection rows (589 clusters)**.
  Manifest and membership hashes are frozen in the config, in
  `experiments/runs/20260805-1554-mobile-scale-sweep-s17/split-receipt.json`, and as sorted ID
  lists (`sweep-train-membership.txt`, `selection-membership.txt`) in the run directory.
- Known bias, declared pre-outcome: candidate-v2 trained on every selection row, so its
  reference score on the selection split is upward-biased and the +3.0-point margin is
  conservative for challenger arms. The reused 756 probe is descriptive-only; the 961 official
  rows stay sealed.

## Roster, transport, and hygiene

Pinned revisions (read-only Hugging Face metadata; receipt at
`experiments/runs/20260805-1554-mobile-scale-sweep-s17/roster-metadata.json`):

| Arm | Revision | Unique trainable params | Claimed pretraining tokens | Context |
| --- | --- | --- | --- | --- |
| `EleutherAI/pythia-70m-deduped` | `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c` | 70,426,624 (untied) | ~3e11 | 2048 |
| `HuggingFaceTB/SmolLM2-135M` | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | 134,515,008 (tied) | ~2e12 | 8192 |
| `HuggingFaceTB/SmolLM2-360M` | `f8027fd0eaeea54caa13c31d31b9fdc459c38b49` | 361,821,120 (tied) | ~4e12 | 8192 |

Pythia's hub safetensors total is 95,592,496, but 25,165,824 of those entries are persisted
non-trainable causal-mask `attention.bias` buffers plus 48 rotary `inv_freq` entries; the
runner's `count_unique_parameters` assertion and the decision ordering use the trainable
70,426,624 (attempt-1 audit defect P1-1, corrected in v2).

All three are Apache-2.0 base checkpoints with no chat template and `<|endoftext|>` (id 0) as
EOS/BOS. Transport is `barun-raw-prompt-transport-v1`: byte-identical model-visible prompts to
candidate-v2, encoded with each arm's own pinned `tokenizer.json` via
`tokenizers.Tokenizer.encode(text, add_special_tokens=False)`; prompt and target encoded
separately exactly as at generation time; exactly one native EOS appended and supervised. The
termination contract (training-appended EOS == generation stop id == pad id) is the preregistered
guard against the SmolLM2 0/756 artifact and is verified per arm by
`verify_termination_contract`.

Gold token-length audit (CPU, frozen in
`experiments/runs/20260805-1554-mobile-scale-sweep-s17/gold-token-audit.json`): the maximum gold
target with EOS is 197 tokens (Barun tokenizer; 150 for the GPT-NeoX tokenizer, 165 for the
SmolLM2 tokenizer). The frozen budget rule (smallest multiple of 64 at or above max + 32) sets
**`max_new_tokens = 256`**, which also supersedes the previous 192-token budget that would have
truncated the 197-token Barun gold maximum during the candidate-v2 reference evaluation. Context
fit is verified for every tokenizer on both splits (longest training row 660 <= 2048; longest
prompt 487 + 256 <= 2048), and the GPU runner recomputes the audit per arm and aborts on drift.

Recipe: identical 7,212 sweep-train rows and presentation budget for every arm, seed 17, one
epoch, per-device batch 21 x accumulation 3 (effective 63), 115 optimizer steps, AdamW with the
frozen candidate-v2/matched-lane hyperparameters, bfloat16, deterministic greedy decoding.
Pre-declared LR screen: {1e-4, 3e-5} per arm, one full fit each, scored only on the fresh
selection split; higher exact-match count wins, ties to the lower rate; failed fits are measured
failures. Truncation, parse failure, and missing predictions are measured failures, never
dropped rows.

## Decision rule

Reference: candidate-v2 evaluated on the same selection split in the same run before any
challenger arm is scored, through the in-run hash-verified path bound in the v3/v4 config's
`reference_evaluation` section (checkpoint pins from `src/barunaction/candidate.py`, greedy
decoding, 256-token budget, same scorer; predictions and scores bound into `result.json`). An
arm passes if its screen-selected fit clears **all** of: exact match at least reference + 3.0
points; schema validity at least 0.95; zero truncations; zero missing predictions; zero
generation failures. The smallest passing arm by unique trainable parameter count is adopted.
Axis conclusions (token: 135M versus 70M; parameter: 360M versus 135M) are reported from
pairwise gaps regardless of adoption; if no arm passes, the scaling hypothesis is falsified at
these budgets and candidate-v2 remains the release checkpoint.

## Phase gates

1. **CPU build (this phase, complete for attempt 5):** split derivation, roster pinning, token
   audit, frozen v5 config with axolotl/3.11.10 attestation plus isolated-venv / flash_attn
   fail-closed preflight, runner, hermetic tests. All authorization flags are false.
2. **Independent prelaunch audit of v5:** a separate adversarial review. The spent attempt-4 go
   does not authorize launch. Only a fresh v5 go unlocks any compute action.
3. **Single GPU launch (not authorized yet):** one fresh exact-ID `barun-scale-sweep-*` H200
   under `template=axolotl` / CPython 3.11.10 with `safe_run --isolated-project-venv` (read-only
   safe inventory first; the protected denylist includes 465072, 465155, 465183, and 465186),
   hard budget 360 minutes, pause-verified by exact ID after artifact download. Never reuse
   465155 / 465183 / 465186.

## Files

- `configs/mobile_scale_sweep_v5.json` — immutable attempt-5 scientific+runtime config (hash
  above; the runner binds it inline).
- `configs/mobile_scale_sweep_v4.json`, `configs/mobile_scale_sweep_v3.json`,
  `configs/mobile_scale_sweep_v2.json`, `configs/mobile_scale_sweep_v1.json` — immutable prior
  configs; never loaded by the active runner.
- `src/barunlm/baselines/mobile_scale_sweep.py` — split derivation, audit, transport,
  termination contract, LR screen, decision rule, machine-ID enforcement, axolotl/CPython
  3.11.10 runtime attestation, isolated-venv / flash_attn preflight, in-run reference
  evaluation, challenger snapshot verification, explicit decoding overrides, per-fit
  measured-failure semantics, GPU runner (binds the v5 config hash).
- `tests/test_mobile_scale_sweep.py` — CPU-hermetic tests for every rule, including the
  attempt-2/3 scientific corrections, attempt-4 axolotl attestation, and attempt-5 isolation
  gate.
- `experiments/runs/20260805-1554-mobile-scale-sweep-s17/` — `preregistration.json` (attempt 1,
  immutable), `preregistration-attempt-2.json` / `preregistration-attempt-3.json` /
  `preregistration-attempt-4.json` (immutable), `preregistration-attempt-5.json`,
  `prelaunch-audit-attempt-1-no-go.json` / `prelaunch-audit-attempt-2-no-go.json` (immutable),
  `prelaunch-audit-attempt-3-go.json` / `prelaunch-audit-attempt-4-go.json` (spent),
  `attempt-1-infrastructure-failure.json`, `attempt-4-infrastructure-failure.json`,
  `split-receipt.json`, membership files, `gold-token-audit.json`, `roster-metadata.json`,
  `snapshot-pins.json`, plus the build scripts.
- Ledger: prior attempt entries plus the attempt-5 prefreeze
  `blocked_pending_independent_prelaunch_audit` entry appended to `experiments/ledger.jsonl`.
