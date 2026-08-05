# Mobile scale sweep: matched-adaptation base-model size/token sweep

Run ID: `20260805-1554-mobile-scale-sweep-s17` (immutable).
Status: attempt-2 CPU prefreeze complete; **blocked pending a fresh independent prelaunch
audit**. No GPU, JarvisLabs resource, CUDA context, training step, or baseline weight download
has occurred in either attempt.

Frozen scientific config (attempt 2, current): `configs/mobile_scale_sweep_v2.json`, SHA-256
`c8d57f84013198094c27d06d35851e5320f66a5106e6f1cbc6407bfd5e78f593`.

Naming note: StrataLM is only the former working name of the base model; the canonical names are
BarunLM-35M (base) and BarunAction-35M (post-trained). The pretraining evidence file
`blog/stratalm-architecture-blog.md` keeps its historical path.

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
challenger arm is scored, through the in-run hash-verified path bound in the v2 config's
`reference_evaluation` section (checkpoint pins from `src/barunaction/candidate.py`, greedy
decoding, 256-token budget, same scorer; predictions and scores bound into `result.json`). An
arm passes if its screen-selected fit clears **all** of: exact match at least reference + 3.0
points; schema validity at least 0.95; zero truncations; zero missing predictions; zero
generation failures. The smallest passing arm by unique trainable parameter count is adopted.
Axis conclusions (token: 135M versus 70M; parameter: 360M versus 135M) are reported from
pairwise gaps regardless of adoption; if no arm passes, the scaling hypothesis is falsified at
these budgets and candidate-v2 remains the release checkpoint.

## Phase gates

1. **CPU build (this phase, complete):** split derivation, roster pinning, token audit, frozen
   config, runner, hermetic tests. All authorization flags are false.
2. **Independent prelaunch audit:** a separate adversarial review of the config, split, runner,
   and hygiene rules. Only its pass unlocks any compute action.
3. **Single GPU launch:** one fresh exact-ID `barun-scale-sweep-*` H200 (read-only safe inventory
   first; the 18-ID protected denylist is copied verbatim from
   `configs/mobile_sub100m_off_the_shelf_v1.json`), hard budget 360 minutes, pause-verified by
   exact ID after artifact download.

## Files

- `configs/mobile_scale_sweep_v2.json` — immutable attempt-2 scientific config (hash above; the
  runner binds it inline).
- `configs/mobile_scale_sweep_v1.json` — immutable rejected attempt-1 config; never loaded.
- `src/barunlm/baselines/mobile_scale_sweep.py` — split derivation, audit, transport,
  termination contract, LR screen, decision rule, machine-ID enforcement, in-run reference
  evaluation, GPU runner (binds the v2 config hash).
- `tests/test_mobile_scale_sweep.py` — 40 CPU-hermetic tests for every rule, including the
  attempt-2 corrections.
- `experiments/runs/20260805-1554-mobile-scale-sweep-s17/` — `preregistration.json` (attempt 1,
  immutable), `preregistration-attempt-2.json`,
  `prelaunch-audit-attempt-1-no-go.json` (immutable), `split-receipt.json`,
  `sweep-train-membership.txt`, `selection-membership.txt`, `gold-token-audit.json`,
  `roster-metadata.json`, plus the build scripts (`pin_roster_metadata.py`,
  `derive_fresh_split_and_audit.py`).
- Ledger: the attempt-1 preregistration entry, the independent reject entry, and the attempt-2
  prefreeze entry appended to `experiments/ledger.jsonl`.
