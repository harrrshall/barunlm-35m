# Mobile scale sweep v1: matched-adaptation base-model size/token sweep

Run ID: `20260805-1554-mobile-scale-sweep-s17` (immutable).
Status: CPU prefreeze complete; **blocked pending an independent prelaunch audit**. No GPU,
JarvisLabs resource, CUDA context, training step, or baseline weight download has occurred.

Frozen scientific config: `configs/mobile_scale_sweep_v1.json`, SHA-256
`d3ee897f9afeefe1e01ec32fe9b2721479b7496785e742d0954ad081758953b8`.

Naming note: StrataLM is only the former working name of the base model; the canonical names are
BarunLM-35M (base) and BarunAction-35M (post-trained). The pretraining evidence file
`blog/stratalm-architecture-blog.md` keeps its historical path.

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

| Arm | Revision | Unique params | Claimed pretraining tokens | Context |
| --- | --- | --- | --- | --- |
| `EleutherAI/pythia-70m-deduped` | `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c` | 95,592,496 (untied) | ~3e11 | 2048 |
| `HuggingFaceTB/SmolLM2-135M` | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | 134,515,008 (tied) | ~2e12 | 8192 |
| `HuggingFaceTB/SmolLM2-360M` | `f8027fd0eaeea54caa13c31d31b9fdc459c38b49` | 361,821,120 (tied) | ~4e12 | 8192 |

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
challenger arm is scored. An arm passes if its screen-selected fit clears **all** of: exact match
at least reference + 3.0 points; schema validity at least 0.95; zero truncations; zero missing
predictions; zero generation failures. The smallest passing arm by unique parameter count is
adopted. Axis conclusions (token: 135M versus 70M; parameter: 360M versus 135M) are reported
from pairwise gaps regardless of adoption; if no arm passes, the scaling hypothesis is falsified
at these budgets and candidate-v2 remains the release checkpoint.

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

- `configs/mobile_scale_sweep_v1.json` — immutable scientific config (hash above).
- `src/barunlm/baselines/mobile_scale_sweep.py` — split derivation, audit, transport,
  termination contract, LR screen, decision rule, GPU runner (binds the config hash).
- `tests/test_mobile_scale_sweep.py` — 28 CPU-hermetic tests for every new rule.
- `experiments/runs/20260805-1554-mobile-scale-sweep-s17/` — `preregistration.json`,
  `split-receipt.json`, `sweep-train-membership.txt`, `selection-membership.txt`,
  `gold-token-audit.json`, `roster-metadata.json`, plus the build scripts
  (`pin_roster_metadata.py`, `derive_fresh_split_and_audit.py`).
- Ledger: one preregistration entry appended to `experiments/ledger.jsonl`.
