# SmolLM2 matched Mobile Actions baseline

Status: preregistered and locally verified; not launched and not scored.

This is the first larger-model matched-adaptation harness for BarunAction-35M. It is designed to
answer a narrow question: what does a public Apache-2.0 instruction model roughly ten times larger
do when it receives the same Mobile Actions training IDs, semantic inputs, targets, presentation
budget, and scorer? It is not an off-the-shelf prompt comparison and it is not yet evidence that
BarunAction beats a larger model.

## Frozen baseline and recipe

The baseline is `HuggingFaceTB/SmolLM2-360M-Instruct` at commit
`a10cc1512eabd3dde888204e902eca88bddb4951`. The preregistered architecture has a 49,152-token
vocabulary, width 960, intermediate width 2,560, 32 layers, 15 query heads, five key/value heads,
and tied input/output embeddings. That gives exactly 361,821,120 unique parameters:

- token embedding: `49,152 * 960 = 47,185,920`;
- each transformer layer: `9,832,320`, or `314,634,240` across 32 layers;
- final RMS normalization: `960`.

The runner loads the exact revision, counts unique storage views, proves the output embedding is
storage-tied to the input embedding, and aborts unless the measured result is exactly 361,821,120.
Remote code is disabled. The model's native chat template is mandatory and its bytes are hashed in
the run evidence.

The only recipe is full-parameter, response-only SFT for seed 17 and one epoch. Its microbatch is
21 with three-step gradient accumulation, giving effective batch 63 and exactly 126 optimizer
steps over all 7,937 rows. It uses BF16, eager attention, AdamW, learning rate `2e-5`, 12 warmup
steps, cosine decay to 10% of the peak, gradient checkpointing, and no packing or truncation. The
final checkpoint is the only checkpoint scored. The frozen recipe is
[`configs/mobile_smollm2_matched_v1.json`](../configs/mobile_smollm2_matched_v1.json), SHA-256
`eca8135ff6cc58e14ddd7ddae210becb43462646e7f32d7de5c57fe624bb2f21`.

## Matched information contract

| Dimension | BarunAction candidate-v2 | SmolLM2 matched baseline |
|---|---:|---:|
| Internal-train training IDs | same 7,937 | same 7,937 |
| Development IDs | same 756 | same 756 |
| Train passes / presentations | 1 / 7,937 | 1 / 7,937 |
| Effective batch / optimizer steps | 63 / 126 | 63 / 126 |
| Seed | 17 | 17 |
| Semantic schemas, order, descriptions, NOW, user text | frozen manifest bytes | identical bytes |
| Canonical Action IR targets | frozen manifest bytes | identical bytes |
| Added teacher, synthetic, preference, or reward data | none | none |
| Loss | assistant response only | assistant response only |
| Model-specific input difference | Barun role tokens | native SmolLM2 chat-control tokens only |
| Scorer | `barun-mobile-actions-score-v1` | the same implementation |

For each row, the adapter's frozen Barun role wrapper is parsed strictly. The unchanged
`ACTION_IR_V1`, timestamp, ordered tool definitions, types, descriptions, required flags, and user
request become system and user messages. `tokenizer.apply_chat_template` supplies only native role
tokens. For training, the unchanged target is the assistant message. The full-message encoding must
have the generation prompt as an exact token prefix; only the suffix is labeled. Any renderer
drift, missing native EOS, or sequence above 2,048 tokens aborts the whole run.
The evidence bundle records each permitted row's source-record hash, rendered native-prompt hash,
prompt/response/full token counts, and no-truncation flag so the renderer comparison is inspectable
at sample level.

Development decoding is deterministic greedy and unconstrained, capped at 192 generated tokens.
The prediction writer does not trim, extract, or repair JSON. Missing, truncated, OOM, context
overflow, schema-invalid, and parse-failed rows remain measured failures. The existing strict
Action IR scorer writes every raw prediction, every sample score, and aggregate metrics.

## Official-test firewall

The runner has no source-dataset, test-manifest, or official-evaluation argument. A clean remote
stage contains only these frozen artifacts:

- train manifest SHA-256
  `131473ccb5bfb51cac0439b42159e72ec4c598025e50364a52b122b056c2e84e`;
- development manifest SHA-256
  `988bdce5874d1f1a775feeb5ba2b58cd2bdc128f57e73cb9a63d535fae7c1d55`;
- adapter audit SHA-256
  `dc756f97c0a7ef706ec8ffefe2d57cf7e16d75a6ccd932f906d16b6a6ee2f83c`.

The audit must prove that all 961 official rows stayed opaque, unparsed, unmaterialized,
un-tokenized, and excluded from overlap calculations. The runner rechecks the exact train/dev file
hashes, ID-membership hashes, row counts, split metadata, disjointness, and audit object before it
downloads the baseline. Do not place the combined `dataset.jsonl` or any official-evaluation
artifact in the clean stage.

## Controlled JarvisLabs launch

Do not run this from the dirty workspace. First construct and hash a clean staging directory with
only `src/`, `tests/`, `pyproject.toml`, the runner, its frozen config and requirements, and copies of
the three input files above under `data/`. Run the exact staged tests locally. Then use the exact-ID
controller, which creates a new project-owned instance, appends its machine ID to the runner,
downloads the compact evidence bundle, and pauses and verifies only that ID:

```bash
uv run python infra/jarvis/safe_run.py run \
  --name barun-mobile-smollm2-matched-1930-s17 \
  --record experiments/runs/20260803-1930-mobile-smollm2-matched-s17/jarvis.json \
  --target /absolute/path/to/clean-stage \
  --script scripts/run_mobile_matched_baseline.py \
  --gpu H200 --num-gpus 1 --storage 100 --region IN2 \
  --requirements requirements/matched-baseline.txt \
  --poll-seconds 60 --max-runtime-minutes 240 \
  --artifact /home/barun-artifacts/20260803-1930-mobile-smollm2-matched-s17/export/essential \
  --artifact-dest experiments/runs/20260803-1930-mobile-smollm2-matched-s17/essential \
  --artifact-recursive --append-jarvis-machine-id -- \
  --run-id 20260803-1930-mobile-smollm2-matched-s17 \
  --train-manifest data/train.jsonl \
  --dev-manifest data/dev.jsonl \
  --audit data/audit.json
```

The command is a reviewed launch template, not evidence that a launch occurred. A read-only
inventory and a new source-snapshot manifest are still required immediately before provisioning.
Instance 463058, Kroda, and every other pre-existing or unrecognized resource remain protected.

## Interpretation limit

The Mobile Actions internal-training population maps every row to `CALL`; it contains no useful
denominator for abstention, clarification, confirmation, unsafe-call, or false-action evaluation.
The development split is public and has already supported BarunAction selection. BarunAction also
consumed three development recipe trials while this first baseline consumes one. Therefore this
run can provide useful matched public-development evidence and error pairs, but it cannot establish
a safety win, official-test win, training-procedure win, or “beats larger models” claim. That claim
still requires matched multi-seed adaptation, the sealed human efficacy and safety suites, paired
uncertainty, and the preregistered practical-effect gates.

The adaptation information is matched; total upstream training is not. SmolLM2 brings its released
instruction-tuning history, and the Mobile Actions exposure of either model's pretraining is
unknown. Report that uncontrolled prior separately from the measured one-epoch adaptation.
