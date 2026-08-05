# BarunAction-35M Mobile development regression gate

This command measures whether a compatible broader post-training checkpoint preserves the
candidate-v2 behavior on the already-derived Mobile Actions development population. It is a
regression gate, not a fourth Mobile development selection trial and not an official-evaluation
result.

## Completed result

Run `20260803-2005-mobile-presto-regression-s17` applied the frozen gate to the PRESTO-stage model
SHA-256 `83c78952e719456574d2bb808fb486d0320519dd4f5df3c67736ae17931ee357`.
It scored **0/756** strict AST exact against candidate-v2's 602/756 and the required 587/756:
0 rows were fixed, all 602 reference-correct rows regressed, 0 were retained correct, and 154 were
retained incorrect. The broader checkpoint was rejected.

Every one of the 756 rows generated a prediction; 47 hit the generation cap and no generation call
failed. The immutable
[`result.json`](../experiments/runs/20260803-2005-mobile-presto-regression-s17/essential/evaluation/result.json)
has SHA-256 `681212a8b46783c2c05aa7b8c6f3bf74d9a4261a423508498570408522fda4ea`;
the paired sample evidence has SHA-256
`708cd0d0b058e33825d12ce9bdebe35675a5cb613501cafdd023d82cabb9b478`.
The expected process exit code was 2 for a complete failed gate. Project-owned H200 instance
463642 downloaded the complete evidence and was verified `Paused`.

This is a reused public-development regression result only. The runner accessed none of the 961
official Mobile Actions rows and no hidden human-suite data, so the result is neither a breakthrough
nor evidence of broad or larger-model superiority.

## Frozen decision

The immutable preregistration is `configs/mobile_regression_v1.json`, SHA-256
`96afa0ac407c53fd375ed0910a08ec66463718e3b826cf02b73f8bbc87a3c47e`.

- Population: 756 examples derived only from the pinned dataset's internal-training rows.
- Manifest SHA-256: `988bdce5874d1f1a775feeb5ba2b58cd2bdc128f57e73cb9a63d535fae7c1d55`.
- Reference: BarunAction-35M candidate-v2 at 602/756 strict AST exact match.
- Gate: pass at 587/756 or better. This is the exact integer boundary for no more than a
  two-percentage-point absolute drop from 602/756; 586/756 fails.
- Decoding: unconstrained greedy, 192 maximum new tokens, batch size 128, CUDA bfloat16, and the
  same H200, PyTorch, CUDA, cuBLAS, and math-SDPA environment recorded for candidate-v2.

The manifest, adapter audit, reference aggregate, reference sample evidence, preregistration, and
the three checkpoint files are hash-verified before model tensors are loaded. The checkpoint's
configuration and tokenizer must remain byte-identical to candidate-v2; only the learned model
weights may differ.

## Frozen command used

The completed attempt used the following command shape from a staged repository tree on the
matching GPU environment:

```bash
python scripts/evaluate_mobile_regression.py \
  --checkpoint /path/to/post-training/checkpoint \
  --checkpoint-hashes /path/to/post-training/checkpoint/checkpoint_manifest.json \
  --output /path/to/immutable/mobile-regression-output
```

`--checkpoint-hashes` may instead name a strict JSON object containing exactly
`barun_config.json`, `model.safetensors`, and `tokenizer.json` SHA-256 values. A destination that
already exists is rejected. Relocated, byte-identical copies of the frozen data or reference
evidence can be supplied with `--manifest`, `--audit`, `--reference-aggregate`, and
`--reference-samples`; their expected hashes cannot be overridden.

The command returns exit status 0 when the gate passes and 2 after writing complete evidence when
the gate fails. Input-contract, hash, environment, or evaluator violations raise an error instead
of producing a score.

## Evidence

Each completed output contains:

- `predictions.jsonl` and its generation manifest with every raw unconstrained output;
- `scores/sample_scores.jsonl` and `scores/aggregate.json` from the strict Mobile Actions scorer;
- `paired-samples.jsonl`, aligned by immutable sample ID, with `retained_correct`, `fixed`,
  `regressed`, or `retained_incorrect` for every example relative to candidate-v2;
- `result.json` with the integer gate, exact delta, paired transition counts, input identities,
  and explicit interpretation limits; and
- `artifact-sha256.json` covering the completed evidence bundle.

The command has no parameter for the 961-row official evaluation split or the original combined
dataset source. It verifies the previously written audit stating that those rows remain opaque,
unparsed, and unmaterialized, and reports an empty list of official-evaluation artifacts accessed.
