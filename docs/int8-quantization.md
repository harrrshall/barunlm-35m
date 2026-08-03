# BarunLM and BarunAction CPU dynamic-int8 format v1

Status: retained, version-bound Darwin ARM64 CPU artifact with a completed 756-row development
accuracy check. It is not a latency, energy, official-test, hidden-safety, or broad-utility claim.

## What is quantized

`barun-cpu-dynamic-int8-v1` converts eligible `torch.nn.Linear` weights to per-tensor symmetric
qint8 through an explicitly selected PyTorch CPU quantized engine. Activations are quantized
dynamically inside the Linear operation. A Linear whose weight is tied to the token embedding is
kept in FP32 so the embedding/output-head alias is preserved. Embeddings, RMS norms, rotary
buffers, attention math, residual operations, and KV caches remain floating point.

For the current BarunAction-35M `candidate-v2`, 87 internal Linear modules are qint8 and `lm_head`
remains an FP32 Linear tied to `embedding.weight`. Calling this an “all-int8 model” would be
incorrect.

The implementation is generic over a compatible BarunLM checkpoint and does not import or
hardcode any candidate identity. Source checkpoint hashes and the chosen engine are mandatory
export inputs.

## Artifact contract

An int8 directory contains exactly four regular, non-symlink files:

- `barun_config.json`;
- `tokenizer.json`;
- `model.int8.pt`, a quantized state dictionary; and
- `quantization_manifest.json`.

The manifest binds the source float hashes, all three payload hashes, algorithm, exact quantized
module inventory, FP32 Linear inventory, parameter counts, payload sizes, PyTorch version, operating
system, CPU architecture, and qengine. Loading requires an out-of-band expected SHA-256 for the
manifest itself. An adjacent, unpinned manifest is not treated as authenticity evidence.

The loader verifies the manifest and every payload hash before calling
`torch.load(..., weights_only=True)`, applies an artifact-size bound, reconstructs the exact
quantized graph, strict-loads the state, and checks every intended weight is qint8. For tied models,
it verifies that serialized embedding and head values are equal before loading and that the FP32
alias remains intact afterward.

There is no float fallback. A missing manifest, wrong hash, extra file, symlink, unsupported engine,
different PyTorch version/platform, state mismatch, or CUDA request is a deterministic failure.
`weights_only=True` narrows pickle execution risk but is not a general defense against denial of
service or lower-level deserializer vulnerabilities; only hash-pinned artifacts from a trusted
distribution channel should be loaded.

## Generic export and verification

The hash file can be either a direct filename-to-SHA object or checked-in provenance with a
`file_sha256` member.

```console
uv run --python 3.11 barunaction export-int8 \
  --source-checkpoint PATH/TO/FLOAT/CHECKPOINT \
  --source-hashes PATH/TO/PINNED-HASHES.json \
  --output PATH/TO/NEW-INT8-CHECKPOINT \
  --qengine qnnpack
```

The exporter refuses an existing output directory. It prints the new manifest SHA-256; retain that
digest outside the artifact directory.

```console
uv run --python 3.11 barunaction verify-int8 \
  --checkpoint PATH/TO/NEW-INT8-CHECKPOINT \
  --manifest-sha256 EXPECTED_SHA256
```

Python callers use `export_dynamic_int8_checkpoint`, `verify_int8_checkpoint`, and
`load_verified_int8_model` from `barunlm.quantization`. BarunAction inference selects the format
explicitly with `checkpoint_format="int8"` and `expected_int8_manifest_sha256=...`.

## Exact Action IR fidelity smoke

`examples/barunaction_int8_smoke.example.json` freezes two simple, expected Action IR cases. The
comparison verifies that the float and int8 artifacts have the same source hashes, runs
unconstrained greedy decoding through the production compiler, and requires both outputs to equal
the expected canonical Action IR. Raw-output equality is reported separately.

```console
uv run --python 3.11 barunaction smoke-int8 \
  --source-checkpoint PATH/TO/FLOAT/CHECKPOINT \
  --source-hashes PATH/TO/PINNED-HASHES.json \
  --int8-checkpoint PATH/TO/INT8/CHECKPOINT \
  --manifest-sha256 EXPECTED_SHA256 \
  --cases examples/barunaction_int8_smoke.example.json \
  --report PATH/TO/NEW-SMOKE-REPORT.json
```

## Current local evidence

Both candidate artifacts were exported on Darwin ARM64 with PyTorch 2.13.0, six CPU threads, and
QNNPACK. The v1 result is retained as historical provisional evidence; v2 is the current candidate.

| Evidence | candidate-v1 | candidate-v2 |
| --- | ---: | ---: |
| Float payload | 141,440,943 bytes | 141,440,943 bytes |
| Int8 payload | 58,615,358 bytes | 58,615,358 bytes |
| Int8 package including manifest | 58,619,256 bytes | 58,619,256 bytes |
| Payload reduction | 58.56% | 58.56% |
| Expected Action IR exact | 2/2 | 2/2 |
| Raw float/int8 output identical | 2/2 | 2/2 |

Current v2 identities:

- source model SHA-256:
  `fdb95ccf58a095e0d321be998924318b35ee59a334f6dd97d8726d2cf80021d3`;
- int8 weight SHA-256:
  `18102649bb4f8507ee754ae0c298e20064580833f05ca515faf5e04eac6f4488`;
- quantization manifest SHA-256:
  `f45c391d18d78758b0d62eeb562d139d24cfe0be3c8a409e3b40c25945d95c6b`; and
- smoke report SHA-256:
  `1007b8ea9c5db56f1298893729a12e065fe21d91f232187864fddcd3882797ab`.

The one-pass v2 smoke measured 0.342 seconds total float inference and 0.590 seconds total int8
inference for the two cases; int8 was slower on this host. Load time was also slightly slower. The
order was not randomized, samples were not repeated, RSS and energy were not measured, and this
Mac/QNNPACK environment is not a frozen target device. These numbers are diagnostic evidence only.
They must not be presented as on-device latency or as evidence that quantization improves speed.

## Completed candidate-v2 ARM64 retention diagnostic

The frozen configuration is
`configs/barunaction/candidate-v2-arm64-int8-retention-v1.json`, SHA-256
`7229572bea461a7899b09c0310a77b9b4d8af74211d1b9ac8b2b5f77dc6b67cb`. Before any int8 score was
read, it bound the exact source and artifact hashes, Darwin ARM64/PyTorch 2.13.0/QNNPACK runtime,
756-row development population, existing 602/756 H200 BF16 reference, unconstrained greedy
decoding, 192-token cap, batch size 16, minimum 587/756 score, and maximum loss of 15 correct rows.

Run `20260803-2224-candidate-v2-arm64-int8-retention-s17` completed all 756 int8 generations with
zero generation failures and zero truncations. The int8 artifact achieved **607/756 (80.29%)** AST
exact and **755/756 (99.87%)** schema validity. Against the frozen 602-row reference it fixed eight
rows and regressed three, a net gain of five. The immutable paired receipt therefore passed every
gate and returned `retain_candidate_v2_int8_artifact`.

An additional non-selection ARM64 FP32 control used the same rows and decoding settings. It scored
603/756; int8 fixed seven of its errors and regressed three, a net gain of four. This control shows
that one row of the difference from the original 602/756 score is attributable to the runtime or
dtype boundary. It does not alter the preregistered gate and should not be interpreted as evidence
that quantization generally improves accuracy.

| Artifact | SHA-256 |
| --- | --- |
| Frozen protocol | `7229572bea461a7899b09c0310a77b9b4d8af74211d1b9ac8b2b5f77dc6b67cb` |
| Int8 aggregate | `481a1e75210fa664cdcd0a55c0cd812393f79f1b74110dd9756f546701c4ef44` |
| Int8 sample scores | `80b4972efe3ea22d2f78887c7c671e6ee2845c642ec3d6b73fecad7c5849a337` |
| Int8 raw predictions | `ad6ea7d41762ccdc34d018ef1158642c6bcada2c2c0682e356bebed9180fb4fc` |
| Paired gate result | `7e74efc58a52e16fc74e9c48eb64d86c1ce5d0bfd6459a7f1fd8ba66218fd688` |
| Paired samples | `e63e48dc8149b615ab03a14fb9fff9b1435ab59c6623e72c16b44d34d7c651ad` |
| Receipt artifact manifest | `ee2e6b98a4b51c0baf3de9616e46761971d62fcb4d52761625e38f5db019671c` |
| ARM64 FP32 control comparison | `83eb9e3967aee973cb1ed17e0602ed2e88838a836141ddb5f9ecd22c87a401e1` |

The exact completed int8 commands were:

```console
uv run --python 3.11 python scripts/evaluate_mobile_actions.py \
  --checkpoint-format int8 \
  --checkpoint experiments/runs/20260803-1845-mobile-followup-retry-s17/essential/arms/batch63/int8-qnnpack \
  --int8-manifest-sha256 f45c391d18d78758b0d62eeb562d139d24cfe0be3c8a409e3b40c25945d95c6b \
  --manifest experiments/runs/20260803-1810-mobile-blind-s17/remote/data/dev.jsonl \
  --manifest-sha256 988bdce5874d1f1a775feeb5ba2b58cd2bdc128f57e73cb9a63d535fae7c1d55 \
  --output experiments/runs/20260803-2224-candidate-v2-arm64-int8-retention-s17/evaluation \
  --device cpu \
  --batch-size 16 \
  --max-new-tokens 192

uv run --python 3.11 python scripts/create_mobile_int8_retention_receipt.py \
  --protocol configs/barunaction/candidate-v2-arm64-int8-retention-v1.json \
  --protocol-sha256 7229572bea461a7899b09c0310a77b9b4d8af74211d1b9ac8b2b5f77dc6b67cb \
  --int8-evaluation experiments/runs/20260803-2224-candidate-v2-arm64-int8-retention-s17/evaluation \
  --output experiments/runs/20260803-2224-candidate-v2-arm64-int8-retention-s17/receipt
```

The receipt verifies complete aligned samples, exact candidate/source hashes, explicit int8
loading, and no float fallback. This was a post-selection deployment diagnostic, not another
Mobile candidate-selection trial. It accessed none of the opaque 961 official rows and supplies no
sealed safety or larger-model-superiority evidence.

## Limitations and deployment blockers

The full retention pass estimates behavior only on the reused all-`CALL` Mobile development
population. It contains no abstention, clarification, confirmation, unsafe, adversarial,
renamed-schema, or contextual-revision cases and says nothing about safety.

PyTorch 2.13 reports the eager `torch.ao.quantization.quantize_dynamic` API and qint8 tensor creation
path as deprecated in favor of torchao. Packed-state compatibility across Torch versions, engines,
operating systems, and architectures is not assumed; format v1 rejects those differences.
Deployment beyond the retained Darwin ARM64/PyTorch 2.13.0/QNNPACK combination requires either a
new validated runtime artifact or a backend-neutral/torchao format. Repeated cold/warm latency,
peak RSS, and energy measurement remain undone.

JarvisLabs training timing is not target-device evidence and cannot close these deployment gates.
