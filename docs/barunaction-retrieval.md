# BarunAction-35M candidate-v2 checkpoint retrieval

Status: public, immutable W&B `v0` release with a hash-bound local manifest, independent fresh
redownload, and anonymous-read verification. Never replace the versions below with `latest`.

## Public immutable artifacts

| Artifact | Immutable identity | W&B digest |
| --- | --- | --- |
| [Float checkpoint](https://wandb.ai/harshalsingh1223-gladium-ai/barunaction-35m/artifacts/model/barunaction-35m-candidate-v2-float/v0) | `harshalsingh1223-gladium-ai/barunaction-35m/barunaction-35m-candidate-v2-float:v0` | `09c2a608c6f3947854a0f55eafc56b8a` |
| [Darwin ARM64 int8 checkpoint](https://wandb.ai/harshalsingh1223-gladium-ai/barunaction-35m/artifacts/model/barunaction-35m-candidate-v2-int8-darwin-arm64-qnnpack/v0) | `harshalsingh1223-gladium-ai/barunaction-35m/barunaction-35m-candidate-v2-int8-darwin-arm64-qnnpack:v0` | `8c3d8d30afc140e8c63461668ecb91a4` |
| [Evaluation evidence](https://wandb.ai/harshalsingh1223-gladium-ai/barunaction-35m/artifacts/evaluation/barunaction-35m-candidate-v2-evidence/v0) | `harshalsingh1223-gladium-ai/barunaction-35m/barunaction-35m-candidate-v2-evidence:v0` | `b5e2cb35698e36bc1a97095c6c2d8565` |

The project uses W&B `USER_READ` access: anyone can read it, while only the owning team can write.
An unauthenticated GraphQL check saw all three committed `v0` records with the exact digests,
counts, and sizes above. With an explicitly empty authorization header, streamed copies of the
448-byte `barun_config.json` and 140,304,464-byte `model.safetensors` returned HTTP 200 and
reproduced their release SHA-256 hashes.

Download the float checkpoint with the pinned W&B client:

```console
uv run --with 'wandb==0.28.1' wandb artifact get \
  --root ./barunaction-35m-candidate-v2-float \
  --type model \
  harshalsingh1223-gladium-ai/barunaction-35m/barunaction-35m-candidate-v2-float:v0

uv run --python 3.11 barunaction verify \
  --checkpoint ./barunaction-35m-candidate-v2-float
```

The local release manifest, upload receipt, public-access record, and independent redownload
receipt are under
`experiments/runs/20260803-2303-candidate-v2-public-release-s17/`. The authoritative manifest is
SHA-256 `cbb29c4921855031bfeee1c1f5e9ed1a33a932b902a875c21f255fac793c2165`; the
fresh-download verifier matched 310 files totaling 336,040,340 bytes.

## Candidate identity

`candidate-v2` is step 126 from arm run `20260803-1845-mob-batch63-s17`, selected by sweep run
`20260803-1845-mobile-followup-retry-s17`. Its current local float directory is:

```text
experiments/runs/20260803-1845-mobile-followup-retry-s17/essential/arms/batch63/checkpoint
```

The approximately 140 MB weight file remains an ignored experiment artifact. It was not copied
into `src/` and should not be committed to the source repository. Preserve the three checkpoint
files together:

| File | SHA-256 |
| --- | --- |
| `barun_config.json` | `9b3a1d71baa95a198744d250f9629231738d942570b8685c44307fd83dd33565` |
| `model.safetensors` | `fdb95ccf58a095e0d321be998924318b35ee59a334f6dd97d8726d2cf80021d3` |
| `tokenizer.json` | `70ded9605fccd09c2340ca7e225361eab0ae8b4dbbb0d6e26343ab5183979db6` |

The local `checkpoint_manifest.json` SHA-256 is
`c743ab7c4d33ae75c6b0aa4547458a961b92766da8fcf85fd148fda2ebb5530a`. It records the three file
hashes and source run. The manifest is a reproducibility record, not a cryptographic signature.

## Verify before loading

From the repository root:

```console
uv run --python 3.11 barunaction verify \
  --checkpoint experiments/runs/20260803-1845-mobile-followup-retry-s17/essential/arms/batch63/checkpoint
```

Verification fails closed if a required file is missing, its digest differs, the tokenizer and
configuration disagree, or the manifest is unsupported. Inference performs the same verification
before model loading.

## Retrieve an alternate compatible checkpoint

The runtime is checkpoint-configurable and does not require the current candidate. For any
alternate compatible checkpoint, keep its three files in one directory and supply a strict JSON
hash object:

```json
{
  "barun_config.json": "<sha256>",
  "model.safetensors": "<sha256>",
  "tokenizer.json": "<sha256>"
}
```

Then pass `--checkpoint-hashes PATH` to `barunaction verify` or `barunaction infer`, or pass the
mapping as `expected_sha256` to `BarunActionCompiler`. A checkpoint with hashes different from the
current manifest is deliberately reported without the `candidate-v2` identity.

## CPU dynamic-int8 artifact

The current local mixed-precision artifact is:

```text
experiments/runs/20260803-1845-mobile-followup-retry-s17/essential/arms/batch63/int8-qnnpack
```

| File | SHA-256 |
| --- | --- |
| `barun_config.json` | `9b3a1d71baa95a198744d250f9629231738d942570b8685c44307fd83dd33565` |
| `model.int8.pt` | `18102649bb4f8507ee754ae0c298e20064580833f05ca515faf5e04eac6f4488` |
| `tokenizer.json` | `70ded9605fccd09c2340ca7e225361eab0ae8b4dbbb0d6e26343ab5183979db6` |

Its externally pinned `quantization_manifest.json` SHA-256 is
`f45c391d18d78758b0d62eeb562d139d24cfe0be3c8a409e3b40c25945d95c6b`. Verify and load it only on
the recorded Darwin ARM64, PyTorch 2.13.0, QNNPACK runtime:

```console
uv run --with 'wandb==0.28.1' wandb artifact get \
  --root ./barunaction-35m-candidate-v2-int8 \
  --type model \
  harshalsingh1223-gladium-ai/barunaction-35m/barunaction-35m-candidate-v2-int8-darwin-arm64-qnnpack:v0
```

Then verify and load it only on the recorded runtime:

```console
uv run --python 3.11 barunaction verify-int8 \
  --checkpoint experiments/runs/20260803-1845-mobile-followup-retry-s17/essential/arms/batch63/int8-qnnpack \
  --manifest-sha256 f45c391d18d78758b0d62eeb562d139d24cfe0be3c8a409e3b40c25945d95c6b

uv run --python 3.11 barunaction infer \
  --checkpoint-format int8 \
  --checkpoint experiments/runs/20260803-1845-mobile-followup-retry-s17/essential/arms/batch63/int8-qnnpack \
  --int8-manifest-sha256 f45c391d18d78758b0d62eeb562d139d24cfe0be3c8a409e3b40c25945d95c6b \
  --tools examples/barunaction_tools.example.json \
  --context examples/barunaction_empty_context.example.json \
  --now 2026-08-03T20:00:00+05:30 \
  --request "Turn on the flashlight" \
  --device cpu
```

The package is 58,619,256 bytes including its manifest. This is dynamic qint8 for internal Linear
weights, not an all-int8 model; embedding, tied output head, and norms remain FP32. Exact export,
fidelity-smoke, security, and portability details are in `docs/int8-quantization.md`.

## Publication and future promotion

The downloadable-checkpoint release gate is complete for candidate-v2. The W&B copy is a public
distribution mirror; the checked-in SHA-256 manifest and receipts remain authoritative. Do not
substitute the public `harrrshall/BarunLM-35M` base checkpoint: it has different model weights.

The superseded `candidate-v1` provenance remains in `configs/barunaction/candidate-v1.json`; its
float and provisional int8 artifacts remain under run `20260803-1810-mobile-blind-s17`. It is
history, not the current default.

If a later development-only sweep promotes another checkpoint, update
`src/barunaction/candidate.py`, add a new immutable checked-in provenance JSON without deleting old
versions, update these cards with the new evidence, and rerun float and int8 verification, exact
Action IR smoke, and package tests. Never select or promote a checkpoint using sealed evaluation
labels.
