# BarunAction-35M candidate-v2 research-release model card

Status: hash-pinned, proposal-only research release for a narrow seven-schema Mobile Actions
domain. It is runnable and quantized, but it is not a production-autonomy, safety, or broad
function-calling release.

## Model summary

BarunAction-35M is a 35,072,768-parameter proposal-only personal-action compiler post-trained from
the `harrrshall/BarunLM-35M` checkpoint. `candidate-v2` identifies step 126 of arm
run `20260803-1845-mob-batch63-s17`, selected by preregistered sweep
`20260803-1845-mobile-followup-retry-s17`. Packaging and quantization did not alter the source
float checkpoint. The dynamic-int8 artifact was derived from those exact hashes and passed its
frozen ARM64 development-retention gate.

The model accepts a request together with caller-supplied tool schemas, JSON context, and a
timezone-aware `NOW`. It proposes exactly one Action IR v1 object or fails closed with a
deterministic input, generation, parse, or schema error. The Python and command-line interfaces do
not invoke tools. Authorization and confirmation remain external policy decisions; a valid model
proposal is never permission to execute it.

Current runtime identity and expected file hashes live in `src/barunaction/candidate.py`; checked-in
provenance lives in `configs/barunaction/candidate-v2.json`. The superseded v1 provenance remains in
`configs/barunaction/candidate-v1.json` rather than being rewritten.

## Intended use

The current package is intended for local research, strict proposal validation, and the included
in-memory simulator. It is suitable for inspecting compact-model action compilation on schemas
close to the Mobile Actions development probe.

It is not approved to contact people, edit calendars, change devices, or perform any other real
side effect. It is not a safety classifier, autonomous assistant, or general function-calling
model. A product integration must add its own authorization, confirmation, identity, permissions,
rate limits, audit trail, and tool-specific validation outside the model.

## Evaluation evidence

The only candidate-selection result is an internal development probe derived from the
`google/mobile-actions` training population. Its 756 examples are all `CALL`; there are no
abstention, clarification, confirmation, unsafe-request, or out-of-scope examples.

| Metric | Base BarunLM-35M | candidate-v1 | v2 H200 BF16 | v2 ARM64 int8 |
| --- | ---: | ---: | ---: | ---: |
| Parse-valid output | 0.00% | 99.87% | 100.00% | 100.00% |
| Strict schema-valid output | 0.00% | 99.47% | 99.87% | 99.87% |
| Action IR AST exact match | 0.00% | 76.46% | 602/756 (79.63%) | 607/756 (80.29%) |
| Truncated at 192 generated tokens | 99.87% | 0.00% | 0/756 | 0/756 |

Generation was unconstrained greedy decoding. The grouped train/dev construction, evaluator, raw
predictions, and aggregate results are recorded under
`experiments/runs/20260803-1810-mobile-blind-s17/` for v1 and
`experiments/runs/20260803-1845-mobile-followup-retry-s17/` for the frozen two-arm follow-up. The
batch-63 arm was the unique AST-exact winner and improved over v1 by 24/756 examples, or 3.17
percentage points, clearing the preregistered one-point promotion margin. No safety metric was used
for selection because this population has no safety-denominator rows. The int8 score was measured
only after selection under a separate frozen deployment-retention protocol; it did not select or
rename the candidate. Its paired gate fixed eight frozen-reference errors, regressed three, and
passed at 607/756 with zero generation failures or truncations. A same-host FP32 control scored
603/756, so the four-row int8-over-FP32 difference is diagnostic rather than a general quantization
claim.

These numbers establish only that the small SFT probe passed its preregistered development gate on
this all-`CALL` Mobile Actions population. The zero-denominator safety fields in the result do not
measure safety. They do not support claims about abstention, ambiguity handling, confirmation,
unseen or renamed schemas, contextual revisions, disfluencies, prompt injection, generality,
deployment readiness, or superiority to larger models. The 961 official Mobile Actions evaluation
rows remained opaque and unparsed.

### Larger-model comparisons and broader post-training result

The independently verified Qwen matched-baseline run
`20260803-2122-mobile-qwen05b-matched-s17` is imported and locally hash-bound. Its selected import
receipt and manifest are stored beside the run evidence; the 988 MB comparison checkpoint itself
is deliberately excluded from the source release.

| Model | BF16 parameters | Tensors | Parse valid | Schema valid | AST exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| BarunAction-35M candidate-v2 | 35,072,768 | — | 756/756 | 755/756 | 602/756 (79.63%) |
| Qwen matched baseline | 494,032,768 (0.494B, not 500B) | 290 | 755/756 | 754/756 | 663/756 (87.70%) |

The paired outcomes are 80 Qwen-only wins, 19 Barun-only wins, and 657 ties. BarunAction is 14.09
times smaller and retains 90.80% of Qwen's exact-match accuracy, but it trails by 61/756 rows, or
8.07 percentage points. Qwen produced 755 parse-valid and 754 schema-valid outputs; the 961 official
Mobile Actions rows remained untouched. The hypothesis that candidate-v2 beats a strong larger
matched baseline therefore failed on this development probe. Because this is still a reused,
all-`CALL` development population, it does not establish hidden-suite performance, safety,
generality, or a universal ordering between the model families.

An earlier preregistered, one-seed comparison adapted
`HuggingFaceTB/SmolLM2-360M-Instruct` revision
`a10cc1512eabd3dde888204e902eca88bddb4951` on the same 7,937 example IDs, canonical targets,
presentation order, one epoch, 126 optimizer steps, and final-only checkpoint rule. Its exact
storage-view-deduplicated parameter count was 361,821,120. Under that particular model and recipe,
SmolLM2 scored 0/756 AST exact, 0/756 schema-valid, and 227/756 parse-valid, with 407 truncations.
The strong Qwen counterexample makes clear that the SmolLM2 outcome is model-and-recipe-specific;
it cannot support a blanket larger-model-superiority or breakthrough claim.

A separate PRESTO stage showed that a 35M checkpoint could reach 10,620/14,288 derived Action IR
exact, 14,285/14,288 schema-valid, 0.97957 abstention F1, and 89/10,407 false calls. It catastrophically
regressed Mobile and was rejected. The frozen recovery/interpolation selector then found no joint
passer. Its closest research-only checkpoint reached 642/756 Mobile and 9,862/14,288 PRESTO while
missing the sole remaining PRESTO exact gate by 140 rows. It is not the released checkpoint. These
negative results prevent representing candidate-v2 as a broad contextual or safety-trained model.

## Input and output contract

The public runtime contract is versioned as `barunaction-local-prompt-v1` and
`barunaction-inference-result-v1`.

Callers must provide all of the following:

- a non-empty user request;
- a non-empty array of strict tool declarations;
- a JSON object for context, including `{}` when no context is available; and
- a timezone-aware ISO-8601 reference timestamp.

Tool declarations use `barunaction-tool-schema-v1`. Every declaration supplies `name`,
`description`, `arguments`, `required`, `additional_arguments`, and `side_effecting`. Every
argument supplies a JSON `type` and `description`; scalar enums and nested typed arrays/objects are
supported. Unknown schema fields, non-finite numbers, duplicate names, undeclared required fields,
and reserved role-token text are rejected.

Empty context is omitted from the rendered prompt. Non-empty context is canonicalized to strict
JSON and inserted as a `CONTEXT` section. Action IR is parsed without extraction, Markdown
stripping, coercion, or repair. A result contains exactly one parsed action or one deterministic
error. On successful parsing, `policy.execution_permitted` is always `false`.

## Python API

Install the repository in an isolated Python 3.10-or-newer environment, then use the hash-verified
checkpoint explicitly:

```python
import json
from pathlib import Path

from barunaction import BarunActionCompiler

root = Path.cwd()
compiler = BarunActionCompiler(
    root
    / "experiments/runs/20260803-1845-mobile-followup-retry-s17/essential/arms/batch63/checkpoint",
    device="cpu",
)
outcome = compiler.infer(
    request="Turn on the flashlight",
    tool_schemas=json.loads(
        (root / "examples/barunaction_tools.example.json").read_text(encoding="utf-8")
    ),
    context={},
    now="2026-08-03T20:00:00+05:30",
)
print(outcome.to_dict())
```

`BarunActionCompiler` defaults to the `candidate-v2` hashes. A compatible alternate checkpoint
must be passed with its own `expected_sha256` mapping; alternate hashes are intentionally not
labeled as `candidate-v2` in results.

## CLI and CPU smoke path

```console
uv run --python 3.11 barunaction verify \
  --checkpoint experiments/runs/20260803-1845-mobile-followup-retry-s17/essential/arms/batch63/checkpoint

uv run --python 3.11 barunaction infer \
  --checkpoint experiments/runs/20260803-1845-mobile-followup-retry-s17/essential/arms/batch63/checkpoint \
  --tools examples/barunaction_tools.example.json \
  --context examples/barunaction_empty_context.example.json \
  --now 2026-08-03T20:00:00+05:30 \
  --request "Turn on the flashlight" \
  --device cpu
```

An input, checkpoint, generation, parse, or schema failure prints structured JSON and exits with
status 2. A valid proposal exits with status 0 but still carries `execution_permitted: false`.

## Sandboxed demonstration

The simulator accepts already generated Action IR and can only append to an in-memory call log. It
has no network, tool handlers, calendar integration, message transport, or device-control path.
Without both explicit sandbox gates it records no simulated calls:

```console
uv run --python 3.11 barunaction simulate-output \
  --tools examples/barunaction_tools.example.json \
  --output '{"calls":[{"args":{},"tool":"turn_on_flashlight"}],"decision":"CALL","mode":"SINGLE"}'
```

`--authorize-sandbox --confirm-sandbox` permits only the in-memory log entry; it never permits an
external side effect.

## Release files

The float package contains `barun_config.json`, `checkpoint_manifest.json`, `model.safetensors`,
`tokenizer.json`, `LICENSE`, `NOTICE`, and `MODEL_CARD.md`. The runtime-bound int8 package replaces
the float weights and checkpoint manifest with `model.int8.pt` and `quantization_manifest.json`.
Exact hashes and immutable retrieval instructions are in `docs/barunaction-retrieval.md`. Data
provenance and split limitations are in `docs/barunaction-data-card.md`.

The public distribution identities are
`harshalsingh1223-gladium-ai/barunaction-35m/barunaction-35m-candidate-v2-float:v0`,
`harshalsingh1223-gladium-ai/barunaction-35m/barunaction-35m-candidate-v2-int8-darwin-arm64-qnnpack:v0`,
and `harshalsingh1223-gladium-ai/barunaction-35m/barunaction-35m-candidate-v2-evidence:v0`.
The W&B project is public-read/team-write. A fresh credentialed redownload reproduced all 310
expected files and hashes; a separate unauthenticated query saw all three committed versions, and
an explicitly unauthenticated 140,304,464-byte float-weight stream reproduced SHA-256
`fdb95ccf58a095e0d321be998924318b35ee59a334f6dd97d8726d2cf80021d3`. The local release manifest
remains authoritative.

The v2 reproducibility trail is contained in
`experiments/runs/20260803-1845-mobile-followup-retry-s17/`: frozen preregistration, source snapshot,
pinned data audit, both arm configurations and curves, selection record, selected-checkpoint
manifest, aggregate metrics, and sample-level predictions. Large artifacts remain represented by
their hashes and retrieval instructions rather than being moved into the Python source tree.

A versioned CPU QNNPACK artifact is retained for Darwin ARM64 with PyTorch 2.13.0. It dynamically quantizes 87 internal Linear
weights to qint8 while retaining FP32 embedding, tied output head, and norms. Its payload is
58,615,358 bytes versus 141,440,943 bytes for the float checkpoint, a 58.56% reduction. Two frozen
smoke requests matched their expected Action IR and the float raw output exactly. The full frozen
development retention check passed at 607/756 exact with 755/756 schema-valid outputs and no
generation failures or truncations. On the measured ARM64 host the earlier int8 smoke was slower
than float, so this release supports no latency claim. See `docs/int8-quantization.md` for exact
hashes, commands, and limitations.

## Known limitations and next gates

The candidate has not passed the project's joint public-data target, independent hidden human
evaluation, target-device performance measurement, or safety co-gate. More decisively, the
imported Qwen matched result means the current larger-model-outperformance hypothesis
failed: parameter efficiency did not become higher task accuracy. The eager PyTorch quantization
implementation is version/platform/engine-bound and deprecated upstream; format v1 fails closed
outside its recorded runtime. “Research release” means the narrow package is runnable, documented,
hash-pinned, and independently retrievable—not that the broader breakthrough hypothesis passed. A
future candidate must be selected on development evidence without consulting sealed labels.

## License and attribution

The source code and released weights are licensed under Apache License 2.0. Candidate-v2 was
post-trained on `google/mobile-actions` revision
`e920309bc2acbc2e99a5e3201cf37df2b9fd9151`, licensed CC BY 4.0. PRESTO revision
`fa47167477453afebe698a287409514df5a7dadf`, also CC BY 4.0, appears only in separately identified
research evidence and rejected checkpoints; it did not train candidate-v2. Preserve the bundled
`LICENSE` and `NOTICE`, including upstream attribution, when redistributing either model artifact.
