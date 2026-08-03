# Qwen2.5-0.5B matched Mobile Actions baseline

Status: completed on 2026-08-03; the scientific recipe was preregistered before any model-weight
download or development scoring, and the downloaded result was independently verified.

This lane measures a second independent larger-model architecture under the same one-epoch Mobile
Actions adaptation contract used for BarunAction-35M candidate-v2 and the existing SmolLM2
comparison. It is comparison-only evidence and cannot change BarunAction weights, rescue selection,
or any official-test claim.

## Frozen baseline identity

The baseline is `Qwen/Qwen2.5-0.5B-Instruct` at immutable Hugging Face commit
`7ae557604adf67be50417f59c2c2f167def9a775`. The repository is public and ungated, uses the
Apache-2.0 license, declares `Qwen2ForCausalLM`, and is supported without remote code by
Transformers 4.48.3.

The exact unique parameter count is 494,032,768, or 14.085936 times the 35,072,768-parameter
BarunLM-35M base. Three independent checks agree: the Hugging Face safetensors metadata, a
range-only sum of all 290 BF16 tensor shapes in the safetensors header, and this config calculation:

```text
151936*896
+ 24*((896*896+896) + 2*(128*896+128) + 896*896
      + 3*896*4864 + 2*896)
+ 896
= 494032768
```

The model has 24 layers, width 896, intermediate width 4,864, 14 query heads, two key/value heads,
and tied input/output embeddings. The runner loads the exact revision, hashes the downloaded
snapshot, counts unique storage views, proves the embedding storage tie, and aborts on any mismatch.

Primary sources are the pinned [model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/README.md),
[config](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/config.json),
[license](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/LICENSE),
and [immutable Hub metadata](https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Instruct/revision/7ae557604adf67be50417f59c2c2f167def9a775).

## Matched adaptation and decoding

The recipe presents the identical 7,937 frozen train IDs and canonical Action IR targets exactly
once, with no augmentation, teacher, preference, reward, packing, truncation, or dropped row. It
uses full-parameter response-only BF16 SFT, seed 17, microbatch 21, gradient accumulation 3,
effective batch 63, exactly 126 optimizer steps, AdamW at `2e-5`, 12 warmup steps, cosine decay to
10%, gradient checkpointing, and the final checkpoint only. The 756 frozen development IDs are
generated once with 192-token unconstrained deterministic decoding and scored by
`barun-mobile-actions-score-v1`.

Qwen's pinned vendor generation config enables sampling and a 1.1 repetition penalty. The runner
explicitly freezes `do_sample=false`, one beam, repetition penalty 1.0, and null temperature,
top-k, and top-p. This neutralization is necessary for plain greedy argmax and is recorded with the
predictions. Missing, duplicate, truncated, OOM, context-overflow, parse-failed, and schema-invalid
rows remain measured failures.

## Evaluation firewall and lifecycle

Only the frozen train manifest, development manifest, and adapter audit enter the stage. Their
required SHA-256 identities are, respectively,
`131473ccb5bfb51cac0439b42159e72ec4c598025e50364a52b122b056c2e84e`,
`988bdce5874d1f1a775feeb5ba2b58cd2bdc128f57e73cb9a63d535fae7c1d55`, and
`dc756f97c0a7ef706ec8ffefe2d57cf7e16d75a6ccd932f906d16b6a6ee2f83c`. The runner accepts no
source-dataset or official-evaluation path and requires the audit to prove all 961 official rows
remained opaque, unparsed, and unmaterialized.

The run uses one fresh H200 instance named with a `barun-` prefix. The exact-ID controller captures
every pre-existing resource as protected, binds only the newly returned ID into the attempt
preregistration, uploads the audited stage, downloads compact evidence, pauses only that exact ID,
and requires a final `Paused` query. Kroda, ID 463058, and every other pre-existing resource remain
untouched.

## Interpretation limit

The public development population contains only `CALL` examples and supplies no useful denominator
for false-action, abstention, clarification, confirmation, or unsafe-action rates. The adaptation
information is matched, but upstream pretraining and instruction tuning are not. A directionally
favorable result for BarunAction-35M would support only a narrow public-development statement, not
a blind, official-test, safety, multi-seed, or universal larger-model result. The observed result
was unfavorable, so no claim that BarunAction-35M exceeded this baseline is supportable.

## Completed result and decision

The single frozen run `20260803-2122-mobile-qwen05b-matched-s17` used all 7,937 training examples
once, completed 126 optimizer steps, generated all 756 development predictions, and had no
truncation, missing prediction, OOM, or infrastructure retry. Training took 73.129 seconds,
terminal generation took 18.842 seconds, and the scientific runner took 117.769 seconds. The full
controller lifecycle, including environment setup, artifact download, and pause verification, took
567.160 seconds. At the frozen H200 rate of INR 378.27 per hour, the lifecycle cost estimate is INR
59.59; the runner-only estimate is INR 12.37.

Qwen achieved 663/756 strict AST exact matches, 755/756 parse-valid outputs, and 754/756
schema-valid outputs. Argument-key F1 was 0.995791 macro and 0.996489 micro; argument-value F1 was
0.927409 macro and 0.945180 micro; tool macro F1 was 0.998431. There were zero catastrophic
unauthorized actions, but the public development set has no non-call denominator, so false-action
or safety-rate conclusions are unavailable.

BarunAction-35M candidate-v2 had 602/756 exact matches on the identical IDs. Candidate minus Qwen
was therefore -61 matches, or -8.0688 percentage points. The paired evidence contains 19
candidate-only wins, 80 Qwen-only wins, 583 cases both correct, and 74 cases both wrong (657 total
ties). The conclusion is `keep`: preserve this as valid second-architecture matched public
development evidence. It rejects any claim that the current BarunAction-35M candidate exceeded
this Qwen baseline under the one-seed protocol and cannot alter candidate weights or rescue
selection.

Independent verification recomputed every sample, accepted only sub-1e-12 derived-float drift,
verified the 494,032,768-parameter checkpoint and all bundle hashes, found no secret markers or
official evaluation artifacts, and confirmed fresh instance 463689 was Paused. Two initial local
verification attempts were preserved as incidents: one exposed a `safetensors` 0.8 iterator API
assumption, and one exposed tuple-to-list normalization across JSON serialization. Both were fixed
with regression tests before the successful clean rerun; neither changed predictions, labels,
sample scores, or scientific settings. The official 961 Mobile Actions evaluation rows were never
read or materialized.
