# BarunAction-35M Generate-Correct SFT v2 population prefreeze

Status: **immutable CPU-contract proposal only**. The typed contract is
`configs/action_correction_population_prefreeze_v2.json`, SHA-256
`b6998a825f403c8b93ba4fa3175348cf4331e6fa8ae7a5714ce2da1fdbec2001`. Run
`20260805-0410-action-correction-population-prefreeze-v2-s17` authorizes no population
enumeration or materialization, prompt or label read, model or tokenizer load, inference,
training, CUDA, network, JarvisLabs inventory or resource action, human access, or release change.

## Decision and boundary

Generate-Correct SFT v1 is closed before population or model access. Its proposed low-hash-bit
assignment produced 8,076 single-fault and 8,308 exact rows rather than the registered
8,192/8,192 split, and its parsed `draft` object could not preserve arbitrary model output. The
terminal feasibility record is
`experiments/runs/20260805-0406-action-correction-v1-feasibility-audit-s17/result.json`, SHA-256
`eec225c1f0e9a88c58446db638b89fa30fb0429ed222c1265e6e6fb225b8a13e`. V2 is a versioned
successor contract, not a repair or retry of v1. The v1 config and design document remain
immutable and hash-bound in the v2 config.

V2 asks a narrower question before paying for nine fits:

> After one full-budget seed-17 correction fit, does the same 35,072,768-parameter
> BarunAction-35M checkpoint improve on frozen one-pass, two-pass, and base controls on a
> disconnected 1,024-row synthetic futility probe?

This is an internal mechanism question. Synthetic T, P, or D evidence is not an official
development result, cannot promote candidate-v2, cannot change the public release, and cannot
support a larger-model or breakthrough claim.

## Exact population allocation

T, P, and D have the same sixteen descriptive strata: thirteen simulator actions plus `ABSTAIN`,
`CLARIFY`, and `CONFIRM`. The finite candidate universe and final selected quota are fixed per
role and stratum before rendering or model output.

| Role | Purpose | Candidates/stratum | Selected/stratum | Exact/fault per stratum | Selected total |
| --- | --- | ---: | ---: | ---: | ---: |
| `T-synth` | Training only | 2,048 | 1,024 | 512 / 512 | 16,384 |
| `P-futility` | Seed-17 futility only | 128 | 64 | 32 / 32 | 1,024 |
| `D-internal` | Final internal synthetic screen | 512 | 256 | 128 / 128 | 4,096 |

The corresponding candidate-universe totals are 32,768, 2,048, and 8,192. Candidate slots are
prefreeze metadata, not authored prompts or labels. This contract did not enumerate them.

The allocation uses strict canonical JSON: UTF-8, sorted keys, separators `,` and `:`, Unicode
preserved, and non-finite numbers forbidden. Let `A` be the 32-byte SHA-256 digest of the exact
canonical prefreeze-plan bytes and `C` the canonical bytes of
`{candidate_ordinal,catalog_coordinates,role,stratum}`. The exact identities are:

```text
slot_id     = SHA256(b"barunaction-gc-slot-v2\0"        + A + C).hexdigest()
roster_rank = SHA256(b"barunaction-gc-roster-rank-v2\0" + bytes.fromhex(slot_id)).hexdigest()
draft_rank  = SHA256(b"barunaction-gc-draft-rank-v2\0"  + bytes.fromhex(slot_id)).hexdigest()
```

Within each role and stratum, select the lowest `(roster_rank, slot_id)` pairs to the fixed quota.
Then sort only those selected slots by `(draft_rank, slot_id)`: the first half receives an exact
draft and the second half a certified single semantic fault. This corrects v1's imbalance without
choosing rows from outcomes. If any selected fault cannot certify, the entire population fails.
There is no backfill, reserve substitution, adaptive identifier, or changed assignment.

Role prefixes do not prove independence. T, P, and D must encode actual generator semantics and
must pass joint provenance-component plus exact, normalized, delexicalized, and near-duplicate
closure. P must be disconnected from both T and D, and all three memberships must freeze jointly
before any model output. A single cross-role lineage or duplicate connection aborts the complete
population. The allocation code and family catalog still require independent audit; nominal rows
are not claimed as independent or power-backed components.

`PAUSE_MEDIA` and `ABSTAIN` have no argument slot. Their fault half is a decision-fault control,
not argument-repair evidence. No pooled argument-repair claim may hide that distinction. Every
other selected fault must be exactly one certified semantic fault, with no second Action IR or
state-transition difference. A scorer for arbitrary model outputs remains mandatory.

## Exact raw-draft boundary

The CPU runtime at `src/barunlm/evaluation/action_correction_runtime.py` replaces v1's parsed
`draft` object with one exact string named `draft_raw`. It encodes the caller value once as strict
UTF-8, retains the detached decoded snapshot without Unicode normalization, and reuses that same
snapshot for transport, hashing, parsing, diagnostics, and fallback evidence. Invalid JSON,
duplicate keys, suffix-bearing output, incomplete output, noncanonical valid JSON, and decomposed
Unicode therefore remain observable instead of being silently canonicalized.

Drafts above 65,536 UTF-8 bytes, invalid UTF-8, non-string values, and any case-insensitive
occurrence of the seven pinned tokenizer control strings fail closed. The original canonical prompt
is capped at 1,048,576 bytes. State is extracted only from that original model-visible prompt.
The runtime accepts no caller-supplied parse, schema, policy, or eligibility booleans.

The model-visible diagnostics remain exactly:

- `parse_valid`;
- `schema_valid`;
- `policy_status`;
- `simulator_status`; and
- sorted, unique stable `error_codes`.

They are recomputed from `draft_raw` and the original-prompt state by the pinned parser, schema,
policy, and sandbox simulator. Gold values, labels, targets, expected answers, fault kinds, and
fault certificates are forbidden. The deterministic fallback derives both pass assessments
itself: eligible pass 2, otherwise eligible pass 1, otherwise canonical
`{"decision":"ABSTAIN"}`. Missing, failed, or truncated generations are never eligible. No real
tool executes.

The audited runtime currently accepts only the existing canonical `T-synth` and `D-internal`
prompt shapes. It does **not** accept a P-shaped prompt, and P must not be silently relabeled as D.
An exact disconnected P renderer and runtime integration are therefore explicit blockers. Static
pass-1 and pass-2 generation caps are also unresolved.

The accepted narrow runtime bytes passed the third independent audit after two preserved no-go
revisions. The attempt-3 receipt is
`experiments/runs/20260805-0418-action-correction-runtime-audit-s17/attempt-3-go.json`, SHA-256
`3d50e9eb0029fefb86301c8aefa08e679ecabea78d648ab503beff1e7b18a6ae`. Its scope is raw
transport only: `PassAssessment` and `TwoPassSelection` remain transport objects rather than
accepted scoring receipts, so this result cannot open P, scoring, model access, or launch.

## One-full-C17 futility ladder

A later, distinct, independently approved model config would train one complete seed-17 C fit
first. This is not a miniature or throwaway fit. It uses 5,744 direct replay rows, 16,384 shared
direct synthetic rows, and 16,384 C-specific rows: 38,512 examples total. With one epoch, batch 64,
no drop-last, and no gradient accumulation, it has 602 optimizer updates, a final batch of 48, and
18 warmup updates. It starts from exact candidate-v2 and retains the complete frozen optimizer,
renderer, target, and safety contract. This v2 CPU config does not authorize that fit.

After its final checkpoint is hash-frozen, P may open once for `B0`, `G0`, `G02`, `C1`, and `C2`.
The exact hard conjunction on 1,024 P rows is:

- `C2 - G0 >= 21` exact rows, at least 2 percentage points;
- `C2 - G02 >= 21` exact rows, at least 2 percentage points;
- `C2 - C1 >= 11` exact rows, at least 1 percentage point;
- `C2 - B0 >= 256` exact rows, at least 25 percentage points;
- at least 973/1,024 C2 raw outputs schema-valid;
- at least 128/512 controlled single faults repaired, reported separately as at least 112/448
  argument faults and 16/64 decision-only controls;
- at least 487/512 controlled correct drafts retained;
- no more than ten additional C2 false actions versus each of `B0`, `G0`, `G02`, and `C1`; and
- zero catastrophic unauthorized actions, missing predictions, generation failures, or
  truncations.

Failure closes the recipe without opening D and without training the other eight fits. Passing P
does not choose a seed, checkpoint, arm, recipe, threshold, or data mixture. The only permissible
continuation is the unchanged conditional continuation already bound in a model-run config frozen
before C17; no post-P config may change the experiment. D remains sealed until all nine final
checkpoints are immutable. Neither P nor D loss may affect training, early stopping, checkpoint
choice, or a rescue refit.

## State machine and unresolved gates

The current state is `CPU_PREFREEZE_ONLY`. Every transition in the typed config is unauthorized
here. A distinct successor and independent go decision are required to freeze a population; a
separate model run is required for C17; and immutable receipts are required before opening P or D.

The remaining blockers are substantive:

1. Freeze and audit actual role-scoped semantic catalog coordinates and prove T/P/D disconnection.
2. Implement P's exact canonical renderer and connect it to the raw runtime without treating P as D.
3. Materialize and hash the complete roster under a separate CPU/data authorization, then certify
   every assigned fault with no replacement.
4. Implement the arbitrary-output one-fault scorer and preserve decision-only controls separately.
5. Freeze exact tagged model-visible bytes, unique training-view IDs, tokenizer revision, complete
   token lengths, static generation caps, and no-truncation proof.
6. Bind and independently audit the evaluator, fallback, provider transform, phase firewall,
   runner, source snapshot, and failure-path receipts.
7. Establish a fresh official-development retention population before any model promotion.

Until all of these pass under a new immutable authorization, the only valid result is CPU contract
evidence. BarunAction-35M candidate-v2 remains the released checkpoint.
