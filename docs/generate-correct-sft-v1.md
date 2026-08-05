# BarunAction-35M Generate-Correct SFT v1

Status: **immutable CPU-prefreeze proposal only**. Run
`20260805-0230-action-correction-forge-screen-s17` does not authorize population
materialization, a model load, CUDA, training, a JarvisLabs resource, human-label access, or a
release change. Its typed contract is `configs/action_correction_forge_screen_v1.json`.

## Decision

The next generator-side question is whether the same BarunAction-35M weights can learn to inspect
one draft and correct an argument-binding error in a second pass:

> Joint direct and draft-correction SFT will let one 35,072,768-parameter BarunAction-35M
> checkpoint recheck argument bindings in a second pass and outperform matched direct
> continuation, masked value reconstruction, frozen candidate-v2, and untrained two-pass controls
> on a hypothesis-fresh internal synthetic screen without increasing unsafe actions.

The primary treatment is `C2`, the two-pass use of arm C. A frozen hierarchy may retain B or A as
a cheaper internal fallback if C2 fails and that arm independently clears its registered margin
and safety gates. Otherwise candidate-v2 remains the release checkpoint.

No result from this template-generated internal screen can support a breakthrough, larger-model
superiority, independent-component confidence statement, or replacement of the public release.

## Why this is distinct

This proposal changes the generator's supervised task and inference protocol. It does not reopen
any closed experiment:

- It does not apply the Month-Boundary Counterfactual transform, reuse the v3 selection or
  confirmation shadows, or rescue a v3 checkpoint.
- It emits canonical Action IR v1 directly. It does not emit PlanIR, use placeholder-v2, invoke a
  PlanIR compiler, or access the closed PlanIR screen.
- It produces one draft and at most one correction. It does not generate K=8 candidates, train a
  verifier or ranker, use a GVS population or receipt, or rescue GVS-v1.
- Simulator and single-fault certification logic may be reused only as hash-pinned, CPU-only data
  construction primitives. Reusing a tested primitive does not import a GVS population or revive
  the GVS experiment.

## Non-negotiable evidence limits

`D-internal` is new only in the narrow sense that its programs and rendered bytes must be created,
split, and frozen for this hypothesis before any model output. It is deterministic synthetic data,
not independently human-authored evidence. Its planned 4,096 rows are a nominal denominator, not
4,096 independent clusters or effective components. No power or uncertainty claim is registered.

The 5,744 replay rows are old construction labels. They may appear only as direct Action IR
training/replay. They are not fresh data, selection data, product evidence, PlanIR targets, or an
MBCF artifact. They derive from the frozen 5,745-row construction file with SHA-256
`800a3ba0a7f0215e5cf95c77c32e47abed48b2f07f375954b0e0b08296a33d10`; the known bad-label row
`mobile-actions-04894-6bd7643b1bc95697` must be excluded.
The immutable source membership/order, exclusion ledger, and accepted membership/order receipts
are respectively
`b70f6dfe79cfe49792b6e6f5d414fd509c889d18101f75cd8df6d4e4f146c588`,
`7593b556b9c096308ca1a9e9d9b89627ebf19710da930bd4dd5794c58879616a`, and
`27f9acb8905db132420ddb5d11a20bed138f2e2927aec41672f6852b3895452d`. A later materializer must
reproduce all three; it may not derive a new replay membership.

Longer correction and reconstruction inputs make exact input-FLOP matching impossible. The causal
budget claims are deliberately limited to identical source IDs, complete gold targets, supervised
target-token totals, example counts, and optimizer updates after materialization. Those are
requirements, not established facts. Input-token and training-compute totals must be reported by
arm and may not be called matched.

## Planned population boundary

No population currently exists. CPU materialization would have to produce:

| Role | Planned rows | Permitted use |
| --- | ---: | --- |
| `T-synth` | 16,384 | New deterministic program-first training sources |
| `T-replay` | 5,744 accepted from 5,745 | Shared direct Action IR training replay only |
| `D-internal` | 4,096 nominal | Hypothesis-fresh internal synthetic screen only |

`T-synth` should balance thirteen sandboxed action families plus `ABSTAIN`, `CLARIFY`, and
`CONFIRM`. Programs, world states, tool schemas, expected state transitions, fault eligibility,
renderers, and quotas must freeze before text rendering. Text must come only from the pinned
deterministic renderer; no model or human may author, paraphrase, label, or adjudicate a row in this
internal population.

The T/D allocation must be role-exclusive before rendering by the connected closure of semantic
program skeleton, renderer or template macrofamily, entity pool, temporal construction and
timezone pool, schema-presentation family, context-state family, source batch, and all declared
duplicate lineage. Exact, normalized, delexicalized, and pinned near-duplicate scans must close
jointly. If exact quotas cannot survive certification and closure, the materialization fails; rows
may not be adaptively replaced.

`D-internal` may contain no lineage from the old 7,937-row fit, reused Mobile 756, v3 selection or
confirmation, PlanIR screen, PRESTO development population, GVS populations, or the Mobile 961
official evaluation. Because the official 961 prompts remain opaque, this internal screen cannot
claim that an overlap scan against them occurred.

## Matched training views

Every fit starts from the exact BarunAction-35M candidate-v2 checkpoint and uses full-parameter
completion-only SFT. The three seeds are 17, 29, and 43, producing exactly nine fits.
The typed config binds candidate-v2 model SHA-256
`fdb95ccf58a095e0d321be998924318b35ee59a334f6dd97d8726d2cf80021d3` and canonical
BarunLM-35M base-model SHA-256
`f2a7c88b9f2c2e3584809081407ab136795d82e30e89b730e007781c45d01447`, together with the exact
configuration and tokenizer identities. Binding a known identity is not model access.

Each fit is planned to contain:

- 5,744 identical direct replay rows;
- 16,384 identical direct synthetic rows; and
- 16,384 arm-specific synthetic rows.

The total is 38,512 rows. With one epoch, batch size 64, `drop_last=false`, and no gradient
accumulation, this is 602 optimizer updates with 48 rows in the final batch.

All arm-specific targets must be the same complete canonical gold Action IR for the same source.
This makes target-token equality testable without padding or claiming equal input compute.

| Arm | Arm-specific input | Target | Ordinary evaluation |
| --- | --- | --- | --- |
| A | Original direct prompt repeated | Complete gold Action IR | One pass |
| B | Original prompt plus one semantically masked Action IR field | Complete gold Action IR | One pass |
| C | Original prompt plus either an exact draft or one certified semantic-fault draft, 50/50 by a frozen hash rule | Complete gold Action IR | Weights evaluated as C1 and C2 |

Arm C's exact split is planned as 8,192 exact-draft and 8,192 certified single-fault rows per fit.
Those counts and every fault certificate remain unverified requirements until materialization.

The proposed fixed optimizer is AdamW in BF16 with response-only cross-entropy, learning rate
`5e-5`, betas `0.9/0.95`, epsilon `1e-8`, weight decay `0.1`, gradient clip `1.0`, exactly 18
warmup updates, cosine decay to 10%, maximum sequence length 2,048, deterministic math SDPA, no
packing, no truncation, no early stopping, and unconditional final checkpoints only. Eighteen of
602 updates is approximately 2.99%; that fraction is descriptive and the integer step count is
authoritative. Materialization must prove all token and row claims before a later config can bind
this recipe.

## Causal and no-fit roles

The contract registers all of the following roles. Only A, B, and C are fits.

| Role | Weights | Passes | Purpose |
| --- | --- | ---: | --- |
| `B0` | Canonical BarunLM-35M | 1 | Fresh base probe needed for the standing +25-point SFT gate |
| `G0` | Candidate-v2 | 1 | Frozen incumbent |
| `G02` | Candidate-v2 | 2 | No-fit control for an untrained correction prompt and fallback |
| `A` | Arm A | 1 | Matched direct-correction SFT control |
| `A2` | Arm A | 2 | Process-only two-pass control without correction training |
| `B` | Arm B | 1 | Masked value-reconstruction auxiliary treatment |
| `C` | Arm C | 0 | Trained weights; not itself a scored deployment mode |
| `C1` | Arm C | 1 | Correction-training effect without the second pass |
| `C2` | Arm C | 2 | Intended Generate-Correct treatment |

`B0` is no fit and cannot run early. It may decode `D-internal` only after D has been frozen, all
nine final checkpoints have been made immutable, and the screen phase has opened. The same phase
barrier applies to every role: no D loss, prediction, or label may influence training, early
stopping, checkpoint selection, a seed, a threshold, or a data mixture.

## Two-pass input and fallback

Pass 1 uses the existing direct Action IR prompt and generates one draft. Pass 2 uses the same
checkpoint and includes the original request, context, reference time, tool schemas, draft, and a
compact diagnostic object recomputed only from the draft by a pinned runtime.

The complete allowed diagnostic vocabulary is:

- `parse_valid`;
- `schema_valid`;
- `policy_status`;
- `simulator_status`; and
- sorted, unique stable `error_codes`.

Gold actions, gold values, targets, labels, expected answers, oracle results, fault kinds, and fault
certificates are forbidden. Training-time fault certification establishes that a constructed draft
is useful supervision; it is not model-visible and does not exist at inference. The current config
sets runtime recomputation verification to false because no runtime integration has been built.

The deployed choice is deterministic:

1. return pass 2 only when it is parse-valid, schema-valid, and a policy-conformant proposal;
2. otherwise return pass 1 only when it meets the same predicate; and
3. otherwise return canonical `{"decision":"ABSTAIN"}`.

Exactly two passes are allowed for G02, A2, and C2. There is no retry, candidate set, selection
model, tool call, external side effect, or silent repair. Policy conformance checks only the
CALL/CONFIRM and side-effect proposal shape; `execution_permitted` is always false and this
predicate never authorizes an external action.

Pass-1 and pass-2 maximum-new-token caps are unresolved in this CPU proposal. Both are serialized
as null with `decoding_caps_resolved=false`. A token-length audit must select and bind exact static
integer caps before any later config can authorize model access; outcomes may not choose them.

## Frozen internal gates

Any missing, malformed, schema-invalid, truncated, failed, or absent prediction is wrong. The
complete conjunction for C2 is:

1. Relative to `B0`, schema validity is at least 95% and exact execution improves by at least 25
   points, satisfying the standing small-SFT probe gate.
2. Mean exact execution improves by at least 5 points over G0 and by at least 3 points over each of
   A, B, A2, and G02.
3. Strict Action IR exact improves by at least 3 points over A and B.
4. C2 improves exact execution by at least 2 points over C1.
5. Argument-value exact improves by at least 5 points over A and B.
6. C2 has a positive comparison against both A and B in at least two of three seeds and in at least
   12 of 16 registered descriptive strata.
7. Correct-draft retention is at least 98%, certified one-fault recovery is at least 40%, fixes
   outnumber regressions by at least 3:1, and no registered stratum regresses by more than 2 points.
8. Parse validity is at least 99.5%, schema validity at least 99%, and the false-action increase no
   more than 1 point.
9. Catastrophic unauthorized actions, missing predictions, generation failures, and truncations
   are all zero.
10. On a separately frozen MacBook Pro M5 Pro protocol, p95 latency is no more than 2.5 times
    candidate-v2, peak RSS no more than 1.25 times, and checkpoint size no more than 1.01 times.

These are exact practical screen thresholds, not power-backed confidence statements. The config
records `evaluated=false` and `passed=null`.

There is an explicit promotion blocker even if C2 passes every internal threshold. The standing
synthetic-tranche gate also requires losing no more than 2 points on an official development
population. This contract authorizes neither the reused Mobile 756 nor the official 961, and no
new official development population exists. It therefore records the retention half as
`evaluable=false` and `passed=null`. The internal screen can reject or retain a research direction;
it cannot clear the standing promotion gate or replace candidate-v2.

The three seeds are replication, not a checkpoint-selection budget. If an arm passes the frozen
hierarchy, only its predesignated seed-17 final checkpoint may be retained as an internal research
preview, and seed 17 must itself pass every applicable arm-specific quality margin plus every
validity, safety, MacBook Pro M5 Pro latency/RSS, and artifact-size hard gate. A passing three-seed
mean cannot rescue a losing seed-17 checkpoint. Seeds 29 and 43 provide replication evidence only.
Weight averaging, best-seed choice, a post-D refit, and a tenth fit are forbidden.

If C2 fails, B is internally retained only if it beats G0 by at least 3 points and A by at least 2
points while satisfying every applicable safety and validity gate. If B also fails, A is retained
only if it beats G0 by at least 3 points and satisfies the same gates. There is no post-result
mixture, on-policy rescue, threshold edit, seed choice, or correction-prompt edit.

## CPU gates before any later model proposal

This run may implement and audit CPU-only contracts, but a later model experiment requires a new
immutable config and run ID. Editing this config can never authorize it. Before even proposing a
fresh resource, the later record must bind:

1. all T/D programs, states, schemas, renderers, prompts, targets, memberships, and hashes;
2. the complete simulator, fault certifier, Action IR parser, diagnostic renderer, evaluator, and
   fallback runtime identities;
3. exact 50/50 clean/fault counts, per-kind eligibility, coverage, and 100% reference state
   transition evidence;
4. joint lineage and duplicate closure with role-exclusive provenance;
5. exact per-arm row, source, target-token, input-token, and optimizer-step audits;
6. a no-gold runtime diagnostic proof, complete token-length audit, and static decoding caps;
7. a phase-firewall test proving all nine checkpoints freeze before the first D decode;
8. complete CPU tests, formatting, source snapshot, dirty-patch receipt, and provider-transform
   rehearsal; and
9. an independent adversarial review of all preceding evidence.

Only then may a separate preregistration decide whether one fresh, exact-ID-controlled H200 is
responsible. This config itself keeps `launch_authorized=false`, `model_cuda_authorized=false`,
`population_materialized=false`, `jarvislabs_resource_creation_authorized=false`, and
`human_label_access_authorized=false`.

## Firewall and maximum claim

The current zero-access contract is exact: zero reused-756 rows, official-961 rows, v3 selection
rows, v3 confirmation rows, PlanIR screen rows, PRESTO development rows, GVS population rows, and
private human labels may be read.

At most, a later passing run could report hypothesis-fresh internal synthetic mechanism evidence
and prepare a research preview. Candidate-v2 remains the public release checkpoint. Fresh
model-independent human selection and confirmation, matched larger-model adaptation, and the
headline safety and simultaneous-bootstrap protocol remain mandatory before any breakthrough or
larger-model statement.
