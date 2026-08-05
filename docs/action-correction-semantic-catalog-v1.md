# BarunAction-35M correction semantic-catalog prefreeze v1

Status: **CPU design draft only; independently unaudited and nonauthorizing**. The exact typed
draft is `configs/action_correction_semantic_catalog_prefreeze_v1.json`, SHA-256
`68955b54e52820aab80f83450ef7aa97da9c50687f703d5b547571b1c1fbc866`. It authorizes no
candidate-universe enumeration, population materialization, existing prompt or label read, model
or tokenizer access, inference, training, network, CUDA, JarvisLabs action, held-out scoring, or
release change.

## Why this exists

The first balanced-roster prototype proved the 43,008-to-21,504 allocation arithmetic, but reused
one global placeholder catalog coordinate for every slot. Role, stratum, ordinal, and a hash can
make identifiers unique without defining what a user asks, what state exists, what program should
run, or which Action IR is correct. That is not a semantic population.

This draft freezes a candidate-level decoder before rendering. For each role, stratum, and ordinal,
an affine permutation over 2,048 positions decodes five actual axes: renderer, wording, entity,
temporal, and context variants. The complete product is `4 x 4 x 8 x 8 x 2 = 2,048`. T uses the
complete domain; P and D take fixed role-specific subsets through their preregistered permutations.
Every axis must change request, state, program, or presentation semantics. An ordinal, role prefix,
or hash alone never satisfies that requirement.

## Program and role boundary

Each future candidate must be program-first and must yield an exact request, simulator world,
program, canonical Action IR target, compiler/reference round trip, lineage payloads, and duplicate
views derived from the same frozen raw record. The thirteen simulator operations and the three
`ABSTAIN`, `CLARIFY`, and `CONFIRM` controls remain the only strata. No real tool executes.

T retains the existing imperative `context.state` shape. D retains the existing declarative
`supplied_context.available_state` shape. P receives a genuinely distinct
`environment.snapshot`/capability-query shape; it must be added to a versioned raw-runtime
successor and independently audited. P must never be relabeled as D merely to reuse an accepted
parser.

Role-specific grammar, schema presentation, time zone, lexicon, temporal epoch, context shape, and
source batch are real generator differences, but they are not automatically proof of
disconnection. A successor firewall must derive exact, normalized, delexicalized, and complete
verified-near evidence from frozen rows and abort on any direct or transitive T/P/D component.
This design makes no effective-component or power claim.

## Required integration order

1. Independently audit the design and its mixed-radix coverage before writing a full renderer.
2. Implement only bounded fixtures with source/runtime identity, program round trips, and exact
   candidate-coordinate commitments.
3. Create a distinct allocation-basis v4-or-later plan that binds this config and implementation
   for every candidate before computing any roster rank.
4. Obtain a separate CPU/data authorization before enumerating or materializing all 43,008
   candidates.
5. Certify every selected fault without backfill; derive the complete duplicate graph; freeze exact
   tokenizer rendering and generation caps; then perform another independent prelaunch audit.

Only a later model-run config may authorize the first seed-17 fit. This draft does not change the
public candidate-v2 release and cannot support a model-quality or breakthrough claim.
