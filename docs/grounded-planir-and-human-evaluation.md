# Grounded PlanIR and human-evaluation contract

Status: **closed without a model result**. The narrow construction-internal mechanism screen was
independently authorized for exactly one attempt on 2026-08-04. Attempt 1 failed closed during its
first remote launch-provenance validation because the JarvisLabs dependency preamble copied
`mobile-planir-screen.txt` into the frozen stage root. The error preceded config decode, the CPU
gate, Torch/CUDA/model access, construction-row reads, and model output. The exact run is
`20260804-0545-mobile-planir-construction-screen-s17`, with config SHA-256
`cca598e6f5a76a5e848a8b9a869cad779bbcbb17999217d2488111b22e2aa34c`. BarunLM-35M is the
canonical 35,072,768-parameter base model; BarunAction-35M candidate-v2 remains the current
post-trained release. No retry, promoted checkpoint, release change, human-label access, or
larger-model-outperformance claim is authorized.

## Frozen hypothesis and terminal decision

The tested proposal was **Grounded PlanIR**: ask BarunLM-35M to predict Action IR with a
compact typed placeholder for calendar datetimes, then compile that placeholder into the unchanged
Action IR contract with a frozen, deterministic, fail-closed compiler. The public product continues
to return Action IR. Any reported result is a model-plus-compiler system result and must separately
disclose compiler coverage, failures, cost, and the direct-Action-IR control.

The attempt produced no such model-plus-compiler result. Under the preregistered rule that any
operational failure ended the run and recipe, placeholder-v2 is closed. Keep the implementation as
CPU-only feasibility evidence and preserve the failure; do not patch the allowlist and relaunch it.

Do not launch the decisive experiment on the former temporal shadows or on a newly salted split
of them. Old construction rows may support training and a labeled internal mechanism ablation
only. They cannot become fresh evaluation evidence by being repartitioned.

Placeholder-v2 is intentionally narrow: its current compiler recognizes the seven frozen Mobile
tools, changes only `create_calendar_event.datetime`, requires empty context, and uses a
timezone-naive reference timestamp. It does not test contextual revisions, timezone-aware
resolution, renamed tools, or unseen schemas. It was eligible only for an internal calendar
mechanism screen, and its one permitted attempt is now spent and closed. It cannot consume the
decisive human confirmation suite or support the project's broad breakthrough claim. Under the frozen
counterfactual, a passing mechanism screen would have licensed work on a generalized
context/timezone/arbitrary-schema PlanIR contract; it would not have promoted a model.

## Why this hypothesis comes before more SFT or distillation

The choice is grounded in prior structured-prediction work, but the claimed transfer to this 35M
model remains a hypothesis. [IRNet](https://aclanthology.org/P19-1444/) separates an intermediate
representation from deterministic domain-aware lowering to reduce the mismatch between natural
language intent and executable syntax. [TripPy](https://aclanthology.org/2020.sigdial-1.4/) shows
the value of copying open-vocabulary slot values from input/context rather than predicting them
from a fixed value list. [Retrieve-and-Fill](https://aclanthology.org/2023.eacl-main.32/) similarly
separates scenario structure from span filling and reports strong size/latency/generalization
tradeoffs with base encoders. Together they motivate testing whether BarunLM-35M should choose a
small symbolic operation and input reference while deterministic code performs literal calendar
resolution.

[PICARD](https://aclanthology.org/2021.emnlp-main.779/) demonstrates that incremental parsing can
prevent invalid formal-language output, but constrained decoding could conceal raw model syntax
quality. The primary experiment therefore keeps unconstrained greedy decoding and counts every
parse/compiler failure. Grammar-constrained decoding may be reported only as a separate product
ablation. More mixture tuning repeats the already closed v3 direction, while immediate teacher
distillation would confound representation and teacher effects. Distillation becomes a separately
preregistered next stage only if placeholder-v2 first beats both matched direct controls.

## Construction-only feasibility evidence

The only dataset read for the schema audit was the former v3 construction manifest, SHA-256
`800a3ba0a7f0215e5cf95c77c32e47abed48b2f07f375954b0e0b08296a33d10`.
It contains 5,745 rows and 7,591 calls, including 2,070 calendar calls. A four-operation date
grammar plus grounded clock references represents 2,069/2,070 calendar calls and should
byte-exactly round-trip 5,744/5,745 complete Action IR labels (99.98%). The accepted-ID membership
SHA-256 is `27f9acb8905db132420ddb5d11a20bed138f2e2927aec41672f6852b3895452d`.

| Date operation | Construction calls |
| --- | ---: |
| `ABSOLUTE_DATE` | 1,333 |
| `MONTH_DAY_NEXT` | 357 |
| `RELATIVE_DAY` | 138 |
| `WEEKDAY` | 241 |

The sole expected rejection is `mobile-actions-04894-6bd7643b1bc95697`. Its reference timestamp
is Thursday, 2025-07-24; the request says “next Tuesday at 10:00 AM,” while the label is Wednesday,
2025-07-30. Preserve `temporal_reference_mismatch` as label-noise evidence. Never correct, coerce,
drop without accounting, or replace this row.

This audit says that the representation is implementable. It says nothing about model quality,
fresh generalization, safety, unseen schemas, or whether PlanIR beats direct Action IR.

## Quote-v1 feasibility baseline

The first, quote-based research profile changes only calendar datetime representation. Every other
argument stays byte-semantically identical to Action IR.

- `schema_version` is exactly `grounded-plan-ir-v1`.
- A span reference is exactly `{"quote":"<exact request substring>","occurrence":0}`. Matching is
  NFC-normalized, exact, case-sensitive, left-to-right, and non-overlapping. The occurrence is
  zero-based.
- Date expressions are exact discriminated unions for `ABSOLUTE_DATE`, `MONTH_DAY_NEXT`,
  `RELATIVE_DAY`, and `WEEKDAY`; weekday ordinal is only 1 or 2.
- A calendar datetime is exactly `{"date":<date expression>,"time":{"ref":<span reference>}}`.
- The compiler may use only the request, supplied reference timestamp, supplied empty v1 context,
  and the seven supplied Mobile tool schemas. It may not use a label, system clock, locale lookup,
  model retry, hidden state, network, or real tool.
- Invalid JSON, shape, reference, date, time, mode, tool, argument, past result, or final Action IR
  produces one stable structured failure. There is no defaulting, guessing, partial call, output
  repair, literal-datetime fallback, or wrapper-generated exact-match credit.
- The research profile accepts only timezone-naive `YYYY-MM-DDTHH:MM:SS` reference timestamps.
  It does not weaken the production API's timezone-aware timestamp boundary.

The oracle may inspect a training label only to derive a supervised PlanIR target. At runtime the
compiler receives no target. Oracle derivation enumerates prompt-derived candidates, chooses by
the frozen operation/shortest/earliest/lexical/occurrence order, compiles the result, and includes
the row only if the compiled canonical Action IR is byte-identical to the source label. Every
rejection remains in an exclusion ledger.

Quote-v1 is no longer the primary treatment. With the pinned tokenizer, its 5,744 accepted targets
contain 735,301 tokens, mean 128.012, versus 514,097 tokens, mean 89.502, for direct Action IR. It
lengthens every accepted target and is about 43% larger in target tokens. Preserve it as immutable
feasibility evidence and, if separately preregistered, a secondary representation baseline. Do not
train it as the primary C arm or call it compact.

## Placeholder-v2 primary treatment

The selected model-facing representation is `PLAN_IR_V2`. It keeps the complete Action IR target
unchanged except that a calendar `datetime` value is one exact typed placeholder:

- `@A:D00:T00` for an absolute date;
- `@M:D00:T00` for a month/day resolved forward;
- `@R:D00:T00` for a relative day;
- `@W1:D00:T00` or `@W2:D00:T00` for the first or second future weekday.

The input contains a deterministic, target-independent reference table derived only from the
NFC-normalized request and frozen reference timestamp before any label access. Its canonical line
is `REFS_V2 {"D":[["<exact quote>",0]],"T":[["<exact quote>",0]]}`. `D00`, `D01`, and so on index
date references; `T00`, `T01`, and so on index clock references. Each entry is an exact quote plus
its zero-based, case-sensitive, left-to-right, non-overlapping occurrence. References are
deduplicated across semantic alternatives and sorted by `(source_start, quote, occurrence)` while
preserving overlapping spans and every recognized distractor. The line is omitted exactly when
both arrays are empty. More than 100 entries in either table is an input-eligibility failure.

The compiler must regenerate the table from the request and reference timestamp and byte-verify
any supplied rendering or digest. It accepts only the complete canonical placeholder grammar,
canonical two-digit in-range IDs, and an opcode/reference/ordinal combination present in the
target-independent semantic enumeration. Literal datetimes, malformed IDs, incompatible
references, ambiguous resolution, extra text, and fallbacks fail the row. The resulting ordinary
Action IR must then pass the frozen tool-schema validator. The separate BarunAction policy wrapper
must assess authorization/confirmation and continue to forbid real execution; compiler success is
never policy approval. There is no repair, nearest candidate, retry, partial call, or
wrapper-generated abstention.

The construction-only tokenizer audit used the same 5,744 accepted rows from the former
construction-training corpus, before the frozen 4,596/1,149 train/screen split, and pinned
tokenizer SHA-256
`70ded9605fccd09c2340ca7e225361eab0ae8b4dbbb0d6e26343ab5183979db6`; it did not read
held-out rows. Exact distributions were:

| Representation | Target tokens | Target mean | Target p90 / max | Full-sequence mean | Sequence p90 / max |
| --- | ---: | ---: | ---: | ---: | ---: |
| A direct, no table | 514,097 | 89.502 | 138 / 196 | 341.787 | 420 / 660 |
| B direct plus table | 514,097 | 89.502 | 138 / 196 | 361.367 | 456 / 689 |
| C quote-v1 | 735,301 | 128.012 | not primary | not primary | not primary |
| C placeholder-v2 plus table | 506,062 | 88.103 | 135 / 192 | 359.968 | 453 / 689 |

The shared compiler/enumerator grammar makes the table nonempty for 2,637/5,744 rows and contains
5,241 date-reference and 3,581 time-reference instances; maxima remain seven date and six time
references per row. B's full-sequence total is 2,075,690 tokens and C's is 2,067,655. These values
supersede an exploratory prototype that omitted several compiler-supported distractors and was 22
prompt tokens shorter across the population.
Placeholder-v2 shortens all 2,069 representable calendar targets by 8,035 tokens total and leaves
every non-calendar target byte-identical. B and C receive identical table bytes and prompt-token
lengths; B-versus-C therefore isolates the representation while A-versus-B measures the table.
These are token and oracle facts, not model-quality evidence.

## Public-source audit

The following source decisions are frozen unless new primary evidence changes them:

- [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k/blob/26d14ebfe18b1f7b524bd39b404b50af5dc97866/README.md)
  at revision
  `26d14ebfe18b1f7b524bd39b404b50af5dc97866` is conditionally acceptable for **training only**.
  It is CC BY 4.0 and contains 60,000 synthetic, three-stage-verified examples over 3,673 APIs, but
  its gated terms must first be accepted. A remote materializer must hash the 96,116,570-byte data
  artifact, validate every answer against its declared schema, preserve attribution, and audit
  duplicates. It has no valid human selection or confirmation split. The generation and verifier
  method is documented in the
  [APIGen paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/61cce86d180b1184949e58939c4f983d-Paper-Datasets_and_Benchmarks_Track.pdf).
- [BFCL](https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard)
  at `ShishirPatil/gorilla@6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` is a public
  compatibility benchmark, not fresh decisive evidence. Its questions, answers, and evaluator are
  public; Live was collected in 2024; released rows lack author, source, batch, edit, and reviewer
  provenance; and upstream pretraining contamination is unknown. Score it only after candidate
  freeze, or clearly disclose any diagnostic tuning against it. The collection description is in
  the [official BFCL Live methodology](https://gorilla.cs.berkeley.edu/blogs/12_bfcl_v2_live.html).
- MASSIVE, TOPv2, and PRESTO are useful public diagnostics but do not satisfy post-release human
  evaluation freshness. PRESTO test remains once-only compatibility evidence; its development
  population has already influenced this project.
- The 961-row Mobile official evaluation remains sealed. It is final-evaluation-only after the
  checkpoint, compiler, evaluator, and claims are frozen; it is not a replacement for a fresh
  selection and confirmation protocol.

## Decisive human collection — not yet commissionable for placeholder-v2

The decisive generalized result requires two separately commissioned, model-independent human
collections. Do not commission or expose them until the generalized compiler, renderer, schemas,
context/timezone semantics, evaluator, claims, and access service are frozen.
No model or teacher may generate, paraphrase, translate, label, review, or adjudicate an evaluation
request. Author cohorts, collection batches, entity pools, and temporal-construction pools are
assigned to a role before authoring and never cross roles.

The selection collection may be smaller but must be powered and frozen before model scoring. The
once-only confirmation collection must contain at least 2,500 independent efficacy clusters and
1,500 independent no-call, ambiguity, confirmation, unsafe, or adversarial clusters, all authored
after the compared model releases. One row may have variants, but variants do not increase the
independent-cluster count.

Each label-free public prompt record must retain:

1. a pseudonymous author ID, source ID, UTC authoring timestamp, collection batch, and explicit
   license/consent plus an attestation of no model assistance;
2. raw request, supplied context, reference timestamp/timezone, complete tool schemas and their
   revisions, and an input-only eligibility decision;
3. schema/API family, paraphrase family, entity-source IDs, temporal-construction ID, authoring
   protocol revision, and immutable raw/transformed hashes;
4. exact/normalized/delexicalized/near-duplicate fingerprints plus the pinned exact-scan,
   near-scan, and scan-code hashes;
5. a model-visible input deterministically recomputed by a pinned renderer from only the public
   structural fields, plus exact renderer/tokenizer identities and recomputed input-token count.

Labels must never be fields in that public schema. Each eligible prompt instead binds to a
separately stored, HMAC-authenticated private envelope containing independent labeler and reviewer
IDs, their separately produced schema-valid Action IR, no-model-assistance attestations,
disagreement state, adjudicator evidence when required, and a recomputed output-token count. A
pinned renderer proves structural reproducibility but cannot prove that arbitrary human-authored
request/context/tool text is semantically free of leaked answers; the collector must audit that
content before sealing it.

Selection and confirmation must be disjoint by author, source, collection batch, schema/API
family where the preregistered lane requires unseen schemas, paraphrase family, entity source, and
temporal construction. Exact and near-duplicate scans run against all permitted training data and
all prior project populations. Eligibility and exclusion rules are correctness-blind and frozen
before labels or model outputs are scored. Excluded rows remain counted and never migrate or get
replaced.

An independent cluster is the connected component induced by declared cluster ID, author, source,
collection batch, paraphrase family, entity source, temporal construction, exact/normalized/
delexicalized input evidence, near-duplicate cluster, and schema family when that lane requires
schema separation. Merely assigning another `cluster_id` therefore cannot increase effective
power. Duplicate scanning must complete before the input-only eligibility decision; eligibility
must precede labeling; the private label envelope must be sealed before the passing selection
receipt; and the receipt must precede disclosure.

Confirmation labels remain access-controlled until an HMAC-authenticated selection receipt proves
that the exact eight preregistered gates passed. The receipt binds the selection and confirmation
population hashes, experiment config, candidate/checkpoint set, compiler, evaluator, source
revision, and code revision. A global HMAC-authenticated hash-chain records one final-label
disclosure per confirmation record across scoring sessions; the label is returned only after an
atomic compare-and-append succeeds. Any selection observation retires that selection version. Any
confirmation access retires that record permanently, and the first confirmation access retires the
population for any rescue experiment. A failed selection retires its unread confirmation rather
than licensing a rescue.

The repository implements protocol validation, signing, renderer/token recount, membership checks,
and ledger verification. It does not implement or claim encryption, key custody, secret storage, or
durable atomic persistence. Deployment must supply those external guarantees and a single-use
selection-scoring signer; a plaintext envelope outside that boundary, a forged callback, or a
reused signer key invalidates the evaluation.
Protocol validators take one canonical caller-detached snapshot and return detached JSON trees;
authorization never validates one container and later performs membership or retirement checks on
the caller-owned version. Only exact built-in lists/tuples are accepted for membership and ledger
inputs, so a custom sequence cannot present different values to iteration and membership checks.

## Stage 1: matched internal mechanism screen

Before fresh human collection, placeholder-v2 may be screened only as a labeled internal ablation
on a preregistered, group-disjoint split of the former v3 construction-training population. Restart
every arm from canonical BarunLM-35M. The minimum causal comparison is:

1. A: direct Action IR;
2. B: direct Action IR with the same prompt-derived grounding table shown to the treatment;
3. C: placeholder-v2 Grounded PlanIR plus the frozen compiler.

The split may claim disjointness only for fields the old manifest actually carries: example ID,
the connected closure of source `cluster_id` and `family_id`, and exact prompt/target content. It
lacks independently auditable author, entity-source, generator-template, and temporal-construction
lineage. Moreover, the full construction labels already informed the PlanIR grammar and oracle
audit. The screen is therefore a retrospective, training-disjoint mechanism check—not a fresh
hypothesis test—even though its A/B/C model fits must never train on their screen rows.

Use identical eligible IDs, semantic labels, presentations, seed set, optimizer budget, precision,
maximum-generation policy, unconstrained greedy decoding, and selection trials. Compiler errors,
parse errors, schema errors, truncations, missing outputs, and generation failures are full row
failures. Report target tokens and estimated training/inference FLOPs; add a token/compute-matched
sensitivity if representation lengths materially differ.

Every prediction receipt must bind the pinned prompt-renderer ID/revision/hash, canonical complete
prompt SHA-256, table SHA-256, model/checkpoint hash, decoding configuration, and raw output hash.
The compiler can verify supplied contract/table evidence, but only the generation/evaluator receipt
can establish which prompt bytes were actually presented to the model. A caller assertion such as
`prompt_contract=PLAN_IR_V2` is not that proof. Any prompt/control-marker ambiguity is an
input-eligibility failure before labels or model output are accessed.

For this screen, C must gain at least three exact-match points versus both controls, at least
five points on row-level argument-value exactness, show positive paired gains in at least two of
three seeds, lose no more than two points in any family or policy class, reach at least 99.5% raw
PlanIR parse validity and 100% compiled Action IR schema validity, and have zero truncations,
missing outputs, generation failures, or catastrophic actions. These thresholds decide only
whether to generalize the mechanism. Construction-internal scores are not fresh evidence, may not
be compared with candidate-v2 or Qwen as a product result, and may not unlock a human label.

## Stage 2: generalized system and fresh matched experiment

Only a passing Stage 1 screen licenses a new compiler that handles pinned arbitrary tool schemas,
explicit grounded-field annotations, supplied context, contextual revisions, timezone-aware
reference timestamps, and renamed/unseen-schema lanes while retaining fail-closed behavior. That
contract must repeat the CPU oracle, coverage, token, and security audits on training-only sources.
After it is frozen, the separately collected selection and confirmation populations above can be
used for a new matched A/B/C experiment. Confirmation must retain at least a two-point overall gain
versus both controls with the preregistered clustered-bootstrap lower bound above zero.

Only after generalized C passes fresh selection should a matched Qwen2.5-0.5B-Instruct PlanIR lane
or a verified distillation ablation begin. Both models must receive the same data, grounding tables,
compiler opportunity, teacher information, seed set, and selection budget. Public BFCL/PRESTO
diagnostics and the sealed Mobile final evaluation follow candidate freeze; they do not select the
candidate.

## Closure and next execution order

1. Preserve quote-v1 and placeholder-v2 only as oracle, token, compiler, and failure evidence. The
   sole remote attempt is documented under
   `experiments/runs/20260804-0545-mobile-planir-construction-screen-s17/attempt-1-launch-provenance-failure/`.
2. Treat the operational outcome as terminal for this run and recipe. Machine `463843` is paused
   and protected. Never retry, resume, rescue, alter the allowlist for a second attempt, or call the
   absence of a score evidence for or against PlanIR quality.
3. Retain BarunAction-35M candidate-v2 as the usable release. There is no active training
   experiment until a genuinely distinct hypothesis and new leakage-controlled population boundary
   pass research, cheap pre-model tests, immutable preregistration, and independent review.
4. Preserve the human prompt/envelope, signed-receipt, and access-ledger implementation as generic
   evaluation infrastructure. External key custody, isolated label storage, single-use scoring,
   and durable atomic persistence remain mandatory before any future fresh human population is
   acquired or disclosed.
5. Every future Jarvis launch rehearsal must simulate provider-side transformations exactly,
   including the managed requirements-file copy into the remote target, before attempt
   authorization. Operational lessons may transfer to a new hypothesis; this recipe and screen may
   not.
