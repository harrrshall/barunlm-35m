# BarunAction Generate-Verify-Select v1 research proposal

Status: **closed at the pre-collection population/inference-contract gate; never launch or rescue
GVS-v1**. No model load, CUDA run, training, JarvisLabs resource, or human-label access occurred or
is authorized. Independent reviews found the mechanism worth testing and
implemented or hardened much of the decoder, simulator, population/support/power, schema
presentation, verifier, calibration, shortcut, schema-partition, and pre-authoring population
plumbing. The final CPU patch-and-rerun review passed, but the later feasibility audit proved that
the requested effective D/S/C sample sizes cannot be constructed honestly under the current
lineage rules. GVS-v1 was distinct from temporal SFT and placeholder-v2 PlanIR, but it never
received a frozen model-experiment run ID, scientific config, dataset receipt, or compute
authorization. Its CPU code is archival nonauthorizing evidence; a future verifier test requires a
new versioned population/inference contract rather than altered identifiers inside v1.

## Decision

The highest-information next question is not whether another SFT mixture can move greedy exact
match. It is whether frozen BarunAction-35M candidate-v2 already places the correct Action IR in a
small candidate set but ranks a plausible argument-value error first.

> Given a frozen candidate-v2 generator, a tiny shared-backbone verifier trained on schema-valid,
> on-policy near-miss actions will select the correct Action IR from eight candidates more
> accurately than frozen candidate-v2 greedy decoding, generator-likelihood selection,
> easy-negative verification, and equal-example correction SFT, without increasing unsafe actions.

This keeps canonical Action IR unchanged. It does not use PlanIR, deterministic value repair,
gold-label access at runtime, real tool execution, or the retired temporal recipe.

## Why this comes next

The audited comparison is already close for the parameter gap: candidate-v2 scored 602/756
(79.63%) while Qwen2.5-0.5B-Instruct scored 663/756 (87.70%). Qwen has exactly 494,032,768
parameters, 14.09 times as many as BarunLM-35M's 35,072,768. Of 154 candidate-v2 failures in that
development comparison, 149 were value-only, and 76 of Qwen's 80 exclusive wins were value-only
BarunAction errors. Parse and schema validity were not the main gap.

The completed temporal intervention strongly repaired its intended cross-month failure but gained
only 1.60 and 2.44 overall points versus its two controls and caused retention/schema/truncation
failures. That makes another correction-SFT pass a necessary control, not the preferred treatment.
Placeholder-v2 produced no model evidence and is closed under its no-retry contract.

Primary research supports the mechanism while also setting limits:

- [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) showed that
  generating candidates and selecting with a learned verifier can improve performance and can
  scale more effectively with data than a finetuning baseline. The task differs, so this motivates
  the causal test rather than predicting a win.
- [SmartAD](https://aclanthology.org/2026.findings-acl.1349/) reports that small-agent distillation
  improves when it selects student-compatible correct trajectories and emphasizes action/final
  decision spans instead of weighting every token uniformly. GVS-v1 applies the same capacity-aware
  principle to candidate errors rather than importing full teacher trajectories.
- [SYNTHAGENT](https://aclanthology.org/2026.acl-long.570/) supports program-first synthetic tasks,
  mock tool environments, and explicit rubric rewards as a way to create diverse, stable tool-use
  training. GVS-v1 uses sampled programs as labels and models only for training-role paraphrases.
- [Distillation-Guided Policy Optimization](https://aclanthology.org/2026.acl-long.1751/) reports
  sparse rewards and unstable RL even for compact 0.5B-1B models without cold-start and continuous
  teacher guidance. BarunLM-35M is smaller, so RL is deferred until supervised candidate support
  and verifier learning are proven.
- [PA-Tool](https://aclanthology.org/2026.acl-long.948/) shows that schema alignment can help small
  models without retraining. That is useful later for renamed/unseen schemas, but the current paired
  evidence points primarily to argument values, so schema renaming is not the first intervention.
- [SCaTR](https://arxiv.org/abs/2604.16535) selects candidates with shallow classifiers over the
  last non-padding token from the penultimate transformer layer. Its reported models were
  1.7B--30B, so the fixed Barun linear/MLP controls test transfer rather than assume it.
- [Tool Calling is Linearly Readable and Steerable](https://arxiv.org/abs/2605.07990) studies
  270M--27B models and reports that its linear tool-selection circuit was absent at 270M and began
  emerging around 1B. This is direct reason to treat a hidden-state ranker at 35M as a cheap
  falsification probe, not the preferred treatment.

## Proposed 35.21M system

Candidate-v2's 35,072,768 generator parameters remain immutable. The intended decoder attempts
exactly `K=8` ordered slots and includes byte-identical greedy output at rank zero. Every slot is
parsed and schema-checked independently; invalid candidates are measured and never repaired. Raw
duplicates remain in the evidence, while `effective_k` counts distinct schema-valid lowered
semantic actions under a frozen order-aware canonicalization. Every row stays in every denominator.
A row-level generation failure is reserved for no
output, no valid candidate, or any truncation; a duplicate or invalid slot reduces effective K and
candidate validity but is not silently dropped or relabeled as a whole-row failure. The verifier
ranks only complete, nontruncated, schema-valid distinct lowered semantic actions. If none remains,
it records a failure rather than inventing an action.

The decoder CPU reference and golden vectors are implemented; they are not yet a frozen population
receipt. Before `D-support` can be exposed, one immutable audited artifact must bind candidate
count, greedy inclusion, beam/group allocation, diversity penalty, length normalization, min/max
new tokens, EOS and truncation behavior, raw-likelihood recomputation, semantic canonicalization
and deduplication, tie-breaking, checkpoint/tokenizer/runtime identities, and failure semantics.
The verifier contract must likewise bind prompt-plus-generated-token bytes, adapter toggling,
pooling token, attention mask/right padding, score serialization, exact-mask alignment, objective,
and semantic-identity tie rules. Candidate order and generator rank/likelihood must be hidden from
the learned verifier.

The verifier reuses the same Barun backbone and adds rank-8 LoRA only to every attention `q_proj`
and `v_proj`, plus one scalar head. With the canonical configuration (`dim=448`, 12 layers, seven
query heads, one key/value head, head dimension 64), the exact unique parameter budget is:

| Component | Calculation | Parameters |
| --- | ---: | ---: |
| `q_proj` LoRA | `12 * 8 * (448 + 448)` | 86,016 |
| `v_proj` LoRA | `12 * 8 * (448 + 64)` | 49,152 |
| Scalar head | `448 + 1` | 449 |
| New verifier parameters | sum | 135,617 |
| Complete generator-plus-verifier system | `35,072,768 + 135,617` | **35,208,385** |

The base weights are stored once. Generation runs with the adapter disabled; candidate scoring
runs with the verifier adapter enabled. No second 35M backbone is counted or required. The count
is conditional on exactly 24 `q_proj`/`v_proj` LoRA targets, rank eight, `bias="none"`, and no new
token, normalization, pooling, or other trainable parameter. CPU tests must enumerate the actual
modules, storage aliases, trainable gradients, and adapter-off byte-identical logits before the
number becomes an artifact claim.

## New data boundary

Old selection, old confirmation, reused-756, the official Mobile 961, and all PlanIR/v3 shadows are
forbidden. A new collection must close connected components jointly across `T-new`, `D-support`,
`S-new`, and `C-new`, using author, source, batch, schema family, program/template, entity pool,
temporal construction, and exact/near-duplicate lineage. A manifest from an external duplicate
scan is evidence to verify, not a trusted assertion.

Proposed populations, subject to provenance and power review before freezing:

| Population | Proposed size | Role |
| --- | ---: | --- |
| `T-new` | 24,000 program clusters | Training-only sampled programs, requests, positives, and hard negatives |
| `D-support` | 2,000 program clusters | Training-role candidate-support and shortcut audit; never fitted |
| `S-new` | 2,000 human clusters | Once-only selection, suggested 1,200 efficacy and 800 policy/safety |
| `C-new` | 4,000 human clusters | Sealed confirmation, suggested 2,500 efficacy and 1,500 policy/safety |

`T-new` and `D-support` sample canonical programs before language generation. Programs cover
context, revisions, disfluency, distractors, timezone-aware dates, multi-action cases, renamed
schemas, and `ABSTAIN`/`CLARIFY`/`CONFIRM` policy outcomes. Labels derive from the sampled program
and deterministic simulator, never from a teacher answer. Model-generated paraphrases are allowed
only in these training-role populations and must retain the sampled semantics under independent
verification.

`S-new` and `C-new` require model-independent human authors and labels. Both prompt bytes and label
bytes must remain access-controlled until the relevant candidate/checkpoint freeze; otherwise
held-out prompts can still guide development. Confirmation must be authored, labeled, encrypted, and
sealed before selection is disclosed. The repository's current human receipt is PlanIR-specific:
it binds `planir_parse_ge_995` and `compiler_sha256`, so it cannot authorize GVS. A GVS-specific
evaluator and receipt, external key custody, isolated prompt/label storage, a single-use signer, and
durable atomic retirement are blocking requirements.

## No-training support gate

No post-training is justified until all of these checks pass:

1. Included programs round-trip program to Action IR to deterministic state transition and back at
   100% exact. The current `barunaction/simulator.py` only blocks or records proposed calls; it is
   not this semantic simulator.
2. Every verifier-training negative is semantically wrong after canonicalization, remains
   schema-valid, and carries an explicit fault taxonomy. Hard negatives differ from the positive by
   exactly one typed fault. Invalid generated candidates remain decoder diagnostics and never enter
   the verifier pair loss.
3. Positive and negative length/token distributions are matched; order is randomized; mutation
   markers, generator fingerprints, and source shortcuts are rejected.
4. Length-only, syntax-only, validity-only, and tool-frequency baselines stay below 60% pair
   accuracy on `D-support`.
5. Token limits, truncation, licenses, source hashes, provenance fields, joint `T/D/S/C`
   duplicate-component closure, and preregistered minimum counts for context, revision, timezone,
   renamed-schema, multi-action, and each policy stratum pass before candidate-v2 is loaded.
6. Decoder bytes, D-support membership, strata, thresholds, and negative taxonomy freeze before
   the one-shot `D-support` candidate run. No D outcome may tune decoding or hyperparameters.
7. Frozen candidate-v2 then runs only on `D-support`. The current provisional aggregate thresholds
   are oracle pass@8 at least 85%, at least 10 points above greedy, recovery of at least 50% of
   greedy failures, and correct support on at least 80% of safety rows. Power review must add frozen
   per-stratum support floors and cluster-level confidence bounds before this becomes an
   authorization gate. Every safety row must also have at least one non-catastrophic fail-closed
   option; the availability target for that property is 100%.
8. Candidate evidence records all eight attempted ranks, per-candidate validity and truncation,
   canonical duplicates, effective K, and row-level failures. Missing/invalid rows cannot be
   excluded or retried after outcomes are visible.
9. A label-free M5 Pro ARM64 smoke test freezes both absolute and relative-to-greedy p95 latency,
   peak RSS, artifact size, end-to-end generation-plus-scoring work, and energy/thermal reporting
   before selection prompts or labels can become accessible.

A failed support gate ends GVS-v1 without a GPU training run. It would show that candidate support,
not ranking, is the bottleneck.

## Causal arms

Use seeds 17, 29, and 43 for the screen. Inference is clustered by prompt, so the primary
uncertainty calculation bootstraps prompt clusters and aggregates the three fixed-seed effects; it
does not count three predictions as three independent samples. A passing screen permits exactly
one T-only final fit with preselected seed 101 and the already frozen recipe. Selection labels may
not choose a seed or enter the fit.

- `G0`, deployed reference: frozen candidate-v2 greedy decoding with no candidate-set selection.
- `R0`, no-fit reference: frozen candidate-v2 generates eight candidates and selects by normalized
  generator likelihood.
- `A`, correction-SFT control: full-parameter response-only cross-entropy on positive `T-new`
  labels, then its own frozen eight-candidate likelihood selection. This measures whether another
  supervised pass is enough.
- `B`, easy-rank control: frozen candidate-v2 plus the exact verifier architecture trained on
  length-matched, schema-valid random or cross-row negatives.
- `C`, hard-rank treatment: the same verifier architecture trained on deterministic schema-valid
  single-fault and on-policy near-miss negatives.

`B` and `C` cannot literally share training candidate pools because the negative intervention is
their causal difference. They instead match prompt IDs, positives, pair counts, token bins,
optimizer, steps, and effort, while using the exact same frozen evaluation candidate pool. Positive
placement is randomized and rank/likelihood metadata is hidden. On-policy positives must come from
actual correct candidates rather than forced gold insertions. `B` versus `C` isolates the
information in hard negatives. `C` versus `A` tests discriminative verification against more
generative SFT. G0 prevents candidate selection from looking useful merely by beating a weak R0.

## Proposed selection and confirmation gates

Exact thresholds remain proposals until dataset power, latency, and simulator audits are frozen.
The recommended selection conjunction is:

- C strict Action IR exact at least +3 points versus G0, A, and B and at least +5 versus R0;
- paired 100,000-resample cluster-bootstrap lower bound above zero versus G0, A, and B;
- positive C-minus-A and C-minus-B gains in at least two of three seeds;
- recovery of at least 50% of available oracle pass@8 headroom;
- argument-value exact at least +5 points versus A and B;
- every preregistered stratum meets its minimum support and no action family or policy class is
  more than two points worse;
- selected-output parse and schema validity at least 99.5%;
- zero catastrophic unauthorized actions, truncations, missing outputs, or generation failures;
- one-sided upper confidence bound for false-action increase below +1 point; and
- frozen ARM64 latency, memory, artifact-size, and energy gates all pass.

Failure closes GVS-v1 without rescue, threshold changes, decoder changes, or seed selection. A
selection pass licenses the fixed seed-101 T-only final fit and one confirmation disclosure only.
Exact confirmation margins, false-action comparator, subgroup minima, and quantized-retention gate
must be frozen before `S-new` is opened. Confirmation must independently retain the registered gain
and safety/efficiency constraints before promotion.

A larger-model claim requires a separately matched Qwen2.5-0.5B-Instruct lane with the same new
training information, `K=8` budget, verifier opportunity, decoding policy, and selection rules. Its
checkpoint identity and predictions must be frozen in the receipt before the one-shot confirmation
disclosure; otherwise it needs a separate fresh confirmation suite. The official 961 rows remain
final-only after candidate freeze; they never select the system.

The intended release artifact is the float generator plus verifier adapter/head, accompanied by an
ARM64 int8 generator-and-adapter path only if compatibility, end-to-end retention, and the frozen
relative latency/RSS/energy ceilings pass before confirmation disclosure. Parameter count alone
does not make K=8 deployment cheap; all eight generations and verifier scores count in the compute
claim.

## Independent audit and implementation state

The first independent review passed only the distinctness of the hypothesis and the static
parameter arithmetic. It explicitly returned **no-go for model/CUDA/human access**. Subsequent CPU
work now recomputes parameter counts, binds K=8 sample/component and per-stratum support evidence,
implements the decoder, state-transition simulator, population firewall, power primitives, and
GVS-specific receipt/claim prototype, and retains exact-rational denominators. These are useful
fail-closed primitives, not a launch receipt. The rank-hidden verifier bridge, schema partition,
component-balanced shortcut controls, and final patched-code re-audits are now complete. Numerical
intervals/floors from a real roster, actual authored populations, external secret custody,
single-use signing, and durable atomic retirement still block access. A generic local rehearsal of
the observed JarvisLabs managed
requirements-file copy also simulates the provider transformation that ended PlanIR on a disposable
tree. Its caller-supplied CLI version and contract hash still require independent binding in a
future preregistration.

The final evidence is
`experiments/runs/20260805-0046-gvs-cpu-prefreeze-audit/result.json`, SHA-256
`6d948c05a14881c761990f070a96b7caeb87a23ac83bc4d38fba1bf9e5aa0468`.
The append-only correction `correction-20260805-0057.json`, SHA-256
`9c374760429b488f9c756d189b6c547876b3446ecbb702e08039ef92234bb3d0`,
clarifies that its displayed 85%/+10-point/50% values are provisional targets, not frozen gates.
The complete repository passed 1,338 tests, and the independent integration slice passed 365
overlapping tests; do not add those counts. Ruff, formatting, `py_compile`, and whitespace checks
passed. The final audit reproduced and closed forged program/membership commitments before this
record was frozen.

The implemented support evidence retains all eight attempted slots, measures invalid and truncated
outputs, preserves duplicates while deriving unique schema-valid `effective_k`, and makes any
no-output, no-valid-candidate, or truncation row fail the preliminary gate. The 57 support, 53
provider-transform, 145 combined, and 863 complete-suite counts below are historical
pre-implementation checkpoints and overlap; never sum them or present them as the current complete
suite. They are superseded by the current evidence above.

A separate read-only decoder audit found that the smallest repository-native v1 is ordinary Torch
beam search rather than a new Transformers dependency: rank zero calls the existing greedy
`BarunLM.generate`, ranks one through seven come from one zero-diversity beam group, and final
likelihood is recomputed from generated-token log probabilities. The CPU reference now implements
that design with no sampling or repair, `max_new_tokens=192`, EOS-only stopping, EOS included in
likelihood, explicit parent/token/rank tie rules, runtime/model/tokenizer identity, and synthetic
vectors. The CPU bridge now proves the structural prompt/candidate bindings, but a future
population still must bind its exact checkpoint, tokenizer, prompt, continuation tokens,
likelihood bytes, adapter-off identity, complete trace, and external custody receipt.

### 2026-08-05 CPU prefreeze delta

The CPU implementation now contains the proposed native K=8 decoder and likelihood selector;
bounded semantic simulator and certified single-fault construction; joint population firewall,
component-conservative support, and exact-rational power primitives; identity/renamed schema
presentation; decoder-to-simulator bridge; exact LoRA verifier; adapter-only serialization; and
two frozen-hidden calibration controls. The control sizes are exactly 449 parameters for a biased
linear projection and 29,793 parameters for a `448 -> 64 -> 16 -> 1` ReLU MLP, compared with
135,617 trainable parameters for the proposed LoRA verifier. These controls do not change the
35,072,768-parameter generator and cannot authenticate feature provenance by themselves.

The LoRA verifier's rank-free selection contract requires exactly eight slots. Each distinct valid
slot uses the SHA-256 of exact canonical lowered Action IR bytes as side metadata; invalid and
semantic-duplicate slots use `null`. Ties resolve by canonical identity, not list position. The
audited bridge proves exact prompt-plus-generated-token construction, canonicalization version,
order-independent semantic deduplication, generator-frozen state, and absence of rank,
beam-origin, likelihood, or list-position features. This is structural evidence only: it has not
seen a real T/D/S/C record or licensed model loading.

A separate GVS receipt/ledger prototype binds the proposed population, custody, generator,
ranker, tokenizer, decoding, rendering, presentation, simulator, candidate, prediction, scorer,
runtime, power, source, and ledger-service commitments. It HMAC-seals exact gate results and claims
the entire D/S/C population before labels may be parsed. Its in-repository validation passed
adversarial review, but external secret custody, a single-use signer, and an atomic durable
compare-and-append service remain required; an in-process Python callback is not durability.

SCaTR motivates the frozen-hidden controls but does not validate them at this scale: its evaluated
models were 1.7B--30B. A separate 2026 mechanistic study reported no linear tool-selection circuit
at 270M and emergence around 1B. BarunLM-35M is much smaller. Therefore the 449-parameter linear
and 29,793-parameter MLP arms are falsification/high-information controls, not favored treatments.
The one-shot D-support oracle gate decides whether ranking is viable. A failure closes GVS without
training and directs the next preregistration toward generator-side correction SFT; a pass licenses
comparison against R0 likelihood, both calibrated controls, the LoRA verifier, and matched
correction SFT. RL stays deferred.

### Bounded semantic simulator

The simulator must be a pure in-memory semantic oracle, not a phone or platform emulator. Its first
version is limited to reminder create/update, calendar create/reschedule, contact lookup, note
create, list add/check, route lookup, simulated message outbox, media play/pause, typed setting set,
and the `ABSTAIN`/`CLARIFY`/`CONFIRM` controls. The model-facing tool names and existing Action IR
schemas stay authoritative.

The immutable world contains only a pinned clock and IANA timezone, reminders, events, contacts,
notes, lists/items, places and an explicit route table, an outbox, a media session, and allowlisted
settings. It uses no wall clock, randomness, network, filesystem, subprocess, real connector, or
floating-point geography. Each case follows one independently checkable path:

```text
semantic program
  -> reference effect
  -> canonical Action IR
  -> existing schema/policy validation
  -> deterministic state transition
  -> terminal state and canonical observation
```

The intended implementation units are `sim_program.py`, `action_simulator.py`, and `gvs_faults.py`.
Every transition receipt binds input state, Action IR, effects, terminal state, and observation.
Controls and rejected batches never mutate state; multi-action execution is atomic. The initial
schema-valid single-fault taxonomy is wrong entity, wrong argument, temporal boundary/offset,
recipient/channel substitution, polarity flip, list-item substitution, operation substitution, or
wrong `ACTION`/`CONFIRM`/`CLARIFY`/`ABSTAIN` decision. A negative is admitted only when exactly one
semantic field or decision changes and the effect, observation, policy result, or terminal state
provably differs. Semantically equivalent alternative serialization is not a negative.

### Authorization and power boundary

A new GVS population manifest must jointly derive connected components across all `T/D/S/C` roles
and reject every cross-role component. Required pre-outcome strata are task class, expected outcome,
action family, and flags for context, revision, disfluency, distractors, timezone/relative time,
multi-action, renamed schema, unseen schema, and unsafe/adversarial content. Component IDs derive
from the sorted evidence graph; caller-supplied cluster IDs never define a denominator.

A separate GVS authorization receipt—not the PlanIR human receipt—must bind population and component
hashes, prompt/label commitments and custody policy, generator/verifier/tokenizer identities,
EOS/pad IDs, renderer/schema/simulator/policy bytes, complete candidate trace roots, all arm/seed
checkpoints and predictions, scorer/bootstrap/runtime identities, and the external one-shot ledger.
The whole `D`, `S`, or `C` population is atomically claimed before any private label is parsed. A
crash retires the whole population; partial-label rescue or a new session is forbidden. A selection
pass licenses only the already specified seed-101 T-only final fit.

The current 85% oracle, +10-point greedy gap, 50% recovery, 80% safety exact support, 60% shortcut,
population sizes, per-stratum floors, bootstrap settings, selection margins, confirmation margins,
and deployment ceilings are deliberately **not frozen**. An independent power specification must
set them from external baseline/discordance assumptions, effective-component sizes and dependence,
multiplicity treatment, target effects, power, and zero post-freeze attrition—never from observed
`D/S/C` outcomes. Structural requirements safe to freeze now are complete K=8 denominators,
byte-identical greedy rank zero, no outcome-dependent exclusion/retry, atomic one-shot retirement,
100% availability of a non-catastrophic fail-closed option on safety components, and zero
catastrophic unauthorized actions.

The live real-roster bridge now proves that the preliminary allocation cannot satisfy that rule.
With 16 rows and eight honest role-scoped schema-family IDs, the firewall derives eight components
of size two, and a minimum of nine fails exactly. A shared collection batch reduces those eight to
one; identifiers shared across roles cause an earlier cross-role rejection. Scaling the same
allocation to 2,000 nominal rows does not increase its maximum eight schema-family components. The
current schema/batch/source allocation is therefore rejected before collection; no model-quality
conclusion exists. Evidence is frozen in
`experiments/runs/20260805-0140-gvs-effective-component-audit-s17/result.json`, SHA-256
`f5285533105ccd867076d344955b10e9160200472bf0f41be5fe0ead127a0ba2`.

Separately, the offline human-collection packager and local SQLite retirement store passed
adversarial CPU reviews after closing concrete leakage, runtime-substitution, path, ownership,
hardlink, race, and rebinding defects. The exact combined record is
`experiments/runs/20260805-0149-gvs-collection-retirement-cpu-audit-s17/result.json`, SHA-256
`38b411c8378a36370cec3b512be27f8e23273feb9839a2fb1f2c2a77ee8c8b76`. Those modules
remain explicitly nonauthorizing: they provide neither real human provenance nor an isolated
signer/custody/anti-rollback service, and no collection has started.

The final feasibility review found that this is not a repairable eight-family allocation bug inside
v1. Under the current union rule, achieving the requested effective counts requires every row to
be a singleton on every author/source/batch/schema/template/paraphrase/entity/temporal axis. The
same identity schema cannot honestly span roles; one alias generator macro does not create
independent schema sources; and the planner's unique slots explicitly are not provenance. Six
thousand S/C singleton authors would also require at least 18,000 author/label/review task
completions before adjudication, while v1 has no reserve and does not account for repeated
labeler/reviewer dependence. The full audit is
`experiments/runs/20260805-0205-gvs-v1-contract-feasibility-audit-s17/result.json`, SHA-256
`27a1f8a536b2c2193091428d13e8b6bd543587fbdb56e48d21d2e76c82105dac`.

## Closure

1. Preserve candidate-v2, the CPU implementation, all audit defects/fixes, and the exact hashes
   above as nonauthorizing evidence.
2. Never collect GVS-v1 data, deploy its custody scaffold, claim `D-support`, load candidate-v2 for
   K=8, create a JarvisLabs resource, train a verifier, or tune identifiers to escape the result.
3. A future verifier experiment must use a new versioned contract with hierarchical ancestry,
   nullable inapplicable axes, role-blind whole-family assignment, a strict contamination graph,
   separately justified multiway/hierarchical inference units, a fixed identity-compatibility lane,
   annotator dependence, and predeclared reserve collection. That is not a GVS-v1 retry.
4. The active next experiment, if pursued instead, must be a genuinely distinct generator-side
   intervention with a fresh internal development boundary and its own immutable preregistration.

GVS-v1 is closed without a model-quality result. Its verifier mechanism remains scientifically
untested, and no larger-model or breakthrough claim follows in either direction.
