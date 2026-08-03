# BarunAction-35M candidate-v2 research-release data card

Status: data provenance for the hash-pinned narrow research release. The development score is not
an independent test or a broad release-quality claim.

## Dataset and license

`candidate-v2` was supervised-fine-tuned only on `google/mobile-actions`, pinned to revision
`e920309bc2acbc2e99a5e3201cf37df2b9fd9151`. The pinned `dataset.jsonl` SHA-256 is
`91d251ee958cfd295af6c4504c236a3a1ad19517de240c3bc680bacfcbf7e7d9`. The source is licensed
CC BY 4.0; downstream distribution must preserve the applicable attribution and license terms.

The source contains 9,654 rows:

- 8,693 internal-training rows were eligible for local grouped train/dev derivation;
- 7,937 derived rows were used for SFT;
- 756 derived rows were used only for development evaluation; and
- 961 official evaluation rows remained opaque and unparsed.

No synthetic, preference, teacher-distillation, or RL data was used for this candidate. The
selected v2 arm trained from the same pinned BarunLM-35M base and the same 7,937-row training
manifest as v1, using batch size 63 for 126 optimizer steps. It was not continued from v1.
Dynamic int8 quantization used no training examples and did not create a new candidate.

## Derivation and hashes

Rows were grouped using ordered gold tool signatures, entity-delexicalized user templates, and
connected components from verified near-duplicate edges before assigning development fold 0 of 10.
This is stronger than a random row split, but it is not proof that the development population is
independent of every training regularity. In particular, all rows share one schema signature and
13-token fragments cross the split.

| Derived artifact | Rows | SHA-256 |
| --- | ---: | --- |
| Training JSONL | 7,937 | `131473ccb5bfb51cac0439b42159e72ec4c598025e50364a52b122b056c2e84e` |
| Development JSONL | 756 | `988bdce5874d1f1a775feeb5ba2b58cd2bdc128f57e73cb9a63d535fae7c1d55` |
| Audit JSON | — | `dc756f97c0a7ef706ec8ffefe2d57cf7e16d75a6ccd932f906d16b6a6ee2f83c` |

The original complete audit is
`experiments/runs/20260803-1810-mobile-blind-s17/remote/data/audit.json`; the follow-up bundle keeps
the same audit under
`experiments/runs/20260803-1845-mobile-followup-retry-s17/essential/data/audit.json`. It records
split membership hashes, duplicate and overlap scans, distributions, tokenizer revision, length
audit, and the machine-checkable official-evaluation firewall.

## Transformation

The adapter serializes each source request and seven available Mobile Actions tools into Action IR
v1. Ordered source call lists map to `SERIAL` because the source supplies no dependency or parallel
annotation. Source `NOW` values are timezone-naive and were preserved during training rather than
given a fabricated offset. No row was truncated or dropped.

Every parsed internal-training example maps to `CALL`. The data contains single and multi-call
requests for calendar events, contacts, Wi-Fi settings, email, maps, and flashlight controls. It
does not contain labels for `ABSTAIN`, `CLARIFY`, or `CONFIRM`, nor examples explicitly labeled as
unsafe, ambiguous, out-of-scope, adversarial, or confirmation-required.

The local runtime is intentionally stricter than the training source: it requires a caller-supplied
timezone-aware `NOW` and explicit JSON context, and it treats all model calls as proposals behind
external policy gates. That runtime policy does not add missing model competence.

## Leakage and evaluation firewall

The 961 official evaluation rows were counted and source-file-hashed but their prompts, schemas,
targets, labels, token lengths, summaries, and overlaps were not parsed or computed. They were not
used for training, prompt selection, checkpoint selection, or the reported development metrics.

The train/dev audit found no cross-split exact, normalized, delexicalized-template, connected-
component, or verified character-5-gram near-duplicate matches at the preregistered threshold. It
did find shared shorter token fragments, and MinHash-LSH candidate generation can have false
negatives; both facts are recorded rather than treated as clean-room proof.

## Appropriate interpretation

The 756-row development probe is an all-`CALL`, single-schema diagnostic. It can measure strict
serialization, known-tool selection, and argument extraction under that adapter. It cannot measure
false-action rate, abstention, clarification, confirmation, unsafe-call behavior, robustness to
renamed or unseen schemas, contextual revisions, disfluencies, or broad personal-action utility.

Do not use this data card or the candidate's score to claim safety, generality, deployment
readiness, or superiority to a larger model. Those claims require the independent human-authored
hidden suite and matched-adaptation evaluation described in `docs/evaluation-protocol.md`.

## Separately distributed PRESTO evidence

The release evidence bundle includes results from a different PRESTO-adapted checkpoint and the
frozen recovery/interpolation experiments. PRESTO is pinned to revision
`fa47167477453afebe698a287409514df5a7dadf` and licensed CC BY 4.0. Only English official-train and
development members were opened; 194,118 official-test rows remained opaque and unread. These
rows did not train the released candidate-v2 weights. They are included to make the rejected broad
post-training lane and the no-joint-passer decision inspectable, not to imply that candidate-v2 has
PRESTO competence.

Redistributed Mobile Actions and PRESTO-derived evaluation records remain under their upstream CC
BY 4.0 terms. Preserve `NOTICE`, cite the respective dataset creators, and do not strip the pinned
source revisions or the development-only labels from derived evidence.

## Candidate selection history

`candidate-v1` was step 62 from `20260803-1810-mobile-blind-s17` and achieved 578/756 AST exact.
The frozen follow-up compared exactly two arms on the same development IDs. The batch-63 arm
achieved 602/756 and the hard-mix arm achieved 566/756; batch-63 was therefore mechanically
promoted as `candidate-v2`. This development-only selection does not turn the development score
into independent test evidence.
