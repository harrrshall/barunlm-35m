# PRESTO user-revision taxonomy correction

This is a post-hoc evaluator-taxonomy correction for the completed
20260803-1920-presto-stage-s17 run. It does not revise that run's preregistration, immutable
essential bundle, recorded result, training intervention, or failed gate.

## Primary-source finding

Section 2, “User Revisions,” on page 2 of the
[PRESTO EMNLP 2023 paper](https://aclanthology.org/2023.emnlp-main.667/) explicitly groups four
dataset tags under the broader user-revision phenomenon:

- `correct-action`;
- `correct-argument`;
- `within-turn-correction`;
- `cancel-action`.

The [README at the pinned dataset revision](https://raw.githubusercontent.com/google-research-datasets/presto/fa47167477453afebe698a287409514df5a7dadf/README.md)
identifies revisions as a target PRESTO phenomenon, documents the raw `linguistic_phenomena`
field, and links the paper that defines the subtype grouping. The source revision remains
`fa47167477453afebe698a287409514df5a7dadf`.

These four exact raw labels form alias set
`barun-presto-user-revision-raw-aliases-v2`. Future v2 evaluator input is normalized for case and
separators, but the historical post-hoc audit may change only those four exact raw spellings in
the pinned evidence. Every other raw label must retain its legacy group.

## Preserved v1 history

The immutable narrow protocol remains at `configs/presto_taxonomy_posthoc_v1.json`, SHA-256
`64a02c519b259c76f3dcee00fef74c0221342797185773602cca72a7154e89d0`. It corrected only
`within-turn-correction` and is now explicitly superseded; it is not rewritten or presented as
the full paper taxonomy.

The durable protocol is `configs/presto_taxonomy_posthoc_v2.json`, SHA-256
`71a15ca2ad31914e21f5a9d75ea8ee68370085434d12ef6feb961da6670bb7ca`. It covers the complete
four-tag family and pins the v1 protocol hash as historical evidence.

The original scorer's v1 grouping recognized labels containing the literal substring
`revision`. None of the four paper-defined raw labels contains that substring, so all 4,140
English development rows were assigned to `other`:

| Raw tag | Rows |
| --- | ---: |
| `cancel-action` | 454 |
| `correct-action` | 230 |
| `correct-argument` | 389 |
| `within-turn-correction` | 3,067 |
| **Full revision family** | **4,140** |

The same v1 function fed focus-replay selection. Its immutable audit consequently records zero
revision-eligible rows, zero revision replay rows, and no original revision development bucket.
The original gate failed its required bucket-representation and hard-gap checks. That remains the
preregistered result. The independent bundle validator retains the legacy v1 function so the
original evidence remains reproducible.

## Separate post-hoc procedure

The v2 audit:

1. verifies both protocol hashes and validates the original bundle with the legacy v1 taxonomy
   plus external preregistration and JarvisLabs trust anchors;
2. requires the original gate to remain failed for the missing revision bucket;
3. verifies the pinned base and post-SFT sample-evidence hashes;
4. reclassifies only existing rows carrying one of the four exact raw aliases;
5. rejects every other taxonomy change and checks each raw-tag count independently;
6. reports base and post-SFT metrics for every raw revision tag;
7. recomputes headline buckets and the diagnostic gate without regenerating predictions; and
8. reports that neither the training intervention nor the original result was corrected.

Run it read-only with:

    uv run python scripts/audit_presto_taxonomy_correction.py \
      --bundle experiments/runs/20260803-1920-presto-stage-s17/essential \
      --preregistration experiments/runs/20260803-1920-presto-stage-s17/preregistration.json \
      --jarvis-record experiments/runs/20260803-1920-presto-stage-s17/jarvis.json

An optional output receipt uses exclusive-create semantics and must be placed outside the
essential bundle.

## Corrected diagnostic result

The audit reclassifies 4,140 rows in both base and post-SFT sample evidence. Overall and safety
metrics do not change:

- derived Action IR AST exact: 10,620 / 14,288 = 0.7432810750279956;
- schema valid: 14,285 / 14,288 = 0.9997900335946248;
- ABSTAIN F1: 0.9795653584171261;
- conservative false CALL: 89 / 10,407 = 0.008551936196790622.

The corrected post-SFT hard buckets are:

- revision AST exact: 3,050 / 4,140 = 0.7367149758454107;
- no-phenomenon AST exact: 5,217 / 6,648 = 0.7847472924187726;
- no-phenomenon minus revision gap: 0.04803231657336193;
- disfluency AST exact: 1,964 / 2,624 = 0.7484756097560976;
- no-phenomenon minus disfluency gap: 0.036271682662675.

The aggregate revision score must be read with its post-SFT subgroups:

| Raw tag | AST exact | Decision accuracy | False CALL on ABSTAIN/CONFIRM |
| --- | ---: | ---: | ---: |
| `cancel-action` | 446 / 454 = 0.9823788546255506 | 446 / 454 = 0.9823788546255506 | 0 / 454 = 0.0 |
| `correct-action` | 220 / 230 = 0.9565217391304348 | 223 / 230 = 0.9695652173913043 | 1 / 228 = 0.0043859649122807015 |
| `correct-argument` | 236 / 389 = 0.6066838046272494 | 365 / 389 = 0.9383033419023136 | 2 / 289 = 0.006920415224913495 |
| `within-turn-correction` | 2,148 / 3,067 = 0.7003586566677535 | 3,031 / 3,067 = 0.9882621454189762 | 9 / 2,013 = 0.004470938897168405 |

All four post-SFT subgroups have 100% schema validity, zero truncations, and zero generation
failures. The base has zero AST exact and zero schema-valid predictions in every revision
subgroup; its per-tag receipt is retained rather than collapsed into the post-SFT aggregate.

All six threshold booleans are true under the corrected full-family taxonomy. This is only a
post-hoc diagnostic gate pass. It is not a retroactive keep decision because the run's revision
focus replay remained 0/0 and therefore did not execute the intended revision-focused training
intervention. In particular, the 60.67% `correct-argument` result remains a clear weakness despite
the aggregate pass.

## Claim limits

This correction does not evaluate the official PRESTO test, native PRESTO semantic-parse exact
match, Mobile Actions preservation, a hidden safety set, additional seeds, or larger models. It
does not establish a usable release or breakthrough, and it does not justify modifying the
original failed result.
