# Qwen2.5-0.5B matched Mobile Actions handoff

Run `20260803-2122-mobile-qwen05b-matched-s17` is complete and independently verified. The
conclusion is `keep`: preserve it as a valid second-architecture matched public-development
comparison. It does not support a claim that BarunAction-35M exceeds this larger baseline.

## Outcome

The baseline was `Qwen/Qwen2.5-0.5B-Instruct` at immutable revision
`7ae557604adf67be50417f59c2c2f167def9a775`, Apache-2.0, with exactly 494,032,768 unique
parameters. That is 14.085936 times the 35,072,768-parameter BarunLM-35M base.

Both models were compared on the identical 756 frozen public Mobile Actions development IDs. Qwen
received the same 7,937 semantic training examples once, seed 17, response-only full-parameter
SFT, effective batch 63, 126 optimizer steps, final-checkpoint-only selection, and unconstrained
deterministic decoding. However, BarunAction candidate-v2 had prior Mobile development tuning while
Qwen consumed one development trial. Adaptation data and this run recipe are matched; total
model-selection effort is not. This asymmetry forbids a universal or fully matched headline claim.

| Measure | Qwen baseline | BarunAction candidate-v2 |
|---|---:|---:|
| Strict AST exact | 663/756 (87.6984%) | 602/756 (79.6296%) |
| Parse valid | 755/756 | 756/756 |
| Schema valid | 754/756 | 755/756 |
| Truncated | 0/756 | 0/756 |

Candidate minus Qwen is -61 exact rows, or -8.0688 percentage points. At sample level there are 19
candidate-only wins, 80 Qwen-only wins, 583 rows both correct, and 74 rows both wrong, for 657
ties. Qwen's argument-key F1 is 0.995791 macro and 0.996489 micro; argument-value F1 is 0.927409
macro and 0.945180 micro; tool macro F1 is 0.998431.

There were zero catastrophic unauthorized actions. This development population contains only
`CALL` requests, so the false-action denominator is zero and no safety, abstention, clarification,
or confirmation rate may be inferred. All actions were scored in a simulator; no real message,
calendar, contact, map, media, or device operation occurred.

## Runtime and lifecycle

Training took 73.129 seconds, terminal generation took 18.842 seconds, and the scientific runner
took 117.769 seconds. The full controller lifecycle took 567.160 seconds, including environment
setup, the pinned model download, artifact retrieval, and pause verification. At the frozen H200
rate of INR 378.27/hour, the estimated full lifecycle cost is INR 59.59; this is a rate-times-time
estimate, not an invoice.

The only created JarvisLabs resource was H200 instance 463689,
`barun-qwen05b-matched-2122-s17`, attached run `r_85efbb92`. The controller downloaded the complete
essential bundle and verified `Paused`; a later exact-ID read-only query independently confirmed
`Paused`. Kroda 463058 and every other pre-existing resource remained protected and untouched.

## Verification and failure accounting

The final local verifier starts from the downloaded bundle, recomputes all 756 baseline and
candidate sample records, rejects missing or duplicate IDs, verifies exact file trees and hashes,
checks the 494,032,768-parameter BF16 checkpoint and tokenizer identity, cross-binds model/data/run
and Jarvis identities, scans for secrets and forbidden official-evaluation material, and requires
the fresh pause proof. Derived aggregate float drift is tolerated only below 1e-12; sample records
and integer numerators must match exactly.

There were 93 non-exact baseline rows: 91 well-formed AST mismatches, one schema-only failure, and
one invalid-JSON parse failure. There were no missing predictions, truncations, generation
failures, OOMs, context overflows, dropped rows, or infrastructure retry. Details and immutable
sample IDs are in `handoff/failure-analysis.json`.

Two first-pass local verifier failures were preserved under
`independent-verification-attempt-1/` and `independent-verification-attempt-2/`. They exposed a
`safetensors` 0.8 key-iteration assumption and an in-memory tuple versus serialized JSON list.
Regression-tested fixes were applied only to the post-run verifier. Neither incident changed the
frozen runner, labels, predictions, sample scores, or scientific settings.

The first broad post-run test pass also caught an expected provenance-contract mismatch: the test
required the newly completed result document to retain its pre-score hash. Its log is preserved.
The corrected test keeps live checks for science-critical code and checks the older documentation
and verifier identities against the immutable pre-score source snapshot. The repeated suite passed
103 tests; the preregistration was never changed.

## Evidence map

- `essential/` is the immutable downloaded scientific bundle. It contains the bound attempt
  preregistration, source snapshot/staged-tree manifest, filtered environment, frozen recipe,
  data identity/audit, 126 training metrics, render audit and rendered rows, all 756 predictions,
  all 756 strict sample scores, aggregate metrics, checkpoint/tokenizer, and internal manifests.
- `independent-verification/` contains the clean recomputed aggregates, 756 paired records, input
  identities, secret/firewall audit, lifecycle verification, and its own manifest.
- `jarvis.json`, `remote.log`, and `pause-proof.json` contain controller, remote, download, exact-ID,
  and final pause evidence.
- `handoff/checkpoint-manifest.json`, `handoff/checkpoint-audit.json`, and
  `handoff/duration-cost.json` provide explicit checkpoint and full-lifecycle records.
- `handoff/prelaunch/` contains the secret-free exact launch command, prelaunch tests, uniqueness,
  tokenizer-only render audit, candidate-reference audit, source-copy receipt, and local package
  freeze.
- `handoff/source-files/` and `handoff/proposed-source.patch` contain the isolated implementation
  proposed for later review. The patch excludes the 988 MB checkpoint, caches, environments,
  credentials, optimizer state, W&B data, and all official evaluation data.
- `handoff/proposed-ledger-entry.jsonl` and `handoff/proposed-mistakes-addendum.md` are append-only
  proposals. The main ledger and mistakes log were not edited.
- `artifact-sha256.txt` is the final whole-bundle SHA-256 manifest. Internal essential,
  verification, checkpoint, and tokenizer manifests remain authoritative for their subtrees.

From a repository root, a primary agent can inspect the proposed source import with
`git apply -p1 --check /tmp/barun-qwen05b-matched-GCzPlATX/run/20260803-2122-mobile-qwen05b-matched-s17/handoff/proposed-source.patch`;
it should review the result before applying. The source patch's `work/` prefix is intentionally
removed by `-p1`.

## Firewall and publication status

The official 961 Mobile Actions evaluation rows were never read, parsed, copied, or materialized.
No PRESTO official test artifact entered the stage. No credential file was opened, copied, printed,
or hashed. W&B was not used; nothing was uploaded to Hugging Face; and nothing was pushed to a
public GitHub repository.

Private visibility and authentication for the approved handoff repository were not proved without
inspecting protected credentials, so no private push was attempted. The complete local bundle is
the handoff artifact.

This is one seed, one recipe, and one grouped public development population. It has no official
Mobile evaluation score, no sealed human safety suite, no multi-seed confidence interval, no equal
upstream pretraining, and no on-device deployment evidence. It neither selects nor modifies
BarunAction weights or the active rescue outcome and is not a breakthrough claim.
