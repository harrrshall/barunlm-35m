# BarunAction-35M candidate-v2 release evidence

Status: published release `candidate-v2-20260803`. The local JSON/JSONL records and their SHA-256
digests are authoritative; the public W&B `v0` artifacts are a distribution mirror, not a
replacement for the frozen experiment ledger.

## What the evidence supports

The released 35,072,768-parameter float checkpoint scored 602/756 strict Action IR AST exact on a
grouped Mobile Actions development population derived only from public training rows. The retained
Darwin ARM64/PyTorch 2.13.0/QNNPACK int8 artifact scored 607/756 under its separately frozen
post-selection protocol and passed the 587-row/minimum, 15-row/maximum-loss gates. The artifact is
usable as a hash-verified proposal compiler behind external policy and confirmation gates.

The evidence does not support a blind breakthrough or a general larger-model-superiority claim.
The Mobile population is reused, all-`CALL`, and limited to seven schemas. The 961 official Mobile
rows remained opaque. More strongly, the independently verified and locally imported Qwen
matched-baseline run scored 663/756 (87.70%) strict AST exact versus candidate-v2's 602/756
(79.63%). Qwen led
by 61 rows, or 8.07 percentage points, with 80 Qwen-only wins, 19 Barun-only wins, and 657 ties; it
produced 755 parse-valid and 754 schema-valid outputs. Its 494,032,768 BF16 parameters across 290
tensors make it a 0.494B—not 500B—checkpoint. BarunAction is 14.09 times smaller and retains 90.80%
of Qwen's exact-match rate, but does not win. The larger-model-outperformance hypothesis failed.

The Qwen evidence is imported as run `20260803-2122-mobile-qwen05b-matched-s17`. Its full source
handoff manifest passed before a safe 63-file subset was copied; `import-receipt.json` and
`selected-artifact-sha256.txt` bind that boundary. The earlier 0/756 SmolLM2 result remains useful
only as a model-and-recipe-specific failure. The four-trial Mobile-plus-PRESTO rescue found no eligible joint
checkpoint; its best research-only recovery result missed the PRESTO gate by 140 rows.

## Included chains

The release evidence artifact contains:

- the candidate-v1, failed-follow-up, and candidate-v2 preregistration/selection chain;
- candidate-v2 and competing-arm training summaries, raw predictions, aggregates, and sample
  scores;
- the PRESTO stage, taxonomy correction record, frozen Mobile regression, interpolation, continual
  recovery, and final no-promotion selector evidence;
- the pinned SmolLM2-360M-Instruct matched-recipe identity, training record, raw predictions, and
  sample scores;
- the Qwen2.5-0.5B matched recipe, predictions, sample scores, paired recomputation, checkpoint
  audit, failure accounting, verification attempts, completion audit, and safe import receipt;
- the candidate-v2 int8 protocol, raw predictions, scores, paired receipt, and same-host FP32
  control; and
- current model/data cards, evaluation protocol, decision record, experiment ledger, mistakes log,
  license, and attribution notice.

Full JarvisLabs inventories, credentials, caches, official-test material, model checkpoints for
rejected arms, optimizer state, and unrelated experiment files are excluded. Resource IDs needed
for scientific provenance remain in the ledger. A later audit found that immutable evidence `v0`
also retains non-access-bearing workstation paths and mentions of unrelated protected resources;
the earlier claim that those names were not redistributed was too strong. No credential marker or
signed endpoint was found, and the scientific payload is unaffected. Do not rewrite or hide `v0`.
A deterministic 296-file local redacted view exists, but its proposed W&B `v1` publication was
stopped before network access because the service's non-atomic version assignment could create an
irreversible `v2` during a race and the anonymous-read gate was not implemented. It is not a
published artifact.

## Dataset attribution and firewalls

Mobile Actions revision `e920309bc2acbc2e99a5e3201cf37df2b9fd9151` and PRESTO revision
`fa47167477453afebe698a287409514df5a7dadf` are CC BY 4.0. Preserve `NOTICE` and the upstream
attribution when redistributing derived sample evidence. Mobile official-evaluation artifacts and
PRESTO official-test contents are absent. Every included score must retain its development-only or
post-hoc label; never present it as an official or hidden-test score.

## Verification

The checked-in W&B release manifest enumerates every uploaded file, size, and SHA-256. The upload
receipt records immutable `vN` artifact identities and remote digests. A separate verification
receipt is produced by downloading each immutable artifact into a fresh external directory,
rejecting symlinks/unlisted files, and recomputing the release-manifest hashes. Retrieval details
and exact immutable identities are in `docs/barunaction-retrieval.md`.

That procedure is complete. The release manifest SHA-256 is
`cbb29c4921855031bfeee1c1f5e9ed1a33a932b902a875c21f255fac793c2165`; its upload receipt is
`6ce9a6095a932195f2caf645993f75e507039e6910f63760326c2860ebe3238b`; and its independent
redownload receipt is `2a3ea05fbcf6e3955f65f92d24816730d1bb6b130d3792983aa3cab0684dfcc4`.
All 310 downloaded files, totaling 336,040,340 bytes, matched. A separate anonymous query proved
public `USER_READ` visibility and returned all three committed `v0` records with their expected
digests. An explicitly unauthenticated stream also reproduced the full 140,304,464-byte float
weights and their SHA-256. The exact proof and URLs are in `public-release-record.json` beside
those receipts.
