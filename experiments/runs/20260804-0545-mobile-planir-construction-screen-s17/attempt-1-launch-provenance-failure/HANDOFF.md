# PlanIR construction screen attempt 1 handoff

## Outcome

Run `20260804-0545-mobile-planir-construction-screen-s17` is closed as an inconclusive,
terminal operational failure. It produced no model or mechanism score. Do not retry, resume,
rescue, relaunch, or assign a new ordinal to this run or the narrow placeholder-v2 recipe.

The one fresh non-spot JarvisLabs H200 was machine `463843`, attached run `r_4c41c40f`. The
controller downloaded the fail-closed essential bundle before pausing the exact machine. The
controller pause and two subsequent read-only status checks all observed `Paused`; Kroda `463058`
remained `Running`, and no pre-existing resource was mutated.

## Exact failure

JarvisLabs' managed dependency preamble copied the pinned requirements file to the uploaded stage
root as `mobile-planir-screen.txt`. The remote stdlib preflight then rejected that provider-created
file because it was outside the frozen scientific allowlist:

```text
LaunchProvenanceError: unexpected file outside the stage allowlist: mobile-planir-screen.txt
```

This happened inside the first launch-provenance validation. Dependencies, including Torch, had
been installed, but the runner had not yet loaded its bound inventory, decoded the frozen config,
started the CPU gate, imported Torch, accessed CUDA,
loaded BarunLM-35M, read a construction train/screen JSON row, generated an output, or created a
checkpoint. Old selection, old confirmation, reused-756, human selection/confirmation, and the
official Mobile 961 population all remained unread.

The allowlist did what it was designed to do: it failed closed on unbound bytes. The missing
integration test was a byte-exact simulation of the managed requirements-file copy into the remote
target before the entrypoint. That lesson is reusable; it is not authority to patch this frozen
recipe and spend another attempt.

## Evidence

- Source commit: `d2f434ac4c598377c12198428f7ea1409c3b3698`
- Source snapshot: `ca090cdaf88555fb0255dd3e7cc5a77ae8f78a51579281805540c9fe1986b5d7`
- Scientific content tree: `9153fb2933d42bcf40eea83005fe6a6c362b45147e6da437bd5cdec905773019`
- Bound attempt: `dd69923ce26b49567daa687805588a6b79438ea9f80e357c5598d88ae6d24eff`
- Controller failure: `bc95353228f93976c13adc18350c425b40651bb83a84d515834746011de7dee1`
- Artifact manifest: `c19c9bd2da9d0e5b544122844e7ab32a17d5b6d9b9f1fa692bd26ce626048894`
- Raw lifecycle record: `648f09662134e65bd6fdb00a391aec99780fe6c48dd0c0bb6754fce29787493f`
- Remote log: `80c28afb9c8cb5b717653207e9bfc330c81cd545f1a1630c9750f46855a58461`

The tracked bundle contains only the safe failure receipt, artifact hash inventory, sanitized
classification, pause proof, and this handoff. The raw lifecycle, remote log, bound attempt, and
source snapshot are preserved privately outside the repository at the sibling path
`../barun-private-evidence/20260804-0545-mobile-planir-construction-screen-s17/attempt-1/` and are
bound by the hashes above. The nine 59 MiB materialized inputs remain local and untracked under
this run's `materialized-v1/`; their exact hashes are in the artifact manifest and frozen config.

## Decision

BarunAction-35M candidate-v2 remains the usable checkpoint at 602/756 exact (79.63%), versus
Qwen2.5-0.5B-Instruct at 663/756 (87.70%). Qwen is exactly 494,032,768 parameters and has 14.09
times as many parameters as BarunLM-35M's 35,072,768. This operational failure changes none of
those results and supplies no PlanIR performance evidence.

The next training proposal must be genuinely distinct, must establish a new leakage-controlled
population boundary before model access, and must include a provider-transform integration test in
its frozen launch rehearsal. It may reuse the operational lesson, but not this recipe, screen, or
attempt.
