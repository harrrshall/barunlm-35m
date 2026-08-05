# SmolLM2 matched Mobile Actions baseline result

Status: completed on 2026-08-03; retained as one-seed matched-adaptation evidence on reused public
development data only.

The preregistration document
[`mobile-matched-baseline.md`](mobile-matched-baseline.md) is frozen at SHA-256
`9d786a55a170f69b3d6b914fe0eb48b04040a7ec8bcf5a8559acb8078297238c` and therefore preserves
its prelaunch wording. This result document supersedes only that historical status; it does not
alter the protocol.

## Completed result

The exact frozen recipe completed in the retry evidence directory
[`experiments/runs/20260803-1955-mobile-smollm2-matched-retry-s17/essential`](../experiments/runs/20260803-1955-mobile-smollm2-matched-retry-s17/essential).
The scientific preregistration and result retain run ID
`20260803-1930-mobile-smollm2-matched-s17`; the earlier controller attempt failed before producing
a score because of an environment import conflict, so the retry did not add a scientific trial.

- SmolLM2 measured exactly 361,821,120 unique trainable parameters, 10.32 times the 35,072,768
  parameters in BarunAction-35M.
- After 7,937 examples, one epoch, and 126 optimizer steps, SmolLM2 scored **0/756** strict AST
  exact. Only 227/756 outputs parsed; none was schema-valid, and 407/756 hit the frozen generation
  cap. BarunAction candidate-v2 scored **602/756** on the same reused public-development manifest.
- The adapted baseline checkpoint has SHA-256
  `17ff9fb0ca95f65c656087c0a7fa6fe50dfd6560f3580b8b791d3b48e65087d4`. Every raw prediction
  and sample score is present; `artifact-sha256.json` has SHA-256
  `d6c58c75f4cbba2704516a42ffa805effcd2285329ff5c7fb4972a8c523ec211`.
- The completed H200 controller run used project-owned instance 463636, downloaded the evidence,
  and verified that exact instance `Paused`. Measured scientific elapsed time was 183.03 seconds
  and estimated cost was INR 19.23.

## Interpretation limit

This striking first result is diagnostic, not a breakthrough or broad superiority result. It uses
one seed and one larger-baseline recipe on a development population already reused for BarunAction
selection, with unequal development-trial budgets and unknown upstream pretraining contamination.
No official Mobile Actions row or hidden human-suite row was scored, and this CALL-only population
cannot establish safety. A “beats larger models” claim still requires the frozen multi-seed,
multi-baseline, hidden-suite, paired-uncertainty, practical-effect, and false-action gates.

## Subsequent larger-baseline context

The independently verified, locally imported Qwen matched-baseline run prevents treating the
SmolLM2 collapse as evidence about larger models as a class. On the aligned 756-row reused
development set, Qwen scored 663/756
(87.70%) strict AST exact and BarunAction candidate-v2 scored 602/756 (79.63%). Qwen led by 61 rows
(8.07 percentage points), with 80 Qwen-only wins, 19 Barun-only wins, and 657 ties. Qwen produced
755 parse-valid and 754 schema-valid outputs; all 961 official Mobile Actions rows remained
untouched.

The Qwen checkpoint contains 494,032,768 BF16 parameters across 290 tensors—0.494B, not 500B.
BarunAction is 14.09 times smaller and retains 90.80% of Qwen's exact-match rate, but trails it.
Accordingly, the larger-model-outperformance hypothesis failed. SmolLM2's 0/756 is a result for
that model and adaptation recipe only. Qwen run `20260803-2122-mobile-qwen05b-matched-s17` is now
bound by its selected import manifest and receipt; its large comparison checkpoint is intentionally
not distributed with the BarunAction source release.
