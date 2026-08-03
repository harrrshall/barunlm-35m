# BarunAction-35M temporal v3 result handoff

## Decision

Month-Boundary Counterfactual SFT v3 is a valid scientific rejection. The remote execution,
provenance, scorer replay, evidence transfer, and exact-ID lifecycle all passed, but the frozen
selection conjunction did not. Keep BarunAction-35M candidate-v2 as the release checkpoint and
close this recipe. Do not retry it, change its thresholds, choose a favorable seed, or preserve
screening weights.

## What completed

All nine A/B/C screening fits at seeds 17, 29, and 43 completed. Independent verification replayed
all 9,216 predictions through the frozen Mobile scorer and reproduced every sample score,
aggregate, temporal subset, comparison, and the failed gate. It also rehashed all 145 manifested
payloads totaling 17,231,542 bytes with no missing, extra, size-mismatched, or hash-mismatched
file.

Arm C corrected the intended shortcut. Cross-month calendar-datetime exact was 123/198 (62.12%),
which is +31.82 points versus standard control A and +51.52 points versus equal-budget repeat
control B. The product gate still failed:

- overall AST gains were +1.60 and +2.44 points, below the required +3.00;
- same-month calendar-datetime exact lost 2.49 points versus B, above the 2-point ceiling;
- parse validity passed at 3,058/3,072;
- schema validity was 3,041/3,072, one valid output short of the 99% threshold; and
- two outputs truncated, while the frozen maximum was zero.

There were no missing predictions, generation failures, or catastrophic unauthorized actions.
Selection failure prevented confirmation, full refit, reused-756 compatibility scoring, and any
official-961 access. No optimizer state, screening checkpoint, or promoted model weight is in the
evidence bundle.

## Lifecycle and evidence

Fresh JarvisLabs H200 `463802` ran managed job `r_fe346e61`, which exited zero. Evidence and logs
were collected before the exact machine was pause-verified. ID 463802 is now permanently protected
and must never be resumed or reused. Raw lifecycle, account inventory, remote log, watchdog files,
attempt records, predictions, sample scores, and training outputs remain in the ignored owner-only
`private/` directory. The tracked files in this directory are the safe aggregate result, gate,
comparison, environment, validation, redacted pause, and independent verification records.

## Next research boundary

There is no active training experiment. The observed v3 selection and untouched v3 confirmation
populations are retired from new selection. The recommended next hypothesis is **Grounded
PlanIR**: post-train BarunLM-35M to emit a compact typed semantic plan, then use a frozen,
fail-closed deterministic compiler to ground input/context references and temporal operators into
the existing Action IR public contract.

Before any GPU launch, derive a new component-disjoint train/selection/confirmation split from
the former v3 construction pool with a new salt, prove at least 95% unambiguous PlanIR coverage,
and prove exact PlanIR-to-Action-IR round trips on every included label. Preregister three matched
arms: direct Action IR, direct Action IR with the same grounding table, and PlanIR plus compiler.
The decisive selection requirement should be at least +3 points compiled AST exact versus both
controls, at least +5 points argument-value exact, no family or policy regression over two points,
raw PlanIR parse validity at least 99.5%, emitted Action IR schema validity 100%, and zero
truncation, missing, generation-failure, or catastrophic-action events. Failure closes that new
hypothesis without a rescue.

BarunLM-35M is the canonical 35,072,768-parameter base-model name; BarunAction-35M is the
post-trained product name. The historical local directory name is not a model identity.
