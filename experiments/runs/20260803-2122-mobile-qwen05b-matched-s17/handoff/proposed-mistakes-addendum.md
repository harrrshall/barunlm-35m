## 2026-08-03 — Qwen matched-baseline verification exposed JSON-runtime assumptions

- The frozen BarunAction-35M candidate-v2 aggregate was recomputed after the Qwen attempt was
  preregistered. All 756 sample records and integer numerators were byte-identical, but Python 3.12
  changed `argument_value_macro_f1` by approximately 4e-16 relative to the frozen aggregate. The
  independent comparison therefore requires exact sample evidence and integer counts while
  allowing at most 1e-12 drift only for derived finite floats; it never rewrites the frozen file.
- The first downloaded-bundle verifier attempt assumed the `safe_open` object in
  `safetensors==0.8.0` was directly iterable. It failed at checkpoint inspection before accepting
  metrics. The failed output directory and exception were preserved, tensor-name access was
  changed to the stable sorted `keys()` interface, and a regression test was added.
- The second verifier attempt compared an in-memory empty tuple from the evaluator with the empty
  list produced by strict JSON serialization for `risk_coverage`. It failed closed before
  accepting metrics. Recomputed records are now normalized through `allow_nan=false` JSON before
  comparison, with sample records still required to match exactly; a regression test covers the
  container conversion.
- The isolated virtual environment was moved out of the remote upload target after creation so it
  could not enter the staged tree. Its generated `pytest` console-script shebang retained the old
  path, so one post-run test command failed before collecting tests. Subsequent checks used the
  environment's interpreter with `python -m pytest`; no scientific artifact or result changed.
- Result documentation and the independent verifier/tests were intentionally amended only after
  terminal scoring. Their post-run hashes differ from the preregistered source snapshot and are
  recorded as a separate verification/handoff delta. The frozen recipe, runner, model module,
  training inputs, evaluator, predictions, and sample scores were not changed.
- The first broad post-run test pass then failed one provenance assertion because that test required
  the live result document to retain its pre-score hash. The failure log was preserved. The test now
  checks live hashes only for science-critical runner/module/requirements and checks pre-score
  documentation/test identities against the immutable source snapshot; the next identical suite
  passed 103 tests. The preregistration and frozen source receipt were not rewritten.
- A second exact-ID query about 30 minutes after pause still reported instance 463689 as `Paused`,
  but its provider metadata showed runtime "0 hours 30 minutes" and cost 0.0. These fields are
  ambiguous and internally unsuitable as historical billing evidence. Report the frozen
  rate-times-controller-lifecycle estimate as an estimate, retain the raw filtered recheck, and do
  not call it a provider invoice.
- The matched public-development result was scientifically negative for the current
  BarunAction-over-Qwen hypothesis: Qwen was exact on 663/756 rows while BarunAction candidate-v2
  was exact on 602/756, a candidate deficit of 61 rows (8.0688 percentage points). Preserve this
  result and reject any current claim that BarunAction-35M exceeds this Qwen baseline; do not tune
  it away or generalize beyond this one-seed, public-development protocol.
