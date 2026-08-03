# Independent PRESTO bundle audit

The script at scripts/validate_presto_bundle.py performs a local, read-only audit after the
JarvisLabs controller downloads a completed PRESTO essential bundle. The validator does not
contact JarvisLabs, Hugging Face, Weights & Biases, GitHub, or any dataset host, and it never
loads the checkpoint into PyTorch.

The downloaded bundle is untrusted input. Two files outside that bundle establish the expected
identity:

- the experiment preregistration frozen before remote launch or model scoring;
- the completed infra/jarvis/safe_run.py record, including the exact created machine, successful
  managed run, artifact download, and verified final pause.

For the preregistered run, invoke:

    uv run python scripts/validate_presto_bundle.py \
      --bundle experiments/runs/20260803-1920-presto-stage-s17/essential \
      --preregistration experiments/runs/20260803-1920-presto-stage-s17/preregistration.json \
      --jarvis-record experiments/runs/20260803-1920-presto-stage-s17/jarvis.json \
      --output experiments/runs/20260803-1920-presto-stage-s17/independent-audit.json

The optional receipt is created with exclusive-create semantics; an existing receipt is never
overwritten. A zero exit code and a true top-level passed field mean the evidence was valid and
the recorded gate decision was reproduced. They do not mean that the research gate itself
passed: inspect recomputed_gate.passed separately.

## Fail-closed checks

The audit rejects the bundle unless all of these conditions hold:

- The artifact manifest names every regular file other than itself, names no unknown file, uses
  safe normalized paths, and matches every byte. Symlinks, duplicate JSON keys, non-finite JSON
  numbers, missing files, and unexpected files are errors.
- Run ID, recipe hash and content, JarvisLabs machine ID, managed-run ID, input checkpoint hashes,
  output checkpoint hashes, tokenizer, and model configuration agree across the independent
  anchors and every relevant bundle record.
- The exact project-created JarvisLabs machine was not pre-existing or protected; its managed run
  succeeded, the download completed, and its final recorded state is Paused.
- The official PRESTO test, combined dataset, and test-partition members show zero opens and zero
  bytes read. The data audit, effective preregistration, focus-view audit, and result must all
  agree on the zero-access firewall.
- Base and post-SFT predictions align one-to-one, without missing, duplicate, or extra sample IDs,
  and share identical development ground truth. Counts must match the pinned data audit,
  generation manifests, aggregates, and trainer manifests.
- Each sample is reparsed. AST exactness, schema validity, decisions, abstention statistics, and
  failure counts are recomputed. A parse-valid top-level CALL on an ABSTAIN or CONFIRM row is
  counted before tool or payload schema validation, so an invalid attempted call remains a false
  call.
- All gate thresholds, observations, bucket gaps, booleans, and the final keep/reject decision are
  recomputed from sample evidence rather than accepted from result.json.
- Training lineage links the frozen input hashes to the training config and trainer manifest, then
  links the selected checkpoint path to the post-SFT generation and exported output hashes.
- No optimizer state file is present. A digest naming remote optimizer state inside a generation
  provenance manifest is not optimizer state and is allowed; the corresponding bytes are not.

## Scope

This validator audits reproducibility and evidence integrity for the derived PRESTO Action IR
development stage. It does not open the official test, rerun GPU inference, assert native PRESTO
semantic-parse performance, establish Mobile Actions preservation, or support a larger-model or
breakthrough claim.
