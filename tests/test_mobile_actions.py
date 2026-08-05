from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from barunlm.datasets.mobile_actions import (
    MANIFEST_SCHEMA_VERSION,
    MobileActionsError,
    PreparationResult,
    PreparedExample,
    SourceSpec,
    SplitPolicy,
    TokenizerIdentity,
    _assign_training_splits,
    _scan_near_duplicates,
    convert_record,
    delexicalize_user,
    derived_split_for_cluster,
    prepare_mobile_actions,
    prepare_source,
)


def _tool(name: str, description: str, properties: dict[str, dict[str, str]], required=()):
    parameters: dict[str, object] = {"type": "OBJECT", "properties": properties}
    if required:
        parameters["required"] = list(required)
    return {
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
    }


TOOLS = [
    _tool(
        "create_calendar_event",
        "Creates a new calendar event.",
        {
            "title": {"type": "STRING", "description": "The title."},
            "datetime": {"type": "STRING", "description": "The date and time."},
        },
        required=("title", "datetime"),
    ),
    _tool(
        "send_email",
        "Sends an email.",
        {
            "to": {"type": "STRING", "description": "The recipient."},
            "subject": {"type": "STRING", "description": "The subject."},
            "body": {"type": "STRING", "description": "The body."},
        },
        required=("to", "subject"),
    ),
]


def _record(
    user: str,
    calls: list[dict[str, object]],
    *,
    split: str = "train",
    now: str = "2026-08-03T16:30:00",
    weekday: str = "Monday",
) -> dict[str, object]:
    return {
        "metadata": split,
        "tools": TOOLS,
        "messages": [
            {
                "role": "developer",
                "content": (
                    f"Current date and time given in YYYY-MM-DDTHH:MM:SS format: {now}\n"
                    f"Day of week is {weekday}\n"
                    "You are a model that can do function calling with the following functions\n"
                ),
            },
            {"role": "user", "content": user},
            {"role": "assistant", "tool_calls": calls},
        ],
    }


def _call(name: str, **arguments: object) -> dict[str, object]:
    return {"function": {"name": name, "arguments": arguments}}


def _source_spec(path: Path, *, train_rows: int, eval_rows: int) -> SourceSpec:
    return SourceSpec(
        dataset_id="test/mobile-actions",
        revision="synthetic-test-revision",
        filename=path.name,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        total_rows=train_rows + eval_rows,
        internal_train_rows=train_rows,
        final_eval_rows=eval_rows,
        license_id="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        source_url="https://example.invalid/synthetic",
        tool_names=frozenset({"create_calendar_event", "send_email"}),
    )


def _conversion_spec() -> SourceSpec:
    return SourceSpec(
        dataset_id="test/mobile-actions",
        revision="synthetic-test-revision",
        filename="dataset.jsonl",
        source_sha256="0" * 64,
        total_rows=1,
        internal_train_rows=1,
        final_eval_rows=0,
        license_id="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        source_url="https://example.invalid/synthetic",
        tool_names=frozenset({"create_calendar_event", "send_email"}),
    )


def _independent_records(count: int = 40) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(count):
        title = f"Task {index}"
        marker = hashlib.sha256(f"independent-family-{index}".encode()).hexdigest()
        records.append(
            _record(
                (
                    f"Schedule {title} on 2026-08-{index % 25 + 4:02d} at 9 AM "
                    f"using immutable marker {marker}."
                ),
                [
                    _call(
                        "create_calendar_event",
                        title=title,
                        datetime=f"2026-08-{index % 25 + 4:02d}T09:00:00",
                    )
                ],
            )
        )
    return records


def _convert_records(records: list[dict[str, object]]) -> list[PreparedExample]:
    examples: list[PreparedExample] = []
    spec = _conversion_spec()
    for index, record in enumerate(records, start=1):
        raw = json.dumps(record, separators=(",", ":")).encode()
        examples.append(
            convert_record(
                record,
                raw_sha256=hashlib.sha256(raw).hexdigest(),
                source_line=index,
                source_spec=spec,
            )
        )
    return examples


def _write_source(path: Path, records: list[dict[str, object]]) -> None:
    payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    )
    path.write_text(payload, encoding="utf-8", newline="")


class _WhitespaceTokenizer:
    def encode(self, sequence: str, add_special_tokens: bool = False):
        assert not add_special_tokens
        return SimpleNamespace(ids=list(range(len(sequence.split()))))


def test_convert_record_renders_prompt_and_canonical_action_ir() -> None:
    record = _record(
        "Schedule Team Sync tomorrow, then email Asha.",
        [
            _call(
                "create_calendar_event",
                title="Team Sync",
                datetime="2026-08-04T09:00:00",
            ),
            _call("send_email", to="asha@example.com", subject="Team Sync", body=None),
        ],
    )
    raw = json.dumps(record, separators=(",", ":")).encode()
    spec = _conversion_spec()
    example = convert_record(
        record,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        source_line=1,
        source_spec=spec,
    )

    assert example.prompt == (
        "<bos><system>\n"
        "ACTION_IR_V1\n"
        "NOW 2026-08-03T16:30:00\n"
        "TOOLS\n"
        "create_calendar_event(title:string!, datetime:string!): Creates a new calendar event.\n"
        "send_email(to:string!, subject:string!, body:string): Sends an email.\n"
        "<user>\n"
        "Schedule Team Sync tomorrow, then email Asha.\n"
        "<assistant>\n"
    )
    assert example.target == (
        '{"calls":[{"args":{"datetime":"2026-08-04T09:00:00","title":"Team Sync"},'
        '"tool":"create_calendar_event"},{"args":{"subject":"Team Sync",'
        '"to":"asha@example.com"},"tool":"send_email"}],"decision":"CALL",'
        '"mode":"SERIAL"}'
    )
    assert example.null_argument_fields_excluded == 1


def test_delexicalized_family_is_grouped_into_one_split() -> None:
    first_calls = [{"tool": "send_email", "args": {"to": "asha@example.com", "subject": "Update"}}]
    second_calls = [{"tool": "send_email", "args": {"to": "li@example.com", "subject": "Status"}}]
    first = delexicalize_user("Email asha@example.com with subject 'Update'.", first_calls)
    second = delexicalize_user("Email li@example.com with subject 'Status'.", second_calls)
    assert first == second

    policy = SplitPolicy(dev_folds=7, dev_fold=3)
    cluster = "ma-family-v1-" + "a" * 64
    assert derived_split_for_cluster(cluster, policy) == derived_split_for_cluster(cluster, policy)


def test_prepare_source_is_deterministic_and_eval_is_opaque_unparsed(
    tmp_path: Path,
) -> None:
    records = _independent_records()
    source = tmp_path / "dataset.jsonl"
    train_payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    ).encode()
    # This row has a valid split prefix but is deliberately invalid JSON.  Successful
    # preparation proves the eval suffix never reaches UTF-8/JSON/record conversion.
    opaque_eval_payload = b'{"metadata":"eval","messages":THIS_IS_NOT_JSON}\n'
    source.write_bytes(train_payload + opaque_eval_payload)
    spec = _source_spec(source, train_rows=len(records), eval_rows=1)
    tokenizer = _WhitespaceTokenizer()
    tokenizer_identity = TokenizerIdentity(identifier="test-whitespace-tokenizer", sha256="1" * 64)

    first = prepare_source(
        source,
        tmp_path / "first",
        source_spec=spec,
        split_policy=SplitPolicy(dev_folds=2),
        tokenizer=tokenizer,
        tokenizer_identity=tokenizer_identity,
        max_seq_len=12,
    )
    assert not hasattr(first, "final_eval_manifest")
    assert not (tmp_path / "first" / "final_eval.jsonl").exists()
    assert first.train_rows + first.dev_rows == len(records)
    assert first.final_eval_rows == 1
    assert set(first.hashes) == {"train", "dev", "audit"}

    train_records = [json.loads(line) for line in first.train_manifest.read_text().splitlines()]
    dev_records = [json.loads(line) for line in first.dev_manifest.read_text().splitlines()]
    assert all(record["schema_version"] == MANIFEST_SCHEMA_VERSION for record in train_records)
    train_clusters = {record["metadata"]["cluster_id"] for record in train_records}
    dev_clusters = {record["metadata"]["cluster_id"] for record in dev_records}
    assert train_clusters.isdisjoint(dev_clusters)
    train_families = {record["metadata"]["family_id"] for record in train_records}
    dev_families = {record["metadata"]["family_id"] for record in dev_records}
    assert train_families.isdisjoint(dev_families)
    assert all("token_lengths" in record["metadata"] for record in train_records + dev_records)

    audit = json.loads(first.audit_path.read_text())
    assert audit["counts"]["records_dropped"] == 0
    assert audit["counts"]["records_truncated"] == 0
    assert "final_eval" not in audit["artifacts"]
    assert audit["official_eval"] == {
        "labels_parsed": False,
        "lengths_computed": False,
        "materialized": False,
        "opaque_unparsed": True,
        "overlaps_computed": False,
        "prompts_parsed": False,
        "rows": 1,
        "source_sha256": spec.source_sha256,
        "summaries_computed": False,
        "targets_parsed": False,
        "tool_schemas_parsed": False,
    }
    assert set(audit["distributions"]) == {"train", "dev"}
    assert audit["tokenization"]["status"] == "computed"
    assert set(audit["tokenization"]["per_split"]) == {"train", "dev"}
    assert (
        audit["overlap"]["family_template"]["cross_split"]["train__dev"]["shared_unique_values"]
        == 0
    )
    assert (
        audit["overlap"]["connected_component"]["cross_split"]["train__dev"]["shared_unique_values"]
        == 0
    )
    assert (
        audit["overlap"]["char_5gram_minhash"]["pairs"]["train__dev"]["near_duplicate_pairs"] == 0
    )
    assert "gold tool names" in audit["overlap_scope"]
    assert "argument values" in audit["overlap_scope"]
    assert (
        sum(
            audit["tokenization"]["per_split"][split_name]["overlength_count"]
            for split_name in ("train", "dev")
        )
        > 0
    )

    second = prepare_source(
        source,
        tmp_path / "second",
        source_spec=spec,
        split_policy=SplitPolicy(dev_folds=2),
        tokenizer=tokenizer,
        tokenizer_identity=tokenizer_identity,
        max_seq_len=12,
    )
    assert first.hashes == second.hashes


def test_near_duplicate_families_share_one_connected_component() -> None:
    common = (
        "Please send the project status email after reviewing the quarterly planning "
        "notes and checking every delivery milestone with the operations team; use the "
        "carefully approved wording and preserve the documented sequence"
    )
    near_records = [
        _record(
            common + " alpha.",
            [_call("send_email", to="one@example.com", subject="One")],
        ),
        _record(
            common + " beta.",
            [_call("send_email", to="two@example.com", subject="Two")],
        ),
    ]
    near_examples = _convert_records(near_records)
    assert near_examples[0].family_id != near_examples[1].family_id
    scan = _scan_near_duplicates(near_examples)
    assert len(scan.verified_pairs) == 1
    assert scan.verified_pairs[0][2] >= 0.80

    all_examples = near_examples + _convert_records(_independent_records())
    assigned, grouping = _assign_training_splits(all_examples, SplitPolicy(dev_folds=2))
    assert assigned[0].cluster_id == assigned[1].cluster_id
    assert assigned[0].derived_split == assigned[1].derived_split
    assert grouping["near_duplicate_verified_edges"] >= 1
    assert grouping["component_count"] < grouping["family_count"]


def test_train_dev_api_has_no_eval_materialization_path() -> None:
    assert "final_eval_output" not in inspect.signature(prepare_source).parameters
    assert "final_eval_output" not in inspect.signature(prepare_mobile_actions).parameters
    assert "final_eval_manifest" not in PreparationResult.__dataclass_fields__

    eval_record = _record(
        "Do not inspect this eval request.",
        [_call("send_email", to="sealed@example.com", subject="Sealed")],
        split="eval",
    )
    raw = json.dumps(eval_record, separators=(",", ":")).encode()
    with pytest.raises(MobileActionsError, match="official eval rows are opaque"):
        convert_record(
            eval_record,
            raw_sha256=hashlib.sha256(raw).hexdigest(),
            source_line=1,
            source_spec=_conversion_spec(),
        )


def test_prepare_source_rejects_source_hash_and_stringified_arguments(tmp_path: Path) -> None:
    bad_record = _record(
        "Email a@example.com.",
        [{"function": {"name": "send_email", "arguments": "{}"}}],
    )
    source = tmp_path / "dataset.jsonl"
    _write_source(source, [bad_record])
    spec = _source_spec(source, train_rows=1, eval_rows=0)
    with pytest.raises(MobileActionsError, match="requires a JSON object"):
        prepare_source(source, tmp_path / "output", source_spec=spec)

    wrong_spec = SourceSpec(
        dataset_id=spec.dataset_id,
        revision=spec.revision,
        filename=spec.filename,
        source_sha256="f" * 64,
        total_rows=spec.total_rows,
        internal_train_rows=spec.internal_train_rows,
        final_eval_rows=spec.final_eval_rows,
        license_id=spec.license_id,
        license_url=spec.license_url,
        source_url=spec.source_url,
        tool_names=spec.tool_names,
    )
    with pytest.raises(MobileActionsError, match="source SHA-256 mismatch"):
        prepare_source(source, tmp_path / "other", source_spec=wrong_spec)
