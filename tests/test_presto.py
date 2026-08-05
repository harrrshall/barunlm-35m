from __future__ import annotations

import hashlib
import json
import zipfile
import zlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from barunlm.datasets.presto import (
    ArchiveMemberSpec,
    PrestoError,
    PrestoSourceSpec,
    TokenizerIdentity,
    action_ir_for_semantic,
    parse_semantic_target,
    prepare_archive,
)
from barunlm.training.data import load_manifest


class _WhitespaceTokenizer:
    def encode(self, sequence: str, add_special_tokens: bool = False):
        assert not add_special_tokens
        return SimpleNamespace(ids=list(range(len(sequence.split()))))


def _record(
    example_key: str,
    user: str,
    target: str,
    *,
    split: str,
    locale: str = "en-US",
) -> dict[str, object]:
    return {
        "inputs": user,
        "targets": target,
        "metadata": {
            "context": "human",
            "example_id": hashlib.sha256(example_key.encode()).hexdigest(),
            "linguistic_phenomena": "correct-argument",
            "locale": locale,
            "previous_turns": [
                {"user_query": "Find my reminder", "response_text": "Which reminder?"}
            ],
            "seeded_contacts": ["Asha"],
            "seeded_lists": [{"name": "Groceries", "items": ["bread"]}],
            "seeded_notes": [{"name": "Reminder", "content": "Call Asha"}],
            "split": split,
        },
    }


def _jsonl(records: list[dict[str, object]]) -> bytes:
    return (
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        )
    ).encode()


def _member_spec(
    name: str,
    payload: bytes,
    *,
    english_rows: int | None,
    runtime_access: str,
) -> ArchiveMemberSpec:
    return ArchiveMemberSpec(
        name=name,
        sha256=hashlib.sha256(payload).hexdigest(),
        rows=payload.count(b"\n"),
        uncompressed_size=len(payload),
        crc32=f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}",
        english_rows=english_rows,
        runtime_access=runtime_access,
    )


def _archive_fixture(path: Path) -> tuple[PrestoSourceSpec, dict[str, bytes]]:
    train = _jsonl(
        [
            _record(
                "train-note",
                "Make a reminder to buy milk",
                "Create_note ( content « Buy milk » device InferFromContext )",
                split="train",
            ),
            _record("train-other", "Sing me a song", "Other ( )", split="train"),
            _record(
                "train-spanish",
                "Canta una canción",
                "Other ( )",
                split="train",
                locale="es-ES",
            ),
        ]
    )
    dev = _jsonl(
        [
            _record(
                "dev-message",
                "Read Asha's latest message",
                (
                    "Get_message_content ( message Electronic_message ( sender "
                    "Personal_contact ( person « Asha » ) ) modality AUDIO )"
                ),
                split="dev",
            )
        ]
    )
    # Deliberately invalid JSON: successful preparation proves these members were not opened.
    combined = b'{"targets":COMBINED_MEMBER_MUST_STAY_OPAQUE}\n'
    test = b'{"targets":OFFICIAL_TEST_MUST_STAY_OPAQUE}\n'
    partition = b'{"targets":TEST_PARTITION_MUST_STAY_OPAQUE}\n'
    payloads = {
        "presto_dataset.jsonl": combined,
        "presto_train.jsonl": train,
        "presto_dev.jsonl": dev,
        "presto_test.jsonl": test,
        "test_partitions/en-US/test.jsonl": partition,
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
    members = (
        _member_spec(
            "presto_dataset.jsonl",
            combined,
            english_rows=None,
            runtime_access="opaque_never_open",
        ),
        _member_spec(
            "presto_train.jsonl",
            train,
            english_rows=2,
            runtime_access="parse_train_dev",
        ),
        _member_spec(
            "presto_dev.jsonl",
            dev,
            english_rows=1,
            runtime_access="parse_train_dev",
        ),
        _member_spec(
            "presto_test.jsonl",
            test,
            english_rows=None,
            runtime_access="opaque_never_open",
        ),
    )
    return (
        PrestoSourceSpec(
            repository="test/presto",
            revision="a" * 40,
            archive_url="https://example.invalid/presto.zip",
            archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            archive_size=path.stat().st_size,
            license_id="CC-BY-4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            members=members,
        ),
        payloads,
    )


def test_semantic_parser_preserves_nested_symbols_and_duplicate_slots() -> None:
    semantic = parse_semantic_target(
        "Add_item_to_list ( items « milk » items « eggs » "
        "list_position List_position ( absolute_position InferFromContext ) )"
    )
    decision, target = action_ir_for_semantic(semantic)
    payload = json.loads(target)

    assert decision == "CONFIRM"
    assert payload["decision"] == "CONFIRM"
    slots = payload["calls"][0]["args"]["slots"]
    assert [slot["name"] for slot in slots] == ["items", "items", "list_position"]
    assert slots[0]["value"] == {"text": "milk"}
    assert slots[2]["value"] == {
        "node": "list_position",
        "slots": [
            {
                "name": "absolute_position",
                "value": {"symbol": "infer_from_context"},
            }
        ],
    }

    _, abstain = action_ir_for_semantic(parse_semantic_target("Other ( )"))
    assert abstain == '{"decision":"ABSTAIN"}'


def test_prepare_archive_keeps_official_test_opaque_and_is_trainer_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "presto.zip"
    source_spec, payloads = _archive_fixture(archive_path)
    original_open = zipfile.ZipFile.open

    def guarded_open(self, name, *args, **kwargs):
        resolved = name.filename if isinstance(name, zipfile.ZipInfo) else str(name)
        if (
            resolved == "presto_dataset.jsonl"
            or resolved == "presto_test.jsonl"
            or resolved.startswith("test_partitions/")
        ):
            raise AssertionError(f"sealed member was opened: {resolved}")
        return original_open(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", guarded_open)
    tokenizer_identity = TokenizerIdentity("test-whitespace", "1" * 64)
    result = prepare_archive(
        archive_path,
        tmp_path / "prepared",
        tokenizer=_WhitespaceTokenizer(),
        tokenizer_identity=tokenizer_identity,
        source_spec=source_spec,
        max_seq_len=16,
    )

    assert result.train_rows == 2
    assert result.dev_rows == 1
    assert result.official_test_rows == 1
    assert set(result.hashes) == {"train", "dev", "audit"}
    train = load_manifest(
        result.train_manifest,
        expected_sha256=result.hashes["train"],
        expected_derived_split="train",
    )
    dev = load_manifest(
        result.dev_manifest,
        expected_sha256=result.hashes["dev"],
        expected_derived_split="dev",
    )
    assert len(train) == 2
    assert len(dev) == 1
    assert json.loads(train[0].target)["decision"] == "CONFIRM"
    assert "CONTEXT" in train[0].prompt
    assert "Call Asha" in train[0].prompt
    assert "Create_note (" not in train[0].target

    audit = json.loads(result.audit_path.read_text())
    assert audit["official_test"]["opaque_unparsed"] is True
    assert audit["official_test"]["member_opened"] is False
    assert audit["official_test"]["bytes_read_from_sensitive_members"] == 0
    assert audit["official_test"]["sensitive_members_opened"] == []
    assert (
        audit["source"]["members"]["presto_test.jsonl"]["sha256"]
        == hashlib.sha256(payloads["presto_test.jsonl"]).hexdigest()
    )
    assert audit["source"]["members"]["presto_test.jsonl"]["runtime_open_count"] == 0
    assert audit["source"]["members"]["presto_test.jsonl"]["runtime_bytes_read"] == 0
    assert audit["counts"]["records_truncated"] == 0
    assert audit["tokenization"]["per_split"]["train"]["overlength_count"] == 2


def test_prepare_archive_rejects_wrong_archive_hash(tmp_path: Path) -> None:
    archive_path = tmp_path / "presto.zip"
    source_spec, _ = _archive_fixture(archive_path)
    wrong_spec = replace(source_spec, archive_sha256="f" * 64)
    with pytest.raises(PrestoError, match="archive SHA-256 mismatch"):
        prepare_archive(
            archive_path,
            tmp_path / "prepared",
            tokenizer=_WhitespaceTokenizer(),
            tokenizer_identity=TokenizerIdentity("test-whitespace", "1" * 64),
            source_spec=wrong_spec,
        )
