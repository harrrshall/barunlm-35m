"""Pinned English PRESTO adapter for BarunAction-35M post-training.

The adapter verifies the complete upstream archive, opens only the official
train and development members, and emits trainer-compatible Action IR manifests.
The official test member, redundant combined dataset member, and all test
partitions remain opaque: only their frozen source-level metadata is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import unicodedata
import urllib.request
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

PRESTO_REPOSITORY = "google-research-datasets/presto"
PRESTO_REVISION = "fa47167477453afebe698a287409514df5a7dadf"
PRESTO_ARCHIVE_URL = "https://storage.googleapis.com/gresearch/presto/presto_v1.zip"
PRESTO_ARCHIVE_SHA256 = "1fc671692cceb31fbda17e351e47f2cc52ee8779042f92dc26674cc0cca2167f"
PRESTO_ARCHIVE_SIZE = 415_990_813
PRESTO_LICENSE = "CC-BY-4.0"
PRESTO_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
PRESTO_LOCALE = "en-US"

MANIFEST_SCHEMA_VERSION = "barun-sft-example-v1"
ADAPTER_SCHEMA_VERSION = "barun-presto-en-adapter-v1"
AUDIT_SCHEMA_VERSION = "barun-presto-en-audit-v1"
PROMPT_CONTRACT_VERSION = "barun-presto-context-v1"
DEFAULT_MAX_SEQ_LEN = 2_048


@dataclass(frozen=True, slots=True)
class ArchiveMemberSpec:
    name: str
    sha256: str
    rows: int
    uncompressed_size: int
    crc32: str
    english_rows: int | None
    runtime_access: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("member sha256 must be a lowercase SHA-256 digest")
        if not re.fullmatch(r"[0-9a-f]{8}", self.crc32):
            raise ValueError("member crc32 must be an eight-digit lowercase digest")
        if self.rows < 1 or self.uncompressed_size < 1:
            raise ValueError("member row count and size must be positive")
        if self.runtime_access not in {"parse_train_dev", "opaque_never_open"}:
            raise ValueError("unsupported member runtime access policy")


PRESTO_MEMBER_SPECS = (
    ArchiveMemberSpec(
        name="presto_dataset.jsonl",
        sha256="d0c03d0d8cef8b80edf489db0ac97c6b01c8edf99105a970854cc3ce4c2967f8",
        rows=552_924,
        uncompressed_size=844_967_222,
        crc32="55b826c4",
        english_rows=None,
        runtime_access="opaque_never_open",
    ),
    ArchiveMemberSpec(
        name="presto_train.jsonl",
        sha256="b93ccf00ef8a7a67e0f1380da39e273791c26c87855babade5c045bd10c0ad65",
        rows=276_259,
        uncompressed_size=347_092_187,
        crc32="21b8668f",
        english_rows=47_806,
        runtime_access="parse_train_dev",
    ),
    ArchiveMemberSpec(
        name="presto_dev.jsonl",
        sha256="dcb8beadcf82802b0aa06d3b9b4ce6b25518b15816b7eb6071116e3bb7ccbd02",
        rows=82_547,
        uncompressed_size=103_487_087,
        crc32="0cd0af31",
        english_rows=14_288,
        runtime_access="parse_train_dev",
    ),
    ArchiveMemberSpec(
        name="presto_test.jsonl",
        sha256="9549050809fdd91fd117f42d62bdaa71db4770f4ddb5ecb3f394f97539bf21ef",
        rows=194_118,
        uncompressed_size=243_717_634,
        crc32="8708e927",
        english_rows=None,
        runtime_access="opaque_never_open",
    ),
)


@dataclass(frozen=True, slots=True)
class PrestoSourceSpec:
    repository: str
    revision: str
    archive_url: str
    archive_sha256: str
    archive_size: int
    license_id: str
    license_url: str
    members: tuple[ArchiveMemberSpec, ...]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("revision must be a full lowercase Git commit")
        if not re.fullmatch(r"[0-9a-f]{64}", self.archive_sha256):
            raise ValueError("archive sha256 must be a lowercase SHA-256 digest")
        names = [member.name for member in self.members]
        if len(names) != len(set(names)):
            raise ValueError("member names must be unique")
        if {"presto_train.jsonl", "presto_dev.jsonl", "presto_test.jsonl"}.difference(names):
            raise ValueError("source spec is missing a required PRESTO split")

    def member(self, name: str) -> ArchiveMemberSpec:
        for member in self.members:
            if member.name == name:
                return member
        raise KeyError(name)


OFFICIAL_SOURCE_SPEC = PrestoSourceSpec(
    repository=PRESTO_REPOSITORY,
    revision=PRESTO_REVISION,
    archive_url=PRESTO_ARCHIVE_URL,
    archive_sha256=PRESTO_ARCHIVE_SHA256,
    archive_size=PRESTO_ARCHIVE_SIZE,
    license_id=PRESTO_LICENSE,
    license_url=PRESTO_LICENSE_URL,
    members=PRESTO_MEMBER_SPECS,
)


class PrestoError(ValueError):
    """The pinned archive or a derived PRESTO artifact violates its contract."""


class TokenizerLike(Protocol):
    def encode(self, sequence: str, add_special_tokens: bool = False) -> Any: ...


@dataclass(frozen=True, slots=True)
class TokenizerIdentity:
    identifier: str
    sha256: str
    revision: str | None = None

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("tokenizer identifier cannot be empty")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("tokenizer sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class SemanticText:
    text: str


@dataclass(frozen=True, slots=True)
class SemanticSymbol:
    symbol: str


@dataclass(frozen=True, slots=True)
class SemanticSlot:
    name: str
    value: SemanticValue


@dataclass(frozen=True, slots=True)
class SemanticNode:
    name: str
    slots: tuple[SemanticSlot, ...]


SemanticValue = SemanticText | SemanticSymbol | SemanticNode


ABSTAIN_ROOTS = frozenset({"Cancel", "Other"})
CALL_ROOTS = frozenset(
    {
        "Check_order_status",
        "Find_parking",
        "GetGenericBusinessType",
        "Get_bill",
        "Get_health_stats",
        "Get_list",
        "Get_message_content",
        "Get_note",
        "Get_product",
        "Get_security_price",
        "Open_app",
        "Play_game",
    }
)
CONFIRM_ROOTS = frozenset(
    {
        "Add_contact",
        "Add_item_to_list",
        "BuyEventTickets",
        "Cancel_ride",
        "Create_list",
        "Create_note",
        "Initiate_call",
        "Log_exercise",
        "Log_nutrition",
        "Order_menu_item",
        "Order_ride",
        "Pause_exercise",
        "Pay_bill",
        "Post_message",
        "Record_video",
        "Resume_exercise",
        "Send_digital_object",
        "Start_exercise",
        "Stop_exercise",
        "Take_photo",
    }
)
PRESTO_ROOTS = ABSTAIN_ROOTS | CALL_ROOTS | CONFIRM_ROOTS
if len(PRESTO_ROOTS) != 34:
    raise AssertionError("the frozen PRESTO root mapping must cover exactly 34 intents")


def _snake_case(value: str) -> str:
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value.replace("-", "_"))
    second = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first)
    return re.sub(r"_+", "_", second).strip("_").casefold()


TOOL_NAMES = tuple(sorted(_snake_case(root) for root in CALL_ROOTS | CONFIRM_ROOTS))
PROMPT_TEMPLATE = (
    "<bos><system>\nACTION_IR_V1\nPRESTO_CONTEXT_V1\n"
    "COMPILE_ONLY never execute an action\n"
    "POLICY Other/Cancel=>ABSTAIN; state-changing=>CONFIRM; read-only=>CALL\n"
    "TOOLS {tools}\n"
    "ARGS slots is an ordered list of name/value objects; values use text, symbol, "
    "or nested node/slots\n"
    "CONTEXT {context}\nDIALOGUE {dialogue}\n<user>\n{user}\n<assistant>\n"
)
PROMPT_TEMPLATE_SHA256 = hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()

_IDENTIFIER_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


class _SemanticParser:
    def __init__(self, value: str) -> None:
        self.value = value
        self.index = 0

    def _whitespace(self) -> None:
        while self.index < len(self.value) and self.value[self.index].isspace():
            self.index += 1

    def _identifier(self) -> str:
        self._whitespace()
        match = _IDENTIFIER_RE.match(self.value, self.index)
        if match is None:
            raise PrestoError(f"semantic parse expected an identifier at character {self.index}")
        self.index = match.end()
        return match.group()

    def _value_after_identifier(self, identifier: str) -> SemanticValue:
        self._whitespace()
        if self.index >= len(self.value) or self.value[self.index] != "(":
            return SemanticSymbol(identifier)
        self.index += 1
        slots: list[SemanticSlot] = []
        while True:
            self._whitespace()
            if self.index < len(self.value) and self.value[self.index] == ")":
                self.index += 1
                return SemanticNode(identifier, tuple(slots))
            slot_name = self._identifier()
            self._whitespace()
            if self.index < len(self.value) and self.value[self.index] == "«":
                self.index += 1
                end = self.value.find("»", self.index)
                if end < 0:
                    raise PrestoError("semantic parse has an unterminated quoted value")
                text = unicodedata.normalize("NFC", self.value[self.index : end].strip())
                self.index = end + 1
                slot_value: SemanticValue = SemanticText(text)
            else:
                child_name = self._identifier()
                slot_value = self._value_after_identifier(child_name)
            slots.append(SemanticSlot(slot_name, slot_value))

    def parse(self) -> SemanticNode:
        root_name = self._identifier()
        parsed = self._value_after_identifier(root_name)
        self._whitespace()
        if self.index != len(self.value):
            raise PrestoError(f"semantic parse has trailing content at character {self.index}")
        if not isinstance(parsed, SemanticNode):
            raise PrestoError("semantic parse root must be a node")
        return parsed


def parse_semantic_target(value: str) -> SemanticNode:
    if not isinstance(value, str) or not value.strip():
        raise PrestoError("semantic target must be a non-empty string")
    return _SemanticParser(value).parse()


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise PrestoError(f"value is not strict JSON: {error}") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise PrestoError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> None:
    raise PrestoError(f"non-finite JSON constant {value!r}")


def _decode_row(raw: bytes, member: str, line_number: int) -> Mapping[str, Any]:
    try:
        text = raw.decode("utf-8", errors="strict")
        decoded = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, PrestoError) as error:
        raise PrestoError(f"{member} line {line_number}: invalid JSON: {error}") from error
    if not isinstance(decoded, dict):
        raise PrestoError(f"{member} line {line_number}: row must be an object")
    return decoded


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    if set(value) != expected:
        raise PrestoError(f"{path}: expected keys {sorted(expected)}, got {sorted(value)}")


def _nfc(value: Any, path: str, *, nonempty: bool = False) -> tuple[str, int]:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        requirement = "non-empty string" if nonempty else "string"
        raise PrestoError(f"{path}: expected a {requirement}")
    normalized = unicodedata.normalize("NFC", value)
    return normalized, int(normalized != value)


def _semantic_value_payload(value: SemanticValue) -> dict[str, Any]:
    if isinstance(value, SemanticText):
        return {"text": value.text}
    if isinstance(value, SemanticSymbol):
        return {"symbol": _snake_case(value.symbol)}
    return {
        "node": _snake_case(value.name),
        "slots": [
            {"name": _snake_case(slot.name), "value": _semantic_value_payload(slot.value)}
            for slot in value.slots
        ],
    }


def _semantic_slots_payload(node: SemanticNode) -> list[dict[str, Any]]:
    return [
        {"name": _snake_case(slot.name), "value": _semantic_value_payload(slot.value)}
        for slot in node.slots
    ]


def action_ir_for_semantic(node: SemanticNode) -> tuple[str, str]:
    if node.name not in PRESTO_ROOTS:
        raise PrestoError(f"unsupported PRESTO root intent {node.name!r}")
    if node.name in ABSTAIN_ROOTS:
        if node.slots:
            raise PrestoError(f"abstention root {node.name!r} unexpectedly has slots")
        return "ABSTAIN", '{"decision":"ABSTAIN"}'
    decision = "CONFIRM" if node.name in CONFIRM_ROOTS else "CALL"
    payload = {
        "calls": [
            {
                "args": {"slots": _semantic_slots_payload(node)},
                "tool": _snake_case(node.name),
            }
        ],
        "decision": decision,
        "mode": "SINGLE",
    }
    return decision, _canonical_json(payload)


def _iter_semantic_text(value: SemanticValue) -> Iterable[str]:
    if isinstance(value, SemanticText):
        yield value.text
    elif isinstance(value, SemanticNode):
        for slot in value.slots:
            yield from _iter_semantic_text(slot.value)


def _semantic_schema(value: SemanticValue) -> Any:
    if isinstance(value, SemanticText):
        return {"kind": "text"}
    if isinstance(value, SemanticSymbol):
        return {"kind": "symbol", "value": _snake_case(value.symbol)}
    return {
        "node": _snake_case(value.name),
        "slots": [
            {"name": _snake_case(slot.name), "value": _semantic_schema(slot.value)}
            for slot in value.slots
        ],
    }


def _normalized_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFC", value).casefold()).strip()


def _delexicalize_input(value: str, node: SemanticNode) -> str:
    output = _normalized_text(value)
    replacements = {
        _normalized_text(text)
        for text in _iter_semantic_text(node)
        if len(_normalized_text(text)) >= 2
    }
    for replacement in sorted(replacements, key=lambda item: (-len(item), item)):
        output = output.replace(replacement, "<value>")
    return _WHITESPACE_RE.sub(" ", output).strip()


@dataclass(frozen=True, slots=True)
class PreparedExample:
    example_id: str
    source_member: str
    source_line: int
    source_split: str
    raw_sha256: str
    prompt: str
    target: str
    native_target_sha256: str
    normalized_input: str
    delexicalized_input: str
    root_intent: str
    decision: str
    context_kind: str
    phenomenon: str
    context_id: str
    group_id: str
    family_id: str
    nfc_normalizations: int
    prompt_tokens: int | None = None
    target_tokens: int | None = None
    total_tokens_with_eos: int | None = None

    def with_lengths(self, prompt_tokens: int, target_tokens: int) -> PreparedExample:
        return replace(
            self,
            prompt_tokens=prompt_tokens,
            target_tokens=target_tokens,
            total_tokens_with_eos=prompt_tokens + target_tokens + 1,
        )

    def to_manifest_record(self, source_spec: PrestoSourceSpec) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
            "context_id": self.context_id,
            "context_kind": self.context_kind,
            "dataset": source_spec.repository,
            "derived_split": self.source_split,
            "family_id": self.family_id,
            "group_id": self.group_id,
            "linguistic_phenomenon": self.phenomenon,
            "locale": PRESTO_LOCALE,
            "native_target_sha256": self.native_target_sha256,
            "policy_decision": self.decision,
            "prompt_contract_sha256": PROMPT_TEMPLATE_SHA256,
            "prompt_contract_version": PROMPT_CONTRACT_VERSION,
            "prompt_sha256": _sha256_bytes(self.prompt.encode("utf-8")),
            "raw_sha256": self.raw_sha256,
            "root_intent": self.root_intent,
            "source_line": self.source_line,
            "source_member": self.source_member,
            "source_revision": source_spec.revision,
            "source_split": self.source_split,
            "target_sha256": _sha256_bytes(self.target.encode("utf-8")),
        }
        if self.prompt_tokens is not None:
            metadata["token_lengths"] = {
                "prompt": self.prompt_tokens,
                "target": self.target_tokens,
                "total_with_eos": self.total_tokens_with_eos,
            }
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "id": self.example_id,
            "prompt": self.prompt,
            "target": self.target,
            "metadata": metadata,
        }


@dataclass(frozen=True, slots=True)
class _SplitRead:
    split: str
    examples: tuple[PreparedExample, ...]
    source_rows: int
    member_sha256: str
    member_bytes_read: int
    locale_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class PreparationResult:
    train_manifest: Path
    dev_manifest: Path
    audit_path: Path
    train_rows: int
    dev_rows: int
    official_test_rows: int
    hashes: Mapping[str, str]


def _context_and_dialogue(
    metadata: Mapping[str, Any], path: str
) -> tuple[dict[str, Any], list[dict[str, str]], int]:
    changes = 0
    contacts_raw = metadata["seeded_contacts"]
    lists_raw = metadata["seeded_lists"]
    notes_raw = metadata["seeded_notes"]
    turns_raw = metadata["previous_turns"]
    if not isinstance(contacts_raw, list):
        raise PrestoError(f"{path}.seeded_contacts: expected a list")
    contacts: list[str] = []
    for index, item in enumerate(contacts_raw):
        normalized, changed = _nfc(item, f"{path}.seeded_contacts[{index}]")
        contacts.append(normalized)
        changes += changed
    if not isinstance(lists_raw, list):
        raise PrestoError(f"{path}.seeded_lists: expected a list")
    lists: list[dict[str, Any]] = []
    for index, item in enumerate(lists_raw):
        item_path = f"{path}.seeded_lists[{index}]"
        if not isinstance(item, dict):
            raise PrestoError(f"{item_path}: expected an object")
        _require_exact_keys(item, {"items", "name"}, item_path)
        name, changed = _nfc(item["name"], f"{item_path}.name")
        changes += changed
        if not isinstance(item["items"], list):
            raise PrestoError(f"{item_path}.items: expected a list")
        items: list[str] = []
        for item_index, raw_item in enumerate(item["items"]):
            normalized, changed = _nfc(raw_item, f"{item_path}.items[{item_index}]")
            items.append(normalized)
            changes += changed
        lists.append({"items": items, "name": name})
    if not isinstance(notes_raw, list):
        raise PrestoError(f"{path}.seeded_notes: expected a list")
    notes: list[dict[str, str]] = []
    for index, item in enumerate(notes_raw):
        item_path = f"{path}.seeded_notes[{index}]"
        if not isinstance(item, dict):
            raise PrestoError(f"{item_path}: expected an object")
        # The pinned rows use ``content``; the upstream README says ``text``.
        _require_exact_keys(item, {"content", "name"}, item_path)
        name, changed = _nfc(item["name"], f"{item_path}.name")
        changes += changed
        content, changed = _nfc(item["content"], f"{item_path}.content")
        changes += changed
        notes.append({"content": content, "name": name})
    if not isinstance(turns_raw, list):
        raise PrestoError(f"{path}.previous_turns: expected a list")
    dialogue: list[dict[str, str]] = []
    for index, item in enumerate(turns_raw):
        item_path = f"{path}.previous_turns[{index}]"
        if not isinstance(item, dict):
            raise PrestoError(f"{item_path}: expected an object")
        _require_exact_keys(item, {"response_text", "user_query"}, item_path)
        user, changed = _nfc(item["user_query"], f"{item_path}.user_query")
        changes += changed
        assistant, changed = _nfc(item["response_text"], f"{item_path}.response_text")
        changes += changed
        dialogue.append({"assistant": assistant, "user": user})
    return {"contacts": contacts, "lists": lists, "notes": notes}, dialogue, changes


def convert_record(
    record: Mapping[str, Any],
    *,
    source_member: str,
    source_split: str,
    source_line: int,
    raw_sha256: str,
) -> PreparedExample | None:
    row_path = f"{source_member} line {source_line}"
    _require_exact_keys(record, {"inputs", "metadata", "targets"}, row_path)
    metadata = record["metadata"]
    if not isinstance(metadata, dict):
        raise PrestoError(f"{row_path}.metadata: expected an object")
    _require_exact_keys(
        metadata,
        {
            "context",
            "example_id",
            "linguistic_phenomena",
            "locale",
            "previous_turns",
            "seeded_contacts",
            "seeded_lists",
            "seeded_notes",
            "split",
        },
        f"{row_path}.metadata",
    )
    if metadata["split"] != source_split:
        raise PrestoError(
            f"{row_path}.metadata.split: expected {source_split!r}, got {metadata['split']!r}"
        )
    locale = metadata["locale"]
    if not isinstance(locale, str):
        raise PrestoError(f"{row_path}.metadata.locale: expected a string")
    if locale != PRESTO_LOCALE:
        return None

    example_id, changes = _nfc(
        metadata["example_id"], f"{row_path}.metadata.example_id", nonempty=True
    )
    if not re.fullmatch(r"[0-9a-f]{64}", example_id):
        raise PrestoError(f"{row_path}.metadata.example_id: expected a lowercase SHA-256 ID")
    user, changed = _nfc(record["inputs"], f"{row_path}.inputs", nonempty=True)
    changes += changed
    native_target, changed = _nfc(record["targets"], f"{row_path}.targets", nonempty=True)
    changes += changed
    context_kind, changed = _nfc(metadata["context"], f"{row_path}.metadata.context", nonempty=True)
    changes += changed
    if context_kind not in {"human", "synthetic"}:
        raise PrestoError(f"{row_path}.metadata.context: unsupported value {context_kind!r}")
    phenomenon, changed = _nfc(
        metadata["linguistic_phenomena"], f"{row_path}.metadata.linguistic_phenomena"
    )
    changes += changed
    context, dialogue, child_changes = _context_and_dialogue(metadata, f"{row_path}.metadata")
    changes += child_changes

    semantic = parse_semantic_target(native_target)
    decision, target = action_ir_for_semantic(semantic)
    context_json = _canonical_json(context)
    dialogue_json = _canonical_json(dialogue)
    prompt = PROMPT_TEMPLATE.format(
        tools=",".join(TOOL_NAMES),
        context=context_json,
        dialogue=dialogue_json,
        user=user,
    )
    context_id = "presto-context-v1-" + _sha256_bytes(context_json.encode("utf-8"))
    if context["contacts"] or context["lists"] or context["notes"] or dialogue:
        group_payload = {"context": context, "dialogue": dialogue}
    else:
        group_payload = {"unclustered_example_id": example_id}
    group_id = "presto-group-v1-" + _sha256_bytes(_canonical_json(group_payload).encode("utf-8"))
    delexicalized = _delexicalize_input(user, semantic)
    family_payload = {
        "delexicalized_input": delexicalized,
        "semantic_schema": _semantic_schema(semantic),
    }
    family_id = "presto-family-v1-" + _sha256_bytes(_canonical_json(family_payload).encode("utf-8"))
    return PreparedExample(
        example_id=f"presto-en-{example_id}",
        source_member=source_member,
        source_line=source_line,
        source_split=source_split,
        raw_sha256=raw_sha256,
        prompt=prompt,
        target=target,
        native_target_sha256=_sha256_bytes(native_target.encode("utf-8")),
        normalized_input=_normalized_text(user),
        delexicalized_input=delexicalized,
        root_intent=semantic.name,
        decision=decision,
        context_kind=context_kind,
        phenomenon=phenomenon,
        context_id=context_id,
        group_id=group_id,
        family_id=family_id,
        nfc_normalizations=changes,
    )


def _verify_member_info(archive: zipfile.ZipFile, member: ArchiveMemberSpec) -> None:
    try:
        info = archive.getinfo(member.name)
    except KeyError as error:
        raise PrestoError(f"archive is missing required member {member.name}") from error
    actual_crc = f"{info.CRC:08x}"
    if info.file_size != member.uncompressed_size or actual_crc != member.crc32:
        raise PrestoError(
            f"archive member metadata mismatch for {member.name}: expected "
            f"size={member.uncompressed_size}, crc32={member.crc32}; got "
            f"size={info.file_size}, crc32={actual_crc}"
        )


def _read_split(
    archive: zipfile.ZipFile,
    member: ArchiveMemberSpec,
    split: str,
    access_log: dict[str, dict[str, int]],
) -> _SplitRead:
    digest = hashlib.sha256()
    examples: list[PreparedExample] = []
    locale_counts: Counter[str] = Counter()
    source_rows = 0
    bytes_read = 0
    access_log[member.name]["open_count"] += 1
    with archive.open(member.name, "r") as handle:
        for line_number, line_with_newline in enumerate(handle, start=1):
            source_rows += 1
            bytes_read += len(line_with_newline)
            digest.update(line_with_newline)
            raw = line_with_newline[:-1] if line_with_newline.endswith(b"\n") else line_with_newline
            if not raw:
                raise PrestoError(f"{member.name} line {line_number}: blank row")
            record = _decode_row(raw, member.name, line_number)
            metadata = record.get("metadata")
            if not isinstance(metadata, dict) or not isinstance(metadata.get("locale"), str):
                raise PrestoError(f"{member.name} line {line_number}: missing metadata.locale")
            locale_counts[metadata["locale"]] += 1
            converted = convert_record(
                record,
                source_member=member.name,
                source_split=split,
                source_line=line_number,
                raw_sha256=_sha256_bytes(raw),
            )
            if converted is not None:
                examples.append(converted)
    access_log[member.name]["bytes_read"] += bytes_read
    actual_sha256 = digest.hexdigest()
    if source_rows != member.rows:
        raise PrestoError(f"{member.name}: expected {member.rows} rows, got {source_rows}")
    if actual_sha256 != member.sha256:
        raise PrestoError(f"{member.name}: expected SHA-256 {member.sha256}, got {actual_sha256}")
    if member.english_rows is None or len(examples) != member.english_rows:
        raise PrestoError(
            f"{member.name}: expected {member.english_rows} English rows, got {len(examples)}"
        )
    return _SplitRead(
        split=split,
        examples=tuple(examples),
        source_rows=source_rows,
        member_sha256=actual_sha256,
        member_bytes_read=bytes_read,
        locale_counts=dict(sorted(locale_counts.items())),
    )


def _encode_ids(tokenizer: TokenizerLike, value: str) -> tuple[int, ...]:
    encoded = tokenizer.encode(value, add_special_tokens=False)
    ids = getattr(encoded, "ids", encoded)
    if not isinstance(ids, (list, tuple)) or any(not isinstance(item, int) for item in ids):
        raise PrestoError("tokenizer.encode must return integer IDs or an object with .ids")
    return tuple(ids)


def _add_lengths(
    examples: Sequence[PreparedExample], tokenizer: TokenizerLike
) -> list[PreparedExample]:
    output: list[PreparedExample] = []
    for example in examples:
        prompt_tokens = len(_encode_ids(tokenizer, example.prompt))
        target_tokens = len(_encode_ids(tokenizer, example.target))
        if prompt_tokens == 0 or target_tokens == 0:
            raise PrestoError(f"{example.example_id}: tokenizer produced a zero-token sequence")
        output.append(example.with_lengths(prompt_tokens, target_tokens))
    return output


def _percentile(values: Sequence[int], quantile: float) -> int:
    ordered = sorted(values)
    index = math.ceil(quantile * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _length_summary(values: Iterable[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize empty lengths")
    return {
        "count": len(ordered),
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 6),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
    }


def _index_overlap(splits: Mapping[str, Sequence[PreparedExample]], getter: Any) -> dict[str, Any]:
    indexes: dict[str, dict[str, list[str]]] = {}
    for split, rows in splits.items():
        index: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            index[getter(row)].append(row.example_id)
        indexes[split] = index
    shared = set(indexes["train"]).intersection(indexes["dev"])
    return {
        "train__dev": {
            "matching_pairs": sum(
                len(indexes["train"][value]) * len(indexes["dev"][value]) for value in shared
            ),
            "shared_unique_values": len(shared),
            "train_rows_affected": len(
                {item for value in shared for item in indexes["train"][value]}
            ),
            "dev_rows_affected": len({item for value in shared for item in indexes["dev"][value]}),
        }
    }


def _contiguous_ngrams(tokens: Sequence[str], n: int) -> set[str]:
    if len(tokens) < n:
        return set()
    return {
        _sha256_bytes(_canonical_json(tokens[index : index + n]).encode("utf-8"))
        for index in range(len(tokens) - n + 1)
    }


def _ngram_overlap(splits: Mapping[str, Sequence[PreparedExample]], *, n: int) -> dict[str, Any]:
    indexes: dict[str, dict[str, set[str]]] = {}
    short_rows: dict[str, int] = {}
    for split, rows in splits.items():
        index: dict[str, set[str]] = defaultdict(set)
        short = 0
        for row in rows:
            grams = _contiguous_ngrams(_TOKEN_RE.findall(row.normalized_input), n)
            if not grams:
                short += 1
            for gram in grams:
                index[gram].add(row.example_id)
        indexes[split] = index
        short_rows[split] = short
    shared = set(indexes["train"]).intersection(indexes["dev"])
    return {
        "n": n,
        "short_rows": short_rows,
        "train__dev": {
            "shared_unique_ngrams": len(shared),
            "train_rows_affected": len(
                {item for gram in shared for item in indexes["train"][gram]}
            ),
            "dev_rows_affected": len({item for gram in shared for item in indexes["dev"][gram]}),
        },
    }


def _membership_hash(examples: Sequence[PreparedExample]) -> str:
    return _sha256_bytes(("\n".join(sorted(row.example_id for row in examples)) + "\n").encode())


def _manifest_bytes(examples: Sequence[PreparedExample], source_spec: PrestoSourceSpec) -> bytes:
    return (
        "".join(
            _canonical_json(example.to_manifest_record(source_spec)) + "\n" for example in examples
        )
    ).encode("utf-8")


def _write_new_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError as error:
        raise PrestoError(f"refusing to overwrite artifact {path}") from error


def _distribution(examples: Sequence[PreparedExample]) -> dict[str, Any]:
    return {
        "context_kind": dict(sorted(Counter(row.context_kind for row in examples).items())),
        "decision": dict(sorted(Counter(row.decision for row in examples).items())),
        "linguistic_phenomenon": dict(sorted(Counter(row.phenomenon for row in examples).items())),
        "root_intent": dict(sorted(Counter(row.root_intent for row in examples).items())),
    }


def _build_audit(
    *,
    source_spec: PrestoSourceSpec,
    train_read: _SplitRead,
    dev_read: _SplitRead,
    train: Sequence[PreparedExample],
    dev: Sequence[PreparedExample],
    tokenizer_identity: TokenizerIdentity,
    max_seq_len: int,
    access_log: Mapping[str, Mapping[str, int]],
    archive_members: Sequence[str],
    artifact_hashes: Mapping[str, str],
) -> dict[str, Any]:
    splits = {"train": train, "dev": dev}
    sensitive_members = sorted(
        name
        for name in archive_members
        if name in {"presto_dataset.jsonl", "presto_test.jsonl"}
        or name.startswith("test_partitions/")
    )
    sensitive_opened = [name for name in sensitive_members if access_log[name]["open_count"] != 0]
    sensitive_bytes = sum(access_log[name]["bytes_read"] for name in sensitive_members)
    if sensitive_opened or sensitive_bytes:
        raise PrestoError("official PRESTO test/combined members were unexpectedly opened")

    tokenization: dict[str, Any] = {
        "eos_tokens_assumed": 1,
        "identity": {
            "identifier": tokenizer_identity.identifier,
            "revision": tokenizer_identity.revision,
            "sha256": tokenizer_identity.sha256,
        },
        "max_seq_len": max_seq_len,
        "policy": "audit only; manifests are never truncated or silently dropped",
        "per_split": {},
    }
    for split, rows in splits.items():
        overlength_ids = [
            row.example_id
            for row in rows
            if row.total_tokens_with_eos is not None and row.total_tokens_with_eos > max_seq_len
        ]
        tokenization["per_split"][split] = {
            "overlength_count": len(overlength_ids),
            "overlength_ids": overlength_ids,
            "prompt": _length_summary(
                row.prompt_tokens for row in rows if row.prompt_tokens is not None
            ),
            "target": _length_summary(
                row.target_tokens for row in rows if row.target_tokens is not None
            ),
            "total_with_eos": _length_summary(
                row.total_tokens_with_eos for row in rows if row.total_tokens_with_eos is not None
            ),
        }

    member_evidence: dict[str, Any] = {}
    for member in source_spec.members:
        runtime = access_log[member.name]
        member_evidence[member.name] = {
            "crc32": member.crc32,
            "english_rows": member.english_rows,
            "runtime_bytes_read": runtime["bytes_read"],
            "runtime_open_count": runtime["open_count"],
            "rows": member.rows,
            "sha256": member.sha256,
            "uncompressed_size": member.uncompressed_size,
            "verification": (
                "runtime_hash_and_count"
                if member.runtime_access == "parse_train_dev"
                else "whole_archive_sha256_and_central_directory_only"
            ),
        }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "adapter": {
            "prompt_contract_sha256": PROMPT_TEMPLATE_SHA256,
            "prompt_contract_version": PROMPT_CONTRACT_VERSION,
            "schema_version": ADAPTER_SCHEMA_VERSION,
        },
        "artifacts": {
            "manifests": {
                split: {
                    "filename": f"{split}.jsonl",
                    "membership_sha256": _membership_hash(rows),
                    "sha256": artifact_hashes[split],
                }
                for split, rows in splits.items()
            }
        },
        "counts": {
            "english_selected": {"dev": len(dev), "train": len(train)},
            "mapping_exclusions": {"dev": 0, "train": 0},
            "non_english_excluded": {
                "dev": dev_read.source_rows - len(dev),
                "train": train_read.source_rows - len(train),
            },
            "source_rows": {"dev": dev_read.source_rows, "train": train_read.source_rows},
            "nfc_normalizations": {
                "dev": sum(row.nfc_normalizations for row in dev),
                "train": sum(row.nfc_normalizations for row in train),
            },
            "records_dropped": 0,
            "records_truncated": 0,
        },
        "distributions": {split: _distribution(rows) for split, rows in splits.items()},
        "license": {"id": source_spec.license_id, "url": source_spec.license_url},
        "mapping": {
            "abstain_roots": sorted(ABSTAIN_ROOTS),
            "call_roots": sorted(CALL_ROOTS),
            "confirm_roots": sorted(CONFIRM_ROOTS),
            "coverage": "all English train/dev rows; no semantic-parse exclusions",
            "native_target_in_manifest": False,
            "native_target_retained_as": "SHA-256 only",
            "representation": (
                "ordered recursive slots preserve duplicate slots; values are typed as text, "
                "bare symbol, or nested node"
            ),
            "upstream_metric_compatibility": (
                "derived Action IR exact match is not the native PRESTO string exact-match metric"
            ),
        },
        "official_test": {
            "archive_payload_hashing_only": True,
            "bytes_read_from_sensitive_members": sensitive_bytes,
            "combined_dataset_member_opened": False,
            "labels_parsed": False,
            "materialized": False,
            "member_opened": False,
            "opaque_unparsed": True,
            "prompts_parsed": False,
            "rows": source_spec.member("presto_test.jsonl").rows,
            "sensitive_member_count": len(sensitive_members),
            "sensitive_members_opened": sensitive_opened,
            "targets_parsed": False,
            "test_partition_members_opened": False,
        },
        "overlap": {
            "context_id": _index_overlap(splits, lambda row: row.context_id),
            "delexicalized_input": _index_overlap(splits, lambda row: row.delexicalized_input),
            "example_id": _index_overlap(splits, lambda row: row.example_id),
            "family_id": _index_overlap(splits, lambda row: row.family_id),
            "group_id": _index_overlap(splits, lambda row: row.group_id),
            "normalized_input": _index_overlap(splits, lambda row: row.normalized_input),
            "whitespace_13_token": _ngram_overlap(splits, n=13),
        },
        "overlap_scope": (
            "official English train/dev only; family grouping uses train/dev native semantic "
            "structure and scalar values for delexicalization. Official test and combined/test "
            "partition members are never opened."
        ),
        "source": {
            "archive": {
                "actual_sha256": source_spec.archive_sha256,
                "expected_sha256": source_spec.archive_sha256,
                "size": source_spec.archive_size,
                "url": source_spec.archive_url,
            },
            "locale_counts": {
                "dev": dict(dev_read.locale_counts),
                "train": dict(train_read.locale_counts),
            },
            "members": member_evidence,
            "repository": source_spec.repository,
            "revision": source_spec.revision,
        },
        "split_policy": {
            "group_key": (
                "SHA-256(canonical structured context + chronological previous turns); "
                "rows with no context/turns use their example ID"
            ),
            "policy": "preserve official PRESTO train/dev splits and report cross-split overlap",
        },
        "tokenization": tokenization,
        "upstream_format_notes": {
            "seeded_notes": (
                "Pinned rows use keys content/name; the revision-pinned README documents "
                "text/name. The adapter validates the pinned content/name representation."
            ),
            "timezone": "PRESTO supplies no frozen reference timestamp or timezone.",
        },
    }


def prepare_archive(
    archive_path: str | Path,
    output_dir: str | Path,
    *,
    tokenizer: TokenizerLike,
    tokenizer_identity: TokenizerIdentity,
    source_spec: PrestoSourceSpec,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
) -> PreparationResult:
    """Prepare English official train/dev manifests; never open official test members."""

    if max_seq_len < 2:
        raise ValueError("max_seq_len must be at least 2")
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise PrestoError(f"archive does not exist: {archive_path}")
    actual_size = archive_path.stat().st_size
    if actual_size != source_spec.archive_size:
        raise PrestoError(
            f"archive size mismatch: expected {source_spec.archive_size}, got {actual_size}"
        )
    actual_archive_sha256 = sha256_file(archive_path)
    if actual_archive_sha256 != source_spec.archive_sha256:
        raise PrestoError(
            f"archive SHA-256 mismatch: expected {source_spec.archive_sha256}, "
            f"got {actual_archive_sha256}"
        )

    output = Path(output_dir)
    train_path = output / "train.jsonl"
    dev_path = output / "dev.jsonl"
    audit_path = output / "audit.json"
    existing = [path for path in (train_path, dev_path, audit_path) if path.exists()]
    if existing:
        raise PrestoError(
            "refusing to overwrite existing artifacts: " + ", ".join(map(str, existing))
        )

    with zipfile.ZipFile(archive_path) as archive:
        archive_members = tuple(archive.namelist())
        access_log: dict[str, dict[str, int]] = defaultdict(
            lambda: {"open_count": 0, "bytes_read": 0}
        )
        for member in source_spec.members:
            _verify_member_info(archive, member)
            access_log[member.name]
        for member_name in archive_members:
            access_log[member_name]
        train_read = _read_split(
            archive, source_spec.member("presto_train.jsonl"), "train", access_log
        )
        dev_read = _read_split(archive, source_spec.member("presto_dev.jsonl"), "dev", access_log)

    all_ids = [row.example_id for row in train_read.examples + dev_read.examples]
    if len(all_ids) != len(set(all_ids)):
        raise PrestoError("English train/dev example IDs are not globally unique")
    train = _add_lengths(train_read.examples, tokenizer)
    dev = _add_lengths(dev_read.examples, tokenizer)
    train_bytes = _manifest_bytes(train, source_spec)
    dev_bytes = _manifest_bytes(dev, source_spec)
    artifact_hashes = {
        "train": _sha256_bytes(train_bytes),
        "dev": _sha256_bytes(dev_bytes),
    }
    audit = _build_audit(
        source_spec=source_spec,
        train_read=train_read,
        dev_read=dev_read,
        train=train,
        dev=dev,
        tokenizer_identity=tokenizer_identity,
        max_seq_len=max_seq_len,
        access_log=access_log,
        archive_members=archive_members,
        artifact_hashes=artifact_hashes,
    )
    audit_bytes = (json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _write_new_file(train_path, train_bytes)
    _write_new_file(dev_path, dev_bytes)
    _write_new_file(audit_path, audit_bytes)
    hashes = dict(artifact_hashes)
    hashes["audit"] = _sha256_bytes(audit_bytes)
    return PreparationResult(
        train_manifest=train_path,
        dev_manifest=dev_path,
        audit_path=audit_path,
        train_rows=len(train),
        dev_rows=len(dev),
        official_test_rows=source_spec.member("presto_test.jsonl").rows,
        hashes=hashes,
    )


def prepare_presto(
    archive_path: str | Path,
    output_dir: str | Path,
    *,
    tokenizer: TokenizerLike,
    tokenizer_identity: TokenizerIdentity,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
) -> PreparationResult:
    return prepare_archive(
        archive_path,
        output_dir,
        tokenizer=tokenizer,
        tokenizer_identity=tokenizer_identity,
        source_spec=OFFICIAL_SOURCE_SPEC,
        max_seq_len=max_seq_len,
    )


def download_pinned_archive(cache_dir: str | Path) -> Path:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / "presto_v1.zip"
    if destination.exists():
        if (
            destination.stat().st_size != PRESTO_ARCHIVE_SIZE
            or sha256_file(destination) != PRESTO_ARCHIVE_SHA256
        ):
            raise PrestoError(
                f"existing cached archive is not the pinned PRESTO source: {destination}"
            )
        return destination
    try:
        with (
            urllib.request.urlopen(PRESTO_ARCHIVE_URL) as response,
            destination.open("xb") as output,
        ):
            for block in iter(lambda: response.read(1024 * 1024), b""):
                output.write(block)
    except Exception:
        if destination.exists():
            destination.unlink()
        raise
    if destination.stat().st_size != PRESTO_ARCHIVE_SIZE:
        raise PrestoError("downloaded PRESTO archive size mismatch")
    if sha256_file(destination) != PRESTO_ARCHIVE_SHA256:
        raise PrestoError("downloaded PRESTO archive SHA-256 mismatch")
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare pinned English PRESTO manifests.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--download-pinned", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--tokenizer-id", default="harrrshall/BarunLM-35M")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.download_pinned:
        if args.cache_dir is None:
            raise PrestoError("--cache-dir is required with --download-pinned")
        archive = download_pinned_archive(args.cache_dir)
    else:
        archive = args.archive
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    result = prepare_presto(
        archive,
        args.output_dir,
        tokenizer=tokenizer,
        tokenizer_identity=TokenizerIdentity(
            identifier=args.tokenizer_id,
            revision=args.tokenizer_revision,
            sha256=sha256_file(args.tokenizer_json),
        ),
        max_seq_len=args.max_seq_len,
    )
    print(
        json.dumps(
            {
                "audit": str(result.audit_path),
                "dev_manifest": str(result.dev_manifest),
                "dev_rows": result.dev_rows,
                "hashes": dict(result.hashes),
                "official_test_rows_opaque_unparsed": result.official_test_rows,
                "train_manifest": str(result.train_manifest),
                "train_rows": result.train_rows,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
