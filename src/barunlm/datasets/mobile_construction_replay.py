"""Fail-closed direct-Action-IR replay for former-v3 construction rows.

This module has one deliberately narrow input boundary: the exact 5,745-row
former-v3 Mobile Actions construction manifest and the exact one-row exclusion
ledger for its known temporal-reference label defect.  It returns immutable
bytes for a 5,744-row, byte-preserving direct-Action-IR training replay plus a
deterministic receipt.  It never reads a development, selection, confirmation,
or official-evaluation population and it never authorizes model access.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from barunlm.evaluation.action_ir import ActionIRError, parse_action_ir
from barunlm.evaluation.mobile_action_schemas import MOBILE_TOOL_SCHEMAS

REPLAY_SCHEMA_VERSION = "barun-mobile-construction-direct-replay-v1"
RECEIPT_SCHEMA_VERSION = "barun-mobile-construction-direct-replay-receipt-v1"

SOURCE_ROWS = 5_745
SOURCE_MANIFEST_SHA256 = "800a3ba0a7f0215e5cf95c77c32e47abed48b2f07f375954b0e0b08296a33d10"
SOURCE_MEMBERSHIP_SHA256 = "b70f6dfe79cfe49792b6e6f5d414fd509c889d18101f75cd8df6d4e4f146c588"
SOURCE_ORDER_SHA256 = "3cc37d0bf7c452c7b7e13066cf2b74f19c1e61d6932b1c4a05fec6759de114bc"

EXCLUSION_LEDGER_ROWS = 1
EXCLUSION_LEDGER_SHA256 = "7593b556b9c096308ca1a9e9d9b89627ebf19710da930bd4dd5794c58879616a"
EXCLUDED_ID = "mobile-actions-04894-6bd7643b1bc95697"
EXCLUDED_CODE = "temporal_reference_mismatch"
EXCLUDED_CONTENT_SHA256 = "f013c85dd59675813385aeaa796ec56a96e4d84471ada9fb7e7f31ed197de753"
EXCLUDED_PROMPT_SHA256 = "2dabdf94322850bb1824dd26902277f80e78f124f8ef30097a9b270dcb6cbfef"
EXCLUDED_TARGET_SHA256 = "c024176db64d33a5e819b4d5d7bbc0b7ff9d742fd34ee4bcc0527816035d3e4c"
EXCLUDED_COMPONENT_ID = (
    "ma-component-v1-2e7e3504d097fdde9776b24ca88096887d8242c6b1d63a1d434f66339838a853"
)
EXCLUDED_FAMILY_ID = "ma-family-v1-3bcd0729a5d858980eef576453cafe64d52a44a8fb72b5df64b1a7d726a3fa09"

OUTPUT_ROWS = 5_744
OUTPUT_MANIFEST_SHA256 = "f964587bbed79f451c2d5360677c22d176ed9a96c750fff6e12852e57530d863"
OUTPUT_MEMBERSHIP_SHA256 = "27f9acb8905db132420ddb5d11a20bed138f2e2927aec41672f6852b3895452d"
OUTPUT_ORDER_SHA256 = "b4a909a1f666cb34f5baccc0e9ebd57c00e05f5a9ff9271cf3bf2c85923ce29e"
DIRECT_TARGET_COMMITMENT_SHA256 = "131b7b3094fcfc3cbf919cdf2cd5fb8062ac7a492fc57a8eff45fbdcfe163389"

V3_SELECTION_ROWS = 1_024
V3_SELECTION_MEMBERSHIP_SHA256 = "b4191540878ec28c6819e08e0b390fbfd57eab836d3361727ad89862a46cf927"
V3_CONFIRMATION_ROWS = 1_168
V3_CONFIRMATION_MEMBERSHIP_SHA256 = (
    "519c22724873ee579d0f59be886cfc297088113bd1a297521b9ca280a79c35b9"
)
V3_SHADOW_AUDIT_SHA256 = "86c38f2b03f5f2e61758079fb1edca698d6e8e658d726004d5d975dd3a1305ae"

_SOURCE_REVISION = "e920309bc2acbc2e99a5e3201cf37df2b9fd9151"
_PROMPT_CONTRACT_SHA256 = "3fd4fcfdecff4aff2a66558da04ca310bc684753da98368e07ecbb9f4e73dc36"
_PLANIR_PLACEHOLDER_RE = re.compile(r"@(?:A|M|R|W[12]):D\d{2}:T\d{2}")
_FORBIDDEN_PROMPT_MARKERS = ("\nREFS_V2 ", "PLAN_IR_V2", "GROUNDED_PLAN_IR")
_FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "counterfactual_transform",
        "grounded_planir",
        "mbcf",
        "mobile_planir",
        "mobile_planir_screen",
        "mobile_temporal_view",
        "month_boundary_counterfactual",
        "plan_ir",
        "planir",
        "transform_receipt",
    }
)
_FORBIDDEN_ROLES = frozenset(
    {
        "confirmation",
        "development",
        "dev",
        "eval",
        "evaluation",
        "final",
        "official",
        "reused",
        "selection",
        "test",
    }
)


class DirectActionReplayError(ValueError):
    """The pinned construction replay boundary did not validate exactly."""


@dataclass(frozen=True, slots=True)
class _ReplayContract:
    source_rows: int
    source_sha256: str
    source_membership_sha256: str
    source_order_sha256: str
    exclusion_ledger_rows: int
    exclusion_ledger_sha256: str
    excluded_id: str
    excluded_code: str
    excluded_content_sha256: str
    excluded_prompt_sha256: str
    excluded_target_sha256: str
    excluded_component_id: str
    excluded_family_id: str
    output_rows: int
    output_sha256: str
    output_membership_sha256: str
    output_order_sha256: str
    direct_target_commitment_sha256: str


_PINNED_CONTRACT = _ReplayContract(
    source_rows=SOURCE_ROWS,
    source_sha256=SOURCE_MANIFEST_SHA256,
    source_membership_sha256=SOURCE_MEMBERSHIP_SHA256,
    source_order_sha256=SOURCE_ORDER_SHA256,
    exclusion_ledger_rows=EXCLUSION_LEDGER_ROWS,
    exclusion_ledger_sha256=EXCLUSION_LEDGER_SHA256,
    excluded_id=EXCLUDED_ID,
    excluded_code=EXCLUDED_CODE,
    excluded_content_sha256=EXCLUDED_CONTENT_SHA256,
    excluded_prompt_sha256=EXCLUDED_PROMPT_SHA256,
    excluded_target_sha256=EXCLUDED_TARGET_SHA256,
    excluded_component_id=EXCLUDED_COMPONENT_ID,
    excluded_family_id=EXCLUDED_FAMILY_ID,
    output_rows=OUTPUT_ROWS,
    output_sha256=OUTPUT_MANIFEST_SHA256,
    output_membership_sha256=OUTPUT_MEMBERSHIP_SHA256,
    output_order_sha256=OUTPUT_ORDER_SHA256,
    direct_target_commitment_sha256=DIRECT_TARGET_COMMITMENT_SHA256,
)


@dataclass(frozen=True, slots=True)
class DirectActionReplayMaterialization:
    """Immutable output bytes; decoded receipts are detached on every access."""

    manifest_bytes: bytes
    receipt_bytes: bytes

    def __post_init__(self) -> None:
        if type(self.manifest_bytes) is not bytes or type(self.receipt_bytes) is not bytes:
            raise TypeError("materialized manifest and receipt must be immutable bytes")

    @property
    def manifest_sha256(self) -> str:
        return _sha256(self.manifest_bytes)

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.receipt_bytes)

    def receipt_dict(self) -> dict[str, Any]:
        """Return a fresh JSON tree so caller mutation cannot change frozen evidence."""

        decoded = json.loads(self.receipt_bytes)
        if type(decoded) is not dict:  # pragma: no cover - construction guarantees this
            raise DirectActionReplayError("internal receipt is not an object")
        return decoded


@dataclass(frozen=True, slots=True)
class _ValidatedRow:
    raw_line: bytes
    payload: dict[str, Any]
    example_id: str
    prompt_sha256: str
    target_sha256: str
    content_sha256: str


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(payload: object) -> str:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise DirectActionReplayError(f"value is not strict canonical JSON: {error}") from error


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise DirectActionReplayError(f"duplicate JSON object key {key!r}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise DirectActionReplayError(f"non-finite JSON constant {value!r} is forbidden")


def _strict_json(raw: str, *, label: str) -> object:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, DirectActionReplayError) as error:
        raise DirectActionReplayError(f"{label} is not strict JSON: {error}") from error


def _membership_sha256(ids: list[str]) -> str:
    if len(ids) != len(set(ids)):
        raise DirectActionReplayError("source contains duplicate example IDs")
    return _sha256(("\n".join(sorted(ids)) + "\n").encode("utf-8"))


def _ordered_sha256(ids: list[str]) -> str:
    payload = "".join(f"{index:06d}\t{example_id}\n" for index, example_id in enumerate(ids, 1))
    return _sha256(payload.encode("utf-8"))


def _target_commitment_sha256(rows: list[_ValidatedRow]) -> str:
    payload = "".join(
        f"{index:06d}\t{row.example_id}\t{row.target_sha256}\n" for index, row in enumerate(rows, 1)
    )
    return _sha256(payload.encode("utf-8"))


def _forbidden_metadata_key(value: object) -> str | None:
    if type(value) is dict:
        for key, child in value.items():
            normalized = key.casefold().replace("-", "_")
            if normalized in _FORBIDDEN_METADATA_KEYS:
                return key
            nested = _forbidden_metadata_key(child)
            if nested is not None:
                return nested
    elif type(value) is list:
        for child in value:
            nested = _forbidden_metadata_key(child)
            if nested is not None:
                return nested
    return None


def _contains_planir_placeholder(value: object) -> bool:
    if type(value) is str:
        return _PLANIR_PLACEHOLDER_RE.search(value) is not None
    if type(value) is dict:
        return any(_contains_planir_placeholder(child) for child in value.values())
    if type(value) is list:
        return any(_contains_planir_placeholder(child) for child in value)
    return False


def _validate_direct_target(target: str, *, example_id: str) -> str:
    decoded = _strict_json(target, label=f"target for {example_id!r}")
    if type(decoded) is not dict:
        raise DirectActionReplayError(f"target for {example_id!r} must be a JSON object")
    if _canonical_json(decoded) != target:
        raise DirectActionReplayError(f"target for {example_id!r} is not canonical JSON")
    if _contains_planir_placeholder(decoded):
        raise DirectActionReplayError(f"target for {example_id!r} contains a PlanIR placeholder")
    try:
        action = parse_action_ir(target, MOBILE_TOOL_SCHEMAS)
    except ActionIRError as error:
        raise DirectActionReplayError(
            f"target for {example_id!r} is not direct Action IR: {error.code}"
        ) from error
    if action.canonical_json() != target:
        raise DirectActionReplayError(
            f"target for {example_id!r} is not canonical direct Action IR"
        )
    return _sha256(target.encode("utf-8"))


def _validate_source_row(raw_line: bytes, *, line_number: int) -> _ValidatedRow:
    if not raw_line.endswith(b"\n") or raw_line.endswith(b"\r\n"):
        raise DirectActionReplayError(f"source line {line_number} must end with one LF")
    raw_record = raw_line[:-1]
    if not raw_record:
        raise DirectActionReplayError(f"source line {line_number} is blank")
    try:
        text = raw_record.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DirectActionReplayError(f"source line {line_number} is not UTF-8") from error
    payload = _strict_json(text, label=f"source line {line_number}")
    if type(payload) is not dict:
        raise DirectActionReplayError(f"source line {line_number} must be an object")
    if _canonical_json(payload) != text:
        raise DirectActionReplayError(f"source line {line_number} is not canonical JSON")
    if set(payload) != {"id", "metadata", "prompt", "schema_version", "target"}:
        raise DirectActionReplayError(f"source line {line_number} fields changed")
    if payload.get("schema_version") != "barun-sft-example-v1":
        raise DirectActionReplayError(f"source line {line_number} schema version changed")

    example_id = payload.get("id")
    prompt = payload.get("prompt")
    target = payload.get("target")
    metadata = payload.get("metadata")
    if type(example_id) is not str or not example_id:
        raise DirectActionReplayError(f"source line {line_number} has an invalid id")
    if type(prompt) is not str or not prompt:
        raise DirectActionReplayError(f"source row {example_id!r} has an invalid prompt")
    if type(target) is not str or not target:
        raise DirectActionReplayError(f"source row {example_id!r} has an invalid target")
    if type(metadata) is not dict:
        raise DirectActionReplayError(f"source row {example_id!r} metadata must be an object")

    required_metadata = {
        "adapter_schema_version": "barun-mobile-actions-adapter-v2",
        "dataset": "google/mobile-actions",
        "derived_split": "train",
        "prompt_contract_sha256": _PROMPT_CONTRACT_SHA256,
        "prompt_contract_version": "barun-action-prompt-v1",
        "source_revision": _SOURCE_REVISION,
        "source_split": "train",
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise DirectActionReplayError(
                f"source row {example_id!r} metadata.{key} must equal {expected!r}"
            )
    for key in ("data_role", "evaluation_role", "population_role", "role", "split_role"):
        value = metadata.get(key)
        if type(value) is str and value.casefold() in _FORBIDDEN_ROLES:
            raise DirectActionReplayError(
                f"source row {example_id!r} belongs to forbidden {value!r} membership"
            )
    forbidden_key = _forbidden_metadata_key(metadata)
    if forbidden_key is not None:
        raise DirectActionReplayError(
            f"source row {example_id!r} contains forbidden metadata field {forbidden_key!r}"
        )
    if "--mbcf-" in example_id.casefold():
        raise DirectActionReplayError(f"source row {example_id!r} is an MBCF derivative")
    if any(marker in prompt for marker in _FORBIDDEN_PROMPT_MARKERS):
        raise DirectActionReplayError(f"source row {example_id!r} contains a PlanIR prompt")

    prompt_sha256 = _sha256(prompt.encode("utf-8"))
    target_sha256 = _validate_direct_target(target, example_id=example_id)
    if metadata.get("prompt_sha256") != prompt_sha256:
        raise DirectActionReplayError(f"source row {example_id!r} prompt hash changed")
    if metadata.get("target_sha256") != target_sha256:
        raise DirectActionReplayError(f"source row {example_id!r} target hash changed")
    content_sha256 = _sha256(text.encode("utf-8"))
    return _ValidatedRow(
        raw_line=bytes(raw_line),
        payload=payload,
        example_id=example_id,
        prompt_sha256=prompt_sha256,
        target_sha256=target_sha256,
        content_sha256=content_sha256,
    )


def _validate_exclusion(snapshot: bytes, contract: _ReplayContract) -> dict[str, Any]:
    if _sha256(snapshot) != contract.exclusion_ledger_sha256:
        raise DirectActionReplayError("exclusion ledger SHA-256 changed")
    lines = snapshot.splitlines(keepends=True)
    if len(lines) != contract.exclusion_ledger_rows:
        raise DirectActionReplayError("exclusion ledger row count changed")
    line = lines[0]
    if not line.endswith(b"\n") or line.endswith(b"\r\n"):
        raise DirectActionReplayError("exclusion ledger must end with one LF")
    try:
        text = line[:-1].decode("utf-8")
    except UnicodeDecodeError as error:
        raise DirectActionReplayError("exclusion ledger is not UTF-8") from error
    record = _strict_json(text, label="exclusion ledger")
    if type(record) is not dict or _canonical_json(record) != text:
        raise DirectActionReplayError("exclusion ledger is not one canonical object")
    expected = {
        "code": contract.excluded_code,
        "component_id": contract.excluded_component_id,
        "excluded_from_arms": ["A", "B", "C"],
        "family_id": contract.excluded_family_id,
        "id": contract.excluded_id,
        "population": "screen",
        "schema_version": "barun-mobile-planir-screen-exclusion-v1",
        "singleton_component": True,
        "source_content_sha256": contract.excluded_content_sha256,
        "source_prompt_sha256": contract.excluded_prompt_sha256,
        "source_target_sha256": contract.excluded_target_sha256,
        "split_version": "barun-mobile-construction-component-screen-split-v1",
    }
    if record != expected:
        raise DirectActionReplayError("exclusion ledger content changed")
    return record


def _materialize_snapshots(
    source_snapshot: bytes | bytearray,
    exclusion_snapshot: bytes | bytearray,
    *,
    contract: _ReplayContract,
) -> DirectActionReplayMaterialization:
    """Internal deterministic builder; snapshots mutable bytearrays exactly once."""

    if type(source_snapshot) not in {bytes, bytearray}:
        raise TypeError("source snapshot must be exact bytes or bytearray")
    if type(exclusion_snapshot) not in {bytes, bytearray}:
        raise TypeError("exclusion snapshot must be exact bytes or bytearray")
    source = bytes(source_snapshot)
    exclusion = bytes(exclusion_snapshot)

    if _sha256(source) != contract.source_sha256:
        raise DirectActionReplayError("construction source SHA-256 changed")
    exclusion_record = _validate_exclusion(exclusion, contract)
    source_lines = source.splitlines(keepends=True)
    if len(source_lines) != contract.source_rows:
        raise DirectActionReplayError("construction source row count changed")

    rows = [
        _validate_source_row(raw_line, line_number=index)
        for index, raw_line in enumerate(source_lines, start=1)
    ]
    source_ids = [row.example_id for row in rows]
    if _membership_sha256(source_ids) != contract.source_membership_sha256:
        raise DirectActionReplayError("construction source membership changed")
    if _ordered_sha256(source_ids) != contract.source_order_sha256:
        raise DirectActionReplayError("construction source ordering changed")

    excluded = [row for row in rows if row.example_id == contract.excluded_id]
    if len(excluded) != 1:
        raise DirectActionReplayError("pinned exclusion must match exactly one source row")
    excluded_row = excluded[0]
    if (
        excluded_row.content_sha256 != contract.excluded_content_sha256
        or excluded_row.prompt_sha256 != contract.excluded_prompt_sha256
        or excluded_row.target_sha256 != contract.excluded_target_sha256
        or excluded_row.payload["metadata"].get("cluster_id") != contract.excluded_component_id
        or excluded_row.payload["metadata"].get("family_id") != contract.excluded_family_id
    ):
        raise DirectActionReplayError("pinned exclusion does not bind the exact source row")

    accepted = [row for row in rows if row.example_id != contract.excluded_id]
    accepted_ids = [row.example_id for row in accepted]
    manifest_bytes = b"".join(row.raw_line for row in accepted)
    if len(accepted) != contract.output_rows:
        raise DirectActionReplayError("direct replay row count changed")
    if _membership_sha256(accepted_ids) != contract.output_membership_sha256:
        raise DirectActionReplayError("direct replay membership changed")
    if _ordered_sha256(accepted_ids) != contract.output_order_sha256:
        raise DirectActionReplayError("direct replay ordering changed")
    if _target_commitment_sha256(accepted) != contract.direct_target_commitment_sha256:
        raise DirectActionReplayError("direct target commitment changed")
    if _sha256(manifest_bytes) != contract.output_sha256:
        raise DirectActionReplayError("direct replay bytes changed")

    receipt = {
        "authorizes_model_access": False,
        "exclusion": {
            "code": exclusion_record["code"],
            "id": exclusion_record["id"],
            "ledger_rows": contract.exclusion_ledger_rows,
            "ledger_sha256": contract.exclusion_ledger_sha256,
            "source_content_sha256": contract.excluded_content_sha256,
            "source_prompt_sha256": contract.excluded_prompt_sha256,
            "source_target_sha256": contract.excluded_target_sha256,
        },
        "forbidden_v3_populations": {
            "confirmation_membership_sha256": V3_CONFIRMATION_MEMBERSHIP_SHA256,
            "confirmation_rows": V3_CONFIRMATION_ROWS,
            "confirmation_rows_in_output": 0,
            "selection_membership_sha256": V3_SELECTION_MEMBERSHIP_SHA256,
            "selection_rows": V3_SELECTION_ROWS,
            "selection_rows_in_output": 0,
            "shadow_audit_sha256": V3_SHADOW_AUDIT_SHA256,
        },
        "fresh_evaluation": False,
        "output": {
            "direct_action_ir_targets": True,
            "direct_target_commitment_sha256": contract.direct_target_commitment_sha256,
            "membership_sha256": contract.output_membership_sha256,
            "ordered_membership_sha256": contract.output_order_sha256,
            "rows": contract.output_rows,
            "sha256": contract.output_sha256,
            "source_order_preserved": True,
            "target_bytes_preserved": True,
        },
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "selection_eligible": False,
        "source": {
            "adapter_schema_version": "barun-mobile-actions-adapter-v2",
            "dataset": "google/mobile-actions",
            "membership_sha256": contract.source_membership_sha256,
            "ordered_membership_sha256": contract.source_order_sha256,
            "revision": _SOURCE_REVISION,
            "rows": contract.source_rows,
            "sha256": contract.source_sha256,
        },
        "training_only": True,
    }
    receipt_bytes = (_canonical_json(receipt) + "\n").encode("utf-8")
    return DirectActionReplayMaterialization(
        manifest_bytes=manifest_bytes,
        receipt_bytes=receipt_bytes,
    )


def materialize_direct_action_replay(
    source_manifest_path: str | Path,
    exclusion_ledger_path: str | Path,
) -> DirectActionReplayMaterialization:
    """Build the one pinned replay from detached file snapshots without writing files."""

    source_path = Path(source_manifest_path)
    exclusion_path = Path(exclusion_ledger_path)
    try:
        source_snapshot = source_path.read_bytes()
    except OSError as error:
        raise DirectActionReplayError(f"cannot read construction source: {error}") from error
    try:
        exclusion_snapshot = exclusion_path.read_bytes()
    except OSError as error:
        raise DirectActionReplayError(f"cannot read exclusion ledger: {error}") from error
    return _materialize_snapshots(
        source_snapshot,
        exclusion_snapshot,
        contract=_PINNED_CONTRACT,
    )


__all__ = [
    "DIRECT_TARGET_COMMITMENT_SHA256",
    "EXCLUDED_CODE",
    "EXCLUDED_ID",
    "EXCLUSION_LEDGER_SHA256",
    "OUTPUT_MANIFEST_SHA256",
    "OUTPUT_MEMBERSHIP_SHA256",
    "OUTPUT_ROWS",
    "RECEIPT_SCHEMA_VERSION",
    "REPLAY_SCHEMA_VERSION",
    "SOURCE_MANIFEST_SHA256",
    "SOURCE_MEMBERSHIP_SHA256",
    "SOURCE_ROWS",
    "DirectActionReplayError",
    "DirectActionReplayMaterialization",
    "materialize_direct_action_replay",
]
