"""Pinned Google Mobile Actions adapter for the BarunAction SFT probe.

The public entry point refuses any source bytes, row counts, or split labels that
do not match the frozen dataset revision.  Tests exercise the same pipeline with
an explicit synthetic ``SourceSpec``; no test downloads public data.

The pinned file differs from its dataset card in two material ways: tool-call
``arguments`` are JSON objects rather than stringified JSON, and call objects omit
the documented ``id`` and ``type`` fields.  This adapter intentionally accepts the
pinned on-disk representation only.  Supporting a second representation would hide
an upstream format change behind coercion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

MOBILE_ACTIONS_DATASET_ID = "google/mobile-actions"
MOBILE_ACTIONS_REVISION = "e920309bc2acbc2e99a5e3201cf37df2b9fd9151"
MOBILE_ACTIONS_SOURCE_FILENAME = "dataset.jsonl"
MOBILE_ACTIONS_SOURCE_SHA256 = "91d251ee958cfd295af6c4504c236a3a1ad19517de240c3bc680bacfcbf7e7d9"
MOBILE_ACTIONS_LICENSE = "CC-BY-4.0"
MOBILE_ACTIONS_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
MOBILE_ACTIONS_SOURCE_URL = (
    f"https://huggingface.co/datasets/google/mobile-actions/tree/{MOBILE_ACTIONS_REVISION}"
)
MOBILE_ACTIONS_TOTAL_ROWS = 9_654
MOBILE_ACTIONS_INTERNAL_TRAIN_ROWS = 8_693
MOBILE_ACTIONS_FINAL_EVAL_ROWS = 961
MOBILE_ACTIONS_TOOL_NAMES = frozenset(
    {
        "create_calendar_event",
        "create_contact",
        "open_wifi_settings",
        "send_email",
        "show_map",
        "turn_off_flashlight",
        "turn_on_flashlight",
    }
)

MANIFEST_SCHEMA_VERSION = "barun-sft-example-v1"
ADAPTER_SCHEMA_VERSION = "barun-mobile-actions-adapter-v2"
AUDIT_SCHEMA_VERSION = "barun-mobile-actions-audit-v2"
PROMPT_CONTRACT_VERSION = "barun-action-prompt-v1"
PROMPT_TEMPLATE = (
    "<bos><system>\nACTION_IR_V1\nNOW {now}\nTOOLS\n{tools}\n<user>\n{user}\n<assistant>\n"
)
PROMPT_TEMPLATE_SHA256 = hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()
FAMILY_POLICY_VERSION = "barun-mobile-actions-family-v1"
COMPONENT_POLICY_VERSION = "barun-mobile-actions-connected-components-v1"
# Retained as the public split-policy name.  A split cluster is now a connected
# component, not merely one exact delexicalized family.
CLUSTER_POLICY_VERSION = COMPONENT_POLICY_VERSION
DEFAULT_DEV_FOLDS = 10
DEFAULT_DEV_FOLD = 0
DEFAULT_MAX_SEQ_LEN = 2_048
MINHASH_PERMUTATIONS = 64
MINHASH_BANDS = 16
CHAR_NGRAM_SIZE = 5
NEAR_DUPLICATE_JACCARD_THRESHOLD = 0.80

_NOW_RE = re.compile(
    r"\ACurrent date and time given in YYYY-MM-DDTHH:MM:SS format: "
    r"(?P<now>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\n"
    r"Day of week is (?P<weekday>Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\n"
    r"You are a model that can do function calling with the following functions\n\Z"
)
_WHITESPACE_RE = re.compile(r"\s+")
_EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
_ISO_DATETIME_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)?\b",
    flags=re.IGNORECASE,
)
_MONTH_DATE_RE = re.compile(
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)?\s*,?\s*"
    r"(?:january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?\b",
    flags=re.IGNORECASE,
)
_RELATIVE_DATE_RE = re.compile(
    r"\b(?:(?:this|next|coming)\s+)?(?:monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday)|\b(?:today|tomorrow|tonight)\b",
    flags=re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"\b(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\b|\b\d{1,2}:\d{2}\b",
    flags=re.IGNORECASE,
)
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{6,}\d)(?!\w)")
_QUOTED_RE = re.compile(r'"[^"\n]+"|(?<!\w)\'[^\'\n]+\'(?!\w)')
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_SOURCE_SPLIT_PREFIX_RE = re.compile(rb'\A\s*\{\s*"metadata"\s*:\s*"(?P<split>train|eval)"\s*,')


class MobileActionsError(ValueError):
    """The pinned source or a derived artifact violates the frozen contract."""


class TokenizerLike(Protocol):
    """Minimal interface shared by tokenizers.Tokenizer and test doubles."""

    def encode(self, sequence: str, add_special_tokens: bool = False) -> Any: ...


@dataclass(frozen=True, slots=True)
class SourceSpec:
    dataset_id: str
    revision: str
    filename: str
    source_sha256: str
    total_rows: int
    internal_train_rows: int
    final_eval_rows: int
    license_id: str
    license_url: str
    source_url: str
    tool_names: frozenset[str]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
        if self.total_rows != self.internal_train_rows + self.final_eval_rows:
            raise ValueError("source row counts do not add up")
        if not self.tool_names:
            raise ValueError("tool_names cannot be empty")


OFFICIAL_SOURCE_SPEC = SourceSpec(
    dataset_id=MOBILE_ACTIONS_DATASET_ID,
    revision=MOBILE_ACTIONS_REVISION,
    filename=MOBILE_ACTIONS_SOURCE_FILENAME,
    source_sha256=MOBILE_ACTIONS_SOURCE_SHA256,
    total_rows=MOBILE_ACTIONS_TOTAL_ROWS,
    internal_train_rows=MOBILE_ACTIONS_INTERNAL_TRAIN_ROWS,
    final_eval_rows=MOBILE_ACTIONS_FINAL_EVAL_ROWS,
    license_id=MOBILE_ACTIONS_LICENSE,
    license_url=MOBILE_ACTIONS_LICENSE_URL,
    source_url=MOBILE_ACTIONS_SOURCE_URL,
    tool_names=MOBILE_ACTIONS_TOOL_NAMES,
)


@dataclass(frozen=True, slots=True)
class SplitPolicy:
    dev_folds: int = DEFAULT_DEV_FOLDS
    dev_fold: int = DEFAULT_DEV_FOLD
    version: str = CLUSTER_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.dev_folds < 2:
            raise ValueError("dev_folds must be at least 2")
        if not 0 <= self.dev_fold < self.dev_folds:
            raise ValueError("dev_fold must be in [0, dev_folds)")


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
class ToolArgument:
    name: str
    type_name: str
    description: str
    required: bool


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    arguments: tuple[ToolArgument, ...]


@dataclass(frozen=True, slots=True)
class PreparedExample:
    example_id: str
    source_line: int
    source_split: str
    derived_split: str
    raw_sha256: str
    prompt: str
    target: str
    user_text: str
    normalized_user: str
    delexicalized_user: str
    family_id: str
    cluster_id: str
    schema_signature: str
    tool_order: tuple[str, ...]
    call_names: tuple[str, ...]
    argument_names: tuple[tuple[str, ...], ...]
    now: str
    null_argument_fields_excluded: int
    nfc_normalizations: int
    prompt_tokens: int | None = None
    target_tokens: int | None = None
    total_tokens_with_eos: int | None = None

    def with_split(self, derived_split: str, *, cluster_id: str) -> PreparedExample:
        return replace(self, derived_split=derived_split, cluster_id=cluster_id)

    def with_token_lengths(
        self,
        *,
        prompt_tokens: int,
        target_tokens: int,
    ) -> PreparedExample:
        return replace(
            self,
            prompt_tokens=prompt_tokens,
            target_tokens=target_tokens,
            total_tokens_with_eos=prompt_tokens + target_tokens + 1,
        )

    def to_manifest_record(self, source_spec: SourceSpec) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
            "call_names": list(self.call_names),
            "cluster_id": self.cluster_id,
            "dataset": source_spec.dataset_id,
            "derived_split": self.derived_split,
            "family_id": self.family_id,
            "prompt_sha256": _sha256_bytes(self.prompt.encode("utf-8")),
            "prompt_contract_version": PROMPT_CONTRACT_VERSION,
            "prompt_contract_sha256": PROMPT_TEMPLATE_SHA256,
            "raw_sha256": self.raw_sha256,
            "schema_signature": self.schema_signature,
            "source_line": self.source_line,
            "source_revision": source_spec.revision,
            "source_split": self.source_split,
            "source_timezone": "UNSPECIFIED",
            "target_sha256": _sha256_bytes(self.target.encode("utf-8")),
            "tool_order": list(self.tool_order),
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
class PreparationResult:
    train_manifest: Path
    dev_manifest: Path
    audit_path: Path
    train_rows: int
    dev_rows: int
    final_eval_rows: int
    hashes: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _SourceRead:
    """Training examples plus source-level evidence about the opaque eval tail."""

    train_examples: tuple[PreparedExample, ...]
    total_rows: int
    internal_train_rows: int
    official_eval_rows: int
    source_sha256: str


@dataclass(frozen=True, slots=True)
class _NearDuplicateScan:
    candidate_pairs: int
    verified_pairs: tuple[tuple[int, int, float], ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise MobileActionsError(f"value is not strict JSON: {exc}") from exc


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise MobileActionsError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> None:
    raise MobileActionsError(f"non-finite JSON constant {value!r}")


def _decode_source_record(raw: bytes, line_number: int) -> Mapping[str, Any]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise MobileActionsError(f"line {line_number}: source is not valid UTF-8") from exc
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, MobileActionsError) as exc:
        raise MobileActionsError(f"line {line_number}: invalid JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise MobileActionsError(f"line {line_number}: source row must be an object")
    return decoded


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise MobileActionsError(f"{path}: expected keys {sorted(expected)}, got {sorted(actual)}")


def _nfc(value: str) -> tuple[str, int]:
    normalized = unicodedata.normalize("NFC", value)
    return normalized, int(normalized != value)


def _parse_tools(raw_tools: Any, path: str) -> tuple[tuple[ToolDefinition, ...], int]:
    if not isinstance(raw_tools, list) or not raw_tools:
        raise MobileActionsError(f"{path}: tools must be a non-empty list")
    definitions: list[ToolDefinition] = []
    seen_names: set[str] = set()
    nfc_changes = 0
    for tool_index, raw_tool in enumerate(raw_tools):
        tool_path = f"{path}[{tool_index}]"
        if not isinstance(raw_tool, dict):
            raise MobileActionsError(f"{tool_path}: tool must be an object")
        _require_exact_keys(raw_tool, {"function"}, tool_path)
        function = raw_tool["function"]
        if not isinstance(function, dict):
            raise MobileActionsError(f"{tool_path}.function: must be an object")
        _require_exact_keys(
            function, {"name", "description", "parameters"}, f"{tool_path}.function"
        )
        raw_name = function["name"]
        raw_description = function["description"]
        parameters = function["parameters"]
        if not isinstance(raw_name, str) or not raw_name:
            raise MobileActionsError(f"{tool_path}.function.name: must be a non-empty string")
        if not isinstance(raw_description, str) or "\n" in raw_description:
            raise MobileActionsError(
                f"{tool_path}.function.description: must be a single-line string"
            )
        name, changed = _nfc(raw_name)
        nfc_changes += changed
        description, changed = _nfc(raw_description)
        nfc_changes += changed
        if name in seen_names:
            raise MobileActionsError(f"{path}: duplicate tool name {name!r}")
        seen_names.add(name)
        if not isinstance(parameters, dict):
            raise MobileActionsError(f"{tool_path}.function.parameters: must be an object")
        allowed_parameter_keys = {"type", "properties", "required"}
        unknown_parameter_keys = set(parameters).difference(allowed_parameter_keys)
        if unknown_parameter_keys or not {"type", "properties"}.issubset(parameters):
            raise MobileActionsError(
                f"{tool_path}.function.parameters: unsupported keys or missing type/properties"
            )
        if parameters["type"] != "OBJECT":
            raise MobileActionsError(
                f"{tool_path}.function.parameters.type: pinned format requires 'OBJECT'"
            )
        properties = parameters["properties"]
        required = parameters.get("required", [])
        if not isinstance(properties, dict):
            raise MobileActionsError(
                f"{tool_path}.function.parameters.properties: must be an object"
            )
        if (
            not isinstance(required, list)
            or any(not isinstance(item, str) or not item for item in required)
            or len(required) != len(set(required))
        ):
            raise MobileActionsError(
                f"{tool_path}.function.parameters.required: must be a unique string list"
            )
        if not set(required).issubset(properties):
            raise MobileActionsError(
                f"{tool_path}.function.parameters.required: contains an undeclared argument"
            )
        arguments: list[ToolArgument] = []
        for raw_argument_name, raw_schema in properties.items():
            argument_path = f"{tool_path}.function.parameters.properties.{raw_argument_name}"
            if not isinstance(raw_argument_name, str) or not raw_argument_name:
                raise MobileActionsError(f"{argument_path}: argument name must be non-empty")
            if not isinstance(raw_schema, dict):
                raise MobileActionsError(f"{argument_path}: schema must be an object")
            _require_exact_keys(raw_schema, {"type", "description"}, argument_path)
            if raw_schema["type"] != "STRING":
                raise MobileActionsError(
                    f"{argument_path}.type: pinned Mobile Actions format requires 'STRING'"
                )
            raw_argument_description = raw_schema["description"]
            if not isinstance(raw_argument_description, str) or "\n" in raw_argument_description:
                raise MobileActionsError(
                    f"{argument_path}.description: must be a single-line string"
                )
            argument_name, changed = _nfc(raw_argument_name)
            nfc_changes += changed
            argument_description, changed = _nfc(raw_argument_description)
            nfc_changes += changed
            arguments.append(
                ToolArgument(
                    name=argument_name,
                    type_name="string",
                    description=argument_description,
                    required=raw_argument_name in required,
                )
            )
        definitions.append(
            ToolDefinition(name=name, description=description, arguments=tuple(arguments))
        )
    return tuple(definitions), nfc_changes


def _normalize_argument_value(value: Any, path: str) -> tuple[Any, int, int]:
    if value is None or isinstance(value, (bool, int)):
        return value, 0, 0
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MobileActionsError(f"{path}: argument number must be finite")
        return value, 0, 0
    if isinstance(value, str):
        normalized, changed = _nfc(value)
        return normalized, 0, changed
    if isinstance(value, list):
        normalized_list: list[Any] = []
        dropped = 0
        nfc_changes = 0
        for index, item in enumerate(value):
            child, child_dropped, child_changes = _normalize_argument_value(
                item, f"{path}[{index}]"
            )
            normalized_list.append(child)
            dropped += child_dropped
            nfc_changes += child_changes
        return normalized_list, dropped, nfc_changes
    if isinstance(value, dict):
        normalized_dict: dict[str, Any] = {}
        dropped = 0
        nfc_changes = 0
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise MobileActionsError(f"{path}: argument object keys must be strings")
            key, changed = _nfc(raw_key)
            nfc_changes += changed
            if key in normalized_dict:
                raise MobileActionsError(f"{path}: duplicate key after NFC normalization {key!r}")
            if item is None:
                dropped += 1
                continue
            child, child_dropped, child_changes = _normalize_argument_value(item, f"{path}.{key}")
            normalized_dict[key] = child
            dropped += child_dropped
            nfc_changes += child_changes
        return normalized_dict, dropped, nfc_changes
    raise MobileActionsError(f"{path}: unsupported argument type {type(value).__name__}")


def _parse_calls(
    raw_calls: Any,
    tools: Sequence[ToolDefinition],
    path: str,
) -> tuple[tuple[dict[str, Any], ...], int, int]:
    if not isinstance(raw_calls, list) or not raw_calls:
        raise MobileActionsError(f"{path}: tool_calls must be a non-empty list")
    tool_map = {tool.name: tool for tool in tools}
    calls: list[dict[str, Any]] = []
    dropped = 0
    nfc_changes = 0
    for call_index, raw_call in enumerate(raw_calls):
        call_path = f"{path}[{call_index}]"
        if not isinstance(raw_call, dict):
            raise MobileActionsError(f"{call_path}: call must be an object")
        _require_exact_keys(raw_call, {"function"}, call_path)
        function = raw_call["function"]
        if not isinstance(function, dict):
            raise MobileActionsError(f"{call_path}.function: must be an object")
        _require_exact_keys(function, {"name", "arguments"}, f"{call_path}.function")
        raw_name = function["name"]
        if not isinstance(raw_name, str) or not raw_name:
            raise MobileActionsError(f"{call_path}.function.name: must be a non-empty string")
        name, changed = _nfc(raw_name)
        nfc_changes += changed
        if name not in tool_map:
            raise MobileActionsError(f"{call_path}.function.name: undeclared tool {name!r}")
        raw_arguments = function["arguments"]
        if not isinstance(raw_arguments, dict):
            raise MobileActionsError(
                f"{call_path}.function.arguments: pinned format requires a JSON object"
            )
        normalized_arguments, child_dropped, child_changes = _normalize_argument_value(
            raw_arguments, f"{call_path}.function.arguments"
        )
        assert isinstance(normalized_arguments, dict)
        dropped += child_dropped
        nfc_changes += child_changes
        definition = tool_map[name]
        declared = {argument.name: argument for argument in definition.arguments}
        unknown = set(normalized_arguments).difference(declared)
        if unknown:
            raise MobileActionsError(
                f"{call_path}.function.arguments: undeclared fields {sorted(unknown)}"
            )
        missing = {
            argument.name
            for argument in definition.arguments
            if argument.required and argument.name not in normalized_arguments
        }
        if missing:
            raise MobileActionsError(
                f"{call_path}.function.arguments: missing required fields {sorted(missing)}"
            )
        for argument_name, value in normalized_arguments.items():
            if not isinstance(value, str):
                raise MobileActionsError(
                    f"{call_path}.function.arguments.{argument_name}: expected string"
                )
        calls.append({"tool": name, "args": normalized_arguments})
    return tuple(calls), dropped, nfc_changes


def _render_tool(tool: ToolDefinition) -> str:
    rendered_arguments = ", ".join(
        f"{argument.name}:{argument.type_name}{'!' if argument.required else ''}"
        for argument in tool.arguments
    )
    return f"{tool.name}({rendered_arguments}): {tool.description}"


def render_prompt(*, now: str, tools: Sequence[ToolDefinition], user_text: str) -> str:
    """Render the frozen Barun prompt without target text or EOS."""

    tool_lines = "\n".join(_render_tool(tool) for tool in tools)
    return PROMPT_TEMPLATE.format(now=now, tools=tool_lines, user=user_text)


def _normalized_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFC", value).casefold()).strip()


def _literal_value_pattern(value: str) -> re.Pattern[str] | None:
    normalized = _normalized_text(value)
    if len(normalized) < 2:
        return None
    escaped_parts = [re.escape(part) for part in normalized.split(" ")]
    return re.compile(r"(?<!\w)" + r"\s+".join(escaped_parts) + r"(?!\w)", re.IGNORECASE)


def delexicalize_user(user_text: str, calls: Sequence[Mapping[str, Any]]) -> str:
    """Create a stable entity-delexicalized template for grouped splitting.

    Exact string arguments are replaced first, longest first, followed by generic
    deterministic patterns for dates, times, email addresses, phone numbers,
    quoted spans, and remaining numbers.  The result is diagnostic grouping data,
    never a training target.
    """

    text = _normalized_text(user_text)
    replacements: list[tuple[str, str]] = []
    for call in calls:
        arguments = call["args"]
        assert isinstance(arguments, Mapping)
        for name, value in arguments.items():
            if isinstance(value, str) and value.strip():
                replacements.append((value, f"<{name}>"))
    for value, marker in sorted(replacements, key=lambda item: (-len(item[0]), item[1])):
        pattern = _literal_value_pattern(value)
        if pattern is not None:
            text = pattern.sub(marker, text)
    for pattern, marker in (
        (_EMAIL_RE, "<email>"),
        (_ISO_DATETIME_RE, "<datetime>"),
        (_MONTH_DATE_RE, "<date>"),
        (_RELATIVE_DATE_RE, "<date>"),
        (_TIME_RE, "<time>"),
        (_PHONE_RE, "<phone>"),
        (_QUOTED_RE, "<quoted>"),
        (_NUMBER_RE, "<number>"),
    ):
        text = pattern.sub(marker, text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _schema_signature(tools: Sequence[ToolDefinition]) -> str:
    semantic_schema = [
        {
            "name": tool.name,
            "arguments": [
                {
                    "name": argument.name,
                    "required": argument.required,
                    "type": argument.type_name,
                }
                for argument in sorted(tool.arguments, key=lambda item: item.name)
            ],
        }
        for tool in sorted(tools, key=lambda item: item.name)
    ]
    return _sha256_bytes(_canonical_json(semantic_schema).encode("utf-8"))


def _family_id(user_template: str, calls: Sequence[Mapping[str, Any]]) -> str:
    call_signature = [call["tool"] for call in calls]
    payload = {
        "call_signature": call_signature,
        "policy": FAMILY_POLICY_VERSION,
        "template": user_template,
    }
    digest = _sha256_bytes(_canonical_json(payload).encode("utf-8"))
    return f"ma-family-v1-{digest}"


def derived_split_for_cluster(cluster_id: str, policy: SplitPolicy | None = None) -> str:
    policy = policy or SplitPolicy()
    digest = hashlib.sha256(f"{policy.version}:{cluster_id}".encode()).digest()
    fold = int.from_bytes(digest[:8], byteorder="big") % policy.dev_folds
    return "dev" if fold == policy.dev_fold else "train"


def convert_record(
    record: Mapping[str, Any],
    *,
    raw_sha256: str,
    source_line: int,
    source_spec: SourceSpec = OFFICIAL_SOURCE_SPEC,
) -> PreparedExample:
    """Convert one internal-training record into Action IR v1.

    The train/dev adapter intentionally has no conversion path for the official
    eval population.  A future evaluator must expose that population through a
    separate, explicit unblind API.
    """

    _require_exact_keys(record, {"metadata", "tools", "messages"}, f"line {source_line}")
    source_split = record["metadata"]
    if source_split != "train":
        raise MobileActionsError(
            f"line {source_line}.metadata: train/dev preparation accepts only 'train'; "
            "official eval rows are opaque"
        )
    tools, nfc_changes = _parse_tools(record["tools"], f"line {source_line}.tools")
    actual_tool_names = frozenset(tool.name for tool in tools)
    if actual_tool_names != source_spec.tool_names:
        raise MobileActionsError(
            f"line {source_line}.tools: expected tool set {sorted(source_spec.tool_names)}, "
            f"got {sorted(actual_tool_names)}"
        )

    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) != 3:
        raise MobileActionsError(f"line {source_line}.messages: expected exactly three messages")
    expected_message_keys = (
        {"role", "content"},
        {"role", "content"},
        {"role", "tool_calls"},
    )
    expected_roles = ("developer", "user", "assistant")
    for index, (message, keys, role) in enumerate(
        zip(messages, expected_message_keys, expected_roles, strict=True)
    ):
        if not isinstance(message, dict):
            raise MobileActionsError(
                f"line {source_line}.messages[{index}]: message must be an object"
            )
        _require_exact_keys(message, keys, f"line {source_line}.messages[{index}]")
        if message["role"] != role:
            raise MobileActionsError(
                f"line {source_line}.messages[{index}].role: expected {role!r}"
            )

    developer_content = messages[0]["content"]
    if not isinstance(developer_content, str):
        raise MobileActionsError(f"line {source_line}.messages[0].content: must be a string")
    now_match = _NOW_RE.fullmatch(developer_content)
    if now_match is None:
        raise MobileActionsError(
            f"line {source_line}.messages[0].content: unsupported pinned timestamp format"
        )
    now = now_match.group("now")
    try:
        parsed_now = datetime.fromisoformat(now)
    except ValueError as exc:
        raise MobileActionsError(
            f"line {source_line}.messages[0].content: invalid timestamp {now!r}"
        ) from exc
    if parsed_now.strftime("%A") != now_match.group("weekday"):
        raise MobileActionsError(
            f"line {source_line}.messages[0].content: weekday disagrees with timestamp"
        )

    raw_user_text = messages[1]["content"]
    if not isinstance(raw_user_text, str) or not raw_user_text.strip():
        raise MobileActionsError(f"line {source_line}.messages[1].content: must be non-empty")
    user_text, changed = _nfc(raw_user_text)
    nfc_changes += changed
    calls, dropped, call_nfc_changes = _parse_calls(
        messages[2]["tool_calls"], tools, f"line {source_line}.messages[2].tool_calls"
    )
    nfc_changes += call_nfc_changes
    mode = "SINGLE" if len(calls) == 1 else "SERIAL"
    target = _canonical_json({"calls": list(calls), "decision": "CALL", "mode": mode})
    prompt = render_prompt(now=now, tools=tools, user_text=user_text)
    delexicalized = delexicalize_user(user_text, calls)
    family_id = _family_id(delexicalized, calls)
    example_id = f"mobile-actions-{source_line:05d}-{raw_sha256[:16]}"
    return PreparedExample(
        example_id=example_id,
        source_line=source_line,
        source_split=source_split,
        derived_split="unassigned",
        raw_sha256=raw_sha256,
        prompt=prompt,
        target=target,
        user_text=user_text,
        normalized_user=_normalized_text(user_text),
        delexicalized_user=delexicalized,
        family_id=family_id,
        cluster_id=family_id,
        schema_signature=_schema_signature(tools),
        tool_order=tuple(tool.name for tool in tools),
        call_names=tuple(call["tool"] for call in calls),
        argument_names=tuple(tuple(sorted(call["args"])) for call in calls),
        now=now,
        null_argument_fields_excluded=dropped,
        nfc_normalizations=nfc_changes,
    )


def _read_source(path: Path, source_spec: SourceSpec) -> _SourceRead:
    """Read internal-train rows while treating official eval rows as opaque bytes.

    The whole-file digest is verified before any row conversion.  On the second
    pass only the leading ``metadata`` split marker is inspected.  Eval rows are
    counted and immediately discarded: they are not UTF-8 decoded, JSON parsed,
    hashed individually, converted, tokenized, summarized, or materialized.
    """

    actual_sha256 = sha256_file(path)
    if actual_sha256 != source_spec.source_sha256:
        raise MobileActionsError(
            f"source SHA-256 mismatch for {path}: expected {source_spec.source_sha256}, "
            f"got {actual_sha256}"
        )
    train_examples: list[PreparedExample] = []
    split_counts: Counter[str] = Counter()
    total_rows = 0
    with path.open("rb") as handle:
        for line_number, line_with_newline in enumerate(handle, start=1):
            raw = line_with_newline[:-1] if line_with_newline.endswith(b"\n") else line_with_newline
            if not raw:
                raise MobileActionsError(f"line {line_number}: blank source rows are not allowed")
            total_rows += 1
            prefix_match = _SOURCE_SPLIT_PREFIX_RE.match(raw)
            if prefix_match is None:
                raise MobileActionsError(
                    f"line {line_number}: expected a leading train/eval metadata split marker"
                )
            source_split = prefix_match.group("split").decode("ascii")
            split_counts[source_split] += 1
            if source_split == "eval":
                continue
            raw_sha256 = _sha256_bytes(raw)
            record = _decode_source_record(raw, line_number)
            train_examples.append(
                convert_record(
                    record,
                    raw_sha256=raw_sha256,
                    source_line=line_number,
                    source_spec=source_spec,
                )
            )
    if total_rows != source_spec.total_rows:
        raise MobileActionsError(
            f"source row count mismatch: expected {source_spec.total_rows}, got {total_rows}"
        )
    if split_counts != Counter(
        {"train": source_spec.internal_train_rows, "eval": source_spec.final_eval_rows}
    ):
        raise MobileActionsError(
            "source split counts mismatch: expected "
            f"train={source_spec.internal_train_rows}, eval={source_spec.final_eval_rows}; "
            f"got {dict(sorted(split_counts.items()))}"
        )
    if len({example.raw_sha256 for example in train_examples}) != len(train_examples):
        raise MobileActionsError("internal-train source contains duplicate raw rows")
    if len({example.example_id for example in train_examples}) != len(train_examples):
        raise MobileActionsError("derived example IDs are not unique")
    return _SourceRead(
        train_examples=tuple(train_examples),
        total_rows=total_rows,
        internal_train_rows=split_counts["train"],
        official_eval_rows=split_counts["eval"],
        source_sha256=actual_sha256,
    )


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, item: int) -> int:
        parent = self._parent[item]
        if parent != item:
            self._parent[item] = self.find(parent)
        return self._parent[item]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        left_rank = self._rank[left_root]
        right_rank = self._rank[right_root]
        if left_rank < right_rank or (left_rank == right_rank and left_root > right_root):
            left_root, right_root = right_root, left_root
            left_rank, right_rank = right_rank, left_rank
        self._parent[right_root] = left_root
        if left_rank == right_rank:
            self._rank[left_root] += 1


def _component_id(family_ids: Iterable[str]) -> str:
    payload = {
        "family_ids": sorted(set(family_ids)),
        "policy": COMPONENT_POLICY_VERSION,
    }
    digest = _sha256_bytes(_canonical_json(payload).encode("utf-8"))
    return f"ma-component-v1-{digest}"


def _assign_training_splits(
    examples: Sequence[PreparedExample], policy: SplitPolicy
) -> tuple[list[PreparedExample], dict[str, Any]]:
    """Split deterministic connected components built from internal train only."""

    if not examples:
        raise MobileActionsError("internal-train source is empty")
    if any(example.source_split != "train" for example in examples):
        raise MobileActionsError("split construction accepts internal-train rows only")

    disjoint_set = _DisjointSet(len(examples))
    first_by_family: dict[str, int] = {}
    for index, example in enumerate(examples):
        previous = first_by_family.setdefault(example.family_id, index)
        disjoint_set.union(previous, index)

    near_scan = _scan_near_duplicates(examples)
    for left_index, right_index, _ in near_scan.verified_pairs:
        disjoint_set.union(left_index, right_index)

    member_indexes: dict[int, list[int]] = defaultdict(list)
    for index in range(len(examples)):
        member_indexes[disjoint_set.find(index)].append(index)

    component_for_index: dict[int, str] = {}
    component_sizes: list[int] = []
    component_splits: dict[str, str] = {}
    for indexes in member_indexes.values():
        component = _component_id(examples[index].family_id for index in indexes)
        derived_split = derived_split_for_cluster(component, policy)
        component_splits[component] = derived_split
        component_sizes.append(len(indexes))
        for index in indexes:
            component_for_index[index] = component

    if set(component_splits.values()) != {"train", "dev"}:
        raise MobileActionsError(
            "component hash produced an empty train or dev split; change policy only before freezing"
        )

    output = [
        example.with_split(
            component_splits[component_for_index[index]],
            cluster_id=component_for_index[index],
        )
        for index, example in enumerate(examples)
    ]
    ordered_sizes = sorted(component_sizes, reverse=True)
    grouping_audit = {
        "component_count": len(member_indexes),
        "family_count": len(first_by_family),
        "largest_component_rows": ordered_sizes[0],
        "near_duplicate_candidate_pairs": near_scan.candidate_pairs,
        "near_duplicate_verified_edges": len(near_scan.verified_pairs),
        "rows_in_components_larger_than_one": sum(size for size in ordered_sizes if size > 1),
        "singleton_components": sum(size == 1 for size in ordered_sizes),
    }
    return output, grouping_audit


def _encode_ids(tokenizer: TokenizerLike, text: str) -> tuple[int, ...]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = getattr(encoded, "ids", encoded)
    if not isinstance(ids, (list, tuple)) or any(not isinstance(item, int) for item in ids):
        raise MobileActionsError("tokenizer.encode must return integer ids or an object with .ids")
    return tuple(ids)


def _add_token_lengths(
    examples: Sequence[PreparedExample], tokenizer: TokenizerLike
) -> list[PreparedExample]:
    output: list[PreparedExample] = []
    for example in examples:
        prompt_tokens = len(_encode_ids(tokenizer, example.prompt))
        target_tokens = len(_encode_ids(tokenizer, example.target))
        if prompt_tokens == 0 or target_tokens == 0:
            raise MobileActionsError(
                f"example {example.example_id}: tokenizer produced a zero-token prompt or target"
            )
        output.append(
            example.with_token_lengths(
                prompt_tokens=prompt_tokens,
                target_tokens=target_tokens,
            )
        )
    return output


def _percentile(sorted_values: Sequence[int], quantile: float) -> int:
    if not sorted_values:
        raise ValueError("cannot summarize an empty sequence")
    index = math.ceil(quantile * len(sorted_values)) - 1
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def _length_summary(values: Iterable[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 6),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
    }


def _split_examples(examples: Sequence[PreparedExample]) -> dict[str, list[PreparedExample]]:
    output = {"train": [], "dev": []}
    for example in examples:
        if example.derived_split not in output:
            raise MobileActionsError(
                f"example {example.example_id}: unexpected derived split {example.derived_split!r}"
            )
        output[example.derived_split].append(example)
    return output


def _index_overlap(
    split_examples: Mapping[str, Sequence[PreparedExample]],
    value_getter: Any,
) -> dict[str, Any]:
    indexes: dict[str, dict[str, list[str]]] = {}
    for split_name, examples in split_examples.items():
        index: dict[str, list[str]] = defaultdict(list)
        for example in examples:
            index[value_getter(example)].append(example.example_id)
        indexes[split_name] = index

    within_split: dict[str, Any] = {}
    for split_name, index in indexes.items():
        duplicate_values = {key: ids for key, ids in index.items() if len(ids) > 1}
        within_split[split_name] = {
            "duplicate_pairs": sum(
                len(ids) * (len(ids) - 1) // 2 for ids in duplicate_values.values()
            ),
            "duplicate_unique_values": len(duplicate_values),
            "rows_affected": len({item for ids in duplicate_values.values() for item in ids}),
        }

    cross_split: dict[str, Any] = {}
    for left, right in (("train", "dev"),):
        shared_values = set(indexes[left]).intersection(indexes[right])
        cross_split[f"{left}__{right}"] = {
            "left_rows_affected": len(
                {item for value in shared_values for item in indexes[left][value]}
            ),
            "matching_pairs": sum(
                len(indexes[left][value]) * len(indexes[right][value]) for value in shared_values
            ),
            "right_rows_affected": len(
                {item for value in shared_values for item in indexes[right][value]}
            ),
            "shared_unique_values": len(shared_values),
        }
    return {"cross_split": cross_split, "within_split": within_split}


def _contiguous_ngrams(tokens: Sequence[Any], n: int) -> set[str]:
    if len(tokens) < n:
        return set()
    return {
        _sha256_bytes(_canonical_json(list(tokens[index : index + n])).encode("utf-8"))
        for index in range(len(tokens) - n + 1)
    }


def _ngram_overlap(
    split_examples: Mapping[str, Sequence[PreparedExample]],
    token_getter: Any,
    *,
    n: int,
) -> dict[str, Any]:
    indexes: dict[str, dict[str, set[str]]] = {}
    short_rows: dict[str, int] = {}
    for split_name, examples in split_examples.items():
        index: dict[str, set[str]] = defaultdict(set)
        short = 0
        for example in examples:
            tokens = token_getter(example)
            grams = _contiguous_ngrams(tokens, n)
            if not grams:
                short += 1
            for gram in grams:
                index[gram].add(example.example_id)
        indexes[split_name] = index
        short_rows[split_name] = short
    cross_split: dict[str, Any] = {}
    for left, right in (("train", "dev"),):
        shared = set(indexes[left]).intersection(indexes[right])
        cross_split[f"{left}__{right}"] = {
            "left_rows_affected": len({item for gram in shared for item in indexes[left][gram]}),
            "right_rows_affected": len({item for gram in shared for item in indexes[right][gram]}),
            "shared_unique_ngrams": len(shared),
        }
    return {"cross_split": cross_split, "n": n, "short_rows": short_rows}


def _character_ngrams(value: str, n: int = CHAR_NGRAM_SIZE) -> frozenset[str]:
    if len(value) < n:
        return frozenset({value})
    return frozenset(value[index : index + n] for index in range(len(value) - n + 1))


_MINHASH_PRIME = (1 << 61) - 1


def _minhash_coefficients() -> tuple[tuple[int, int], ...]:
    coefficients: list[tuple[int, int]] = []
    for index in range(MINHASH_PERMUTATIONS):
        seed = hashlib.sha256(f"barun-minhash-v1:{index}".encode("ascii")).digest()
        a = int.from_bytes(seed[:8], "big") % (_MINHASH_PRIME - 1) + 1
        b = int.from_bytes(seed[8:16], "big") % _MINHASH_PRIME
        coefficients.append((a, b))
    return tuple(coefficients)


_MINHASH_COEFFICIENTS = _minhash_coefficients()


def _minhash_signature(grams: frozenset[str]) -> tuple[int, ...]:
    base_hashes = [
        int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big")
        % _MINHASH_PRIME
        for gram in grams
    ]
    return tuple(
        min((a * value + b) % _MINHASH_PRIME for value in base_hashes)
        for a, b in _MINHASH_COEFFICIENTS
    )


def _scan_near_duplicates(examples: Sequence[PreparedExample]) -> _NearDuplicateScan:
    """Find verified char-5-gram neighbors from deterministic MinHash candidates.

    Candidate generation is deliberately approximate; every candidate edge is
    verified with exact Jaccard similarity before it can join a component.
    """

    rows_per_band = MINHASH_PERMUTATIONS // MINHASH_BANDS
    row_data: list[tuple[frozenset[str], tuple[int, ...]]] = []
    for example in examples:
        grams = _character_ngrams(example.normalized_user)
        row_data.append((grams, _minhash_signature(grams)))
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for row_index, (_, signature) in enumerate(row_data):
        for band in range(MINHASH_BANDS):
            start = band * rows_per_band
            buckets[(band, signature[start : start + rows_per_band])].append(row_index)

    candidates: set[tuple[int, int]] = set()
    for indexes in buckets.values():
        if len(indexes) < 2:
            continue
        for position, left_index in enumerate(indexes[:-1]):
            candidates.update((left_index, right_index) for right_index in indexes[position + 1 :])

    verified: list[tuple[int, int, float]] = []
    for left_index, right_index in sorted(candidates):
        left_grams = row_data[left_index][0]
        right_grams = row_data[right_index][0]
        union = left_grams.union(right_grams)
        similarity = len(left_grams.intersection(right_grams)) / len(union)
        if similarity >= NEAR_DUPLICATE_JACCARD_THRESHOLD:
            verified.append((left_index, right_index, similarity))
    return _NearDuplicateScan(
        candidate_pairs=len(candidates),
        verified_pairs=tuple(verified),
    )


def _near_duplicate_overlap(
    split_examples: Mapping[str, Sequence[PreparedExample]],
) -> dict[str, Any]:
    rows: dict[str, list[tuple[str, frozenset[str], tuple[int, ...]]]] = {}
    rows_per_band = MINHASH_PERMUTATIONS // MINHASH_BANDS
    for split_name, examples in split_examples.items():
        split_rows = []
        for example in examples:
            grams = _character_ngrams(example.normalized_user)
            split_rows.append((example.example_id, grams, _minhash_signature(grams)))
        rows[split_name] = split_rows

    pair_results: dict[str, Any] = {}
    for left, right in (("train", "dev"),):
        left_buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
        right_buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
        for side_rows, buckets in ((rows[left], left_buckets), (rows[right], right_buckets)):
            for row_index, (_, _, signature) in enumerate(side_rows):
                for band in range(MINHASH_BANDS):
                    start = band * rows_per_band
                    key = (band, signature[start : start + rows_per_band])
                    buckets[key].append(row_index)
        candidates: set[tuple[int, int]] = set()
        for key in set(left_buckets).intersection(right_buckets):
            candidates.update(
                (left_index, right_index)
                for left_index in left_buckets[key]
                for right_index in right_buckets[key]
            )
        near_pairs: list[tuple[str, str, float]] = []
        for left_index, right_index in sorted(candidates):
            left_id, left_grams, _ = rows[left][left_index]
            right_id, right_grams, _ = rows[right][right_index]
            union = left_grams.union(right_grams)
            similarity = len(left_grams.intersection(right_grams)) / len(union)
            if similarity >= NEAR_DUPLICATE_JACCARD_THRESHOLD:
                near_pairs.append((left_id, right_id, similarity))
        pair_results[f"{left}__{right}"] = {
            "candidate_pairs": len(candidates),
            "left_rows_affected": len({item[0] for item in near_pairs}),
            "near_duplicate_pairs": len(near_pairs),
            "right_rows_affected": len({item[1] for item in near_pairs}),
            "sample_pair_ids": [
                {"left": left_id, "right": right_id, "jaccard": round(score, 6)}
                for left_id, right_id, score in near_pairs[:20]
            ],
        }
    return {
        "bands": MINHASH_BANDS,
        "candidate_generation_exhaustive": False,
        "character_ngram_size": CHAR_NGRAM_SIZE,
        "exact_jaccard_verification": True,
        "jaccard_threshold": NEAR_DUPLICATE_JACCARD_THRESHOLD,
        "lsh_false_negatives_possible": True,
        "method": "64-permutation affine MinHash; 16x4 LSH candidates",
        "pairs": pair_results,
        "permutations": MINHASH_PERMUTATIONS,
    }


def _distribution(examples: Sequence[PreparedExample]) -> dict[str, Any]:
    call_count = Counter()
    tool_calls = Counter()
    arguments = Counter()
    for example in examples:
        call_count[str(len(example.call_names))] += 1
        for tool_name, argument_names in zip(
            example.call_names, example.argument_names, strict=True
        ):
            tool_calls[tool_name] += 1
            for argument_name in argument_names:
                arguments[f"{tool_name}.{argument_name}"] += 1
    return {
        "argument_presence": dict(sorted(arguments.items())),
        "call_count": dict(sorted(call_count.items())),
        "tool_calls": dict(sorted(tool_calls.items())),
    }


def _membership_hash(examples: Sequence[PreparedExample]) -> str:
    payload = "\n".join(sorted(example.example_id for example in examples)) + "\n"
    return _sha256_bytes(payload.encode("utf-8"))


def _manifest_bytes(examples: Sequence[PreparedExample], source_spec: SourceSpec) -> bytes:
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
    except FileExistsError as exc:
        raise MobileActionsError(f"refusing to overwrite existing artifact {path}") from exc


def _build_audit(
    *,
    examples: Sequence[PreparedExample],
    source_read: _SourceRead,
    source_spec: SourceSpec,
    split_policy: SplitPolicy,
    grouping_audit: Mapping[str, Any],
    tokenizer: TokenizerLike | None,
    tokenizer_identity: TokenizerIdentity | None,
    max_seq_len: int,
    artifact_hashes: Mapping[str, str],
) -> dict[str, Any]:
    split_examples = _split_examples(examples)
    overlap = {
        "connected_component": _index_overlap(split_examples, lambda example: example.cluster_id),
        "delexicalized_user": _index_overlap(
            split_examples, lambda example: example.delexicalized_user
        ),
        "exact_user": _index_overlap(split_examples, lambda example: example.user_text),
        "family_template": _index_overlap(split_examples, lambda example: example.family_id),
        "normalized_user": _index_overlap(split_examples, lambda example: example.normalized_user),
        "schema_signature": _index_overlap(
            split_examples, lambda example: example.schema_signature
        ),
        "whitespace_13_token": _ngram_overlap(
            split_examples,
            lambda example: tuple(_TOKEN_RE.findall(example.normalized_user)),
            n=13,
        ),
        "char_5gram_minhash": _near_duplicate_overlap(split_examples),
    }
    if tokenizer is None:
        overlap["barun_13_token"] = {
            "reason": "tokenizer_not_provided",
            "status": "not_computed",
        }
        tokenization: dict[str, Any] = {
            "reason": "tokenizer_not_provided",
            "status": "not_computed",
        }
    else:
        assert tokenizer_identity is not None
        overlap["barun_13_token"] = _ngram_overlap(
            split_examples,
            lambda example: _encode_ids(tokenizer, example.user_text),
            n=13,
        )
        per_split_lengths: dict[str, Any] = {}
        for split_name, rows in split_examples.items():
            total_lengths = [
                example.total_tokens_with_eos
                for example in rows
                if example.total_tokens_with_eos is not None
            ]
            prompt_lengths = [
                example.prompt_tokens for example in rows if example.prompt_tokens is not None
            ]
            target_lengths = [
                example.target_tokens for example in rows if example.target_tokens is not None
            ]
            assert all(isinstance(item, int) for item in total_lengths)
            assert all(isinstance(item, int) for item in prompt_lengths)
            assert all(isinstance(item, int) for item in target_lengths)
            per_split_lengths[split_name] = {
                "overlength_ids": [
                    example.example_id
                    for example in rows
                    if example.total_tokens_with_eos is not None
                    and example.total_tokens_with_eos > max_seq_len
                ],
                "prompt": _length_summary(prompt_lengths),
                "target": _length_summary(target_lengths),
                "total_with_eos": _length_summary(total_lengths),
            }
            per_split_lengths[split_name]["overlength_count"] = len(
                per_split_lengths[split_name]["overlength_ids"]
            )
        tokenization = {
            "eos_tokens_assumed": 1,
            "identity": {
                "identifier": tokenizer_identity.identifier,
                "revision": tokenizer_identity.revision,
                "sha256": tokenizer_identity.sha256,
            },
            "max_seq_len": max_seq_len,
            "per_split": per_split_lengths,
            "policy": "audit only; manifests are never truncated or silently dropped",
            "status": "computed",
        }

    counts = {name: len(rows) for name, rows in split_examples.items()}
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "adapter": {
            "prompt_contract_sha256": PROMPT_TEMPLATE_SHA256,
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        },
        "artifacts": {
            "manifests": {
                name: {
                    "filename": f"{name}.jsonl",
                    "membership_sha256": _membership_hash(split_examples[name]),
                    "sha256": artifact_hashes[name],
                }
                for name in ("train", "dev")
            },
        },
        "counts": {
            "derived": counts,
            "nfc_normalizations": sum(example.nfc_normalizations for example in examples),
            "null_argument_fields_excluded": sum(
                example.null_argument_fields_excluded for example in examples
            ),
            "records_dropped": 0,
            "records_truncated": 0,
            "source": {
                "official_eval_opaque_unparsed": source_read.official_eval_rows,
                "total": source_read.total_rows,
                "train": source_read.internal_train_rows,
            },
        },
        "distributions": {
            split_name: _distribution(rows) for split_name, rows in split_examples.items()
        },
        "format_notes": {
            "arguments": (
                "Parsed internal-train rows contain JSON objects; the dataset card says "
                "stringified JSON. No coercion is performed. The official eval rows were "
                "not parsed, so this audit makes no claim about their argument encoding."
            ),
            "call_fields": (
                "Parsed internal-train call rows contain only the function field; the dataset "
                "card also documents id and type. The official eval rows were not parsed."
            ),
            "multi_call_mode": (
                "Mapped to SERIAL to preserve Google's ordered call-list target because the "
                "source has no dependency/parallel annotation."
            ),
            "official_metric": (
                "Google's cookbook compares ordered function-name lists and per-call argument "
                "dictionaries after sorting dictionary keys; it does not execute calls or score "
                "Action IR policy decisions."
            ),
            "policy_decision": (
                "Every parsed internal-train row is mapped to CALL because the internal-train "
                "population has no abstention, clarification, confirmation, or unsafe-request "
                "labels. It is not safety-policy training data."
            ),
            "timezone": (
                "Source NOW timestamps are naive and provide no timezone. They are preserved "
                "without fabricating an offset."
            ),
        },
        "license": {
            "id": source_spec.license_id,
            "url": source_spec.license_url,
        },
        "official_eval": {
            "labels_parsed": False,
            "lengths_computed": False,
            "materialized": False,
            "opaque_unparsed": True,
            "overlaps_computed": False,
            "prompts_parsed": False,
            "rows": source_read.official_eval_rows,
            "source_sha256": source_read.source_sha256,
            "summaries_computed": False,
            "targets_parsed": False,
            "tool_schemas_parsed": False,
        },
        "overlap": overlap,
        "overlap_scope": (
            "Internal-train-derived train/dev only. Internal-train gold tool names and "
            "argument values are used for exact-family grouping and entity delexicalization; "
            "normalized internal-train user text is used for near-duplicate edges. Official "
            "eval rows are opaque and excluded from every overlap calculation."
        ),
        "source": {
            "dataset_id": source_spec.dataset_id,
            "expected_sha256": source_spec.source_sha256,
            "filename": source_spec.filename,
            "revision": source_spec.revision,
            "sha256": source_spec.source_sha256,
            "url": source_spec.source_url,
        },
        "split_policy": {
            "component_construction": {
                "candidate_generation_exhaustive": False,
                "character_ngram_size": CHAR_NGRAM_SIZE,
                "edge_rule": (
                    "exact Jaccard >= threshold after deterministic MinHash-LSH candidacy"
                ),
                "family_rule": (
                    "ordered gold tool-name signature plus entity-delexicalized user template; "
                    "delexicalization uses internal-train gold argument values"
                ),
                "jaccard_threshold": NEAR_DUPLICATE_JACCARD_THRESHOLD,
                "lsh_bands": MINHASH_BANDS,
                "lsh_false_negatives_possible": True,
                "minhash_permutations": MINHASH_PERMUTATIONS,
                "scope": "internal train only; official eval is never consulted",
                "transitivity": (
                    "split groups are deterministic connected components, so bridge edges "
                    "may connect rows whose direct Jaccard is below threshold"
                ),
            },
            "dev_fold": split_policy.dev_fold,
            "dev_folds": split_policy.dev_folds,
            "observed": dict(grouping_audit),
            "source": "internal train only",
            "version": split_policy.version,
        },
        "tokenization": tokenization,
    }


def prepare_source(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    source_spec: SourceSpec,
    split_policy: SplitPolicy | None = None,
    tokenizer: TokenizerLike | None = None,
    tokenizer_identity: TokenizerIdentity | None = None,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
) -> PreparationResult:
    """Prepare immutable train/dev manifests without unblinding official eval rows."""

    split_policy = split_policy or SplitPolicy()
    if max_seq_len < 2:
        raise ValueError("max_seq_len must be at least 2")
    if (tokenizer is None) != (tokenizer_identity is None):
        raise ValueError("tokenizer and tokenizer_identity must be supplied together")
    source = Path(source_path)
    if not source.is_file():
        raise MobileActionsError(f"source file does not exist: {source}")
    output = Path(output_dir)
    train_path = output / "train.jsonl"
    dev_path = output / "dev.jsonl"
    audit_path = output / "audit.json"
    intended_paths = [train_path, dev_path, audit_path]
    if len({path.resolve() for path in intended_paths}) != len(intended_paths):
        raise MobileActionsError("artifact output paths must be distinct")
    existing = [path for path in intended_paths if path.exists()]
    if existing:
        raise MobileActionsError(
            "refusing to overwrite existing artifacts: " + ", ".join(str(path) for path in existing)
        )

    source_read = _read_source(source, source_spec)
    examples, grouping_audit = _assign_training_splits(source_read.train_examples, split_policy)
    if tokenizer is not None:
        examples = _add_token_lengths(examples, tokenizer)
    split_examples = _split_examples(examples)
    train_bytes = _manifest_bytes(split_examples["train"], source_spec)
    dev_bytes = _manifest_bytes(split_examples["dev"], source_spec)
    artifact_hashes = {
        "train": _sha256_bytes(train_bytes),
        "dev": _sha256_bytes(dev_bytes),
    }
    audit = _build_audit(
        examples=examples,
        source_read=source_read,
        source_spec=source_spec,
        split_policy=split_policy,
        grouping_audit=grouping_audit,
        tokenizer=tokenizer,
        tokenizer_identity=tokenizer_identity,
        max_seq_len=max_seq_len,
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
        train_rows=len(split_examples["train"]),
        dev_rows=len(split_examples["dev"]),
        final_eval_rows=source_read.official_eval_rows,
        hashes=hashes,
    )


def prepare_mobile_actions(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    tokenizer: TokenizerLike | None = None,
    tokenizer_identity: TokenizerIdentity | None = None,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
) -> PreparationResult:
    """Prepare only the pinned official Google Mobile Actions revision."""

    return prepare_source(
        source_path,
        output_dir,
        source_spec=OFFICIAL_SOURCE_SPEC,
        tokenizer=tokenizer,
        tokenizer_identity=tokenizer_identity,
        max_seq_len=max_seq_len,
    )


def download_pinned_source(cache_dir: str | Path | None = None) -> Path:
    """Download the exact pinned source through huggingface_hub, then verify its bytes."""

    from huggingface_hub import hf_hub_download

    downloaded = Path(
        hf_hub_download(
            repo_id=MOBILE_ACTIONS_DATASET_ID,
            filename=MOBILE_ACTIONS_SOURCE_FILENAME,
            repo_type="dataset",
            revision=MOBILE_ACTIONS_REVISION,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    )
    actual = sha256_file(downloaded)
    if actual != MOBILE_ACTIONS_SOURCE_SHA256:
        raise MobileActionsError(
            f"downloaded source SHA-256 mismatch: expected {MOBILE_ACTIONS_SOURCE_SHA256}, got {actual}"
        )
    return downloaded


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the pinned Google Mobile Actions manifests and audit."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source", type=Path, help="Path to the pinned dataset.jsonl")
    source_group.add_argument(
        "--download-pinned",
        action="store_true",
        help="Download the exact pinned revision through huggingface_hub",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--tokenizer-id", default="harrrshall/BarunLM-35M")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source_path = download_pinned_source(args.cache_dir) if args.download_pinned else args.source
    tokenizer = None
    tokenizer_identity = None
    if args.tokenizer_json is not None:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
        tokenizer_identity = TokenizerIdentity(
            identifier=args.tokenizer_id,
            revision=args.tokenizer_revision,
            sha256=sha256_file(args.tokenizer_json),
        )
    result = prepare_mobile_actions(
        source_path,
        args.output_dir,
        tokenizer=tokenizer,
        tokenizer_identity=tokenizer_identity,
        max_seq_len=args.max_seq_len,
    )
    summary = {
        "audit": str(result.audit_path),
        "dev_manifest": str(result.dev_manifest),
        "dev_rows": result.dev_rows,
        "official_eval_rows_opaque_unparsed": result.final_eval_rows,
        "hashes": dict(result.hashes),
        "train_manifest": str(result.train_manifest),
        "train_rows": result.train_rows,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
