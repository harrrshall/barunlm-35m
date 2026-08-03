"""Command-line interface for verified BarunAction-35M local inference and simulation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

from barunlm.evaluation.generation import GenerationError, verify_checkpoint
from barunlm.quantization import (
    QuantizationError,
    export_dynamic_int8_checkpoint,
    verify_int8_checkpoint,
)

from .candidate import (
    CANDIDATE_CHECKPOINT_SHA256,
    CANDIDATE_MANIFEST_SHA256,
    candidate_identity,
)
from .inference import (
    DEFAULT_MAX_NEW_TOKENS,
    RESULT_SCHEMA_VERSION,
    BarunActionCompiler,
    InferenceOutcome,
    validate_action_output,
)
from .quantization import (
    QuantizationSmokeError,
    compare_int8_action_ir,
    parse_int8_smoke_cases,
)
from .schema import ContractError, ToolDeclaration, parse_tool_declarations
from .simulator import simulate_action


class CLIError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise CLIError("duplicate_key", f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _reject_constant(value: str) -> NoReturn:
    raise CLIError("non_finite_number", f"non-finite JSON value {value!r}")


def _load_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        return json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_pairs,
            parse_constant=_reject_constant,
        )
    except CLIError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CLIError("invalid_json_file", f"cannot read strict JSON from {source}") from error


def _hashes(path: Path | None) -> Mapping[str, str]:
    if path is None:
        return CANDIDATE_CHECKPOINT_SHA256
    value = _load_json(path)
    if not isinstance(value, Mapping):
        raise CLIError("invalid_hashes", "checkpoint hash file must be a JSON object")
    if "file_sha256" in value:
        value = value["file_sha256"]
        if not isinstance(value, Mapping):
            raise CLIError("invalid_hashes", "file_sha256 must be a JSON object")
    hashes = {str(name): str(digest) for name, digest in value.items()}
    return hashes


def _read_text(value: str | None, path: Path | None, *, name: str) -> str:
    if (value is None) == (path is None):
        raise CLIError("ambiguous_input", f"provide exactly one --{name} or --{name}-file")
    if value is not None:
        return value
    assert path is not None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise CLIError("invalid_text_file", f"cannot read UTF-8 text from {path}") from error


def _print(payload: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise CLIError("refuse_overwrite", f"refusing to overwrite report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _verify(args: argparse.Namespace) -> int:
    hashes = verify_checkpoint(args.checkpoint, expected_sha256=_hashes(args.checkpoint_hashes))
    candidate_id, candidate_run_id = candidate_identity(hashes)
    _print(
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": hashes,
            "candidate_id": candidate_id,
            "candidate_run_id": candidate_run_id,
            "manifest_sha256": (CANDIDATE_MANIFEST_SHA256 if candidate_id is not None else None),
            "ok": True,
        }
    )
    return 0


def _infer(args: argparse.Namespace) -> int:
    if args.checkpoint_format == "int8" and args.checkpoint_hashes is not None:
        raise CLIError(
            "conflicting_checkpoint_identity",
            "--checkpoint-hashes is valid only with --checkpoint-format float",
        )
    expected_sha256 = _hashes(args.checkpoint_hashes) if args.checkpoint_format == "float" else None
    compiler = BarunActionCompiler(
        args.checkpoint,
        expected_sha256=expected_sha256,
        checkpoint_format=args.checkpoint_format,
        expected_int8_manifest_sha256=args.int8_manifest_sha256,
        device=args.device,
    )
    outcome = compiler.infer(
        request=_read_text(args.request, args.request_file, name="request"),
        tool_schemas=_load_json(args.tools),
        context=_load_json(args.context),
        now=args.now,
        max_new_tokens=args.max_new_tokens,
    )
    _print(outcome.to_dict())
    return 0 if outcome.ok else 2


def _export_int8(args: argparse.Namespace) -> int:
    info = export_dynamic_int8_checkpoint(
        args.source_checkpoint,
        args.output,
        expected_source_sha256=_hashes(args.source_hashes),
        qengine=args.qengine,
    )
    _print({"ok": True, "quantized_checkpoint": info.to_dict()})
    return 0


def _verify_int8(args: argparse.Namespace) -> int:
    info = verify_int8_checkpoint(
        args.checkpoint,
        expected_manifest_sha256=args.manifest_sha256,
    )
    _print({"ok": True, "quantized_checkpoint": info.to_dict()})
    return 0


def _smoke_int8(args: argparse.Namespace) -> int:
    cases = parse_int8_smoke_cases(_load_json(args.cases))
    report = compare_int8_action_ir(
        float_checkpoint=args.source_checkpoint,
        expected_float_sha256=_hashes(args.source_hashes),
        int8_checkpoint=args.int8_checkpoint,
        expected_int8_manifest_sha256=args.manifest_sha256,
        cases=cases,
        max_new_tokens=args.max_new_tokens,
    )
    if args.report is not None:
        _write_report(args.report, report)
    _print(report)
    return 0 if report["all_action_ir_exact"] else 2


def _validated_output(
    args: argparse.Namespace,
) -> tuple[tuple[ToolDeclaration, ...], InferenceOutcome]:
    declarations = parse_tool_declarations(_load_json(args.tools))
    raw_output = _read_text(args.output, args.output_file, name="output")
    return declarations, validate_action_output(
        raw_output,
        declarations=declarations,
        checkpoint_sha256={},
    )


def _validate_output(args: argparse.Namespace) -> int:
    _, outcome = _validated_output(args)
    _print(outcome.to_dict())
    return 0 if outcome.ok else 2


def _simulate_output(args: argparse.Namespace) -> int:
    declarations, outcome = _validated_output(args)
    simulation = None
    if outcome.action is not None:
        simulation = simulate_action(
            outcome.action,
            declarations=declarations,
            externally_authorized=args.authorize_sandbox,
            externally_confirmed=args.confirm_sandbox,
        ).to_dict()
    _print(
        {
            "inference": outcome.to_dict(),
            "ok": outcome.ok,
            "simulation": simulation,
        }
    )
    return 0 if outcome.ok else 2


def _add_text_source(parser: argparse.ArgumentParser, name: str) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(f"--{name}")
    group.add_argument(f"--{name}-file", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="barunaction",
        description="Verified local BarunAction-35M proposal inference; never executes real tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser(
        "verify", help="verify an immutable checkpoint without inference"
    )
    verify.add_argument("--checkpoint", type=Path, required=True)
    verify.add_argument("--checkpoint-hashes", type=Path)
    verify.set_defaults(func=_verify)

    infer = subparsers.add_parser("infer", help="run deterministic local proposal inference")
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--checkpoint-format", choices=("float", "int8"), default="float")
    infer.add_argument("--checkpoint-hashes", type=Path)
    infer.add_argument("--int8-manifest-sha256")
    infer.add_argument("--tools", type=Path, required=True)
    infer.add_argument("--context", type=Path, required=True)
    infer.add_argument("--now", required=True)
    infer.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    infer.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    _add_text_source(infer, "request")
    infer.set_defaults(func=_infer)

    export_int8 = subparsers.add_parser(
        "export-int8",
        help="export a new explicitly pinned CPU dynamic-int8 checkpoint",
    )
    export_int8.add_argument("--source-checkpoint", type=Path, required=True)
    export_int8.add_argument("--source-hashes", type=Path, required=True)
    export_int8.add_argument("--output", type=Path, required=True)
    export_int8.add_argument("--qengine", required=True)
    export_int8.set_defaults(func=_export_int8)

    verify_int8 = subparsers.add_parser(
        "verify-int8",
        help="verify a CPU dynamic-int8 checkpoint without loading weights",
    )
    verify_int8.add_argument("--checkpoint", type=Path, required=True)
    verify_int8.add_argument("--manifest-sha256", required=True)
    verify_int8.set_defaults(func=_verify_int8)

    smoke_int8 = subparsers.add_parser(
        "smoke-int8",
        help="compare float and int8 outputs against expected exact Action IR",
    )
    smoke_int8.add_argument("--source-checkpoint", type=Path, required=True)
    smoke_int8.add_argument("--source-hashes", type=Path, required=True)
    smoke_int8.add_argument("--int8-checkpoint", type=Path, required=True)
    smoke_int8.add_argument("--manifest-sha256", required=True)
    smoke_int8.add_argument("--cases", type=Path, required=True)
    smoke_int8.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    smoke_int8.add_argument("--report", type=Path)
    smoke_int8.set_defaults(func=_smoke_int8)

    validate = subparsers.add_parser(
        "validate-output", help="strictly validate an existing model output"
    )
    validate.add_argument("--tools", type=Path, required=True)
    _add_text_source(validate, "output")
    validate.set_defaults(func=_validate_output)

    simulate = subparsers.add_parser(
        "simulate-output", help="validate and apply an output only to an in-memory call log"
    )
    simulate.add_argument("--tools", type=Path, required=True)
    simulate.add_argument("--authorize-sandbox", action="store_true")
    simulate.add_argument("--confirm-sandbox", action="store_true")
    _add_text_source(simulate, "output")
    simulate.set_defaults(func=_simulate_output)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ContractError as error:
        _print(
            {
                "error": error.to_dict(),
                "ok": False,
                "schema_version": RESULT_SCHEMA_VERSION,
            }
        )
        return 2
    except CLIError as error:
        _print(
            {
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "path": "$",
                    "stage": "input",
                },
                "ok": False,
                "schema_version": RESULT_SCHEMA_VERSION,
            }
        )
        return 2
    except GenerationError as error:
        _print(
            {
                "error": {
                    "code": "checkpoint_error",
                    "message": str(error),
                    "path": "$.checkpoint",
                    "stage": "checkpoint",
                },
                "ok": False,
                "schema_version": RESULT_SCHEMA_VERSION,
            }
        )
        return 2
    except (QuantizationError, QuantizationSmokeError) as error:
        _print(
            {
                "error": {
                    "code": "quantization_error",
                    "message": str(error),
                    "path": "$.checkpoint",
                    "stage": "quantization",
                },
                "ok": False,
                "schema_version": RESULT_SCHEMA_VERSION,
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
