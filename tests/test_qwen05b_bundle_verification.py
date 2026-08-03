from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_qwen05b_matched_bundle.py"
_SPEC = importlib.util.spec_from_file_location("qwen05b_bundle_verifier", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_VERIFIER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _VERIFIER
_SPEC.loader.exec_module(_VERIFIER)
VerificationError = _VERIFIER.VerificationError
assert_json_equivalent = _VERIFIER.assert_json_equivalent
json_native = _VERIFIER.json_native
file_map = _VERIFIER.file_map
parse_strict_json = _VERIFIER.parse_strict_json
safe_manifest_member = _VERIFIER.safe_manifest_member
safetensor_shapes = _VERIFIER.safetensor_shapes


def test_cross_python_aggregate_comparison_allows_only_tiny_float_drift() -> None:
    drift = assert_json_equivalent(
        {"count": 602, "metric": 0.9033383723859914},
        {"count": 602, "metric": 0.9033383723859918},
        label="aggregate",
    )

    assert drift == pytest.approx(4e-16)


def test_cross_python_aggregate_comparison_rejects_metric_or_integer_changes() -> None:
    with pytest.raises(VerificationError, match="float differs"):
        assert_json_equivalent(
            {"metric": 0.90},
            {"metric": 0.91},
            label="aggregate",
        )

    with pytest.raises(VerificationError, match="differs"):
        assert_json_equivalent(
            {"numerator": 601},
            {"numerator": 602},
            label="aggregate",
        )


def test_cross_python_aggregate_comparison_rejects_shape_and_type_changes() -> None:
    with pytest.raises(VerificationError, match="object keys differ"):
        assert_json_equivalent({"value": 1.0, "extra": 0}, {"value": 1.0}, label="aggregate")

    with pytest.raises(VerificationError, match="changed type"):
        assert_json_equivalent({"value": 1}, {"value": 1.0}, label="aggregate")


def test_safetensor_shapes_uses_keys_interface(monkeypatch: pytest.MonkeyPatch) -> None:
    class Slice:
        def __init__(self, shape: list[int]) -> None:
            self.shape = shape

        def get_shape(self) -> list[int]:
            return self.shape

        def get_dtype(self) -> str:
            return "BF16"

    class Handle:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def keys(self) -> list[str]:
            return ["weight", "bias"]

        def get_slice(self, name: str) -> Slice:
            return Slice([2, 3] if name == "weight" else [3])

    monkeypatch.setattr(_VERIFIER, "safe_open", lambda *args, **kwargs: Handle())

    assert safetensor_shapes(Path("unused.safetensors")) == {
        "weight": [2, 3],
        "bias": [3],
    }


def test_json_native_matches_written_json_container_types() -> None:
    assert json_native({"risk_coverage": (), "nested": (1, 2)}) == {
        "risk_coverage": [],
        "nested": [1, 2],
    }

    with pytest.raises(ValueError, match="Out of range float values"):
        json_native({"metric": float("nan")})


@pytest.mark.parametrize(
    "payload, pattern",
    [
        ('{"value":1,"value":2}', "duplicate JSON key"),
        ('{"value":NaN}', "non-finite JSON constant"),
        ('{"value":1e9999}', "non-finite JSON float"),
    ],
)
def test_strict_json_rejects_duplicate_and_nonfinite_values(
    payload: str,
    pattern: str,
) -> None:
    with pytest.raises(VerificationError, match=pattern):
        parse_strict_json(payload, label="fixture")


def test_manifest_paths_reject_escape_and_noncanonical_names(tmp_path: Path) -> None:
    (tmp_path / "safe.json").write_text("{}", encoding="utf-8")
    assert safe_manifest_member(tmp_path, "safe.json", label="fixture") == (tmp_path / "safe.json")

    for unsafe in ("../safe.json", "/safe.json", "./safe.json", "nested/../safe.json"):
        with pytest.raises(VerificationError, match="unsafe path"):
            safe_manifest_member(tmp_path, unsafe, label="fixture")


def test_file_map_exclusion_is_root_relative_not_basename_based(tmp_path: Path) -> None:
    (tmp_path / "artifact-sha256.json").write_text("{}", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "artifact-sha256.json").write_text("{}", encoding="utf-8")

    mapped = file_map(tmp_path, excluded={"artifact-sha256.json"})

    assert set(mapped) == {"nested/artifact-sha256.json"}
