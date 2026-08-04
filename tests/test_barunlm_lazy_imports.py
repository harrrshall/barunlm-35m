from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import barunlm

ROOT = Path(__file__).resolve().parents[1]


def _isolated_import_report(statement: str) -> dict[str, object]:
    script = (
        "import json,sys;"
        f"{statement};"
        "print(json.dumps({"
        "'torch': 'torch' in sys.modules,"
        "'model': 'barunlm.model' in sys.modules,"
        "'quantization': 'barunlm.quantization' in sys.modules,"
        "'config': 'barunlm.config' in sys.modules"
        "}, sort_keys=True))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_package_and_cpu_submodule_import_do_not_import_torch_or_model_code() -> None:
    assert _isolated_import_report("import barunlm") == {
        "config": False,
        "model": False,
        "quantization": False,
        "torch": False,
    }
    assert _isolated_import_report("import barunlm.evaluation.action_correction_contract") == {
        "config": False,
        "model": False,
        "quantization": False,
        "torch": False,
    }


def test_config_is_lazy_and_model_import_remains_explicit() -> None:
    assert _isolated_import_report("from barunlm import BarunConfig") == {
        "config": True,
        "model": False,
        "quantization": False,
        "torch": False,
    }
    model_report = _isolated_import_report("from barunlm import BarunLM")
    assert model_report["model"] is True
    assert model_report["torch"] is True


def test_public_api_and_attribute_errors_remain_stable() -> None:
    assert set(barunlm.__all__) == {
        "INT8_FORMAT_VERSION",
        "BarunConfig",
        "BarunLM",
        "Int8CheckpointInfo",
        "QuantizationError",
        "export_dynamic_int8_checkpoint",
        "load_verified_int8_model",
        "verify_int8_checkpoint",
    }
    assert set(barunlm.__all__) <= set(dir(barunlm))
    try:
        _ = barunlm.not_a_real_export
    except AttributeError as error:
        assert "not_a_real_export" in str(error)
    else:  # pragma: no cover - exact negative API invariant
        raise AssertionError("unknown top-level attributes must fail")
