"""Remote entrypoint for the frozen mobile scale-sweep v6 recipe.

Loads the scientific runner via a stdlib-only file-path bootstrap so the
isolation preflight runs before ``barunlm.baselines`` package init can import
Torch, then runs the CPU test gate, then delegates to the scientific runner.
Does not invent a parallel launcher.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Deterministic cuBLAS configuration must precede torch and project imports.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def main() -> None:
    staged_repo_root = Path(__file__).resolve().parents[1]
    if sys.path:
        # File execution normally inserts ``<repo>/scripts``.  Replace it with
        # the exact staged repository root so the preflight can use a closed
        # repo/source-root allowlist rather than admitting arbitrary subpaths.
        sys.path[0] = str(staged_repo_root)
    # Establish the live interpreter boundary before pytest or any Torch-bearing
    # project module is allowed to import through ``baselines`` package init.
    from barunlm.scale_sweep_stdlib_loader import load_mobile_scale_sweep_stdlib

    sweep = load_mobile_scale_sweep_stdlib()
    preflight_parser = argparse.ArgumentParser(add_help=False)
    preflight_parser.add_argument("--config", type=Path, default=sweep.CONFIG_PATH)
    preflight_args, _ = preflight_parser.parse_known_args(sys.argv[1:])
    sweep.stdlib_isolation_preflight(preflight_args.config, require_torch_absent=True)

    gate = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_mobile_scale_sweep.py"],
        check=False,
    )
    if gate.returncode != 0:
        raise SystemExit(gate.returncode)
    from barunlm.baselines.mobile_scale_sweep import main as run_main

    run_main()


if __name__ == "__main__":
    main()
