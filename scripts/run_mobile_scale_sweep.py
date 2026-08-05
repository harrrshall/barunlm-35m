"""Remote entrypoint for the frozen mobile scale-sweep v5 recipe.

Runs the CPU test gate, then delegates to the scientific runner in
``barunlm.baselines.mobile_scale_sweep``. Does not invent a parallel launcher.
"""

from __future__ import annotations

import os
import subprocess
import sys

# Deterministic cuBLAS configuration must precede torch and project imports.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def main() -> None:
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
