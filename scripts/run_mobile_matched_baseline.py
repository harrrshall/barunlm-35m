"""Run the frozen SmolLM2 matched Mobile Actions recipe on one project-owned GPU."""

from __future__ import annotations

# Deterministic cuBLAS configuration must precede torch and project imports.
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main() -> None:
    from barunlm.baselines.mobile_matched import main as run_main

    run_main()


if __name__ == "__main__":
    main()
