"""Deterministic continual-learning recovery utilities for BarunAction-35M."""

from .continual_recovery import (
    RECOVERY_CONFIG_SHA256,
    RECOVERY_CONFIG_VERSION,
    RecoveryError,
    evaluate_recovery_presto_gate,
    load_recovery_config,
    materialize_recovery_view,
    train_continual_recovery,
)

__all__ = [
    "RECOVERY_CONFIG_SHA256",
    "RECOVERY_CONFIG_VERSION",
    "RecoveryError",
    "evaluate_recovery_presto_gate",
    "load_recovery_config",
    "materialize_recovery_view",
    "train_continual_recovery",
]
