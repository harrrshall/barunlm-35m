"""In-memory-only BarunAction simulator; no tool handler or network access exists here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from barunlm.evaluation.action_ir import ActionIR, Decision

from .inference import assess_policy
from .schema import ToolDeclaration

SIMULATOR_SCHEMA_VERSION = "barunaction-sandbox-simulator-v1"


@dataclass(frozen=True, slots=True)
class SimulationResult:
    status: str
    blocked_reasons: tuple[str, ...]
    simulated_calls: tuple[dict[str, Any], ...]
    external_side_effects: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocked_reasons": list(self.blocked_reasons),
            "external_side_effects": self.external_side_effects,
            "schema_version": SIMULATOR_SCHEMA_VERSION,
            "simulated_calls": list(self.simulated_calls),
            "status": self.status,
        }


def simulate_action(
    action: ActionIR,
    *,
    declarations: tuple[ToolDeclaration, ...],
    externally_authorized: bool = False,
    externally_confirmed: bool = False,
) -> SimulationResult:
    """Apply an action only to a returned in-memory log after explicit external gates."""

    if type(externally_authorized) is not bool or type(externally_confirmed) is not bool:
        raise TypeError("sandbox gate values must be bool")
    registry = {item.schema.name: item.schema for item in declarations}
    policy = assess_policy(action, registry)
    if action.decision not in (Decision.CALL, Decision.CONFIRM):
        return SimulationResult(
            status="no_calls_proposed",
            blocked_reasons=(f"decision_{action.decision.value.lower()}",),
            simulated_calls=(),
        )
    blocked: list[str] = []
    if policy.authorization_required and not externally_authorized:
        blocked.append("external_authorization_missing")
    if policy.confirmation_required and not externally_confirmed:
        blocked.append("external_confirmation_missing")
    if blocked:
        return SimulationResult(
            status="blocked",
            blocked_reasons=tuple(blocked),
            simulated_calls=(),
        )
    return SimulationResult(
        status="simulated",
        blocked_reasons=(),
        simulated_calls=tuple(call.to_dict() for call in action.calls),
    )


__all__ = ["SIMULATOR_SCHEMA_VERSION", "SimulationResult", "simulate_action"]
