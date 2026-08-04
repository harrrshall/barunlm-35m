"""Frozen seven-tool Mobile Actions schema registry shared by direct Action IR code."""

from __future__ import annotations

from .action_ir import JSONType, ToolSchema, ValueSchema

_STRING = ValueSchema(JSONType.STRING)

MOBILE_TOOL_SCHEMAS: tuple[ToolSchema, ...] = (
    ToolSchema(
        "create_calendar_event",
        {"datetime": _STRING, "title": _STRING},
        required=frozenset({"datetime", "title"}),
    ),
    ToolSchema(
        "create_contact",
        {
            "email": _STRING,
            "first_name": _STRING,
            "last_name": _STRING,
            "phone_number": _STRING,
        },
        required=frozenset({"first_name", "last_name"}),
    ),
    ToolSchema("open_wifi_settings", {}),
    ToolSchema(
        "send_email",
        {"body": _STRING, "subject": _STRING, "to": _STRING},
        required=frozenset({"subject", "to"}),
    ),
    ToolSchema("show_map", {"query": _STRING}, required=frozenset({"query"})),
    ToolSchema("turn_off_flashlight", {}),
    ToolSchema("turn_on_flashlight", {}),
)

__all__ = ["MOBILE_TOOL_SCHEMAS"]
