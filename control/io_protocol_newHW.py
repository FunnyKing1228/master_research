"""newHW I/O protocol boundary.

No communication protocol has been provided.  This file deliberately contains
no parser, writer, transport guess, or P302 compatibility fallback.
"""

from __future__ import annotations

from typing import Any


REQUIRED_BEFORE_IMPLEMENTATION = (
    "TODO(newHW): transport/interface type (file, serial, CAN, Modbus, API, etc.)",
    "TODO(newHW): measurement message schema, units, timestamps, and update rate",
    "TODO(newHW): command schema, units, limits, acknowledgement, and timeout",
    "TODO(newHW): BMS/MPPT/controller ownership of charge and discharge limits",
    "TODO(newHW): emergency stop, disconnect, stale-data, and fail-safe behavior",
    "TODO(newHW): representative captured input/output messages",
    "TODO(newHW): hardware acceptance owner and procedure",
)


class NewHWProtocolUnavailable(NotImplementedError):
    """Raised until the real newHW I/O contract is supplied."""


def parse_measurement_newHW(payload: Any) -> dict[str, Any]:
    """Parse one real newHW measurement after its protocol is specified."""
    raise NewHWProtocolUnavailable(
        "newHW measurement protocol is unknown; see REQUIRED_BEFORE_IMPLEMENTATION "
        "and docs/handover/newHW_pending_data.md"
    )


def encode_command_newHW(command: dict[str, Any]) -> Any:
    """Encode one real newHW command after its protocol is specified."""
    raise NewHWProtocolUnavailable(
        "newHW command protocol is unknown; do not reuse P302 Data.txt/Command.txt"
    )
