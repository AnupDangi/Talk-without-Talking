"""Shared, read-only inventory of audio endpoints visible to PortAudio.

This provider exposes runtime device indices, not stable host UIDs. Keep those
indices explicitly transient and leave stable IDs empty rather than deriving
them from names.
"""

from __future__ import annotations

from typing import Any

import sounddevice as sd


def enumerate_audio_devices(sd_api: Any | None = None) -> list[dict[str, Any]]:
    """Return every visible endpoint, including input-only and output-only devices."""
    api = sd_api or sd
    try:
        raw_devices = api.query_devices()
    except Exception as exc:
        raise RuntimeError(f"Could not enumerate audio endpoints: {exc}") from exc

    try:
        host_apis = api.query_hostapis()
    except Exception:
        host_apis = []

    try:
        default_input, default_output = api.default.device
    except Exception:
        default_input, default_output = None, None

    endpoints: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_devices):
        host_api_index = raw.get("hostapi")
        host_api = None
        if host_api_index is not None and 0 <= int(host_api_index) < len(host_apis):
            host_api = str(host_apis[int(host_api_index)].get("name") or "") or None

        try:
            samplerate = float(raw.get("default_samplerate"))
        except (TypeError, ValueError):
            samplerate = None

        endpoints.append(
            {
                "index": index,
                "stable_id": None,
                "stable_id_available": False,
                "name": str(raw.get("name") or f"Audio endpoint {index}").strip(),
                "host_api": host_api,
                "max_input_channels": int(raw.get("max_input_channels") or 0),
                "max_output_channels": int(raw.get("max_output_channels") or 0),
                "default_samplerate": samplerate,
                "supported_sample_rates": None,
                "sample_formats": None,
                "format_metadata_available": False,
                "default_input": index == default_input,
                "default_output": index == default_output,
            }
        )
    return endpoints


def input_devices(sd_api: Any | None = None) -> list[dict[str, Any]]:
    """Return input endpoints with the legacy fields used by JawLab and the UI."""
    inputs = []
    for endpoint in enumerate_audio_devices(sd_api):
        channels = endpoint["max_input_channels"]
        if channels < 1:
            continue
        inputs.append(
            {
                **endpoint,
                "channels": channels,
                "rate": round(endpoint["default_samplerate"] or 0),
                "default": endpoint["default_input"],
            }
        )
    return inputs


def resolve_input_device(
    device: int | str | None, sd_api: Any | None = None
) -> tuple[dict[str, Any], int]:
    """Resolve a runtime index or name to a currently visible input device."""
    api = sd_api or sd
    available = input_devices(api)
    if not available:
        raise RuntimeError("No audio input devices found. Connect or enable a microphone, then refresh devices.")

    if device is None or device == "":
        selected = next((item for item in available if item["default"]), available[0])
    elif isinstance(device, int) or str(device).strip().isdigit():
        index = int(device)
        selected = next((item for item in available if item["index"] == index), None)
        if selected is None:
            raise RuntimeError(f"Audio input index {index} is no longer available. Refresh the device list.")
    else:
        needle = str(device).strip().lower()
        exact = [item for item in available if item["name"].lower() == needle]
        matches = exact or [item for item in available if needle in item["name"].lower()]
        if not matches:
            names = ", ".join(item["name"] for item in available)
            raise RuntimeError(f"No input matching {device!r}. Available inputs: {names}")
        selected = matches[0]

    index = selected["index"]
    try:
        info = api.query_devices(index)
    except Exception as exc:
        raise RuntimeError(f"Audio input {selected['name']!r} is no longer available. Refresh devices.") from exc
    if int(info.get("max_input_channels") or 0) < 1:
        raise RuntimeError(f"Audio endpoint {selected['name']!r} has no input channels.")
    return info, index
