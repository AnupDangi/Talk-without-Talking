"""Bounded JawLab trials recorded from the server's existing microphone stream."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

import jawlab


CONDITIONS = {"still", "jaw_open", "clench", "silent", "whisper", "audible"}
MIN_TRIAL_SECONDS = 1.0
MAX_TRIAL_SECONDS = 10.0


class AudioTrialRecorder:
    """Collect labeled slices from the one active Listener/MicSource stream."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self._lock = threading.RLock()
        self._active: dict[str, Any] | None = None
        self._processing = False
        self._session_id: str | None = None
        self._session_dir: Path | None = None
        self._session_key: tuple[int | None, str] | None = None
        self._metadata: dict[str, Any] | None = None
        self._report: dict[str, Any] | None = None
        self._trials: list[dict[str, Any]] = []
        self._last_error: str | None = None

    def start_trial(
        self,
        device: dict[str, Any] | None,
        *,
        condition: str,
        word: str | None,
        duration_sec: float,
    ) -> dict[str, Any]:
        if device is None or not device.get("name"):
            raise ValueError("Start microphone capture and select an input before recording a trial.")
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition {condition!r}.")
        normalized_word = (word or "").strip().lower() or None
        if condition == "silent" and not normalized_word:
            raise ValueError("Enter the word label before recording a silent-mouthing trial.")
        try:
            duration = float(duration_sec)
        except (TypeError, ValueError) as exc:
            raise ValueError("Trial duration must be a number between 1 and 10 seconds.") from exc
        if not MIN_TRIAL_SECONDS <= duration <= MAX_TRIAL_SECONDS:
            raise ValueError("Trial duration must be between 1 and 10 seconds.")

        with self._lock:
            if self._active is not None or self._processing:
                raise RuntimeError("A JawLab trial is already active. Stop it before starting another.")
            session_key = (device.get("index"), str(device["name"]))
            if session_key != self._session_key:
                self._begin_session(device)
            trial_id = str(uuid4())
            self._active = {
                "id": trial_id,
                "condition": condition,
                "word": normalized_word,
                "duration_sec": duration,
                "started_monotonic": time.monotonic(),
                "chunks": [],
                "samples": 0,
                "rate": None,
            }
            self._last_error = None
            return {"trial_id": trial_id, "duration_sec": duration}

    def push_block(self, samples: np.ndarray, sample_rate: int) -> None:
        """Fast callback hook. Copy only the active trial's samples and never block on disk."""
        with self._lock:
            active = self._active
            if active is None or active["samples"] / max(1, sample_rate) >= active["duration_sec"]:
                return
            if active["rate"] is None:
                active["rate"] = int(sample_rate)
            if active["rate"] != int(sample_rate):
                return
            block = np.asarray(samples, dtype=np.float32).reshape(-1).copy()
            remaining = max(0, int(active["duration_sec"] * sample_rate) - active["samples"])
            if remaining:
                block = block[:remaining]
                active["chunks"].append(block)
                active["samples"] += int(block.size)

    def stop_trial(self, trial_id: str) -> dict[str, Any]:
        with self._lock:
            active = self._active
            if active is None or active["id"] != trial_id:
                raise RuntimeError("That trial is no longer active. Refresh the experiment state.")
            rate = int(active["rate"] or 0)
            audio = np.concatenate(active["chunks"]) if active["chunks"] else np.zeros(0, dtype=np.float32)
            duration = audio.size / rate if rate else 0.0
            if duration < MIN_TRIAL_SECONDS:
                raise RuntimeError(
                    f"Only {duration:.1f}s of audio was captured. Keep the mic running for at least "
                    f"{MIN_TRIAL_SECONDS:.0f}s, then stop the trial."
                )
            self._active = None
            self._processing = True

        try:
            metrics = jawlab.spectral_metrics(audio, rate)
            index = len(self._trials) + 1
            condition = active["condition"]
            word = active["word"]
            suffix = f"_{word}" if word else ""
            filename = f"{index:03d}_{condition}{suffix}"
            assert self._session_dir is not None and self._metadata is not None
            jawlab.save_trial(self._session_dir / filename, audio, rate)
            trial = {
                "file": filename,
                "condition": condition,
                "word": word,
                "start": time.time(),
                "duration": round(duration, 3),
            }
            with self._lock:
                self._metadata["trials"].append(trial)
                self._metadata["updated"] = datetime.now().isoformat(timespec="seconds")
                self._session_dir.joinpath("session.json").write_text(
                    json.dumps(self._metadata, indent=2)
                )
            report = jawlab.analyze_session(self._session_dir)
            public_trial = {
                "condition": condition,
                "word": word,
                "device": self._metadata["device"],
                "duration_sec": round(duration, 3),
                "metrics": metrics,
            }
            with self._lock:
                self._report = report
                self._trials.append(public_trial)
                return {"trial": public_trial, **self._state_locked()}
        finally:
            with self._lock:
                self._processing = False

    def state(self) -> dict[str, Any]:
        with self._lock:
            return self._state_locked()

    def cancel_active(self, reason: str) -> None:
        with self._lock:
            if self._active is not None:
                self._active = None
                self._last_error = reason

    def _begin_session(self, device: dict[str, Any]) -> None:
        session_id = datetime.now().strftime("browser-%Y%m%d-%H%M%S-") + uuid4().hex[:6]
        session_dir = self.output_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=False)
        rate = device.get("default_samplerate") or device.get("rate")
        self._session_id = session_id
        self._session_dir = session_dir
        self._session_key = (device.get("index"), str(device["name"]))
        self._metadata = {
            "version": 1,
            "subject": "browser-user",
            "device": str(device["name"]),
            "device_index": device.get("index"),
            "stable_id": device.get("stable_id"),
            "rate": int(rate or 0),
            "created": datetime.now().isoformat(timespec="seconds"),
            "capture_source": "existing-server-input-stream",
            "trials": [],
        }
        self._report = None
        self._trials = []

    def _state_locked(self) -> dict[str, Any]:
        return {
            "active": self._active is not None or self._processing,
            "active_trial": (
                {
                    "id": self._active["id"],
                    "condition": self._active["condition"],
                    "word": self._active["word"],
                    "duration_sec": self._active["duration_sec"],
                    "captured_sec": round(
                        self._active["samples"] / max(1, self._active["rate"] or 1), 2
                    ),
                }
                if self._active is not None
                else None
            ),
            "trial_count": len(self._trials),
            "session_id": self._session_id,
            "error": self._last_error,
            "trials": list(self._trials),
            "report": _public_report(self._report),
        }


def _public_report(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    keys = (
        "counts",
        "groups",
        "comparisons",
        "word_accuracy",
        "word_chance",
        "word_confusion",
        "verdict",
    )
    return {key: report.get(key) for key in keys}
