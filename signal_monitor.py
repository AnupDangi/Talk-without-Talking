"""Short-window, signal-only summaries for the live microphone stream.

Raw samples are held only in a bounded in-memory analysis window. This module
does not write audio or infer words.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

import numpy as np


BANDS: tuple[tuple[float, float], ...] = (
    (20, 80),
    (80, 160),
    (160, 320),
    (320, 640),
    (640, 1250),
    (1250, 2500),
    (2500, 4000),
    (4000, 8000),
)
BAND_LABELS = ("20–80", "80–160", "160–320", "320–640", "640–1.25k", "1.25–2.5k", "2.5–4k", "4–8k")


def _dbfs(rms: float) -> float:
    return 20.0 * math.log10(max(float(rms), 1e-9))


class LiveSignalMonitor:
    """Compute a bounded rolling acoustic frame from existing input blocks."""

    def __init__(
        self,
        *,
        window_seconds: float = 0.256,
        modulation_seconds: float = 4.0,
        publish_interval_seconds: float = 0.1,
    ) -> None:
        self.window_seconds = max(0.128, float(window_seconds))
        self.modulation_seconds = max(2.0, float(modulation_seconds))
        self.publish_interval_seconds = max(0.0, float(publish_interval_seconds))
        self._lock = threading.RLock()
        self._rate: int | None = None
        self._device: str | None = None
        self._chunks: deque[np.ndarray] = deque()
        self._sample_count = 0
        self._envelope: deque[float] = deque(maxlen=max(20, int(self.modulation_seconds * 10)))
        self._last_publish = 0.0
        self._sequence = 0
        self._latest: dict[str, Any] | None = None

    def reset(self) -> None:
        """Drop transient signal history when the microphone stream changes."""
        with self._lock:
            self._rate = None
            self._device = None
            self._chunks.clear()
            self._sample_count = 0
            self._envelope.clear()
            self._last_publish = 0.0
            self._latest = None

    def update(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        device: str | None,
        noise_floor_db: float,
        calibrated: bool,
        now: float | None = None,
        force: bool = False,
    ) -> dict[str, Any] | None:
        """Add one existing stream block and publish a compact feature frame."""
        rate = int(sample_rate)
        block = np.asarray(samples, dtype=np.float32).reshape(-1)
        if rate <= 0 or not block.size:
            return None

        instant = time.monotonic() if now is None else float(now)
        with self._lock:
            if self._rate != rate or self._device != device:
                self._chunks.clear()
                self._sample_count = 0
                self._envelope.clear()
                self._rate = rate
                self._device = device
                self._last_publish = 0.0

            copy = block.copy()
            self._chunks.append(copy)
            self._sample_count += int(copy.size)
            max_samples = max(1, int(round(rate * self.window_seconds)))
            while self._chunks and self._sample_count - self._chunks[0].size >= max_samples:
                self._sample_count -= int(self._chunks.popleft().size)
            if self._chunks and self._sample_count > max_samples:
                excess = self._sample_count - max_samples
                self._chunks[0] = self._chunks[0][excess:]
                self._sample_count -= excess

            if not force and instant - self._last_publish < self.publish_interval_seconds:
                return None
            self._last_publish = instant
            window = np.concatenate(tuple(self._chunks)) if self._chunks else copy
            rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))
            peak = float(np.max(np.abs(window)))
            spectrum = _band_dbfs(window, rate)
            bands = spectrum["bands_dbfs"]
            low_dbfs = _combined_band_dbfs(bands[:4])
            self._envelope.append(10.0 ** (low_dbfs / 20.0))
            modulation_ratio, cycle_strength, cycle_hz = self._low_band_context()
            voicing = _voicing_score(window, rate)

            floor = float(noise_floor_db) if math.isfinite(float(noise_floor_db)) else -60.0
            activity_threshold = max(-60.0, floor + 6.0)
            if not calibrated:
                activity = "calibrating"
            elif _dbfs(rms) < activity_threshold:
                activity = "near_silence"
            elif voicing >= 0.45:
                activity = "voiced_like_activity"
            else:
                activity = "acoustic_activity_uncertain"

            self._sequence += 1
            self._latest = {
                "type": "signal",
                "seq": self._sequence,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "device": device,
                "sample_rate": rate,
                "window_seconds": round(window.size / rate, 3),
                "rms_dbfs": round(_dbfs(rms), 1),
                "peak_dbfs": round(_dbfs(peak), 1),
                "bands_dbfs": bands,
                "low_band_dbfs": round(low_dbfs, 1),
                "voicing_score": round(voicing, 3),
                "low_modulation_ratio": round(modulation_ratio, 3),
                "low_cycle_strength": round(cycle_strength, 3),
                "low_cycle_hz": round(cycle_hz, 2),
                "modulation_frames": len(self._envelope),
                "periodic_low_band_pattern": bool(
                    len(self._envelope) >= 25
                    and modulation_ratio >= 0.28
                    and cycle_strength >= 0.25
                ),
                "activity": activity,
            }
            return dict(self._latest)

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._latest) if self._latest is not None else None

    def _low_band_context(self) -> tuple[float, float, float]:
        envelope = np.asarray(self._envelope, dtype=np.float64)
        if envelope.size < 16:
            return 0.0, 0.0, 0.0
        envelope -= envelope.mean()
        if float(np.dot(envelope, envelope)) <= 1e-16:
            return 0.0, 0.0, 0.0
        frame_rate = 1.0 / self.publish_interval_seconds if self.publish_interval_seconds > 0 else 10.0
        frame_rate = min(20.0, max(5.0, frame_rate))
        windowed = envelope * np.hanning(envelope.size)
        modulation = np.abs(np.fft.rfft(windowed)) ** 2
        frequencies = np.fft.rfftfreq(envelope.size, 1.0 / frame_rate)
        target = (frequencies >= 0.5) & (frequencies <= 3.0)
        reference = (frequencies >= 0.1) & (frequencies <= min(4.5, frame_rate / 2))
        ratio = float(modulation[target].sum() / (modulation[reference].sum() + 1e-12)) if target.any() else 0.0

        ac = np.correlate(envelope, envelope, mode="full")[envelope.size - 1 :]
        ac = ac / (ac[0] + 1e-12)
        min_lag = max(1, int(0.5 * frame_rate))
        max_lag = min(ac.size - 1, int(1.5 * frame_rate))
        if max_lag <= min_lag:
            return ratio, 0.0, 0.0
        segment = ac[min_lag:max_lag]
        best = int(np.argmax(segment))
        lag = min_lag + best
        return ratio, float(segment[best]), frame_rate / lag


def _band_dbfs(samples: np.ndarray, sample_rate: int) -> dict[str, list[float]]:
    n = max(32, int(samples.size))
    nfft = 1 << (n - 1).bit_length()
    signal = np.zeros(nfft, dtype=np.float64)
    signal[:n] = np.asarray(samples, dtype=np.float64)
    window = np.hanning(n)
    windowed = signal.copy()
    windowed[:n] *= window
    transform = np.fft.rfft(windowed)
    power = np.square(np.abs(transform)) / (nfft * max(float(np.square(window).sum()), 1e-12))
    if power.size > 2:
        power[1:-1] *= 2.0
    frequencies = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    levels: list[float] = []
    nyquist = sample_rate / 2.0
    for low, high in BANDS:
        mask = (frequencies >= low) & (frequencies < min(high, nyquist))
        band_rms = math.sqrt(float(power[mask].sum())) if mask.any() else 0.0
        levels.append(round(_dbfs(band_rms), 1))
    total_rms = math.sqrt(float(power.sum()))
    return {"bands_dbfs": levels, "window_rms_dbfs": round(_dbfs(total_rms), 1)}


def _combined_band_dbfs(bands_dbfs: list[float]) -> float:
    powers = [10.0 ** (value / 10.0) for value in bands_dbfs if math.isfinite(value)]
    return 10.0 * math.log10(sum(powers) + 1e-18) if powers else -180.0


def _voicing_score(samples: np.ndarray, sample_rate: int) -> float:
    """A coarse periodicity cue; not a speech or whisper classifier."""
    signal = np.asarray(samples, dtype=np.float64)
    if signal.size < int(sample_rate * 0.08):
        return 0.0
    signal = signal - signal.mean()
    energy = float(np.dot(signal, signal))
    if energy <= 1e-12:
        return 0.0
    nfft = 1 << (2 * signal.size - 1).bit_length()
    spectrum = np.fft.rfft(signal, nfft)
    ac = np.fft.irfft(spectrum * np.conjugate(spectrum), nfft)[: signal.size]
    ac /= ac[0] + 1e-12
    min_lag = max(1, int(sample_rate / 400))
    max_lag = min(signal.size - 1, int(sample_rate / 70))
    if max_lag <= min_lag:
        return 0.0
    return float(np.clip(np.max(ac[min_lag : max_lag + 1]), 0.0, 1.0))
