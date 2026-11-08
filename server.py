#!/usr/bin/env python3
"""Local real-time speech-to-text server.

Captures any input exposed by the active PortAudio host API (built-in mic,
Bluetooth headset, virtual loopback), detects speech, transcribes it with
faster-whisper and streams
the text to a small browser UI over WebSocket.

Audio never leaves the machine after the model is downloaded.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import queue
import re
import threading
import time
import wave
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from pydantic import BaseModel

from audio_experiment import AudioTrialRecorder
from device_discovery import enumerate_audio_devices, input_devices, resolve_input_device
from signal_monitor import LiveSignalMonitor

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

TARGET_RATE = 16000          # whisper input rate
BLOCK_SEC = 0.03             # audio callback / segmenter block size
PRE_ROLL_SEC = 0.35          # audio kept before speech onset
END_SILENCE_SEC = 0.70       # silence that closes an utterance
MAX_UTTERANCE_SEC = 18.0     # force-flush long monologues
PARTIAL_EVERY_SEC = 1.0      # partial transcript cadence
MIN_SPEECH_SEC = 0.25        # ignore shorter blips

JUNK = {
    "",
    ".",
    "..",
    "...",
    "you",
    "[blank_audio]",
    "(silence)",
    "[silence]",
}


# --------------------------------------------------------------------------
# Audio helpers
# --------------------------------------------------------------------------


def dbfs(block: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
    return 20.0 * math.log10(rms + 1e-9)


def resample(audio: np.ndarray, src_rate: int, dst_rate: int = TARGET_RATE) -> np.ndarray:
    if src_rate == dst_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    n_out = int(round(audio.size * dst_rate / src_rate))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    src_x = np.arange(audio.size, dtype=np.float64)
    dst_x = np.linspace(0.0, audio.size - 1.0, n_out, dtype=np.float64)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


# --------------------------------------------------------------------------
# Audio sources
# --------------------------------------------------------------------------


class MicSource:
    """Live audio from any input device exposed by PortAudio."""

    is_file = False

    def __init__(self, device: int | str | None):
        self.info, self.index = resolve_input_device(device)
        self.info = {**self.info, "index": self.index}
        self.name = str(self.info["name"]).strip()
        self.label = self.name
        self.samplerate = int(self.info["default_samplerate"])
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=600)
        self.level = 0.0
        self.level_dbfs: float | None = None
        self._stream: sd.InputStream | None = None
        self._status_error: str | None = None
        self.block_sink = None

    def start(self) -> None:
        blocksize = max(128, int(self.samplerate * BLOCK_SEC))

        def callback(indata, frames, time_info, status):
            if status:
                self._status_error = str(status)
            mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
            self.level_dbfs = dbfs(mono)
            self.level = max(0.0, min(1.0, (self.level_dbfs + 60.0) / 55.0))
            if self.block_sink is not None:
                try:
                    self.block_sink(mono, self.samplerate)
                except Exception as exc:
                    self._status_error = f"experiment callback: {exc}"
            try:
                self.queue.put_nowait(mono)
            except queue.Full:
                pass  # never block the audio thread

        try:
            self._stream = sd.InputStream(
                device=self.index,
                channels=1,
                samplerate=self.samplerate,
                dtype="float32",
                blocksize=blocksize,
                latency="low",
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot open input {self.name!r} ({self.samplerate} Hz): {exc}. "
                "Check microphone permission for the app running this server."
            ) from exc

        if self._status_error:
            print(f"[audio] stream warning: {self._status_error}", flush=True)

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        self.level = 0.0
        self.level_dbfs = None


class FileSource:
    """Replay a 16-bit PCM wav in real time. Useful for testing without a mic."""

    is_file = True

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with wave.open(str(self.path), "rb") as wav:
            if wav.getsampwidth() != 2:
                raise RuntimeError("Only 16-bit PCM wav files are supported")
            self.samplerate = wav.getframerate()
            channels = wav.getnchannels()
            raw = wav.readframes(wav.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        self.audio = audio
        self.name = f"file:{self.path.name}"
        self.label = self.name
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=600)
        self.level = 0.0
        self.level_dbfs: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="file-source", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        block = max(128, int(self.samplerate * BLOCK_SEC))
        chunks = [
            self.audio[i : i + block]
            for i in range(0, self.audio.size, block)
        ]
        # A second of silence lets the segmenter close the last utterance.
        chunks.extend(
            np.zeros(block, dtype=np.float32)
            for _ in range(int(1.0 / BLOCK_SEC))
        )
        for chunk in chunks:
            if self._stop.is_set():
                break
            self.level_dbfs = dbfs(chunk)
            self.level = max(0.0, min(1.0, (self.level_dbfs + 60.0) / 55.0))
            try:
                self.queue.put_nowait(chunk)
            except queue.Full:
                pass
            self._stop.wait(len(chunk) / self.samplerate)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self.level = 0.0
        self.level_dbfs = None


# --------------------------------------------------------------------------
# Transcription pipeline
# --------------------------------------------------------------------------


class Listener:
    def __init__(self, args: argparse.Namespace):
        self.model_name = args.model
        self.language = None if args.language in (None, "", "auto") else args.language
        if self.model_name.endswith(".en") and self.language not in (None, "en"):
            print("[model] *.en model selected; forcing language=en", flush=True)
            self.language = "en"

        self.threshold_db = args.threshold
        self.input_file = args.input_file
        self.device_arg = args.device
        self.no_transcription = getattr(args, "no_transcription", False)

        self.model: WhisperModel | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[WebSocket] = set()

        self._lock = threading.RLock()
        self._listen_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._jobs: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._finalized_utt = -1
        self._utt = 0

        self.status = "loading"
        self.error: str | None = None
        self.transcription_error: str | None = None
        self.device_name: str | None = None
        self.started_at: float | None = None
        self.source: MicSource | FileSource | None = None
        self.signal_monitor = LiveSignalMonitor()
        self.segments: list[dict[str, Any]] = []
        self.run_segments: list[dict[str, Any]] = []
        self._segmenter: threading.Thread | None = None
        self._worker: threading.Thread | None = None
        self.audio_trials = AudioTrialRecorder(ROOT / "jawdata")

    # -- lifecycle --------------------------------------------------------

    def load_model(self) -> None:
        started = time.time()
        print(f"[model] loading {self.model_name!r} (first run downloads it) ...", flush=True)
        self.model = WhisperModel(
            self.model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=min(8, os.cpu_count() or 4),
        )
        print(f"[model] ready in {time.time() - started:.1f}s", flush=True)
        self.status = "idle"
        self._emit(self.snapshot())

    @property
    def transcription_status(self) -> str:
        if self.no_transcription:
            return "disabled"
        if self.transcription_error:
            return "error"
        return "ready" if self.model is not None else "loading"

    def start(self, device: int | str | None = None) -> dict[str, Any]:
        with self._listen_lock:
            if self.status == "listening":
                if device is None or str(device) == str(self.device_arg):
                    return self.snapshot()
                self._stop_locked()

            previous_device = self.device_arg
            if device is not None:
                self.device_arg = device

            self._stop_event = threading.Event()
            self._jobs = queue.Queue()
            self.signal_monitor.reset()
            self._finalized_utt = -1
            self._utt = 0
            self.error = None
            with self._lock:
                self.run_segments = []

            try:
                source = (
                    FileSource(self.input_file)
                    if self.input_file
                    else MicSource(self.device_arg)
                )
                self.source = source
                if isinstance(source, MicSource):
                    source.block_sink = self.audio_trials.push_block
                if self.model is not None:
                    self._worker = threading.Thread(
                        target=self._worker_loop,
                        args=(self.model, self._stop_event, self._jobs),
                        name="whisper-worker",
                        daemon=True,
                    )
                    self._worker.start()
                source.start()
                self._segmenter = threading.Thread(
                    target=self._segment_loop,
                    args=(source, self._stop_event, self._jobs),
                    name="segmenter",
                    daemon=True,
                )
                self._segmenter.start()
            except Exception as exc:
                self.error = str(exc)
                self.status = "error"
                self.device_arg = previous_device
                self._stop_event.set()
                if self._segmenter is not None:
                    self._segmenter.join(timeout=2)
                try:
                    self._jobs.put_nowait(None)
                except queue.Full:
                    pass
                if self._worker is not None:
                    self._worker.join(timeout=5)
                self._cleanup_source()
                self._emit(self.snapshot())
                raise

            self.status = "listening"
            self.device_name = source.label
            self.started_at = time.time()
            print(f"[listen] source={source.label} rate={source.samplerate} Hz", flush=True)
            self._emit(self.snapshot())
            return self.snapshot()

    def _stop_locked(self) -> None:
        source = self.source
        if source is None:
            self.status = "idle"
            return

        self.audio_trials.cancel_active("Microphone capture stopped before this JawLab trial was saved.")
        self._stop_event.set()
        source.stop()
        if self._segmenter is not None:
            self._segmenter.join(timeout=6)
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            self._jobs.put(None)
        if self._worker is not None:
            self._worker.join(timeout=40)

        self._cleanup_source()
        self.status = "idle"
        self.device_name = None
        self.started_at = None
        self._emit(self.snapshot())

    def _cleanup_source(self) -> None:
        if self.source is not None:
            try:
                self.source.stop()
            except Exception:
                pass
        self.source = None
        self._segmenter = None
        self._worker = None
        self._utt = 0

    def stop(self) -> dict[str, Any]:
        with self._listen_lock:
            if self.status != "listening":
                return self.snapshot()
            self._stop_locked()
            return self.snapshot()

    def shutdown(self) -> None:
        try:
            self.stop()
        except Exception:
            pass

    # -- segmenter --------------------------------------------------------

    def _segment_loop(
        self,
        source: MicSource | FileSource,
        stop_event: threading.Event,
        jobs: queue.Queue[dict[str, Any] | None],
    ) -> None:
        rate = source.samplerate
        pre_roll = deque(maxlen=max(1, int(PRE_ROLL_SEC / BLOCK_SEC)))
        buffer: list[np.ndarray] = []
        speaking = False
        utt = 0
        start_ts = 0.0
        silence_sec = 0.0
        speech_sec = 0.0
        last_voice = 0.0
        last_partial = 0.0
        candidate = 0

        noise_db = -60.0
        calibration: list[float] = []
        threshold = self.threshold_db if self.threshold_db is not None else -45.0

        def threshold_now() -> float:
            if self.threshold_db is not None:
                return self.threshold_db
            return max(-48.0, min(-28.0, noise_db + 10.0))

        def finalize() -> None:
            nonlocal buffer, speaking, silence_sec, speech_sec, start_ts, last_partial
            if buffer and speech_sec >= MIN_SPEECH_SEC:
                audio = resample(np.concatenate(buffer), rate)
                self._submit(jobs, "final", utt, audio, start_ts, speech_sec)
            buffer = []
            speaking = False
            silence_sec = 0.0
            speech_sec = 0.0
            last_partial = 0.0

        while not stop_event.is_set():
            try:
                block = source.queue.get(timeout=0.1)
            except queue.Empty:
                if speaking and (time.time() - last_voice) >= END_SILENCE_SEC:
                    finalize()
                continue

            block = np.asarray(block, dtype=np.float32).reshape(-1)
            if block.size == 0:
                continue
            block_db = dbfs(block)

            if isinstance(source, MicSource):
                self.signal_monitor.update(
                    block,
                    rate,
                    device=source.name,
                    noise_floor_db=noise_db,
                    calibrated=len(calibration) >= 34,
                )

            if not speaking:
                if len(calibration) < 34:
                    calibration.append(block_db)
                    if len(calibration) == 34:
                        noise_db = float(np.percentile(calibration, 20))
                elif block_db < noise_db + 12:
                    noise_db = 0.995 * noise_db + 0.005 * block_db
                threshold = threshold_now()

                if block_db > threshold:
                    candidate += 1
                else:
                    candidate = 0

                pre_roll.append(block)
                if candidate < 2:
                    continue

                onset_blocks = candidate
                speaking = True
                utt += 1
                buffer = list(pre_roll)
                pre_roll.clear()
                candidate = 0
                speech_sec = onset_blocks * BLOCK_SEC
                silence_sec = 0.0
                start_ts = time.time() - len(buffer) * BLOCK_SEC
                last_voice = time.time()
                last_partial = last_voice
                continue

            buffer.append(block)
            if block_db > threshold:
                speech_sec += BLOCK_SEC
                silence_sec = 0.0
                last_voice = time.time()
            else:
                silence_sec += BLOCK_SEC

            duration = len(buffer) * BLOCK_SEC
            if silence_sec >= END_SILENCE_SEC or duration >= MAX_UTTERANCE_SEC:
                finalize()
                threshold = threshold_now()
                continue

            now = time.time()
            if now - last_partial >= PARTIAL_EVERY_SEC and speech_sec >= 1.0:
                last_partial = now
                partial_audio = resample(np.concatenate(buffer), rate)
                self._submit(jobs, "partial", utt, partial_audio, start_ts, speech_sec)

        if speaking:
            finalize()

    def _submit(
        self,
        jobs: queue.Queue[dict[str, Any] | None],
        kind: str,
        utt: int,
        audio: np.ndarray,
        start_ts: float,
        speech_sec: float,
    ) -> None:
        if self.model is None or audio.size < TARGET_RATE * MIN_SPEECH_SEC:
            return
        job = {
            "kind": kind,
            "utt": utt,
            "audio": audio,
            "start": start_ts,
            "speech": speech_sec,
        }
        if kind == "partial" and jobs.qsize() >= 3:
            return
        try:
            jobs.put_nowait(job)
        except queue.Full:
            if kind != "final":
                return
            try:
                jobs.get_nowait()
            except queue.Empty:
                pass
            try:
                jobs.put_nowait(job)
            except queue.Full:
                print("[warn] transcription queue full; dropped a segment", flush=True)

    # -- whisper worker ---------------------------------------------------

    def _worker_loop(
        self,
        model: WhisperModel,
        stop_event: threading.Event,
        jobs: queue.Queue[dict[str, Any] | None],
    ) -> None:
        while True:
            job = jobs.get()
            if job is None:
                break
            kind = job["kind"]
            utt = int(job["utt"])
            if kind == "partial" and utt <= self._finalized_utt:
                continue
            try:
                text = self._transcribe(model, job["audio"], partial=(kind == "partial"))
            except Exception as exc:
                print(f"[warn] transcription failed: {exc}", flush=True)
                continue
            if not text:
                continue

            if kind == "partial":
                if utt <= self._finalized_utt or stop_event.is_set():
                    continue
                self._emit(
                    {
                        "type": "partial",
                        "utt": utt,
                        "text": text,
                        "start": job["start"],
                    }
                )
                continue

            self._finalized_utt = max(self._finalized_utt, utt)
            segment = {
                "id": f"{utt}-{int(job['start'] * 1000)}",
                "text": text,
                "start": job["start"],
                "duration": round(len(job["audio"]) / TARGET_RATE, 2),
                "speech_duration": round(float(job["speech"]), 2),
            }
            with self._lock:
                self.segments.append(segment)
                self.run_segments.append(segment)
            self._emit({"type": "final", "utt": utt, "segment": segment})

    def _transcribe(self, model: WhisperModel, audio: np.ndarray, partial: bool) -> str:
        segments, _info = model.transcribe(
            audio,
            language=self.language,
            beam_size=1 if partial else 5,
            best_of=1 if partial else 5,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=not partial,
            vad_parameters={"min_silence_duration_ms": 250, "speech_pad_ms": 120}
            if not partial
            else None,
            without_timestamps=True,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        text = " ".join(text.split())
        if text.lower() in JUNK:
            return ""
        return text

    # -- state / events ---------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            segments = list(self.segments)
            run_segments = list(self.run_segments)
        recognized_words = sum(
            len(re.findall(r"\b[\w’'-]+\b", str(segment.get("text", ""))))
            for segment in run_segments
        )
        speech_seconds = sum(
            float(segment.get("speech_duration") or segment.get("duration") or 0)
            for segment in run_segments
        )
        return {
            "type": "snapshot",
            "status": self.status,
            "model": self.model_name,
            "language": self.language,
            "transcription_status": self.transcription_status,
            "device": self.device_name,
            "error": self.error,
            "transcription_error": self.transcription_error,
            "started_at": self.started_at,
            "segments": segments,
            "input_file": self.input_file,
            "samplerate": self.source.samplerate if isinstance(self.source, MicSource) else None,
            "level": self.level,
            "level_dbfs": getattr(self.source, "level_dbfs", None),
            "signal": self.signal_monitor.latest() if self.status == "listening" else None,
            "metrics": {
                "recognized_words": recognized_words,
                "recognized_word_count": recognized_words,
                "recognized_speech_seconds": round(speech_seconds, 2),
                "recognized_wpm": round(recognized_words * 60 / speech_seconds, 1)
                if speech_seconds > 0
                else None,
            },
        }

    def transcript_text(self) -> str:
        with self._lock:
            segments = list(self.segments)
        lines = []
        for seg in segments:
            stamp = datetime.fromtimestamp(seg["start"]).strftime("%H:%M:%S")
            lines.append(f"[{stamp}] {seg['text']}")
        return "\n".join(lines) + ("\n" if lines else "")

    def clear(self) -> None:
        with self._lock:
            self.segments = []
            self.run_segments = []
        self._emit({"type": "cleared"})

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    async def broadcast(self, payload: dict[str, Any]) -> None:
        dead = []
        for client in list(self.clients):
            try:
                await client.send_json(payload)
            except Exception:
                dead.append(client)
        for client in dead:
            self.clients.discard(client)

    def _emit(self, payload: dict[str, Any]) -> None:
        if self.loop is None or self.loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self.broadcast(payload), self.loop)
        except RuntimeError:
            pass

    @property
    def level(self) -> float:
        return self.source.level if self.source is not None else 0.0

    @property
    def level_dbfs(self) -> float | None:
        return getattr(self.source, "level_dbfs", None)


# --------------------------------------------------------------------------
# HTTP app
# --------------------------------------------------------------------------


class StartRequest(BaseModel):
    device: int | str | None = None


class TrialStartRequest(BaseModel):
    condition: str
    word: str | None = None
    duration_sec: float = 3.0


class TrialStopRequest(BaseModel):
    trial_id: str


def create_app(listener: Listener, args: argparse.Namespace) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        listener.attach_loop(asyncio.get_running_loop())
        if listener.no_transcription:
            listener.status = "idle"
            listener.transcription_error = "Whisper is disabled. Audio capture and JawLab trials are available."
        else:
            try:
                await asyncio.to_thread(listener.load_model)
            except Exception as exc:
                listener.status = "idle"
                listener.transcription_error = f"Could not load model {listener.model_name!r}: {exc}"
                print(f"[model] {listener.transcription_error}", flush=True)
        listener._emit(listener.snapshot())

        if not args.no_autostart and listener.model is not None:
            try:
                await asyncio.to_thread(listener.start)
            except Exception as exc:
                print(f"[listen] autostart failed: {exc}", flush=True)

        async def level_loop():
            while True:
                await asyncio.sleep(0.1)
                if listener.clients:
                    await listener.broadcast(
                        {
                            "type": "level",
                            "value": round(listener.level, 3),
                            "dbfs": round(listener.level_dbfs, 1)
                            if listener.level_dbfs is not None
                            else None,
                            "signal": listener.signal_monitor.latest()
                            if listener.status == "listening"
                            else None,
                        }
                    )

        task = asyncio.create_task(level_loop())
        try:
            yield
        finally:
            task.cancel()
            await asyncio.to_thread(listener.shutdown)

    app = FastAPI(title="Live Transcript", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/experiment", include_in_schema=False)
    async def experiment():
        return FileResponse(STATIC / "experiment.html")

    @app.get("/api/devices")
    async def devices():
        try:
            endpoints = enumerate_audio_devices()
            inputs = input_devices()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        defaults = {
            "input_index": next((d["index"] for d in endpoints if d["default_input"]), None),
            "output_index": next((d["index"] for d in endpoints if d["default_output"]), None),
        }
        return {"endpoints": endpoints, "devices": inputs, "defaults": defaults}

    @app.get("/api/experiment/capabilities")
    async def experiment_capabilities():
        snapshot = listener.snapshot()
        return {
            "capture": {
                "status": snapshot["status"],
                "device": snapshot["device"],
                "samplerate": snapshot["samplerate"],
                "error": snapshot["error"],
                "transcription_error": snapshot["transcription_error"],
                "transcription_status": snapshot["transcription_status"],
            },
            "motion": {
                "status": "native_only",
                "browser_access": False,
                "message": (
                    "This browser page cannot access native headphone motion sensors. "
                    "Open the native Subvox app and confirm Streaming with a nonzero sample rate."
                ),
            },
            "jawlab": listener.audio_trials.state(),
        }

    @app.get("/api/jawlab/state")
    async def jawlab_state():
        return listener.audio_trials.state()

    @app.post("/api/jawlab/trial/start")
    async def jawlab_trial_start(request: TrialStartRequest):
        source = listener.source
        if listener.status != "listening" or not isinstance(source, MicSource):
            raise HTTPException(
                status_code=409,
                detail="Start the selected microphone first. JawLab trials use that same server-owned stream.",
            )
        try:
            result = listener.audio_trials.start_trial(
                source.info,
                condition=request.condition,
                word=request.word,
                duration_sec=request.duration_sec,
            )
            return {**result, "state": listener.audio_trials.state()}
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jawlab/trial/stop")
    async def jawlab_trial_stop(request: TrialStopRequest):
        try:
            return await asyncio.to_thread(listener.audio_trials.stop_trial, request.trial_id)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/state")
    async def state():
        return listener.snapshot()

    @app.get("/api/transcript.txt", response_class=PlainTextResponse)
    async def transcript_txt():
        return PlainTextResponse(listener.transcript_text())

    @app.post("/api/listen/start")
    async def start(request: StartRequest):
        try:
            return await asyncio.to_thread(listener.start, request.device)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/listen/stop")
    async def stop():
        return await asyncio.to_thread(listener.stop)

    @app.post("/api/clear")
    async def clear():
        listener.clear()
        return {"ok": True}

    @app.websocket("/ws")
    async def websocket(client: WebSocket):
        await client.accept()
        listener.clients.add(client)
        try:
            await client.send_json(listener.snapshot())
            while True:
                message = await client.receive_text()
                if message == "ping":
                    await client.send_text("pong")
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            listener.clients.discard(client)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time local speech-to-text server")
    parser.add_argument("--model", default=os.getenv("WHISPER_MODEL", "base.en"),
                        help="faster-whisper model name or local path (default: base.en)")
    parser.add_argument("--language", default=os.getenv("WHISPER_LANGUAGE", "en"),
                        help="language code, or 'auto' (default: en)")
    parser.add_argument("--device", default=None,
                        help="input device name or index (default: system default)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="speech threshold in dBFS, e.g. -40 (default: adaptive)")
    parser.add_argument("--input-file", default=None,
                        help="stream a 16-bit PCM wav instead of the microphone (testing)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-autostart", action="store_true",
                        help="load the model but wait for the UI to start listening")
    parser.add_argument("--no-transcription", action="store_true",
                        help="skip Whisper loading; keep audio capture and JawLab tests available")
    parser.add_argument("--list-devices", action="store_true",
                        help="print input devices and exit")
    parser.add_argument("--all-devices", action="store_true",
                        help="with --list-devices, include output-only endpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_devices:
        devices = enumerate_audio_devices() if args.all_devices else input_devices()
        if not devices:
            print("No audio endpoints found." if args.all_devices else "No input devices found.")
            return
        for item in devices:
            if args.all_devices:
                markers = []
                if item["default_input"]:
                    markers.append("default input")
                if item["default_output"]:
                    markers.append("default output")
                marker = f" [{', '.join(markers)}]" if markers else ""
                print(
                    f"[{item['index']}] {item['name']} — "
                    f"{item['max_input_channels']} in / {item['max_output_channels']} out, "
                    f"{item['default_samplerate'] or 'unknown'} Hz, "
                    f"{item['host_api'] or 'unknown host API'}; runtime ID only{marker}"
                )
            else:
                mark = "*" if item["default"] else " "
                print(f"{mark} [{item['index']}] {item['name']} "
                      f"({item['channels']} ch, {item['default_samplerate']} Hz)")
        return

    listener = Listener(args)
    app = create_app(listener, args)

    print(f"[server] http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
