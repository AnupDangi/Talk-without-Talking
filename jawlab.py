#!/usr/bin/env python3
"""JawLab - can the microphone you already own sense jaw movement?

Three commands:

  devices   list input devices
  probe     quick 13-second guided test with an immediate verdict
  record    full labelled recording session for model work
  analyze   compare conditions / train a small word classifier
  monitor   live low-band and modulation meters

Everything is local; recordings are saved under ./jawdata.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

from device_discovery import input_devices, resolve_input_device

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "jawdata"

BANDS = [
    (20, 80),
    (80, 160),
    (160, 320),
    (320, 640),
    (640, 1280),
    (1280, 2560),
    (2560, 5120),
    (5120, 8000),
]
LOW_BANDS = slice(0, 3)   # 20-320 Hz
MID_BANDS = slice(3, 6)   # 320-2560 Hz
HIGH_BANDS = slice(6, 8)  # 2560-8000 Hz

PROTOCOL = [
    ("still", None, "Sit still, mouth closed. Do not move your jaw."),
    ("jaw_open", None, "Open and close your jaw about once per second, like slow chewing."),
    ("clench", None, "Clench and unclench your jaw about once per second."),
    ("silent", "hello", "Silently mouth 'hello' about once per second. No voice, no whisper."),
    ("silent", "yes", "Silently mouth 'yes' about once per second."),
    ("silent", "no", "Silently mouth 'no' about once per second."),
    ("silent", "water", "Silently mouth 'water' about once per second."),
    ("silent", "help", "Silently mouth 'help' about once per second."),
    ("whisper", "hello", "Whisper 'hello' about once per second."),
    ("audible", "hello", "Say 'hello' out loud, normally, about once per second."),
]


# --------------------------------------------------------------------------
# audio input
# --------------------------------------------------------------------------


def resolve_device(device: int | str | None):
    info, index = resolve_input_device(device)
    try:
        default_index = sd.default.device[0]
    except Exception:
        default_index = None
    return {
        "index": index,
        "name": str(info["name"]).strip(),
        "channels": int(info.get("max_input_channels", 0)),
        "rate": int(info.get("default_samplerate", 0)),
        "default": index == default_index,
    }


def capture(device: int | str | None, seconds: float, warmup: float = 1.5) -> tuple[np.ndarray, dict]:
    info = resolve_device(device)
    rate = info["rate"]
    blocks: list[np.ndarray] = []
    error: list[str] = []

    def callback(indata, frames, time_info, status):
        if status:
            error.append(str(status))
        blocks.append(np.asarray(indata[:, 0], dtype=np.float32).copy())

    blocksize = max(128, int(rate * 0.02))
    try:
        with sd.InputStream(
            device=info["index"],
            channels=1,
            samplerate=rate,
            dtype="float32",
            blocksize=blocksize,
            latency="low",
            callback=callback,
        ):
            time.sleep(warmup)
            blocks.clear()
            time.sleep(seconds)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot record from {info['name']!r}: {exc}. "
            "Check macOS microphone permission for your terminal."
        ) from exc

    if not blocks:
        raise RuntimeError("No audio received from the device")
    audio = np.concatenate(blocks)
    if error:
        print(f"[warn] stream status: {error[0]}", file=sys.stderr)
    return audio, info


def countdown(seconds: int = 3, label: str = "") -> None:
    if label:
        print(f"\n{label}")
    for n in range(seconds, 0, -1):
        print(f"  {n}...", flush=True)
        time.sleep(1)
    print("  REC", flush=True)


def save_trial(path: Path, audio: np.ndarray, rate: int) -> None:
    np.save(path.with_suffix(".npy"), audio.astype(np.float32))
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path.with_suffix(".wav")), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def load_trial(path: Path) -> tuple[np.ndarray, int]:
    npy = path.with_suffix(".npy")
    if npy.exists():
        audio = np.load(npy).astype(np.float32)
    else:
        with wave.open(str(path.with_suffix(".wav")), "rb") as handle:
            rate = handle.getframerate()
            raw = handle.readframes(handle.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, rate
    with wave.open(str(path.with_suffix(".wav")), "rb") as handle:
        rate = handle.getframerate()
    return audio, rate


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------


def dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20.0 * math.log10(rms + 1e-12)


def frames_of(audio: np.ndarray, rate: int, win_sec: float = 0.032, hop_sec: float = 0.010):
    n_win = max(8, int(round(win_sec * rate)))
    hop = max(1, int(round(hop_sec * rate)))
    if audio.size < n_win:
        audio = np.pad(audio, (0, n_win - audio.size))
    count = 1 + (audio.size - n_win) // hop
    index = np.arange(n_win)[None, :] + hop * np.arange(count)[:, None]
    return audio[index]


def spectral_metrics(audio: np.ndarray, rate: int) -> dict:
    audio = np.asarray(audio, dtype=np.float32)
    frames = frames_of(audio, rate)
    n_win = frames.shape[1]
    window = np.hanning(n_win).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_win, 1.0 / rate)
    scale = float(n_win) ** 2

    band_power = []
    for low, high in BANDS:
        mask = (freqs >= low) & (freqs < high)
        band_power.append(spec[:, mask].sum(axis=1) / scale)
    band_power = np.stack(band_power, axis=1)  # frames x bands
    band_db = 10.0 * np.log10(band_power + 1e-12)

    total_power = spec.sum(axis=1) / scale
    total_db = 10.0 * np.log10(total_power + 1e-12)

    env = np.sqrt(band_power[:, LOW_BANDS].sum(axis=1))
    env_centered = env - env.mean()
    frame_rate = rate / max(1, int(round(0.010 * rate)))

    mod_ratio = 0.0
    mod_peak = 0.0
    if env_centered.size >= 16:
        env_window = env_centered * np.hanning(env_centered.size)
        mod_spec = np.abs(np.fft.rfft(env_window)) ** 2
        mod_freqs = np.fft.rfftfreq(env_centered.size, 1.0 / frame_rate)
        jaw_band = (mod_freqs >= 0.5) & (mod_freqs <= 3.0)
        ref_band = (mod_freqs >= 0.1) & (mod_freqs <= 10.0)
        mod_ratio = float(mod_spec[jaw_band].sum() / (mod_spec[ref_band].sum() + 1e-12))
        if jaw_band.any():
            mod_peak = float(mod_freqs[jaw_band][np.argmax(mod_spec[jaw_band])])

    cycle_strength = 0.0
    cycle_hz = 0.0
    if env_centered.size >= 16:
        ac = np.correlate(env_centered, env_centered, mode="full")[env_centered.size - 1 :]
        ac = ac / (ac[0] + 1e-12)
        min_lag = max(1, int(0.5 * frame_rate))
        max_lag = min(ac.size - 1, int(1.5 * frame_rate))
        if max_lag > min_lag:
            segment = ac[min_lag:max_lag]
            best = int(np.argmax(segment))
            cycle_strength = float(segment[best])
            cycle_hz = frame_rate / (min_lag + best)

    flux = float(np.mean(np.abs(np.diff(band_db, axis=0)))) if band_db.shape[0] > 1 else 0.0
    zcr = float(np.mean(np.abs(np.diff(np.signbit(audio).astype(np.int8)))))

    low_db = float(band_db[:, LOW_BANDS].mean())
    mid_db = float(band_db[:, MID_BANDS].mean())
    high_db = float(band_db[:, HIGH_BANDS].mean())
    rms = dbfs(audio)

    return {
        "rms_db": rms,
        "peak_db": 20.0 * math.log10(float(np.max(np.abs(audio))) + 1e-12),
        "low_db": low_db,
        "mid_db": mid_db,
        "high_db": high_db,
        "low_rel_db": low_db - rms,
        "mid_rel_db": mid_db - rms,
        "high_rel_db": high_db - rms,
        "band_db": [float(v) for v in band_db.mean(axis=0)],
        "band_rel_db": [float(v) - rms for v in band_db.mean(axis=0)],
        "mod_ratio": mod_ratio,
        "mod_peak_hz": mod_peak,
        "cycle_strength": cycle_strength,
        "cycle_hz": cycle_hz,
        "flux": flux,
        "zcr": zcr,
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return 0.0
    pooled = math.sqrt(
        ((a.size - 1) * np.var(a, ddof=1) + (b.size - 1) * np.var(b, ddof=1))
        / (a.size + b.size - 2)
    )
    return float((a.mean() - b.mean()) / (pooled + 1e-12))


def permutation_p(a: np.ndarray, b: np.ndarray, n: int = 4000, seed: int = 7) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 1.0
    observed = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[: a.size].mean() - pool[a.size :].mean()) >= observed:
            hits += 1
    return (hits + 1) / (n + 1)


def auc_score(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return 0.5
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return wins / (pos.size * neg.size)


def trial_features(metrics: dict) -> np.ndarray:
    return np.array(
        list(metrics["band_rel_db"])
        + [
            metrics["low_rel_db"] / 60.0,
            metrics["mid_rel_db"] / 60.0,
            metrics["high_rel_db"] / 60.0,
            metrics["mod_ratio"],
            metrics["mod_peak_hz"] / 3.0,
            metrics["cycle_strength"],
            metrics["cycle_hz"] / 2.0,
            metrics["flux"] / 10.0,
            metrics["zcr"],
        ],
        dtype=float,
    )


def loo_word_accuracy(features: np.ndarray, labels: list[str]) -> tuple[float, dict]:
    labels_arr = np.asarray(labels)
    classes = sorted(set(labels))
    correct = 0
    predictions = []
    for i in range(len(labels)):
        train = np.arange(len(labels)) != i
        x_train = features[train]
        y_train = labels_arr[train]
        x_test = features[i]

        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0) + 1e-9
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std

        best_label = classes[0]
        best_dist = float("inf")
        for label in classes:
            if not np.any(y_train == label):
                continue
            centroid = x_train[y_train == label].mean(axis=0)
            dist = float(np.linalg.norm(x_test - centroid))
            if dist < best_dist:
                best_dist = dist
                best_label = label
        predictions.append(best_label)
        if best_label == labels[i]:
            correct += 1

    accuracy = correct / len(labels) if labels else 0.0
    confusion: dict[str, dict[str, int]] = {}
    for truth, pred in zip(labels, predictions):
        confusion.setdefault(truth, {})
        confusion[truth][pred] = confusion[truth].get(pred, 0) + 1
    return accuracy, confusion


def analyze_session(session_dir: Path, out_path: Path | None = None) -> dict:
    session_dir = Path(session_dir)
    meta_path = session_dir / "session.json"
    if not meta_path.exists():
        raise RuntimeError(f"No session.json in {session_dir}")
    meta = json.loads(meta_path.read_text())

    metrics_by_trial = []
    for trial in meta["trials"]:
        audio, rate = load_trial(session_dir / trial["file"])
        metrics = spectral_metrics(audio, rate)
        metrics_by_trial.append({"trial": trial, "metrics": metrics})

    groups: dict[str, list[dict]] = {}
    for item in metrics_by_trial:
        condition = item["trial"]["condition"]
        groups.setdefault(condition, []).append(item["metrics"])

    def values(condition: str, key: str):
        return np.array([m[key] for m in groups.get(condition, [])], dtype=float)

    comparisons = []
    def compare(name: str, left: str, right: str, key: str):
        a, b = values(left, key), values(right, key)
        if a.size < 2 or b.size < 2:
            return
        comparisons.append(
            {
                "name": name,
                "metric": key,
                "left": left,
                "right": right,
                "left_mean": float(a.mean()),
                "right_mean": float(b.mean()),
                "d": cohens_d(a, b),
                "p": permutation_p(a, b),
                "auc": auc_score(a, b),
            }
        )

    for key in ["low_rel_db", "mod_ratio", "cycle_strength", "flux"]:
        compare("still vs jaw_open", "jaw_open", "still", key)
        compare("still vs clench", "clench", "still", key)
        compare("still vs silent", "silent", "still", key)
        compare("silent vs audible", "audible", "silent", key)
        compare("silent vs whisper", "whisper", "silent", key)

    silent_items = [m for m in metrics_by_trial if m["trial"]["condition"] == "silent"]
    word_accuracy = None
    word_confusion = None
    word_chance = None
    if len(silent_items) >= 6:
        features = np.stack([trial_features(m["metrics"]) for m in silent_items])
        labels = [str(m["trial"]["word"] or "") for m in silent_items]
        if len(set(labels)) >= 2:
            word_accuracy, word_confusion = loo_word_accuracy(features, labels)
            word_chance = 1.0 / len(set(labels))

    jaw_p = min(
        [c["p"] for c in comparisons if c["name"] == "still vs jaw_open"] or [1.0]
    )
    jaw_metrics = [
        c
        for c in comparisons
        if c["name"] == "still vs jaw_open"
        and c["metric"] in ("low_rel_db", "mod_ratio", "cycle_strength")
    ]
    jaw_d = max([abs(c["d"]) for c in jaw_metrics] or [0.0])
    jaw_auc = max([c["auc"] for c in jaw_metrics] or [0.5])
    cycle = values("jaw_open", "cycle_strength")
    cycle_hz = values("jaw_open", "cycle_hz")
    jaw_cycle = float(np.median(cycle)) if cycle.size else 0.0
    jaw_cycle_hz = float(np.median(cycle_hz)) if cycle_hz.size else 0.0

    if jaw_d >= 1.0 and jaw_p < 0.05:
        verdict = "YES - jaw movement is detectable in this microphone"
    elif jaw_d >= 0.5 or jaw_auc >= 0.7 or jaw_cycle >= 0.25:
        verdict = "WEAK - some jaw-correlated signal, not yet reliable"
    else:
        verdict = "NO - this microphone does not show usable jaw-motion signal"

    report = {
        "session": str(session_dir),
        "subject": meta.get("subject"),
        "device": meta.get("device"),
        "rate": meta.get("rate"),
        "created": meta.get("created"),
        "counts": {k: len(v) for k, v in groups.items()},
        "groups": {
            condition: {
                key: float(np.mean([m[key] for m in items]))
                for key in [
                    "rms_db",
                    "low_db",
                    "low_rel_db",
                    "mid_db",
                    "high_db",
                    "mod_ratio",
                    "mod_peak_hz",
                    "cycle_strength",
                    "cycle_hz",
                    "flux",
                    "zcr",
                ]
            }
            for condition, items in groups.items()
        },
        "comparisons": comparisons,
        "word_accuracy": word_accuracy,
        "word_chance": word_chance,
        "word_confusion": word_confusion,
        "verdict": {
            "summary": verdict,
            "jaw_effect_d": jaw_d,
            "jaw_p": jaw_p,
            "jaw_auc": jaw_auc,
            "jaw_cycle_strength": jaw_cycle,
            "jaw_cycle_hz": jaw_cycle_hz,
        },
    }

    out_path = out_path or session_dir / "report.json"
    out_path.write_text(json.dumps(report, indent=2))
    markdown = render_markdown(report)
    out_path.with_suffix(".md").write_text(markdown)
    return report


def render_markdown(report: dict) -> str:
    lines = []
    lines.append("# JawLab report\n")
    lines.append(f"- Session: `{report['session']}`")
    lines.append(f"- Subject: {report['subject']}")
    lines.append(f"- Device: {report['device']} @ {report['rate']} Hz")
    lines.append(f"- Trials: {report['counts']}\n")

    lines.append("## Condition means\n")
    lines.append("| condition | n | low dB | low rel | mod 0.5-3 Hz | cycle strength | cycle Hz |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for condition, values in report["groups"].items():
        lines.append(
            f"| {condition} | {report['counts'][condition]} | "
            f"{values['low_db']:.1f} | {values['low_rel_db']:.1f} | "
            f"{values['mod_ratio']:.3f} | "
            f"{values['cycle_strength']:.3f} | {values['cycle_hz']:.2f} |"
        )

    lines.append("\n## Tests\n")
    lines.append("| comparison | metric | mean A | mean B | Cohen d | p | AUC |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for item in report["comparisons"]:
        lines.append(
            f"| {item['name']} | {item['metric']} | {item['left_mean']:.3f} | "
            f"{item['right_mean']:.3f} | {item['d']:.2f} | {item['p']:.4f} | "
            f"{item['auc']:.2f} |"
        )

    lines.append("\n## Verdict\n")
    verdict = report["verdict"]
    lines.append(f"**{verdict['summary']}**\n")
    lines.append(f"- Jaw effect size (best metric): d = {verdict['jaw_effect_d']:.2f}")
    lines.append(f"- Permutation p: {verdict['jaw_p']:.4f}")
    lines.append(f"- AUC: {verdict['jaw_auc']:.2f}")
    lines.append(
        f"- Jaw cycle strength in jaw_open: {verdict['jaw_cycle_strength']:.2f} "
        f"at {verdict['jaw_cycle_hz']:.2f} Hz"
    )

    if report["word_accuracy"] is not None:
        lines.append(
            f"\nSilent-word leave-one-out accuracy: **{report['word_accuracy']:.0%}** "
            f"(chance {report['word_chance']:.0%})"
        )
        lines.append("\nConfusion (rows = truth, columns = predicted):")
        lines.append("```")
        lines.append(json.dumps(report["word_confusion"], indent=2))
        lines.append("```")

    lines.append("\n## Next step\n")
    if verdict["summary"].startswith("YES"):
        lines.append(
            "Collect the full protocol with several reps, train a per-user "
            "10-word classifier, then test whether accuracy improves with a "
            "contact mic or in-ear mic."
        )
    elif verdict["summary"].startswith("WEAK"):
        lines.append(
            "Repeat with the mic closer to the jaw or use a contact mic taped "
            "to the jaw angle. Do not invest in a large model yet."
        )
    else:
        lines.append(
            "Mic-only silent articulation is below the noise floor. Switch to a "
            "contact mic / in-ear mic or sEMG before building a classifier."
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_devices(_args) -> None:
    devices = input_devices()
    if not devices:
        print("No input devices found.")
        return
    for item in devices:
        mark = "*" if item["default"] else " "
        print(f"{mark} [{item['index']}] {item['name']} ({item['rate']} Hz)")


def cmd_probe(args) -> None:
    info = resolve_device(args.device)
    print(f"\nJawLab probe - input: {info['name']} ({info['rate']} Hz)")
    print("This takes about 15 seconds. Follow the prompts exactly.\n")
    input("Press Enter when ready... ")

    trials = []
    for condition, label, instruction in [
        ("still", "STILL", "Keep still. Mouth closed. Do not move your jaw."),
        ("jaw_open", "JAW", "Open and close your jaw about once per second."),
        ("silent", "SILENT", "Silently mouth: hello ... yes ... no ... (no voice)."),
        ("still", "STILL", "Keep still again."),
    ]:
        print(f"\n[{condition}] {instruction}")
        if condition == "silent":
            time.sleep(1.5)
        countdown(3)
        audio, info = capture(info["index"], args.seconds)
        metrics = spectral_metrics(audio, info["rate"])
        trials.append({"condition": condition, "label": label, "metrics": metrics})
        print(
            f"  low {metrics['low_db']:6.1f} dB (rel {metrics['low_rel_db']:5.1f}) | "
            f"mod {metrics['mod_ratio']:.3f} | "
            f"cycle {metrics['cycle_strength']:.2f} @ {metrics['cycle_hz']:.2f} Hz"
        )

    still_low = np.array(
        [t["metrics"]["low_rel_db"] for t in trials if t["condition"] == "still"]
    )
    jaw_low = np.array(
        [t["metrics"]["low_rel_db"] for t in trials if t["condition"] == "jaw_open"]
    )
    d = cohens_d(jaw_low, still_low)
    p = permutation_p(jaw_low, still_low)
    cycle = max(t["metrics"]["cycle_strength"] for t in trials if t["condition"] == "jaw_open")

    print("\n--- quick verdict ---")
    print(f"jaw low-band relative effect: d={d:.2f}, p={p:.3f}, cycle={cycle:.2f}")
    if d >= 0.8 or cycle >= 0.25:
        print("Signal present. Run `record` for the full dataset.")
    else:
        print("No usable jaw signal with this mic placement.")
        print("Next: contact mic on the jaw angle, or in-ear mic.")

    out_dir = Path(args.out) / ("probe-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe.json").write_text(
        json.dumps({"device": info["name"], "trials": trials, "d": d, "p": p}, indent=2)
    )
    print(f"saved: {out_dir}")


def cmd_record(args) -> None:
    info = resolve_device(args.device)
    out_dir = Path(args.out) / (args.subject + "-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nJawLab recording - input: {info['name']} ({info['rate']} Hz)")
    print(f"Output: {out_dir}")
    print(f"{len(PROTOCOL)} conditions x {args.reps} reps x {args.seconds:.1f}s")
    print("\nRules:")
    print("  - Keep the mic and your head in the same position.")
    print("  - Silent means silent: no voice, no whisper, only mouth movement.")
    print("  - Keep a steady rhythm for jaw_open and clench.")
    input("\nPress Enter to begin... ")

    meta = {
        "version": 1,
        "subject": args.subject,
        "device": info["name"],
        "device_index": info["index"],
        "rate": info["rate"],
        "created": datetime.now().isoformat(timespec="seconds"),
        "seconds_per_trial": args.seconds,
        "reps": args.reps,
        "trials": [],
    }

    index = 0
    for rep in range(args.reps):
        for condition, word, instruction in PROTOCOL:
            index += 1
            print(f"\n[{index}] rep {rep + 1}/{args.reps} - {condition}" + (f" / {word}" if word else ""))
            print(f"    {instruction}")
            countdown(3)
            audio, _ = capture(info["index"], args.seconds)
            name = f"{index:03d}_{condition}" + (f"_{word}" if word else "")
            save_trial(out_dir / name, audio, info["rate"])
            meta["trials"].append(
                {
                    "file": name,
                    "condition": condition,
                    "word": word,
                    "start": time.time(),
                    "duration": len(audio) / info["rate"],
                }
            )
            print("    saved", name)

    (out_dir / "session.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSession complete: {out_dir}")
    print(f"Analyze with: .venv/bin/python jawlab.py analyze {out_dir}")


def cmd_analyze(args) -> None:
    out = Path(args.out) if args.out else None
    report = analyze_session(Path(args.session), out)
    print(render_markdown(report))


def cmd_monitor(args) -> None:
    info = resolve_device(args.device)
    rate = info["rate"]
    blocks: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)

    def callback(indata, frames, time_info, status):
        try:
            blocks.put_nowait(np.asarray(indata[:, 0], dtype=np.float32).copy())
        except queue.Full:
            pass

    print(f"\nJawLab monitor - {info['name']} ({rate} Hz)")
    print("Try: still -> open/close jaw -> clench -> silently mouth words.")
    print("Watch LOW and MOD. Ctrl+C to stop.\n")

    ring: list[np.ndarray] = []
    ring_seconds = 0.6
    last_print = 0.0
    started = time.time()

    with sd.InputStream(
        device=info["index"],
        channels=1,
        samplerate=rate,
        dtype="float32",
        blocksize=max(128, int(rate * 0.02)),
        latency="low",
        callback=callback,
    ):
        try:
            while args.duration <= 0 or time.time() - started < args.duration:
                try:
                    block = blocks.get(timeout=0.5)
                except queue.Empty:
                    continue
                ring.append(block)
                total = sum(len(x) for x in ring)
                while total > rate * ring_seconds and len(ring) > 1:
                    total -= len(ring.pop(0))
                now = time.time()
                if now - last_print < 0.1:
                    continue
                last_print = now
                audio = np.concatenate(ring)
                metrics = spectral_metrics(audio, rate)
                bars = int(max(0.0, min(1.0, (metrics["low_db"] + 80) / 60)) * 30)
                print(
                    f"\rLOW {metrics['low_db']:6.1f} dB "
                    f"(rel {metrics['low_rel_db']:5.1f}) {'#' * bars:<30} "
                    f"MOD {metrics['mod_ratio']:.3f} CYCLE {metrics['cycle_strength']:.2f}",
                    end="",
                    flush=True,
                )
        except KeyboardInterrupt:
            pass
        finally:
            print("\nstopped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test whether a microphone can sense jaw movement")
    sub = parser.add_subparsers(dest="command", required=True)

    dev = sub.add_parser("devices", help="list audio inputs")
    dev.set_defaults(func=cmd_devices)

    probe = sub.add_parser("probe", help="quick guided jaw-signal test")
    probe.add_argument("--device", default=None)
    probe.add_argument("--seconds", type=float, default=3.0)
    probe.add_argument("--out", default=str(DEFAULT_OUT))
    probe.set_defaults(func=cmd_probe)

    record = sub.add_parser("record", help="record the labelled protocol")
    record.add_argument("--device", default=None)
    record.add_argument("--subject", default="subject")
    record.add_argument("--reps", type=int, default=4)
    record.add_argument("--seconds", type=float, default=3.0)
    record.add_argument("--out", default=str(DEFAULT_OUT))
    record.set_defaults(func=cmd_record)

    analyze = sub.add_parser("analyze", help="analyze a recorded session")
    analyze.add_argument("session")
    analyze.add_argument("--out", default=None, help="report path (default: inside session)")
    analyze.set_defaults(func=cmd_analyze)

    monitor = sub.add_parser("monitor", help="live low-band / modulation meter")
    monitor.add_argument("--device", default=None)
    monitor.add_argument("--duration", type=float, default=0, help="seconds (0 = until Ctrl+C)")
    monitor.set_defaults(func=cmd_monitor)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
