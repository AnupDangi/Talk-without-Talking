# Talk Without Talking — live speech-to-text

> Testing the silent jaw-motion side of the project? See **[JAWLAB.md](JAWLAB.md)**
> and run `python jawlab.py probe --device "WH-CH720N"`.
> For the Subvox motion route, WH-CH720N compatibility, and a cross-modal test,
> see **[SUBVOX_EXPERIMENT.md](SUBVOX_EXPERIMENT.md)**. Upstream Subvox source is
> retained under `experiments/subvox/` with its MIT license.
> The runnable device experiment is available at `/experiment` while the local
> server is running.

Local real-time transcription server. Your laptop microphone (or another input
exposed by the installed PortAudio host, including Bluetooth call inputs when
the host provides them) is captured continuously, speech is detected,
transcribed on-device with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), and the text
streams to a browser UI over WebSocket.

Audio never leaves the machine. The only network access is the one-time model
download from Hugging Face.

## Requirements

- macOS with Python 3.10+ and `uv` (or plain `pip`)
- `portaudio` for microphone capture: `brew install portaudio`
- ~1 GB free for the app + model (`base.en` is ~145 MB)

`ffmpeg` is only needed if you convert audio files for the test mode.

## Setup

```bash
cd "Talk-Without-Talking"
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
```

Or with pip:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
./run.sh
# or
.venv/bin/python server.py
```

Open <http://127.0.0.1:8765> and press **Start listening**.

First launch downloads the model (`base.en` by default) and takes a minute.
After that startup is ~1–2 seconds.

### macOS microphone permission

The first capture triggers a permission prompt for the app running the server
(Terminal, iTerm, OpenCode…). If no prompt appears or capture fails:

**System Settings → Privacy & Security → Microphone** → enable your terminal.
Restart the terminal after granting access.

## Device Signal Lab

Open <http://127.0.0.1:8765/experiment> to compare the system's microphone
audio route with Subvox's separate native headphone-motion route. The page
enumerates host-visible input and output endpoints without a model allowlist,
selects an input, starts the existing server-owned capture stream on demand,
shows whether Whisper is loading, ready, disabled, or failed to load, and
displays transcription pace when Whisper is available. The transcript preview
links to the full live transcript at `/`; both pages use this same server and
microphone stream. Its live signal
monitor plots 30 seconds of RMS level and frequency-band energy, with cautious
activity indicators. JawLab trials tee that same audio stream; they do not open
a second microphone stream.

The monitor keeps only a short in-memory analysis window for display. Feature
logging is off by default; start it explicitly while the mic is active to keep
up to 30 minutes of derived features in the page, then export a CSV. It never
saves raw audio. Clicking **Record trial** in JawLab is a separate opt-in that
saves that short controlled trial locally under `jawdata/`. The monitor cannot
recognize silently mouthed words; see [SPEC-live-signal-monitor.md](SPEC-live-signal-monitor.md).

To compare inputs, connect the Bluetooth headset and refresh the inventory. If
its microphone is listed, select it and press **Start microphone**; repeat with
the computer's built-in mic. The inventory is host-driven and does not filter
by model, so a missing endpoint means the current audio host did not expose it
at that time. The monitor distinguishes near-silence from acoustic activity
and shows a periodicity cue, but it cannot reliably label whisper versus noise
or prove that low-frequency activity came from the jaw. Speak normally and
whisper as acoustic controls, then try silent trials separately with the
correct user-selected label. Do not expect the monitor to output words.

For a lightweight audio-only run without loading Whisper:

```bash
./run.sh --no-transcription --no-autostart
```

For normal transcription, run `./run.sh --no-autostart`. The experiment page
and the live transcript share this server. If the page says Whisper is disabled,
the running server was started with `--no-transcription`; stop that server and
restart with the normal command above. `--no-autostart` loads Whisper without
opening the microphone, so capture still waits for **Start microphone**. A trial
needs an active microphone stream and saves local artifacts in
`jawdata/browser-*`. The closed-set result becomes available after at least six
silent-mouthing trials across two or more word labels. It is a pilot metric,
not open-vocabulary silent speech recognition.

The browser cannot read Apple's CoreMotion headphone stream. Use the native
Subvox app for that route and confirm that it reports `Streaming` with a
nonzero sample rate before collecting motion trials. See
[SUBVOX_EXPERIMENT.md](SUBVOX_EXPERIMENT.md) for the protocol and current
platform limitations.

### Choose the microphone

```bash
./run.sh --list-devices
./run.sh --device "WH-CH720N"          # name (partial match works)
./run.sh --device 2                    # or index
```

The UI also has an input dropdown; switching devices restarts the stream.

### Bluetooth headsets

While a headset mic is active, macOS switches it to the HFP profile, so the mic
runs at 16 kHz and playback quality drops. That is normal. If transcription is
poor, switch the input dropdown back to the built-in mic and use the headset
only for audio output.

### Capture system audio ("anything playing on the laptop")

macOS does not expose system output as a microphone by default. Install a
virtual loopback device, then select it as the input:

```bash
brew install blackhole-2ch
```

1. Create a **Multi-Output Device** in *Audio MIDI Setup* containing your
   speakers + BlackHole 2ch (so you can still hear it).
2. Set that Multi-Output as system output.
3. Select **BlackHole 2ch** in the server/UI input dropdown.

Everything playing on the machine is then transcribed. The same approach works
with Loopback or any other virtual audio driver.

## Model / accuracy tuning

```bash
./run.sh --model tiny.en     # fastest, least accurate
./run.sh --model base.en     # default
./run.sh --model small.en    # slower, noticeably better
./run.sh --model large-v3    # best, heavy on CPU
```

Other useful flags:

| Flag | Meaning |
| --- | --- |
| `--language auto` | Detect language (use a multilingual model, e.g. `small`) |
| `--threshold -40` | Fixed speech threshold in dBFS (default: adaptive) |
| `--no-autostart` | Load the model but wait for the UI to start capture |
| `--no-transcription` | Skip Whisper model loading; keep microphone/JawLab capture available |
| `--all-devices` | With `--list-devices`, show both inputs and outputs |
| `--host 0.0.0.0` | Expose on the LAN (no auth — trusted networks only) |
| `--port 8765` | Change the port |

## Test without a microphone

Stream a 16-bit PCM wav as if it were live audio:

```bash
say -o /tmp/test.aiff "Hello, this is a real time transcription test."
ffmpeg -y -i /tmp/test.aiff -ac 1 -ar 48000 -c:a pcm_s16le /tmp/test.wav
./run.sh --input-file /tmp/test.wav
```

The file is replayed in real time and appears in the UI/transcript.

## HTTP API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/state` | Status, source, full transcript |
| `GET /api/devices` | Legacy input list plus all input/output endpoints |
| `GET /api/experiment/capabilities` | Current audio and motion route status |
| `GET /api/jawlab/state` | Trial status and audio-only JawLab report |
| `POST /api/jawlab/trial/start` | Record one labeled trial from the active microphone stream |
| `POST /api/jawlab/trial/stop` | Stop the current trial |
| `GET /api/transcript.txt` | Plain-text transcript |
| `POST /api/listen/start` | `{"device": 2}` or `{"device": "Mic name"}` |
| `POST /api/listen/stop` | Stop capture |
| `POST /api/clear` | Clear transcript |
| `WS /ws` | Live `snapshot`, `state`, `partial`, `final`, `level` events |

## How it works

- Opens the selected input at its native rate (44.1/48 kHz), keeps 30 ms blocks.
- Energy-based VAD with automatic noise-floor calibration; 0.35 s pre-roll,
  0.7 s silence closes an utterance, 18 s max utterance.
- While you speak, a partial hypothesis is produced about once per second.
- On silence, the utterance is resampled to 16 kHz and transcribed with
  beam size 5 and Whisper's own VAD; the partial row is replaced by the final.
- One worker thread owns the model, so the audio callback never blocks.

## Troubleshooting

**`Cannot open input ...` / silence** — microphone permission, or the device is
busy in another app. Check `./run.sh --list-devices`.

**Bluetooth device opens but captures nothing** — toggle the input to the
built-in mic once, then back. Some headsets need a second to enter HFP mode.

**Repeated hallucinated lines in silence** — raise the threshold, e.g.
`--threshold -38`, or move closer to the mic.

**Slow / laggy** — use `--model tiny.en`; close other CPU-heavy apps. The
`base.en` model on Apple Silicon is usually well under real time.

**Port already in use** — `./run.sh --port 8766`.
