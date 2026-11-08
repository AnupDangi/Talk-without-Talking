(function () {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const els = {
    connection: $("#connectionBadge"),
    notice: $("#globalNotice"),
    refresh: $("#refreshButton"),
    input: $("#inputSelect"),
    mic: $("#micButton"),
    micHelp: $("#micHelp"),
    endpointCount: $("#endpointCount"),
    endpoints: $("#endpointList"),
    level: $("#levelValue"),
    levelFill: $("#levelFill"),
    levelTrack: $(".level-track"),
    captureState: $("#captureState"),
    wpm: $("#wpmValue"),
    wordCount: $("#wordCount"),
    transcriptionHelp: $("#transcriptionHelp"),
    whisperStatus: $("#whisperStatus"),
    whisperStatusText: $("#whisperStatusText"),
    sampleRate: $("#sampleRate"),
    recognized: $("#recognizedWords"),
    motionState: $("#motionState"),
    motionMessage: $("#motionMessage"),
    condition: $("#conditionSelect"),
    wordField: $("#wordField"),
    word: $("#wordSelect"),
    duration: $("#durationSelect"),
    trial: $("#trialButton"),
    jawlabState: $("#jawlabState"),
    trialCount: $("#trialCount"),
    trialRows: $("#trialRows"),
    accuracy: $("#accuracyValue"),
    chance: $("#chanceValue"),
    accuracyDetail: $("#accuracyDetail"),
    signalActivity: $("#signalActivity"),
    signalDevice: $("#signalDevice"),
    signalRate: $("#signalRate"),
    signalLevels: $("#signalLevels"),
    signalVoicing: $("#signalVoicing"),
    signalModulation: $("#signalModulation"),
    signalCycle: $("#signalCycle"),
    signalCue: $("#signalCue"),
    levelHistory: $("#levelHistory"),
    bandHistory: $("#bandHistory"),
    featureLogStart: $("#featureLogStart"),
    featureLogStop: $("#featureLogStop"),
    featureLogExport: $("#featureLogExport"),
    featureLogStatus: $("#featureLogStatus"),
  };

  const SIGNAL_HISTORY_FRAMES = 300; // 30 s at the server's 10 Hz monitor cadence.
  const FEATURE_LOG_MAX_FRAMES = 18000; // 30 minutes at 10 Hz.
  const BAND_NAMES = ["20-80", "80-160", "160-320", "320-640", "640-1250", "1250-2500", "2500-4000", "4000-8000"];

  const state = {
    endpoints: [],
    legacyDevices: [],
    capture: "idle",
    jawlabActive: false,
    activeTrialId: null,
    activeTrialTimer: null,
    busy: false,
    pollBusy: false,
    signalHistory: [],
    lastSignalSequence: null,
    featureLogging: false,
    featureLog: [],
  };

  function setConnection(online) {
    els.connection.classList.toggle("online", online);
    els.connection.classList.toggle("offline", !online);
    els.connection.querySelector("span:last-child").textContent = online ? "Server online" : "Server offline";
  }

  function setNotice(message, kind) {
    if (!message) {
      els.notice.textContent = "";
      els.notice.className = "notice hidden";
      return;
    }
    els.notice.textContent = message;
    els.notice.className = "notice " + (kind || "error");
  }

  function setMicHelp(message) {
    els.micHelp.textContent = message || "";
    els.micHelp.classList.toggle("hidden", !message);
  }

  async function requestJson(path, options) {
    const response = await fetch(path, {
      headers: options && options.body ? { "Content-Type": "application/json" } : undefined,
      ...options,
    });
    let payload = null;
    try { payload = await response.json(); } catch (_) { /* Non-JSON server errors use status text. */ }
    if (!response.ok) {
      const detail = payload && (payload.detail || payload.error || payload.message);
      throw new Error(typeof detail === "string" ? detail : response.statusText || "Request failed");
    }
    return payload || {};
  }

  function endpointDirection(endpoint) {
    const input = Number(endpoint.max_input_channels || 0) > 0;
    const output = Number(endpoint.max_output_channels || 0) > 0;
    if (input && output) return "Input + output";
    if (input) return "Input only";
    if (output) return "Output only";
    return "No audio channels reported";
  }

  function stableIdText(endpoint) {
    if (endpoint.stable_id_available && endpoint.stable_id) return "Stable ID · " + endpoint.stable_id;
    return "Stable ID not exposed · session index [" + endpoint.index + "]";
  }

  function endpointCard(endpoint) {
    const card = document.createElement("article");
    card.className = "endpoint-card";

    const top = document.createElement("div");
    top.className = "endpoint-top";
    const name = document.createElement("strong");
    name.textContent = endpoint.name || "Unnamed audio endpoint";
    const tag = document.createElement("span");
    tag.className = "endpoint-tag";
    tag.textContent = endpointDirection(endpoint);
    top.append(name, tag);

    const meta = document.createElement("div");
    meta.className = "endpoint-meta";
    const channels = document.createElement("span");
    channels.textContent = "In " + Number(endpoint.max_input_channels || 0) + " ch · Out " + Number(endpoint.max_output_channels || 0) + " ch";
    const rate = document.createElement("span");
    rate.textContent = endpoint.default_samplerate ? Number(endpoint.default_samplerate).toLocaleString() + " Hz default" : "Default rate not reported";
    meta.append(channels, rate);

    const id = document.createElement("p");
    id.className = "endpoint-id";
    id.textContent = stableIdText(endpoint);
    const host = document.createElement("span");
    host.className = "endpoint-host";
    host.textContent = (endpoint.host_api || "Host API not reported") +
      (endpoint.format_metadata_available ? " · format metadata available" : " · format metadata unavailable");

    const defaults = [];
    if (endpoint.default_input) defaults.push("Default input");
    if (endpoint.default_output) defaults.push("Default output");
    if (defaults.length) {
      const badge = document.createElement("span");
      badge.className = "default-badge";
      badge.textContent = defaults.join(" · ");
      top.append(badge);
    }
    card.append(top, meta, id, host);
    return card;
  }

  function renderEndpoints() {
    const inputs = state.endpoints.filter((endpoint) => Number(endpoint.max_input_channels || 0) > 0);
    const previous = els.input.value;
    els.input.replaceChildren();
    const defaultOption = document.createElement("option");
    defaultOption.value = "";
    defaultOption.textContent = "System default input";
    els.input.append(defaultOption);
    for (const endpoint of inputs) {
      const option = document.createElement("option");
      option.value = String(endpoint.index);
      option.textContent = endpoint.name + (endpoint.default_input ? " · default" : "");
      els.input.append(option);
    }
    if (inputs.some((endpoint) => String(endpoint.index) === previous)) els.input.value = previous;
    else if (!inputs.some((endpoint) => endpoint.default_input)) els.input.value = "";

    els.endpointCount.textContent = state.endpoints.length + (state.endpoints.length === 1 ? " endpoint" : " endpoints");
    els.endpoints.replaceChildren();
    if (!state.endpoints.length) {
      const empty = document.createElement("p");
      empty.className = "empty-note";
      empty.textContent = "The audio host has not exposed any endpoints to the server. Connect or enable a device in system audio settings, then refresh.";
      els.endpoints.append(empty);
    } else {
      for (const endpoint of state.endpoints) els.endpoints.append(endpointCard(endpoint));
    }

    if (!inputs.length) {
      defaultOption.textContent = "No microphone inputs found";
      els.mic.disabled = true;
      setMicHelp("No microphone input is currently listed. Connect or enable an input in system audio settings, then refresh the device list.");
    } else {
      els.mic.disabled = state.busy;
      setMicHelp("");
    }
    updateControls();
  }

  function normalizeWords(metrics, segments) {
    const words = metrics && metrics.recognized_words;
    if (Array.isArray(words)) {
      return words.map((item) => {
        if (typeof item === "string") return item;
        if (item && typeof item === "object") return item.word || item.text || "";
        return "";
      }).filter(Boolean);
    }
    if (typeof words === "string") return words.trim().split(/\s+/).filter(Boolean);
    const transcript = (segments || []).map((segment) => segment && segment.text || "").join(" ").trim();
    return transcript ? transcript.split(/\s+/).filter(Boolean) : [];
  }

  function renderWords(words, transcriptionStatus) {
    els.recognized.replaceChildren();
    if (!words.length) {
      const empty = document.createElement("span");
      empty.className = "transcript-empty";
      if (transcriptionStatus === "disabled") {
        empty.textContent = "Whisper is disabled for this server run. Restart with ./run.sh --no-autostart to enable speech transcription; JawLab can still measure audio.";
      } else if (transcriptionStatus === "error") {
        empty.textContent = "Whisper could not load. Check the server log, then restart the server.";
      } else if (transcriptionStatus === "loading") {
        empty.textContent = "Waiting for the Whisper model to finish loading…";
      } else {
        empty.textContent = state.capture === "listening"
          ? "Listening. Speak normally or whisper audibly; recognition depends on the microphone and audio conditions."
          : "Whisper is ready. Start the microphone, then speak normally or whisper audibly.";
      }
      els.recognized.append(empty);
      return;
    }
    const text = document.createElement("p");
    text.textContent = words.join(" ");
    els.recognized.append(text);
  }

  function renderCapture(snapshot, capability) {
    const metrics = snapshot && snapshot.metrics || {};
    const cap = capability && capability.capture || {};
    state.capture = cap.status || snapshot && snapshot.status || "idle";
    const listening = state.capture === "listening";
    const error = cap.error || snapshot && snapshot.error;
    const transcriptionError = cap.transcription_error || snapshot && snapshot.transcription_error;
    const transcriptionStatus = cap.transcription_status || snapshot && snapshot.transcription_status || "loading";
    renderWhisperStatus(transcriptionStatus, transcriptionError, listening);
    if (!listening && state.featureLogging) {
      state.featureLogging = false;
      els.featureLogStatus.textContent = "Microphone stopped; feature logging paused. Export the in-memory log before closing this page.";
    }
    if (!listening && !state.signalHistory.length) {
      els.signalActivity.textContent = "WAITING FOR MIC";
      els.signalDevice.textContent = "No active input";
      els.signalRate.textContent = "No sample rate";
    } else if (!listening && state.signalHistory.length) {
      els.signalActivity.textContent = "MICROPHONE STOPPED";
    }
    els.captureState.textContent = state.capture === "error" ? "ERROR" : listening ? "CAPTURING" : "IDLE";
    els.captureState.classList.toggle("state-error", state.capture === "error");
    els.captureState.classList.toggle("state-live", listening);
    els.mic.textContent = listening ? "Stop microphone" : "Start microphone";
    els.mic.classList.toggle("button-stop", listening);
    els.mic.disabled = state.busy || (!listening && !state.endpoints.some((endpoint) => Number(endpoint.max_input_channels || 0) > 0));
    const selectedName = cap.device || snapshot && snapshot.device;
    if (selectedName) {
      const selected = state.endpoints.find((endpoint) => endpoint.name === selectedName);
      if (selected && selected.index != null) els.input.value = String(selected.index);
    }
    const rate = cap.samplerate || metrics.samplerate;
    els.sampleRate.textContent = listening && rate ? Number(rate).toLocaleString() + " Hz" : listening ? "Rate not reported" : "No stream";

    const words = normalizeWords(metrics, snapshot && snapshot.segments);
    const wpm = metrics.recognized_wpm;
    els.wpm.textContent = Number.isFinite(Number(wpm)) && wpm !== null ? Math.round(Number(wpm)) + " WPM" : "—";
    const count = metrics.recognized_word_count != null ? metrics.recognized_word_count : words.length;
    els.wordCount.textContent = String(count);
    els.transcriptionHelp.textContent = transcriptionError
      ? transcriptionError + " JawLab audio trials remain available; transcript pace will stay blank."
      : "Pace is recognized words per VAD speech minute; it is not a measure of transcript accuracy.";
    renderWords(words, transcriptionStatus);

    if (snapshot && (snapshot.level != null || snapshot.level_dbfs != null)) {
      updateLevel(snapshot.level, snapshot.level_dbfs);
    }
    if (snapshot && snapshot.signal) addSignalFrame(snapshot.signal);
    if (error) setMicHelp(formatCaptureError(error));
    else if (listening && transcriptionError) setMicHelp("Capturing from " + (selectedName || "the selected system input") + ". Whisper is unavailable; JawLab audio trials still work.");
    else if (listening) setMicHelp("Capturing from " + (selectedName || "the selected system input") + ". Speak normally for transcription, then run labeled JawLab trials.");
    else if (state.endpoints.some((endpoint) => Number(endpoint.max_input_channels || 0) > 0)) setMicHelp("");
    updateControls();
  }

  function renderWhisperStatus(status, message, listening) {
    const labels = {
      loading: "WHISPER LOADING",
      ready: listening ? "WHISPER READY · LISTENING" : "WHISPER READY",
      disabled: "WHISPER DISABLED",
      error: "WHISPER ERROR",
    };
    const label = labels[status] || "WHISPER STATUS UNKNOWN";
    els.whisperStatus.textContent = label;
    els.whisperStatus.classList.toggle("state-live", status === "ready");
    els.whisperStatus.classList.toggle("state-error", status === "error");
    els.whisperStatus.classList.toggle("state-disabled", status === "disabled");
    els.whisperStatus.classList.toggle("state-neutral", status === "loading" || !labels[status]);

    if (status === "disabled") {
      els.whisperStatusText.textContent = (message || "Whisper is disabled.") + " Start the server with ./run.sh --no-autostart to enable transcription.";
    } else if (status === "loading") {
      els.whisperStatusText.textContent = "Loading the local Whisper model. Microphone capture remains off until you start it.";
    } else if (status === "error") {
      els.whisperStatusText.textContent = message || "Whisper failed to load. Check the server log.";
    } else if (status === "ready") {
      els.whisperStatusText.textContent = listening
        ? "Transcription is running on the selected mic stream. Audibly whispered speech may be recognized; silent mouthing is not supported by this ASR path."
        : "Model loaded; microphone is idle. Start it to transcribe audible speech or whisper. The full transcript shares this stream.";
    } else {
      els.whisperStatusText.textContent = "Waiting for a supported Whisper status from the server.";
    }
  }

  function updateLevel(rawLevel, dbfs) {
    const level = Number(rawLevel);
    const levelValue = Number.isFinite(level) ? Math.max(0, Math.min(1, level)) : 0;
    const decibels = Number(dbfs);
    els.level.textContent = Number.isFinite(decibels)
      ? decibels.toFixed(1) + " dBFS"
      : levelValue > 0 ? Math.round(levelValue * 100) + "%" : "— dBFS";
    els.levelFill.style.width = Math.round(levelValue * 100) + "%";
    els.levelTrack.setAttribute("aria-valuenow", String(Math.round(levelValue * 100)));
  }

  function activityLabel(activity) {
    if (activity === "calibrating") return "CALIBRATING NOISE FLOOR";
    if (activity === "near_silence") return "NEAR-SILENCE / BELOW FLOOR";
    if (activity === "voiced_like_activity") return "VOICED-LIKE ACOUSTIC ACTIVITY";
    if (activity === "acoustic_activity_uncertain") return "ACOUSTIC ACTIVITY · VOICING UNCERTAIN";
    return "WAITING FOR SIGNAL";
  }

  function addSignalFrame(frame) {
    if (!frame || !Number.isFinite(Number(frame.seq))) return;
    const sequence = Number(frame.seq);
    if (sequence === state.lastSignalSequence) return;
    state.lastSignalSequence = sequence;
    const previous = state.signalHistory[state.signalHistory.length - 1];
    if (previous && (previous.device !== frame.device || previous.sample_rate !== frame.sample_rate)) {
      state.signalHistory = [];
    }
    state.signalHistory.push(frame);
    if (state.signalHistory.length > SIGNAL_HISTORY_FRAMES) {
      state.signalHistory.splice(0, state.signalHistory.length - SIGNAL_HISTORY_FRAMES);
    }

    els.signalActivity.textContent = activityLabel(frame.activity);
    els.signalDevice.textContent = frame.device || "Input name unavailable";
    els.signalRate.textContent = frame.sample_rate ? Number(frame.sample_rate).toLocaleString() + " Hz" : "Rate unavailable";
    els.signalLevels.textContent = Number(frame.rms_dbfs).toFixed(1) + " / " + Number(frame.peak_dbfs).toFixed(1) + " dBFS";
    els.signalVoicing.textContent = Number(frame.voicing_score).toFixed(2) + " · periodicity only";
    els.signalModulation.textContent = Number(frame.low_modulation_ratio).toFixed(2);
    els.signalCycle.textContent = Number(frame.low_cycle_strength).toFixed(2) + " · " + Number(frame.low_cycle_hz).toFixed(2) + " Hz";
    if (frame.periodic_low_band_pattern) {
      els.signalCue.textContent = "Periodic low-band pattern present. This is not specific to jaw or mouth movement.";
    } else if (Number(frame.modulation_frames) < 25) {
      els.signalCue.textContent = "Collecting low-band context. A periodic pattern, if present, is not specific to jaw or mouth movement.";
    } else {
      els.signalCue.textContent = "No strong periodic low-band pattern in this window. Microphone audio alone cannot confirm mouth movement.";
    }

    if (state.featureLogging) {
      state.featureLog.push({
        timestamp: frame.timestamp,
        device: frame.device || "",
        sample_rate: frame.sample_rate,
        activity: frame.activity,
        rms_dbfs: frame.rms_dbfs,
        peak_dbfs: frame.peak_dbfs,
        voicing_score: frame.voicing_score,
        low_band_dbfs: frame.low_band_dbfs,
        low_modulation_ratio: frame.low_modulation_ratio,
        low_cycle_strength: frame.low_cycle_strength,
        low_cycle_hz: frame.low_cycle_hz,
        periodic_low_band_pattern: frame.periodic_low_band_pattern,
        bands_dbfs: Array.isArray(frame.bands_dbfs) ? frame.bands_dbfs.slice() : [],
      });
      if (state.featureLog.length >= FEATURE_LOG_MAX_FRAMES) {
        state.featureLogging = false;
        els.featureLogStatus.textContent = "Feature log reached its 30-minute safety limit; export the in-memory log.";
      } else {
        els.featureLogStatus.textContent = "Logging derived features in page memory · " + state.featureLog.length + " frames · no raw audio.";
      }
    }
    drawSignalPlots();
    updateSignalLogControls();
  }

  function fitCanvas(canvas, height) {
    if (!canvas) return null;
    const width = Math.max(1, Math.floor(canvas.clientWidth));
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    return { ctx, width, height };
  }

  function drawSignalPlots() {
    const level = fitCanvas(els.levelHistory, 150);
    const bands = fitCanvas(els.bandHistory, 150);
    if (!level || !bands) return;
    const frames = state.signalHistory;
    const columns = SIGNAL_HISTORY_FRAMES;
    const gridLevels = [-90, -70, -50, -30, -10];
    level.ctx.lineWidth = 1;
    level.ctx.strokeStyle = "rgba(164,175,165,.16)";
    for (const db of gridLevels) {
      const y = level.height - ((db + 90) / 80) * level.height;
      level.ctx.beginPath();
      level.ctx.moveTo(0, y);
      level.ctx.lineTo(level.width, y);
      level.ctx.stroke();
    }
    if (frames.length > 1) {
      level.ctx.beginPath();
      frames.forEach((frame, index) => {
        const x = level.width - ((frames.length - 1 - index) / columns) * level.width;
        const db = Math.max(-90, Math.min(-10, Number(frame.rms_dbfs)));
        const y = level.height - ((db + 90) / 80) * level.height;
        if (index === 0) level.ctx.moveTo(x, y);
        else level.ctx.lineTo(x, y);
      });
      level.ctx.strokeStyle = "#b9e7bd";
      level.ctx.lineWidth = 2;
      level.ctx.stroke();
    }

    const cellWidth = bands.width / columns;
    const cellHeight = bands.height / BAND_NAMES.length;
    frames.forEach((frame, index) => {
      const values = Array.isArray(frame.bands_dbfs) ? frame.bands_dbfs : [];
      const x = bands.width - (frames.length - index) * cellWidth;
      values.forEach((value, bandIndex) => {
        const normalized = Math.max(0, Math.min(1, (Number(value) + 90) / 75));
        const lightness = 9 + normalized * 48;
        bands.ctx.fillStyle = "hsl(" + (132 - normalized * 15) + " 42% " + lightness + "%)";
        const y = bands.height - (bandIndex + 1) * cellHeight;
        bands.ctx.fillRect(x, y, Math.ceil(cellWidth + 0.5), Math.ceil(cellHeight + 0.5));
      });
    });
  }

  function updateSignalLogControls() {
    const listening = state.capture === "listening";
    els.featureLogStart.disabled = !listening || state.featureLogging || state.featureLog.length >= FEATURE_LOG_MAX_FRAMES;
    els.featureLogStop.disabled = !state.featureLogging;
    els.featureLogExport.disabled = !state.featureLog.length;
  }

  function csvCell(value) {
    const text = String(value == null ? "" : value);
    return '"' + text.replace(/"/g, '""') + '"';
  }

  function exportSignalLog() {
    if (!state.featureLog.length) return;
    const headers = [
      "timestamp", "device", "sample_rate", "activity", "rms_dbfs", "peak_dbfs",
      "voicing_score", "low_band_dbfs", "low_modulation_ratio", "low_cycle_strength",
      "low_cycle_hz", "periodic_low_band_pattern", ...BAND_NAMES.map((name) => "band_" + name + "_dbfs"),
    ];
    const lines = [headers.map(csvCell).join(",")];
    for (const row of state.featureLog) {
      lines.push([
        row.timestamp, row.device, row.sample_rate, row.activity, row.rms_dbfs, row.peak_dbfs,
        row.voicing_score, row.low_band_dbfs, row.low_modulation_ratio, row.low_cycle_strength,
        row.low_cycle_hz, row.periodic_low_band_pattern,
        ...BAND_NAMES.map((_, index) => row.bands_dbfs[index]),
      ].map(csvCell).join(","));
    }
    const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "signal-features-" + new Date().toISOString().replace(/[:.]/g, "-") + ".csv";
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function formatCaptureError(error) {
    const message = String(error);
    const lower = message.toLowerCase();
    if (/permission|not authorized|denied/.test(lower)) {
      return message + " Allow microphone access for the app running this server, then restart the server if needed.";
    }
    if (/no input|device|open/.test(lower)) {
      return message + " Refresh endpoints, reconnect the device, and verify the input in system audio settings.";
    }
    return message;
  }

  function renderMotion(capability) {
    const motion = capability && capability.motion || {};
    const message = motion.message || "This browser cannot access native headphone motion sensors. Use Subvox to check whether a supported stream is available.";
    els.motionMessage.querySelector("strong").textContent = motion.status === "streaming" ? "Native motion samples reported" : "Check in native Subvox";
    els.motionMessage.querySelector("span:not(.status-symbol)").textContent = message;
    els.motionState.textContent = motion.browser_access === false ? "NATIVE APP ONLY" : String(motion.status || "NATIVE APP").toUpperCase();
  }

  function metricValue(metrics, key, digits) {
    const value = metrics && metrics[key];
    return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
  }

  function renderTrials(trials) {
    const rows = Array.isArray(trials) ? trials.slice().reverse() : [];
    els.trialCount.textContent = rows.length + (rows.length === 1 ? " trial" : " trials");
    els.trialRows.replaceChildren();
    if (!rows.length) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 6;
      cell.className = "table-empty";
      cell.textContent = state.capture === "listening" ? "No trials yet. Choose a condition and record a take." : "No trials yet. Start the mic to begin.";
      row.append(cell);
      els.trialRows.append(row);
      return;
    }
    for (const trial of rows) {
      const row = document.createElement("tr");
      const values = [
        trial.condition || "—",
        trial.word || "—",
        Number(trial.duration_sec || 0).toFixed(1) + " s",
        metricValue(trial.metrics, "low_rel_db", 1) + " dB",
        metricValue(trial.metrics, "mod_ratio", 3),
        metricValue(trial.metrics, "cycle_strength", 2),
      ];
      for (const value of values) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      els.trialRows.append(row);
    }
  }

  function renderJawlab(jawlab) {
    const data = jawlab || {};
    state.jawlabActive = Boolean(data.active);
    els.jawlabState.textContent = data.active ? "TRIAL IN PROGRESS" : state.capture === "listening" ? "MIC READY" : "READY WHEN MIC IS ON";
    els.jawlabState.classList.toggle("state-live", data.active);
    const trials = data.trials || [];
    renderTrials(trials);

    const report = data.report || data.latest_report || null;
    const accuracy = report && report.word_accuracy;
    const chance = report && report.word_chance;
    const silentCount = trials.filter((trial) => trial.condition === "silent").length;
    if (Number.isFinite(Number(accuracy)) && accuracy !== null) {
      els.accuracy.textContent = Math.round(Number(accuracy) * 100) + "% LOO accuracy";
      els.chance.textContent = "Chance baseline " + (Number.isFinite(Number(chance)) ? Math.round(Number(chance) * 100) + "%" : "not reported");
      const verdict = report.verdict && report.verdict.summary;
      const labeledWords = new Set(trials.filter((trial) => trial.condition === "silent").map((trial) => trial.word).filter(Boolean)).size;
      const basis = " Based on " + silentCount + " silent trials across " + labeledWords + " user-labeled words.";
      els.accuracyDetail.textContent = verdict
        ? verdict + basis + " This is not open-vocabulary silent-speech recognition."
        : "Leave-one-out classification on labeled mic-audio features; compare with chance." + basis + " This is not open-vocabulary silent-speech recognition.";
    } else {
      els.accuracy.textContent = "Not enough data";
      els.chance.textContent = "Chance baseline —";
      const enough = silentCount >= 6 && new Set(trials.filter((trial) => trial.condition === "silent").map((trial) => trial.word)).size >= 2;
      els.accuracyDetail.textContent = enough
        ? "Enough labeled trials are present; the session report will appear when the backend analyzer has a result."
        : "Collect at least six silent trials across two or more labels. JawLab then reports leave-one-out accuracy and chance for this closed word set.";
    }
    updateControls();
  }

  function updateControls() {
    const hasInput = state.endpoints.some((endpoint) => Number(endpoint.max_input_channels || 0) > 0);
    const listening = state.capture === "listening";
    els.mic.disabled = state.busy || state.jawlabActive || (!listening && !hasInput);
    els.input.disabled = state.busy || listening;
    els.refresh.disabled = state.busy || listening;
    els.trial.disabled = state.busy || !listening || (state.jawlabActive && !state.activeTrialId);
    els.trial.textContent = state.jawlabActive ? "Stop trial" : "Record trial";
    const needsWord = ["silent", "whisper", "audible"].includes(els.condition.value);
    els.wordField.classList.toggle("hidden", !needsWord);
    updateSignalLogControls();
  }

  async function refreshDevices() {
    els.refresh.disabled = true;
    els.endpointCount.textContent = "Refreshing…";
    setNotice(null);
    try {
      const data = await requestJson("/api/devices");
      state.endpoints = Array.isArray(data.endpoints) ? data.endpoints : [];
      const legacyDevices = Array.isArray(data.devices) ? data.devices : [];
      renderEndpoints();
      setConnection(true);
      if (!state.endpoints.length && !legacyDevices.length) {
        setNotice("The audio host did not report any endpoints to the server. Connect or enable an input, then refresh. This does not establish whether a specific device is incompatible.", "warning");
      }
    } catch (error) {
      setConnection(false);
      els.endpointCount.textContent = "Unavailable";
      els.endpoints.innerHTML = "";
      const note = document.createElement("p");
      note.className = "empty-note";
      note.textContent = "Endpoint discovery failed: " + error.message;
      els.endpoints.append(note);
      setNotice("Could not reach the local experiment API. Start the project server and reload this page.", "error");
    } finally {
      els.refresh.disabled = false;
      updateControls();
    }
  }

  async function refreshState() {
    if (state.pollBusy) return;
    state.pollBusy = true;
    try {
      const results = await Promise.all([
        requestJson("/api/experiment/capabilities"),
        requestJson("/api/state"),
        requestJson("/api/jawlab/state"),
      ]);
      const [capability, snapshot, jawlab] = results;
      setConnection(true);
      renderMotion(capability);
      renderCapture(snapshot, capability);
      renderJawlab(jawlab);
      if (capability.capture && capability.capture.error) setMicHelp(formatCaptureError(capability.capture.error));
    } catch (_) {
      setConnection(false);
    } finally {
      state.pollBusy = false;
    }
  }

  function connectStatusSocket() {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(scheme + "://" + location.host + "/ws");
    socket.onopen = () => setConnection(true);
    socket.onmessage = (event) => {
      let message;
      try { message = JSON.parse(event.data); } catch (_) { return; }
      if (message.type === "level") updateLevel(message.value, message.dbfs);
      if (message.type === "level" && message.signal) addSignalFrame(message.signal);
      if (message.type === "final" || message.type === "cleared" || message.type === "state") refreshState();
    };
    socket.onclose = () => window.setTimeout(connectStatusSocket, 1800);
    socket.onerror = () => socket.close();
  }

  async function toggleMicrophone() {
    if (state.busy) return;
    state.busy = true;
    updateControls();
    setNotice(null);
    try {
      if (state.capture === "listening") {
        await requestJson("/api/listen/stop", { method: "POST" });
      } else {
        const value = els.input.value;
        await requestJson("/api/listen/start", {
          method: "POST",
          body: JSON.stringify({ device: value === "" ? null : Number(value) }),
        });
      }
      await refreshState();
    } catch (error) {
      setMicHelp(formatCaptureError(error.message));
      setNotice(formatCaptureError(error.message), "error");
    } finally {
      state.busy = false;
      updateControls();
    }
  }

  async function finishTrial(trialId) {
    if (!trialId) return;
    if (state.activeTrialTimer) clearTimeout(state.activeTrialTimer);
    state.activeTrialTimer = null;
    state.busy = true;
    updateControls();
    try {
      await requestJson("/api/jawlab/trial/stop", {
        method: "POST",
        body: JSON.stringify({ trial_id: trialId }),
      });
    } catch (error) {
      setNotice("Could not finish the JawLab trial: " + error.message, "error");
    } finally {
      state.activeTrialId = null;
      state.busy = false;
      await refreshState();
      updateControls();
    }
  }

  async function recordTrial() {
    if (state.jawlabActive && state.activeTrialId) {
      await finishTrial(state.activeTrialId);
      return;
    }
    if (state.busy || state.capture !== "listening") return;
    state.busy = true;
    updateControls();
    setNotice(null);
    try {
      const condition = els.condition.value;
      const needsWord = ["silent", "whisper", "audible"].includes(condition);
      const durationSec = Number(els.duration.value);
      const response = await requestJson("/api/jawlab/trial/start", {
        method: "POST",
        body: JSON.stringify({
          condition,
          word: needsWord ? els.word.value : null,
          duration_sec: durationSec,
        }),
      });
      state.activeTrialId = response.trial_id || response.id || null;
      if (!state.activeTrialId) throw new Error("The server did not return a trial ID.");
      state.activeTrialTimer = setTimeout(() => finishTrial(state.activeTrialId), durationSec * 1000 + 300);
      await refreshState();
    } catch (error) {
      setNotice("Could not start trial: " + error.message, "error");
    } finally {
      state.busy = false;
      updateControls();
    }
  }

  els.refresh.addEventListener("click", async () => {
    await refreshDevices();
    await refreshState();
  });
  els.mic.addEventListener("click", toggleMicrophone);
  els.trial.addEventListener("click", recordTrial);
  els.condition.addEventListener("change", updateControls);
  els.featureLogStart.addEventListener("click", () => {
    if (state.capture !== "listening") return;
    state.featureLogging = true;
    els.featureLogStatus.textContent = "Logging derived features in page memory · " + state.featureLog.length + " frames · no raw audio.";
    updateSignalLogControls();
  });
  els.featureLogStop.addEventListener("click", () => {
    state.featureLogging = false;
    els.featureLogStatus.textContent = "Feature log paused · " + state.featureLog.length + " frames in page memory · export before closing.";
    updateSignalLogControls();
  });
  els.featureLogExport.addEventListener("click", exportSignalLog);
  els.input.addEventListener("change", () => {
    if (state.capture === "listening") {
      setMicHelp("Stop the current microphone stream before changing inputs, then start again with the new selection.");
    }
  });

  setConnection(false);
  updateControls();
  drawSignalPlots();
  window.addEventListener("resize", drawSignalPlots);
  Promise.all([refreshDevices(), refreshState()]);
  connectStatusSocket();
  window.setInterval(refreshState, 1400);
})();
