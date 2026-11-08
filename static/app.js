/* Live transcript UI */
(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);

  const els = {
    statusDot: $("#statusDot"),
    statusText: $("#statusText"),
    modelText: $("#modelText"),
    deviceText: $("#deviceText"),
    connBadge: $("#connBadge"),
    deviceSelect: $("#deviceSelect"),
    toggleBtn: $("#toggleBtn"),
    clearBtn: $("#clearBtn"),
    downloadBtn: $("#downloadBtn"),
    meterFill: $("#meterFill"),
    errorBox: $("#errorBox"),
    transcript: $("#transcript"),
    emptyState: $("#emptyState"),
    partialRow: $("#partialRow"),
    partialTime: $("#partialTime"),
    partialText: $("#partialText"),
    countText: $("#countText"),
  };

  const state = {
    status: "loading",
    model: "",
    device: "",
    error: null,
    segments: [],
    partial: null,
    busy: false,
    models: [],
  };

  let ws = null;
  let reconnectTimer = null;
  let stickToBottom = true;

  const STATUS_LABEL = {
    loading: "Loading model…",
    idle: "Idle",
    listening: "Listening",
    error: "Error",
  };

  // ---------------------------------------------------------------- utils

  function formatTime(ts) {
    if (!ts) return "";
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  }

  function escapeText(text) {
    return String(text == null ? "" : text);
  }

  function nearBottom(el) {
    return el.scrollHeight - el.scrollTop - el.clientHeight < 70;
  }

  function setError(message) {
    state.error = message || null;
    if (message) {
      els.errorBox.textContent = message;
      els.errorBox.classList.remove("hidden");
    } else {
      els.errorBox.textContent = "";
      els.errorBox.classList.add("hidden");
    }
  }

  function setBusy(busy, label) {
    state.busy = busy;
    els.toggleBtn.disabled = busy || state.status === "loading";
    if (busy) els.toggleBtn.textContent = label || "Working…";
    else renderStatus();
  }

  function renderStatus() {
    const status = state.status;
    els.statusDot.className = "dot " + status;
    els.statusText.textContent = STATUS_LABEL[status] || status;
    els.modelText.textContent = state.model || "model";
    els.deviceText.textContent = state.device || "no source";

    const listening = status === "listening";
    els.toggleBtn.classList.toggle("stop", listening);
    els.toggleBtn.textContent = listening ? "Stop listening" : "Start listening";
    els.toggleBtn.disabled = state.busy || status === "loading";
    els.clearBtn.disabled = state.segments.length === 0;
    els.downloadBtn.disabled = state.segments.length === 0;
  }

  // ----------------------------------------------------------- transcript

  function segmentEl(segment) {
    const row = document.createElement("div");
    row.className = "segment";
    row.dataset.id = segment.id;

    const time = document.createElement("span");
    time.className = "time";
    time.textContent = formatTime(segment.start);

    const text = document.createElement("span");
    text.className = "text";
    text.textContent = escapeText(segment.text);

    row.append(time, text);
    return row;
  }

  function renderAll() {
    els.transcript.querySelectorAll(".segment").forEach((node) => node.remove());

    if (state.segments.length === 0) {
      els.emptyState.classList.remove("hidden");
    } else {
      els.emptyState.classList.add("hidden");
      const frag = document.createDocumentFragment();
      for (const segment of state.segments) frag.append(segmentEl(segment));
      els.transcript.append(frag);
    }

    els.countText.textContent =
      state.segments.length === 1 ? "1 segment" : state.segments.length + " segments";
    renderStatus();
    scrollDown(true);
  }

  function appendSegment(segment) {
    if (state.segments.some((item) => item.id === segment.id)) return;
    state.segments.push(segment);
    els.emptyState.classList.add("hidden");
    const atBottom = stickToBottom;
    els.transcript.append(segmentEl(segment));
    els.countText.textContent =
      state.segments.length === 1 ? "1 segment" : state.segments.length + " segments";
    clearPartial();
    renderStatus();
    if (atBottom) scrollDown();
  }

  function clearPartial() {
    state.partial = null;
    els.partialRow.classList.add("hidden");
    els.partialText.textContent = "";
  }

  function showPartial(message) {
    state.partial = message;
    els.partialTime.textContent = formatTime(message.start);
    els.partialText.textContent = message.text;
    els.partialRow.classList.remove("hidden");
    if (stickToBottom) scrollDown();
  }

  function scrollDown(force) {
    if (force) els.transcript.scrollTop = els.transcript.scrollHeight;
  }

  // ---------------------------------------------------------------- state

  function applySnapshot(message) {
    state.status = message.status || "idle";
    state.model = message.model || "";
    state.device = message.device || "";
    state.segments = Array.isArray(message.segments) ? message.segments : [];
    setError(message.error || message.transcription_error);
    clearPartial();
    renderAll();
  }

  function applyState(message) {
    state.status = message.status || state.status;
    if (message.model) state.model = message.model;
    state.device = message.device || "";
    setError(message.error || message.transcription_error);
    renderStatus();
  }

  // ------------------------------------------------------------------ net

  function connect() {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(scheme + "://" + location.host + "/ws");

    ws.onopen = () => {
      els.connBadge.textContent = "online";
      els.connBadge.classList.add("online");
      if (reconnectTimer) clearTimeout(reconnectTimer);
    };

    ws.onclose = () => {
      els.connBadge.textContent = "offline";
      els.connBadge.classList.remove("online");
      reconnectTimer = setTimeout(connect, 1500);
    };

    ws.onerror = () => {
      try { ws.close(); } catch (_) {}
    };

    ws.onmessage = (event) => {
      let message;
      try { message = JSON.parse(event.data); } catch (_) { return; }
      switch (message.type) {
        case "snapshot":
          applySnapshot(message);
          break;
        case "state":
          applyState(message);
          break;
        case "partial":
          if (["listening", "idle"].includes(state.status)) showPartial(message);
          break;
        case "final":
          appendSegment(message.segment);
          break;
        case "cleared":
          state.segments = [];
          clearPartial();
          renderAll();
          break;
        case "level":
          els.meterFill.style.width = Math.round((message.value || 0) * 100) + "%";
          break;
        default:
          break;
      }
    };
  }

  async function loadDevices() {
    try {
      const res = await fetch("/api/devices");
      const data = await res.json();
      state.models = data.devices || [];
      els.deviceSelect.innerHTML = "";
      const def = document.createElement("option");
      def.value = "";
      def.textContent = "System default";
      els.deviceSelect.append(def);
      for (const device of state.models) {
        const option = document.createElement("option");
        option.value = String(device.index);
        option.textContent =
          device.name + (device.default ? "  (default)" : "");
        els.deviceSelect.append(option);
      }
      if (state.device) {
        const match = state.models.find((d) => d.name === state.device);
        if (match) els.deviceSelect.value = String(match.index);
      }
    } catch (err) {
      setError("Could not list audio devices: " + err.message);
    }
  }

  async function api(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const data = await res.json();
        detail = data.detail || detail;
      } catch (_) {}
      throw new Error(detail);
    }
    return res.json();
  }

  // --------------------------------------------------------------- actions

  async function toggleListening() {
    if (state.busy) return;
    const listening = state.status === "listening";
    setBusy(true, listening ? "Stopping…" : "Starting…");
    setError(null);

    try {
      if (listening) {
        await api("/api/listen/stop");
      } else {
        const selected = els.deviceSelect.value;
        const device = selected === "" ? null : Number(selected);
        await api("/api/listen/start", { device });
      }
    } catch (err) {
      setError(err.message);
      renderStatus();
    } finally {
      setBusy(false);
    }
  }

  async function changeDevice() {
    if (state.status !== "listening") return;
    const selected = els.deviceSelect.value;
    const device = selected === "" ? null : Number(selected);
    setBusy(true, "Switching…");
    setError(null);
    try {
      await api("/api/listen/start", { device });
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function clearTranscript() {
    try {
      await api("/api/clear");
      state.segments = [];
      clearPartial();
      renderAll();
    } catch (err) {
      setError(err.message);
    }
  }

  async function downloadTranscript() {
    try {
      const res = await fetch("/api/transcript.txt");
      const text = await res.text();
      const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
      link.href = url;
      link.download = "transcript-" + stamp + ".txt";
      document.body.append(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError("Download failed: " + err.message);
    }
  }

  // ----------------------------------------------------------------- init

  els.toggleBtn.addEventListener("click", toggleListening);
  els.clearBtn.addEventListener("click", clearTranscript);
  els.downloadBtn.addEventListener("click", downloadTranscript);
  els.deviceSelect.addEventListener("change", changeDevice);
  els.transcript.addEventListener("scroll", () => {
    stickToBottom = nearBottom(els.transcript);
  });

  renderStatus();
  loadDevices();
  connect();
})();
