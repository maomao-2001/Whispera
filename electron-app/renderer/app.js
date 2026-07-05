const els = {
  minBtn: document.getElementById("minBtn"),
  maxBtn: document.getElementById("maxBtn"),
  closeBtn: document.getElementById("closeBtn"),
  pingBtn: document.getElementById("pingBtn"),
  connectBtn: document.getElementById("connectBtn"),
  disconnectBtn: document.getElementById("disconnectBtn"),
  backendState: document.getElementById("backendState"),
  orb: document.getElementById("orb"),
  orbCaption: document.getElementById("orbCaption"),
  flowListening: document.getElementById("flowListening"),
  flowSpeechEnd: document.getElementById("flowSpeechEnd"),
  flowAsr: document.getElementById("flowAsr"),
  flowLlm: document.getElementById("flowLlm"),
  userStage: document.getElementById("userStage"),
  assistantStage: document.getElementById("assistantStage"),
  conversationScroll: document.getElementById("conversationScroll"),
  httpBase: document.getElementById("httpBase"),
  wsUrl: document.getElementById("wsUrl"),
  appVersion: document.getElementById("appVersion"),
  detectResourcesBtn: document.getElementById("detectResourcesBtn"),
  resourceLlamaState: document.getElementById("resourceLlamaState"),
  resourceGgufState: document.getElementById("resourceGgufState"),
  resourceAsrState: document.getElementById("resourceAsrState"),
  resourceTtsState: document.getElementById("resourceTtsState"),
  resourceHint: document.getElementById("resourceHint"),
  eventLog: document.getElementById("eventLog"),
  logCaptureToggle: document.getElementById("logCaptureToggle"),
  logCaptureHint: document.getElementById("logCaptureHint"),
  openServiceLogsBtn: document.getElementById("openServiceLogsBtn"),
  llmModelPath: document.getElementById("llmModelPath"),
  browseModelBtn: document.getElementById("browseModelBtn"),
  llmModelInfo: document.getElementById("llmModelInfo"),
  llmModelName: document.getElementById("llmModelName"),
  llmModelSize: document.getElementById("llmModelSize"),
  applyLlmBtn: document.getElementById("applyLlmBtn"),
  llmStatus: document.getElementById("llmStatus"),
  ttsLoraSelect: document.getElementById("ttsLoraSelect"),
  refreshLoraBtn: document.getElementById("refreshLoraBtn"),
  promptAudioPath: document.getElementById("promptAudioPath"),
  browseAudioBtn: document.getElementById("browseAudioBtn"),
  promptText: document.getElementById("promptText"),
  ttsCfgValue: document.getElementById("ttsCfgValue"),
  ttsInferenceTimesteps: document.getElementById("ttsInferenceTimesteps"),
  ttsSeed: document.getElementById("ttsSeed"),
  applyTtsBtn: document.getElementById("applyTtsBtn"),
  ttsStatus: document.getElementById("ttsStatus"),
  toolPanel: document.getElementById("toolPanel"),
  toolTabs: [...document.querySelectorAll("[data-panel-target]")],
  toolViews: [...document.querySelectorAll("[data-panel-view]")],
  textComposer: document.getElementById("textComposer"),
  textPromptInput: document.getElementById("textPromptInput"),
  sendTextBtn: document.getElementById("sendTextBtn")
};

const DEFAULT_TEXT_PLACEHOLDER = "Type a message...";
const OFFLINE_TEXT_PLACEHOLDER = "请先启动服务";
const TEXT_STARTUP_PLACEHOLDER_FRAMES = ["正在启动", "正在启动.", "正在启动..", "正在启动..."];

const state = {
  ws: null,
  micStream: null,
  micContext: null,
  micSource: null,
  micProcessor: null,
  playbackContext: null,
  playbackAnalyser: null,
  playbackNextTime: 0,
  playbackSources: new Set(),
  phase: "idle",
  speechActive: false,
  serviceBusy: false,
  sessionMode: "idle",
  sessionReady: false,
  textInputEnabled: false,
  textInputLoading: false,
  textPlaceholderTimer: null,
  textPlaceholderFrame: 0,
  activeToolPanel: null,
  conversationMessages: [],
  nextMessageId: 1,
  currentUserMessageId: null,
  currentAssistantMessageId: null,
  pendingTextRequest: null,
  pendingTextRetryTimer: null,
  textInterruptPending: false,
  textInputComposing: false,
  backend: {
    httpBase: "",
    wsUrl: ""
  },
  llm: {
    modelPath: null,
    modelName: null,
    modelSize: null
  },
  resourceStatus: null,
  logCaptureEnabled: false,
  serviceLogsDir: "",
  turnCaptureDir: "",
  ttsOptions: {
    lora_selection: null,
    prompt_audio_path: null,
    prompt_text: null,
    cfg_value: 2.0,
    inference_timesteps: 10,
    seed: -1
  }
};

function setActiveToolPanel(panelName) {
  const nextPanel = state.activeToolPanel === panelName ? null : panelName;
  state.activeToolPanel = nextPanel;
  els.toolPanel.classList.toggle("tool-panel--hidden", !nextPanel);

  for (const tab of els.toolTabs) {
    const isActive = Boolean(nextPanel && tab.dataset.panelTarget === nextPanel);
    tab.classList.toggle("is-active", isActive);
    tab.setAttribute("aria-pressed", String(isActive));
  }

  for (const view of els.toolViews) {
    view.hidden = !nextPanel || view.dataset.panelView !== nextPanel;
  }
}

function isRealtimeSocketOpen() {
  return Boolean(state.ws && state.ws.readyState === WebSocket.OPEN);
}

function isVoiceSessionActive() {
  return state.sessionMode === "voice";
}

function hasActiveTurn() {
  return (
    state.speechActive ||
    state.playbackSources.size > 0 ||
    state.phase === "speech-ended" ||
    state.phase === "asr" ||
    state.phase === "llm"
  );
}

function getReadyTurnPhase() {
  return isVoiceSessionActive() ? "listening" : "idle";
}

function getReadyOrbMode() {
  return isVoiceSessionActive() ? "listening" : "live";
}

function getReadyOrbMessage() {
  return isVoiceSessionActive()
    ? "Realtime channel connected. Microphone is streaming."
    : "Realtime channel connected. Text input is ready.";
}

function clearPendingTextRetry() {
  if (state.pendingTextRetryTimer) {
    clearTimeout(state.pendingTextRetryTimer);
    state.pendingTextRetryTimer = null;
  }
}

function renderTextComposerPlaceholder() {
  if (!els.textPromptInput) {
    return;
  }
  if (state.textInputLoading) {
    const frameIndex = state.textPlaceholderFrame % TEXT_STARTUP_PLACEHOLDER_FRAMES.length;
    els.textPromptInput.placeholder = TEXT_STARTUP_PLACEHOLDER_FRAMES[frameIndex];
    return;
  }
  els.textPromptInput.placeholder = state.sessionReady ? DEFAULT_TEXT_PLACEHOLDER : OFFLINE_TEXT_PLACEHOLDER;
}

function clearTextComposerLoadingTimer() {
  if (state.textPlaceholderTimer) {
    clearInterval(state.textPlaceholderTimer);
    state.textPlaceholderTimer = null;
  }
}

function setTextComposerLoading(loading) {
  const nextLoading = Boolean(loading);
  if (state.textInputLoading === nextLoading) {
    renderTextComposerPlaceholder();
    return;
  }

  state.textInputLoading = nextLoading;
  clearTextComposerLoadingTimer();
  state.textPlaceholderFrame = 0;
  renderTextComposerPlaceholder();

  if (nextLoading) {
    state.textPlaceholderTimer = window.setInterval(() => {
      state.textPlaceholderFrame = (state.textPlaceholderFrame + 1) % TEXT_STARTUP_PLACEHOLDER_FRAMES.length;
      renderTextComposerPlaceholder();
    }, 420);
  }
}

function updateTextComposerState() {
  const hasDraft = Boolean(String(els.textPromptInput.value || "").trim());
  const isLocked = !state.textInputEnabled;
  const isBusy = state.serviceBusy || Boolean(state.pendingTextRequest) || state.textInterruptPending;
  els.textComposer.classList.toggle("text-composer--disabled", isLocked);
  els.textComposer.classList.toggle("text-composer--pending", isBusy);
  els.textComposer.classList.toggle("text-composer--live", state.sessionReady);
  els.textComposer.setAttribute("aria-disabled", String(isLocked));
  els.textComposer.setAttribute("aria-busy", String(isBusy));
  els.textPromptInput.disabled = isLocked;
  els.sendTextBtn.disabled = isLocked || !hasDraft || state.textInputComposing;
  els.sendTextBtn.setAttribute(
    "aria-label",
    isLocked
      ? "Text input is unavailable until the voice session is ready"
      : state.textInterruptPending
        ? "Queue text message while interrupting the current turn"
        : state.pendingTextRequest
          ? "Queue text message"
          : "Send text message"
  );
  renderTextComposerPlaceholder();
}

function clearPendingTextRequest() {
  clearPendingTextRetry();
  state.pendingTextRequest = null;
  state.textInterruptPending = false;
  updateTextComposerState();
}

function restorePendingTextDraft() {
  const pending = state.pendingTextRequest;
  if (pending && !pending.dispatched && !String(els.textPromptInput.value || "").trim()) {
    els.textPromptInput.value = pending.text;
  }
  if (pending && !pending.dispatched) {
    removeConversationMessage(pending.userMessageId);
  }
  clearPendingTextRequest();
}

function queueTextRequest(text) {
  const value = String(text || "").trim();
  if (!value) {
    return null;
  }

  const existing = state.pendingTextRequest;
  let userMessageId = existing && !existing.dispatched ? existing.userMessageId : null;

  if (userMessageId) {
    updateConversationMessage(userMessageId, { text: value });
  } else {
    const message = beginConversationMessage("user", value, "Queued", "active");
    userMessageId = message.id;
  }
  state.pendingTextRequest = {
    text: value,
    userMessageId,
    dispatched: false,
    attempts: 0
  };
  setConversationStage("user", "Queued", "active");
  updateTextComposerState();
  return state.pendingTextRequest;
}

function sendRealtimeJson(payload) {
  if (!isRealtimeSocketOpen()) {
    return false;
  }
  state.ws.send(JSON.stringify(payload));
  return true;
}

function flushPendingTextRequest(reason = "direct") {
  const pending = state.pendingTextRequest;
  if (!pending || pending.dispatched) {
    updateTextComposerState();
    return false;
  }
  if (!state.sessionReady || !isRealtimeSocketOpen() || state.serviceBusy || state.textInterruptPending || hasActiveTurn()) {
    return false;
  }

  state.currentUserMessageId = pending.userMessageId;
  state.currentAssistantMessageId = null;
  pending.dispatched = true;
  pending.attempts += 1;

  const sent = sendRealtimeJson({
    type: "text.input",
    text: pending.text,
    tts_options: state.ttsOptions
  });

  if (!sent) {
    pending.dispatched = false;
    return false;
  }

  setConversationStage("user", "Done", "done");
  setOrbMode("live", "Text input sent. Waiting for the assistant.");
  appendLog("text.input.sent", {
    reason,
    attempt: pending.attempts,
    length: pending.text.length
  });
  updateTextComposerState();
  return true;
}

function schedulePendingTextFlush(reason, delayMs = 0) {
  clearPendingTextRetry();
  if (!state.pendingTextRequest) {
    updateTextComposerState();
    return;
  }
  state.pendingTextRetryTimer = window.setTimeout(() => {
    state.pendingTextRetryTimer = null;
    const sent = flushPendingTextRequest(reason);
    if (
      !sent &&
      state.pendingTextRequest &&
      !state.pendingTextRequest.dispatched &&
      !state.textInterruptPending &&
      state.sessionReady &&
      isRealtimeSocketOpen() &&
      !state.serviceBusy &&
      hasActiveTurn()
    ) {
      schedulePendingTextFlush(reason, 160);
    }
  }, delayMs);
  updateTextComposerState();
}

function retryPendingTextAfterBusyError(message) {
  const pending = state.pendingTextRequest;
  const detail = String(message?.message || "");
  if (!pending || !pending.dispatched || !/assistant request is still running/i.test(detail)) {
    return false;
  }
  pending.dispatched = false;
  setOrbMode("live", "The current turn is finishing. Queued text will send next.");
  schedulePendingTextFlush("backend.busy", 180);
  return true;
}

function shouldSuppressBusyBackendError(message) {
  const detail = String(message?.message || "");
  if (!/assistant request is still running/i.test(detail)) {
    return false;
  }
  return state.speechActive || state.phase === "speech-ended" || state.phase === "asr";
}

function handleTextComposerSubmit(event) {
  event.preventDefault();
  if (!state.textInputEnabled) {
    return;
  }
  if (state.textInputComposing) {
    return;
  }
  const text = String(els.textPromptInput.value || "").trim();
  if (!text) {
    els.textPromptInput.focus();
    return;
  }
  queueTextRequest(text);
  els.textPromptInput.value = "";
  updateTextComposerState();

  if (!state.ws) {
    void connectRealtime({ withMicrophone: false });
    return;
  }

  if (hasActiveTurn()) {
    state.textInterruptPending = true;
    updateTextComposerState();
    stopPlaybackQueue();
    if (sendRealtimeJson({ type: "interrupt", request_id: `interrupt-${Date.now()}` })) {
      setOrbMode("live", "Interrupting the current turn before sending the queued text.");
      appendLog("text.input.interrupt_requested", {
        length: text.length
      });
    } else {
      state.textInterruptPending = false;
      schedulePendingTextFlush("interrupt.send_failed", 0);
    }
    return;
  }

  schedulePendingTextFlush("submit", 0);
}

function getConversationMessage(messageId) {
  return state.conversationMessages.find((message) => message.id === messageId) || null;
}

function removeConversationMessage(messageId) {
  const index = state.conversationMessages.findIndex((message) => message.id === messageId);
  if (index < 0) {
    return false;
  }
  state.conversationMessages.splice(index, 1);
  if (state.currentUserMessageId === messageId) {
    state.currentUserMessageId = null;
  }
  if (state.currentAssistantMessageId === messageId) {
    state.currentAssistantMessageId = null;
  }
  renderConversation();
  return true;
}

function trimConversationMessages() {
  while (state.conversationMessages.length > 6) {
    const removed = state.conversationMessages.shift();
    if (removed.id === state.currentUserMessageId) {
      state.currentUserMessageId = null;
    }
    if (removed.id === state.currentAssistantMessageId) {
      state.currentAssistantMessageId = null;
    }
  }
}

function renderConversation() {
  els.conversationScroll.replaceChildren();

  for (const message of state.conversationMessages) {
    const article = document.createElement("article");
    article.className = `message message--${message.role}`;

    const bubble = document.createElement("div");
    bubble.className = `message__bubble transcript transcript--${message.role}`;
    bubble.textContent = message.text || "";

    article.appendChild(bubble);
    els.conversationScroll.appendChild(article);
  }

  els.conversationScroll.scrollTop = els.conversationScroll.scrollHeight;
}

function clearConversation() {
  clearPendingTextRequest();
  state.conversationMessages = [];
  state.currentUserMessageId = null;
  state.currentAssistantMessageId = null;
  state.nextMessageId = 1;
  renderConversation();
}

function beginConversationMessage(role, text = "", stageText = "Idle", stageTone = null) {
  const message = {
    id: state.nextMessageId,
    role,
    text,
    stageText,
    stageTone
  };
  state.nextMessageId += 1;
  state.conversationMessages.push(message);
  if (role === "user") {
    state.currentUserMessageId = message.id;
  } else {
    state.currentAssistantMessageId = message.id;
  }
  trimConversationMessages();
  renderConversation();
  return message;
}

function updateConversationMessage(messageId, updates) {
  const message = getConversationMessage(messageId);
  if (!message) {
    return null;
  }
  Object.assign(message, updates);
  renderConversation();
  return message;
}

function getCurrentConversationMessage(role) {
  const messageId = role === "user" ? state.currentUserMessageId : state.currentAssistantMessageId;
  return getConversationMessage(messageId);
}

function setConversationStage(role, text, tone) {
  const stageEl = role === "user" ? els.userStage : els.assistantStage;
  const label = role === "user" ? "User" : "AI";
  stageEl.textContent = `${label} ${text}`;
  stageEl.className = "stage-pill header-stage";
  if (tone) {
    stageEl.classList.add(`stage-pill--${tone}`);
  }
}

function setConversationText(role, text, stageText, stageTone) {
  const message = getCurrentConversationMessage(role);
  if (!message) {
    beginConversationMessage(role, text || "", stageText, stageTone);
    if (stageText) {
      setConversationStage(role, stageText, stageTone);
    }
    return;
  }
  updateConversationMessage(message.id, {
    text: text || ""
  });
  if (stageText) {
    setConversationStage(role, stageText, stageTone);
  }
}

function appendConversationText(role, text) {
  const value = text || "";
  if (!value) {
    return;
  }
  const message = getCurrentConversationMessage(role) || beginConversationMessage(role, "", "Streaming", "active");
  updateConversationMessage(message.id, {
    text: `${message.text || ""}${value}`
  });
}

function updateLlmModelDisplay() {
  const { modelPath, modelName, modelSize } = state.llm;
  els.llmModelPath.value = modelPath || "";
  
  if (modelName) {
    els.llmModelName.textContent = modelName;
    els.llmModelSize.textContent = modelSize || "";
    els.llmModelInfo.classList.add("llm-model-info--loaded");
    els.llmStatus.textContent = `已选择: ${modelName}${modelSize ? ` (${modelSize})` : ""}`;
  } else {
    els.llmModelName.textContent = "未选择模型";
    els.llmModelSize.textContent = "";
    els.llmModelInfo.classList.remove("llm-model-info--loaded");
    els.llmStatus.textContent = "请选择一个 GGUF 格式的大语言模型文件。";
  }
}

function setResourceBadge(el, mode, text) {
  el.textContent = text;
  el.className = "resource-badge";
  if (mode) {
    el.classList.add(`resource-badge--${mode}`);
  }
}

function renderResourceStatus() {
  const status = state.resourceStatus;
  if (!status) {
    setResourceBadge(els.resourceLlamaState, "", "Checking");
    setResourceBadge(els.resourceGgufState, "", "Checking");
    setResourceBadge(els.resourceAsrState, "", "Checking");
    setResourceBadge(els.resourceTtsState, "", "Checking");
    els.resourceHint.textContent = "Checking installation assets...";
    return;
  }

  setResourceBadge(
    els.resourceLlamaState,
    status.llama.exists ? "ok" : "warn",
    status.llama.exists ? "Installed" : "Missing"
  );
  setResourceBadge(
    els.resourceAsrState,
    status.asr.exists ? "ok" : "warn",
    status.asr.exists ? "Installed" : "Missing"
  );
  setResourceBadge(
    els.resourceTtsState,
    status.tts.exists ? "ok" : "warn",
    status.tts.exists ? "Installed" : "Missing"
  );

  if (status.llm.exists) {
    setResourceBadge(els.resourceGgufState, "ok", "Ready");
  } else if (status.llm.source === "assets.multiple") {
    setResourceBadge(els.resourceGgufState, "warn", "Select Model");
  } else {
    setResourceBadge(els.resourceGgufState, "warn", "Missing");
  }

  const missing = [];
  if (!status.llama.exists) missing.push("llama runtime");
  if (!status.llm.exists) missing.push(status.llm.source === "assets.multiple" ? "GGUF selection" : "GGUF");
  if (!status.asr.exists) missing.push("ASR");
  if (!status.tts.exists) missing.push("TTS");

  if (!missing.length) {
    const ggufSource = status.llm.source === "assets.auto" ? "auto-selected GGUF" : "configured GGUF";
    els.resourceHint.textContent = `Core assets detected. Using ${ggufSource} from assets.`;
    return;
  }

  els.resourceHint.textContent = `Missing: ${missing.join(", ")}. Expected under ${status.assetsRoot}.`;
}

async function detectResources() {
  if (!window.desktopApp.getResourceStatus) {
    return;
  }

  els.detectResourcesBtn.disabled = true;
  els.resourceHint.textContent = "Checking installation assets...";

  try {
    const status = await window.desktopApp.getResourceStatus();
    state.resourceStatus = status;
    renderResourceStatus();
    appendLog("assets.status", status);
  } catch (error) {
    appendLog("assets.status_error", String(error));
    els.resourceHint.textContent = "Resource detection failed.";
  } finally {
    els.detectResourcesBtn.disabled = false;
  }
}

async function loadLlmModelPath() {
  try {
    const modelPath = await window.desktopApp.getLlmModelPath();
    if (modelPath) {
      const info = await window.desktopApp.getLlmModelInfo(modelPath);
      state.llm.modelPath = modelPath;
      state.llm.modelName = info.name || null;
      state.llm.modelSize = info.size || null;
    } else {
      state.llm.modelPath = null;
      state.llm.modelName = null;
      state.llm.modelSize = null;
    }
    updateLlmModelDisplay();
  } catch (error) {
    appendLog("llm.config.load_error", String(error));
  }
}

async function browseForModel() {
  try {
    const result = await window.desktopApp.showModelFileDialog();
    if (result.canceled || !result.filePaths || result.filePaths.length === 0) {
      return;
    }
    const filePath = result.filePaths[0];
    const info = await window.desktopApp.getLlmModelInfo(filePath);
    if (!info.exists) {
      appendLog("llm.model.not_found", { path: filePath });
      return;
    }
    state.llm.modelPath = filePath;
    state.llm.modelName = info.name;
    state.llm.modelSize = info.size;
    updateLlmModelDisplay();
    appendLog("llm.model.selected", { path: filePath, name: info.name, size: info.size });
  } catch (error) {
    appendLog("llm.browse.error", String(error));
  }
}

async function applyLlmConfig() {
  const { modelPath } = state.llm;
  if (!modelPath) {
    appendLog("llm.config.no_model", "Please select a model file first.");
    return;
  }
  try {
    const saved = await window.desktopApp.setLlmModelPath(modelPath);
    if (saved) {
      appendLog("llm.config.saved", { modelPath });
      void detectResources();
      els.llmStatus.textContent = `配置已保存: ${state.llm.modelName}${state.llm.modelSize ? ` (${state.llm.modelSize})` : ""}。重新启动会话后生效。`;
    } else {
      appendLog("llm.config.save_failed", "Failed to save LLM configuration.");
    }
  } catch (error) {
    appendLog("llm.config.error", String(error));
  }
}

async function browseForAudio() {
  try {
    const result = await window.desktopApp.showAudioFileDialog();
    if (result.canceled || !result.filePaths || result.filePaths.length === 0) {
      return;
    }
    const filePath = result.filePaths[0];
    els.promptAudioPath.value = filePath;

    // Extract filename without extension and fill Reference Text
    const fileNameWithExt = filePath.split(/[\\\/]/).pop();
    const lastDotIndex = fileNameWithExt.lastIndexOf(".");
    const fileName = lastDotIndex !== -1 ? fileNameWithExt.substring(0, lastDotIndex) : fileNameWithExt;
    els.promptText.value = fileName;

    applyTtsOptions(false);
    appendLog("tts.audio.selected", { path: filePath });
  } catch (error) {
    appendLog("tts.audio.browse_error", String(error));
  }
}

function readNumberInput(el, fallback, min, max) {
  const value = Number(el.value);
  const safe = Number.isFinite(value) ? value : fallback;
  return Math.max(min ?? safe, Math.min(max ?? safe, safe));
}

function readTtsOptions() {
  const promptAudioPath = String(els.promptAudioPath.value || "").trim();
  const promptText = String(els.promptText.value || "").trim();
  const hasCompleteReference = Boolean(promptAudioPath && promptText);
  return {
    lora_selection: String(els.ttsLoraSelect.value || "").trim() || null,
    prompt_audio_path: hasCompleteReference ? promptAudioPath : null,
    prompt_text: hasCompleteReference ? promptText : null,
    cfg_value: readNumberInput(els.ttsCfgValue, 2.0, 0.1, 10),
    inference_timesteps: Math.round(readNumberInput(els.ttsInferenceTimesteps, 10, 1, 100)),
    seed: Math.round(readNumberInput(els.ttsSeed, -1))
  };
}

function applyTtsConfigToUi(config) {
  els.promptAudioPath.value = config.prompt_audio_path || "";
  els.promptText.value = config.prompt_text || "";

  if (config.lora_selection) {
    const options = [...els.ttsLoraSelect.options];
    if (options.some(opt => opt.value === config.lora_selection)) {
      els.ttsLoraSelect.value = config.lora_selection;
    }
  } else {
    els.ttsLoraSelect.value = "";
  }
  if (config.cfg_value !== undefined && config.cfg_value !== null) {
    els.ttsCfgValue.value = config.cfg_value;
  }
  if (config.inference_timesteps !== undefined && config.inference_timesteps !== null) {
    els.ttsInferenceTimesteps.value = config.inference_timesteps;
  }
  if (config.seed !== undefined && config.seed !== null) {
    els.ttsSeed.value = config.seed;
  }
  state.ttsOptions = readTtsOptions();
  updateTtsStatus();
}

async function loadTtsConfigFromStorage() {
  try {
    const config = await window.desktopApp.getTtsConfig();
    if (config) {
      applyTtsConfigToUi(config);
      await normalizeTtsReferencePath();
      appendLog("tts.config.loaded", config);
    }
  } catch (error) {
    appendLog("tts.config.load_error", String(error));
  }
}

async function saveTtsConfigToStorage() {
  try {
    const config = state.ttsOptions;
    const saved = await window.desktopApp.saveTtsConfig(config);
    if (saved) {
      appendLog("tts.config.saved", config);
    }
  } catch (error) {
    appendLog("tts.config.save_error", String(error));
  }
}

function updateTtsStatus() {
  const options = state.ttsOptions;
  const lora = options.lora_selection || "base voice";
  const seed = Number(options.seed) >= 0 ? options.seed : "random";
  const typedReferencePath = String(els.promptAudioPath.value || "").trim();
  const typedReferenceText = String(els.promptText.value || "").trim();
  const referenceStatus = options.prompt_audio_path
    ? "reference on"
    : typedReferencePath && !typedReferenceText
      ? "reference ignored: text required"
      : "base voice";
  els.ttsStatus.textContent = `LoRA: ${lora}. ${referenceStatus}. CFG ${options.cfg_value}, ${options.inference_timesteps} steps, seed ${seed}.`;
}

async function normalizeTtsReferencePath() {
  const promptAudioPath = String(els.promptAudioPath.value || "").trim();
  if (!promptAudioPath || !window.desktopApp.getPathInfo) {
    return false;
  }

  try {
    const pathInfo = await window.desktopApp.getPathInfo(promptAudioPath);
    if (pathInfo.exists && pathInfo.type === "file") {
      return false;
    }
  } catch (error) {
    appendLog("tts.reference.check_error", String(error));
    return false;
  }

  els.promptAudioPath.value = "";
  state.ttsOptions = readTtsOptions();
  updateTtsStatus();
  await saveTtsConfigToStorage();
  appendLog("tts.reference.cleared_missing", { path: promptAudioPath });
  els.ttsStatus.textContent = "Reference audio was cleared because the file no longer exists.";
  return true;
}

function applyTtsOptions(sendToBackend = true) {
  state.ttsOptions = readTtsOptions();
  updateTtsStatus();
  saveTtsConfigToStorage();
  if (sendToBackend && state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: "tts.configure", tts_options: state.ttsOptions }));
  }
  appendLog("tts.config.local", state.ttsOptions);
}

async function refreshLoraCatalog() {
  const current = els.ttsLoraSelect.value;
  let models = [];
  let source = "local";

  try {
    const localResult = await window.desktopApp.scanLoraLocal();
    models = localResult.models || [];
    appendLog("tts.lora.local_scan", { count: models.length });
  } catch (error) {
    appendLog("tts.lora.local_scan_error", String(error));
  }

  if (models.length === 0 && state.backend.httpBase) {
    try {
      const response = await fetch(`${state.backend.httpBase}/tts/lora/catalog`);
      if (response.ok) {
        const payload = await response.json();
        models = payload.models || [];
        source = "backend";
        appendLog("tts.lora.backend_scan", { count: models.length });
      }
    } catch (error) {
      appendLog("tts.lora.backend_scan_error", String(error));
    }
  }

  els.ttsLoraSelect.innerHTML = '<option value="">Base voice</option>';
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.path;
    option.textContent = model.base_model ? `${model.label} [${model.base_model}]` : model.label;
    els.ttsLoraSelect.appendChild(option);
  }
  if ([...els.ttsLoraSelect.options].some((option) => option.value === current)) {
    els.ttsLoraSelect.value = current;
  }
  state.ttsOptions = readTtsOptions();
  updateTtsStatus();
  appendLog("tts.lora.catalog", { count: models.length, source });
}

function setBackendState(text, tone) {
  els.backendState.textContent = text;
  els.backendState.className = "status-pill";
  if (tone) {
    els.backendState.classList.add(`status-pill--${tone}`);
  }
}

function setOrbMode(mode, caption) {
  if (!els.orb || !els.orbCaption) {
    return;
  }

  els.orb.className = "orb";
  if (mode) {
    els.orb.classList.add(`orb--${mode}`);
  }
  els.orbCaption.textContent = caption;
}

function setFlowStep(el, stateName) {
  if (!el) {
    return;
  }

  el.className = "flow-step";
  if (stateName) {
    el.classList.add(`flow-step--${stateName}`);
  }
}

function setTurnPhase(phase) {
  state.phase = phase;
  const done = "done";
  const active = "active";

  setFlowStep(els.flowListening, "");
  setFlowStep(els.flowSpeechEnd, "");
  setFlowStep(els.flowAsr, "");
  setFlowStep(els.flowLlm, "");

  if (phase === "idle") {
    setConversationStage("user", "Idle");
    setConversationStage("assistant", "Idle");
    return;
  }

  if (phase === "listening") {
    setFlowStep(els.flowListening, active);
    setConversationStage("user", "Listening", "active");
    setConversationStage("assistant", "Idle");
    return;
  }

  if (phase === "speech-ended") {
    setFlowStep(els.flowListening, done);
    setFlowStep(els.flowSpeechEnd, active);
    setConversationStage("user", "Speech End", "active");
    setConversationStage("assistant", "Idle");
    return;
  }

  if (phase === "asr") {
    setFlowStep(els.flowListening, done);
    setFlowStep(els.flowSpeechEnd, done);
    setFlowStep(els.flowAsr, active);
    setConversationStage("user", "ASR", "active");
    setConversationStage("assistant", "Waiting");
    return;
  }

  if (phase === "llm") {
    setFlowStep(els.flowListening, done);
    setFlowStep(els.flowSpeechEnd, done);
    setFlowStep(els.flowAsr, done);
    setFlowStep(els.flowLlm, active);
    setConversationStage("user", "Done", "done");
    setConversationStage("assistant", "LLM", "active");
    return;
  }

  if (phase === "completed") {
    setFlowStep(els.flowListening, done);
    setFlowStep(els.flowSpeechEnd, done);
    setFlowStep(els.flowAsr, done);
    setFlowStep(els.flowLlm, done);
    setConversationStage("user", "Done", "done");
    setConversationStage("assistant", "Done", "done");
  }
}

const LOG_TIME_FORMATTER = new Intl.DateTimeFormat("zh-CN", {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit"
});

function formatLogTimestamp(date = new Date()) {
  return LOG_TIME_FORMATTER.format(date);
}

function getLogTone(type) {
  const normalized = String(type || "").toLowerCase();

  if (/(error|failed|timeout|not_found)/.test(normalized)) {
    return "error";
  }

  if (/(missing|cleared|close|stop)/.test(normalized)) {
    return "warn";
  }

  if (/(ready|ok|saved|selected|configured)/.test(normalized)) {
    return "success";
  }

  return "info";
}

function appendLog(type, payload) {
  const entry = document.createElement("div");
  entry.className = `log-entry log-entry--${getLogTone(type)}`;
  let displayPayload = payload;
  if (payload && payload.type === "assistant.audio.chunk") {
    displayPayload = {
      ...payload,
      data: `<base64 pcm_f32le ${payload.num_samples || 0} samples>`
    };
  }
  const safe = typeof displayPayload === "string" ? displayPayload : JSON.stringify(displayPayload, null, 2);

  const meta = document.createElement("div");
  meta.className = "log-entry__meta";

  const typeBadge = document.createElement("span");
  typeBadge.className = "log-entry__type";
  typeBadge.textContent = type;

  const timestamp = document.createElement("time");
  timestamp.className = "log-entry__time";
  timestamp.textContent = formatLogTimestamp();

  const content = document.createElement("pre");
  content.textContent = safe;

  meta.append(typeBadge, timestamp);
  entry.append(meta, content);
  els.eventLog.prepend(entry);
}

function renderLogCaptureState() {
  if (els.logCaptureToggle) {
    els.logCaptureToggle.checked = state.logCaptureEnabled;
  }
  if (!els.logCaptureHint) {
    return;
  }
  const captureLocation = state.turnCaptureDir || state.serviceLogsDir || "";
  els.logCaptureHint.textContent = state.logCaptureEnabled
    ? "Capture is on. Service logs and per-turn user audio/transcripts are being written for this runtime."
    : "Capture is off. No service logs or per-turn user audio/transcripts are being written.";
  els.logCaptureHint.title = captureLocation || "";
}

function applyVoiceLoggingState(loggingState) {
  if (!loggingState || typeof loggingState !== "object") {
    renderLogCaptureState();
    return;
  }
  state.logCaptureEnabled = Boolean(loggingState.enabled);
  state.serviceLogsDir = String(loggingState.logsDir || state.serviceLogsDir || "");
  state.turnCaptureDir = String(loggingState.turnCaptureDir || state.turnCaptureDir || "");
  renderLogCaptureState();
}

async function setVoiceLoggingEnabled(enabled) {
  const nextEnabled = Boolean(enabled);
  try {
    if (window.desktopApp.setVoiceLoggingEnabled) {
      const loggingState = await window.desktopApp.setVoiceLoggingEnabled(nextEnabled);
      applyVoiceLoggingState(loggingState);
    } else {
      state.logCaptureEnabled = nextEnabled;
      renderLogCaptureState();
    }
    if (isRealtimeSocketOpen()) {
      sendRealtimeJson({
        type: "logging.configure",
        enabled: state.logCaptureEnabled
      });
    }
  } catch (error) {
    appendLog("logging.capture_error", String(error));
    renderLogCaptureState();
  }
}

function setConnected(connected) {
  const voiceConnected = connected && state.sessionMode === "voice";
  els.connectBtn.disabled = voiceConnected || state.serviceBusy;
  els.disconnectBtn.disabled = (!connected && !state.serviceBusy) || state.serviceBusy;
  updateTextComposerState();
}

function applyBackendConfig(backendConfig) {
  state.backend = backendConfig;
  els.httpBase.textContent = backendConfig.httpBase;
  els.wsUrl.textContent = backendConfig.wsUrl;
}

function setServiceBusy(busy) {
  state.serviceBusy = busy;
  setConnected(Boolean(state.ws));
}

async function enableVoiceInput() {
  if (isVoiceSessionActive() && state.micStream) {
    return true;
  }

  setServiceBusy(true);
  setBackendState("Preparing Audio", "warn");
  setOrbMode("", "Requesting microphone access...");

  try {
    await ensureMicrophone();
    state.sessionMode = "voice";
    setConnected(Boolean(state.ws));
    setBackendState("WebSocket Live", "live");
    if (!hasActiveTurn()) {
      setTurnPhase("listening");
      setOrbMode("live", "Microphone is ready. Waiting for speech.");
    } else {
      setOrbMode("live", "Microphone is ready. Voice interruption is available.");
    }
    return true;
  } catch (error) {
    appendLog("mic.error", String(error));
    setBackendState("Microphone Error", "warn");
    setOrbMode("", "Microphone access failed.");
    return false;
  } finally {
    setServiceBusy(false);
  }
}

async function ensurePlaybackContext() {
  if (!state.playbackContext || state.playbackContext.state === "closed") {
    state.playbackContext = new AudioContext();
    const analyser = state.playbackContext.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.82;
    analyser.connect(state.playbackContext.destination);
    state.playbackAnalyser = analyser;
    window.__ttsAudioAnalyser = analyser;
    state.playbackNextTime = 0;
  }
  if (state.playbackContext.state === "suspended") {
    await state.playbackContext.resume();
  }
  return state.playbackContext;
}

function decodeBase64PcmF32(data) {
  const binary = atob(data || "");
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Float32Array(bytes.buffer);
}

async function playPcmF32Chunk(data, sampleRate) {
  const audioContext = await ensurePlaybackContext();
  const pcm = decodeBase64PcmF32(data);
  if (!pcm.length) {
    return;
  }

  const buffer = audioContext.createBuffer(1, pcm.length, sampleRate || 44100);
  buffer.copyToChannel(pcm, 0);

  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(state.playbackAnalyser || audioContext.destination);
  state.playbackSources.add(source);
  source.onended = () => {
    state.playbackSources.delete(source);
  };

  const startAt = Math.max(audioContext.currentTime + 0.02, state.playbackNextTime || 0);
  source.start(startAt);
  state.playbackNextTime = startAt + buffer.duration;
}

function stopPlaybackQueue() {
  for (const source of state.playbackSources) {
    try {
      source.stop();
    } catch (_) {
      // Already stopped or not yet started.
    }
  }
  state.playbackSources.clear();
  if (state.playbackContext && state.playbackContext.state !== "closed") {
    state.playbackNextTime = state.playbackContext.currentTime;
  } else {
    state.playbackNextTime = 0;
  }
}

async function releasePlayback() {
  stopPlaybackQueue();
  state.playbackNextTime = 0;
  if (state.playbackAnalyser) {
    try {
      state.playbackAnalyser.disconnect();
    } catch (_) {
      // Ignore analyser disconnect failures during shutdown.
    }
    state.playbackAnalyser = null;
  }
  window.__ttsAudioAnalyser = null;
  if (state.playbackContext) {
    try {
      await state.playbackContext.close();
    } catch (_) {
      // Ignore close failures during shutdown.
    }
    state.playbackContext = null;
  }
}

async function ensureMicrophone() {
  if (state.micStream && state.micContext && state.micProcessor && state.micSource) {
    return;
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true
    }
  });

  const audioContext = new AudioContext({ sampleRate: 16000 });
  await audioContext.resume();
  const source = audioContext.createMediaStreamSource(stream);
  const processor = audioContext.createScriptProcessor(4096, 1, 1);

  processor.onaudioprocess = (event) => {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
      return;
    }
    const input = event.inputBuffer.getChannelData(0);
    const pcm = new Int16Array(input.length);
    for (let i = 0; i < input.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, input[i]));
      pcm[i] = sample < 0 ? sample * 32768 : sample * 32767;
    }
    state.ws.send(pcm.buffer);
  };

  source.connect(processor);
  processor.connect(audioContext.destination);

  state.micStream = stream;
  state.micContext = audioContext;
  state.micSource = source;
  state.micProcessor = processor;
}

async function releaseMicrophone() {
  if (state.micProcessor) {
    state.micProcessor.disconnect();
    state.micProcessor.onaudioprocess = null;
    state.micProcessor = null;
  }
  if (state.micSource) {
    state.micSource.disconnect();
    state.micSource = null;
  }
  if (state.micContext) {
    try {
      await state.micContext.close();
    } catch (_) {
      // Ignore close failures during shutdown.
    }
    state.micContext = null;
  }
  if (state.micStream) {
    for (const track of state.micStream.getTracks()) {
      track.stop();
    }
    state.micStream = null;
  }
}

async function bootstrap() {
  const [appInfo, backendConfig] = await Promise.all([
    window.desktopApp.getInfo(),
    window.desktopApp.getBackendConfig()
  ]);

  applyBackendConfig(backendConfig);
  if (window.desktopApp.setVoiceLoggingEnabled) {
    try {
      const loggingState = await window.desktopApp.setVoiceLoggingEnabled(false);
      applyVoiceLoggingState(loggingState);
    } catch (error) {
      appendLog("logging.capture_init_error", String(error));
      renderLogCaptureState();
    }
  } else {
    renderLogCaptureState();
  }
  els.appVersion.textContent = `${appInfo.name} ${appInfo.version}`;
  appendLog("app.ready", { appInfo, backendConfig });
  if (window.desktopApp.onVoiceServicesEvent) {
    window.desktopApp.onVoiceServicesEvent((event) => {
      appendLog(`services.${event.type}`, event);
      if (event.type === "logging.capture_changed") {
        applyVoiceLoggingState(event);
      }
      if (event.type === "process.output" && state.serviceBusy) {
        setBackendState(`${event.service} Starting`, "warn");
      } else if (event.type === "warmup.begin" && state.serviceBusy) {
        setBackendState("Warming Models", "warn");
        setOrbMode("", "ASR and TTS warm-up are running before the first turn.");
      } else if (event.type === "warmup.completed" && state.serviceBusy) {
        const totalMs = Number(event.total_ms);
        const asrMs = Number(event.asr_inference_ms);
        const ttsMs = Number(event.tts_inference_ms);
        const totalSummary = Number.isFinite(totalMs) ? `${Math.round(totalMs)} ms` : "completed";
        const detailSummary = [
          Number.isFinite(asrMs) ? `ASR ${Math.round(asrMs)} ms` : "",
          Number.isFinite(ttsMs) ? `TTS ${Math.round(ttsMs)} ms` : "",
        ].filter(Boolean).join(", ");
        setBackendState("Models Warmed", "ok");
        setOrbMode("live", `${detailSummary ? `${detailSummary}. ` : ""}Warm-up ${totalSummary}. Opening microphone next.`);
      } else if (event.type === "warmup.error" && state.serviceBusy) {
        setBackendState("Warmup Failed", "warn");
        setOrbMode("", event.message || "ASR/TTS warm-up failed.");
      }
    });
  }
  await loadLlmModelPath();
  await detectResources();
  await refreshLoraCatalog().catch((error) => appendLog("tts.lora.catalog_error", String(error)));
  await loadTtsConfigFromStorage();
  setTextComposerLoading(false);
  updateTextComposerState();
  setConnected(false);
}

async function pingBackend() {
  if (!state.backend.httpBase) {
    setBackendState("Backend Not Configured", "warn");
    return;
  }

  setBackendState("Checking Backend", "warn");
  setOrbMode("", "Pinging Python backend...");

  try {
    const res = await fetch(`${state.backend.httpBase}/health`);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json();
    appendLog("health.ok", data);
    setBackendState("Backend Reachable", "ok");
    setOrbMode("live", "Backend is reachable. Realtime WebSocket can be opened next.");
    void refreshLoraCatalog().catch((error) => appendLog("tts.lora.catalog_error", String(error)));
  } catch (error) {
    appendLog("health.error", String(error));
    setBackendState("Backend Offline", "warn");
    setOrbMode("", "Health check failed. Start the Python backend first.");
  }
}

function handleRealtimeMessage(message) {
  appendLog(message.type || "message", message);

  switch (message.type) {
    case "session.ready":
      state.sessionReady = true;
      state.textInputEnabled = true;
      setTextComposerLoading(false);
      setBackendState("WebSocket Live", "live");
      setOrbMode("live", getReadyOrbMessage());
      setTurnPhase(getReadyTurnPhase());
      schedulePendingTextFlush("session.ready", 0);
      break;
    case "server.ready":
      setBackendState("WebSocket Live", "live");
      setOrbMode("live", "Realtime transport connected.");
      break;
    case "vad":
      if (message.speaking) {
        state.speechActive = true;
        if (state.phase === "llm" || state.playbackSources.size > 0) {
          stopPlaybackQueue();
          setTurnPhase("listening");
          setConversationStage("assistant", "Interrupted", "warn");
          setOrbMode("listening", "User speech interrupted the assistant.");
        }
        if (state.phase === "idle" || state.phase === "completed" || state.phase === "listening") {
          setTurnPhase("listening");
        }
        setOrbMode("listening", "User speech detected.");
      } else if (state.speechActive && state.phase === "listening") {
        state.speechActive = false;
        setTurnPhase("speech-ended");
        setOrbMode("live", "Speech ended. Waiting for ASR.");
      } else if (state.phase === "idle" || state.phase === "completed" || state.phase === "listening") {
        setOrbMode("live", "Waiting for the next speech segment.");
      }
      break;
    case "asr.started":
      state.speechActive = false;
      state.currentAssistantMessageId = null;
      beginConversationMessage("user", "Transcribing...", "ASR", "active");
      setTurnPhase("asr");
      setOrbMode("listening", "Speech segment captured. ASR is transcribing.");
      break;
    case "asr.completed":
    case "user_text":
      setConversationText("user", message.text || "", "Done", "done");
      break;
    case "assistant.started":
    case "generating":
      if (state.pendingTextRequest?.dispatched) {
        clearPendingTextRequest();
      }
      if (!getCurrentConversationMessage("assistant")) {
        beginConversationMessage("assistant", "", "LLM", "active");
      }
      setTurnPhase("llm");
      setOrbMode("live", "LLM is thinking...");
      break;
    case "assistant.delta":
      appendConversationText("assistant", message.text);
      break;
    case "text":
      appendConversationText("assistant", message.content);
      break;
    case "assistant.audio.start":
      void ensurePlaybackContext();
      setConversationStage("assistant", "Speaking", "active");
      setOrbMode("speaking", "TTS streaming has started.");
      break;
    case "assistant.audio.chunk":
      void playPcmF32Chunk(message.data, message.sample_rate).catch((error) => {
        appendLog("audio.play_error", String(error));
      });
      break;
    case "assistant.audio.completed":
      if (message.interrupted) {
        stopPlaybackQueue();
      }
      setConversationStage("assistant", message.interrupted ? "Interrupted" : "Spoken", message.interrupted ? "warn" : "done");
      break;
    case "assistant.audio.error":
      setConversationStage("assistant", "Audio Error", "warn");
      setOrbMode("", "TTS failed while generating audio.");
      break;
    case "tts.configured":
      appendLog("tts.configured", message);
      break;
    case "logging.configured":
      applyVoiceLoggingState({
        enabled: message.enabled,
        turnCaptureDir: message.capture_dir
      });
      break;
    case "assistant.completed":
    case "done":
      if (message.interrupted) {
        stopPlaybackQueue();
        setTurnPhase(getReadyTurnPhase());
        setConversationStage("assistant", "Interrupted", "warn");
        setOrbMode(getReadyOrbMode(), "Assistant interrupted. Ready for the queued turn.");
        schedulePendingTextFlush("assistant.completed.interrupted", 80);
      } else {
        const assistantMessage = getCurrentConversationMessage("assistant");
        if (message.text && (!assistantMessage || !(assistantMessage.text || "").trim())) {
          setConversationText("assistant", message.text, "Done", "done");
        } else if (assistantMessage && !(assistantMessage.text || "").trim()) {
          setConversationText("assistant", "LLM returned no speakable text.", "Error", "warn");
          setBackendState("LLM Empty", "warn");
          setOrbMode("", "LLM completed without text. Check whether thinking mode is disabled.");
          break;
        }
        setTurnPhase("completed");
        setOrbMode("live", "Turn completed. Ready for the next user utterance.");
        schedulePendingTextFlush("assistant.completed", 0);
      }
      break;
    case "assistant.error":
      setTurnPhase(getReadyTurnPhase());
      setConversationText("assistant", `LLM error: ${message.message || "unknown error"}`, "Error", "warn");
      setBackendState("LLM Error", "warn");
      setOrbMode("", "LLM request failed. Check the local LLM server.");
      schedulePendingTextFlush("assistant.error", 120);
      break;
    case "asr.error":
      setTurnPhase(getReadyTurnPhase());
      setConversationText("user", `ASR error: ${message.message || "unknown error"}`, "Error", "warn");
      setBackendState("ASR Error", "warn");
      setOrbMode("", "ASR failed while processing the speech segment.");
      schedulePendingTextFlush("asr.error", 120);
      break;
    case "interrupt.ack":
      state.textInterruptPending = false;
      if (message.accepted) {
        stopPlaybackQueue();
        setTurnPhase(getReadyTurnPhase());
        setConversationStage("assistant", "Interrupted", "warn");
        setOrbMode(getReadyOrbMode(), "Interrupt acknowledged. Preparing the queued turn.");
        schedulePendingTextFlush("interrupt.ack", 180);
      } else {
        setOrbMode("live", "Interrupt acknowledged.");
        schedulePendingTextFlush("interrupt.ack.noop", 0);
      }
      updateTextComposerState();
      break;
    case "error":
      if (retryPendingTextAfterBusyError(message)) {
        break;
      }
      if (shouldSuppressBusyBackendError(message)) {
        appendLog("backend.busy_ignored", {
          message: message.message || "",
          phase: state.phase,
          speechActive: state.speechActive
        });
        setOrbMode("listening", "The interrupted turn is still winding down. Waiting for the new voice turn.");
        break;
      }
      setTurnPhase(getReadyTurnPhase());
      setConversationText("assistant", `Backend error: ${message.message || "unknown error"}`, "Error", "warn");
      setBackendState("Backend Error", "warn");
      setOrbMode("", "The backend reported an error.");
      break;
    default:
      break;
  }
}

async function connectRealtime(options = {}) {
  const withMicrophone = options.withMicrophone !== false;
  if (!state.backend.wsUrl || state.serviceBusy) {
    return false;
  }
  if (state.ws) {
    if (withMicrophone) {
      return enableVoiceInput();
    }
    return true;
  }

  state.sessionMode = withMicrophone ? "voice" : "text";
  state.sessionReady = false;
  state.textInputEnabled = false;
  setTextComposerLoading(true);
  updateTextComposerState();

  try {
    await ensurePlaybackContext();
    appendLog("audio.playback_ready", { state: state.playbackContext?.state || "unknown" });
  } catch (error) {
    appendLog("audio.context_error", String(error));
    setBackendState("Audio Output Error", "warn");
    setOrbMode("", "Failed to initialize speaker playback.");
    setTextComposerLoading(false);
    restorePendingTextDraft();
    return false;
  }

  setServiceBusy(true);
  setBackendState("Starting Services", "warn");
  setOrbMode("", "Starting llama-server and realtime backend...");

  try {
    if (window.desktopApp.startVoiceServices) {
      const services = await window.desktopApp.startVoiceServices(state.ttsOptions);
      appendLog("services.ready", services);
      if (services.backend) {
        applyBackendConfig(services.backend);
      }
    }
    await refreshLoraCatalog().catch((error) => appendLog("tts.lora.catalog_error", String(error)));
  } catch (error) {
    appendLog("services.start_error", String(error));
    setBackendState("Service Start Failed", "warn");
    setOrbMode("", "Failed to start local services. Check the service log.");
    setServiceBusy(false);
    setTextComposerLoading(false);
    restorePendingTextDraft();
    return false;
  }

  if (withMicrophone) {
    setBackendState("Preparing Audio", "warn");
    setOrbMode("", "Services are ready. Requesting microphone access...");

    try {
      await ensureMicrophone();
    } catch (error) {
      appendLog("mic.error", String(error));
      setBackendState("Microphone Error", "warn");
      setOrbMode("", "Microphone access failed.");
      if (window.desktopApp.stopVoiceServices) {
        await window.desktopApp.stopVoiceServices().catch((stopError) => appendLog("services.stop_error", String(stopError)));
      }
      setServiceBusy(false);
      setTextComposerLoading(false);
      restorePendingTextDraft();
      return false;
    }
  }

  setBackendState("Opening WebSocket", "warn");
  setOrbMode("", withMicrophone ? "Microphone is ready. Opening realtime connection..." : "Services are ready. Opening realtime text session...");

  const ws = new WebSocket(state.backend.wsUrl);
  state.ws = ws;
  ws.binaryType = "arraybuffer";
  setTurnPhase("idle");

  ws.onopen = () => {
    setServiceBusy(false);
    setConnected(true);
    appendLog("ws.open", { url: state.backend.wsUrl });
    setTurnPhase(getReadyTurnPhase());
    setOrbMode("live", withMicrophone ? "Realtime session opened. Waiting for speech." : "Realtime text session opened.");
    applyTtsOptions(false);
    ws.send(JSON.stringify({
      type: "session.start",
      client: "electron-shell",
      tts_options: state.ttsOptions,
      logging_enabled: state.logCaptureEnabled
    }));
  };

  ws.onmessage = (event) => {
    try {
      const message = JSON.parse(event.data);
      handleRealtimeMessage(message);
    } catch (error) {
      appendLog("ws.parse_error", String(error));
    }
  };

  ws.onerror = () => {
    setServiceBusy(false);
    setTextComposerLoading(false);
    setBackendState("WebSocket Error", "warn");
    setOrbMode("", "WebSocket failed to connect.");
  };

  ws.onclose = async () => {
    appendLog("ws.close", { url: state.backend.wsUrl });
    state.ws = null;
    state.speechActive = false;
    state.sessionReady = false;
    state.sessionMode = "idle";
    state.textInputEnabled = false;
    state.textInterruptPending = false;
    setTextComposerLoading(false);
    restorePendingTextDraft();
    setTurnPhase("idle");
    setConnected(false);
    setBackendState("Backend Offline", "warn");
    setOrbMode("", "Realtime connection closed.");
    await releaseMicrophone();
    await releasePlayback();
  };
  return true;
}

async function closeRealtimeSocket() {
  if (state.ws) {
    const ws = state.ws;
    const closed = await new Promise((resolve) => {
      let settled = false;
      const finish = (value) => {
        if (!settled) {
          settled = true;
          clearTimeout(timer);
          resolve(value);
        }
      };
      const timer = setTimeout(() => finish(false), 3000);
      const originalOnClose = ws.onclose;
      ws.onclose = async (event) => {
        if (originalOnClose) {
          await originalOnClose.call(ws, event);
        }
        finish(true);
      };
      try {
        ws.send(JSON.stringify({ type: "session.stop" }));
      } catch (_) {
        // Ignore close-path send failures.
      }
      ws.close();
    });
    if (!closed && state.ws === ws) {
      appendLog("ws.close_timeout", { url: state.backend.wsUrl });
      state.ws = null;
      state.speechActive = false;
      state.sessionReady = false;
      state.sessionMode = "idle";
      state.textInputEnabled = false;
      state.textInterruptPending = false;
      setTextComposerLoading(false);
      restorePendingTextDraft();
      setTurnPhase("idle");
      setConnected(false);
      await releaseMicrophone();
      await releasePlayback();
    }
    return;
  }
  await releaseMicrophone();
  await releasePlayback();
}

async function disconnectRealtime() {
  if (state.serviceBusy) {
    return;
  }

  setServiceBusy(true);
  clearPendingTextRequest();
  state.sessionReady = false;
  state.sessionMode = "idle";
  state.textInputEnabled = false;
  setTextComposerLoading(false);
  setBackendState("Stopping Session", "warn");
  setOrbMode("", "Stopping realtime session and local services...");
  clearConversation();

  try {
    await closeRealtimeSocket();
    if (window.desktopApp.stopVoiceServices) {
      const services = await window.desktopApp.stopVoiceServices();
      appendLog("services.stopped", services);
    }
    setBackendState("Services Stopped", "warn");
    setOrbMode("", "Voice session stopped. Managed local services were cleaned up.");
  } catch (error) {
    appendLog("services.stop_error", String(error));
    setBackendState("Stop Failed", "warn");
    setOrbMode("", "Failed to stop local services cleanly.");
  } finally {
    setServiceBusy(false);
    setConnected(false);
  }
}

els.minBtn.addEventListener("click", () => window.desktopWindow.minimize());
els.maxBtn.addEventListener("click", () => window.desktopWindow.toggleMaximize());
els.closeBtn.addEventListener("click", () => window.desktopWindow.close());
// els.pingBtn.addEventListener("click", () => void pingBackend());
els.detectResourcesBtn.addEventListener("click", () => void detectResources());
els.connectBtn.addEventListener("click", connectRealtime);
els.disconnectBtn.addEventListener("click", () => void disconnectRealtime());
els.logCaptureToggle?.addEventListener("change", (event) => {
  void setVoiceLoggingEnabled(event.target.checked);
});
els.openServiceLogsBtn.addEventListener("click", () => {
  window.desktopApp.openVoiceServicesLogFolder().catch((error) => appendLog("services.open_log_folder_error", String(error)));
});
els.browseModelBtn.addEventListener("click", () => void browseForModel());
els.applyLlmBtn.addEventListener("click", () => void applyLlmConfig());
els.browseAudioBtn.addEventListener("click", () => void browseForAudio());
els.refreshLoraBtn.addEventListener("click", () => {
  refreshLoraCatalog().catch((error) => appendLog("tts.lora.catalog_error", String(error)));
});
els.applyTtsBtn.addEventListener("click", () => applyTtsOptions(true));
els.ttsLoraSelect.addEventListener("change", () => applyTtsOptions(false));
els.promptAudioPath.addEventListener("input", () => applyTtsOptions(false));
els.promptAudioPath.addEventListener("change", () => applyTtsOptions(false));
els.promptText.addEventListener("change", () => applyTtsOptions(false));
els.ttsCfgValue.addEventListener("change", () => applyTtsOptions(false));
els.ttsInferenceTimesteps.addEventListener("change", () => applyTtsOptions(false));
els.ttsSeed.addEventListener("change", () => applyTtsOptions(false));
for (const tab of els.toolTabs) {
  tab.addEventListener("click", () => setActiveToolPanel(tab.dataset.panelTarget));
}
els.textPromptInput.addEventListener("input", updateTextComposerState);
els.textPromptInput.addEventListener("compositionstart", () => {
  state.textInputComposing = true;
  updateTextComposerState();
});
els.textPromptInput.addEventListener("compositionend", () => {
  state.textInputComposing = false;
  updateTextComposerState();
});
els.textComposer.addEventListener("submit", handleTextComposerSubmit);

bootstrap().catch((error) => {
  appendLog("bootstrap.error", String(error));
});
