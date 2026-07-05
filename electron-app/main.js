const { app, BrowserWindow, ipcMain, shell, dialog } = require("electron");
const { spawn, spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const BACKEND_HTTP_BASE = process.env.MINIMIND_BACKEND_HTTP_BASE || "http://127.0.0.1:8011";
const BACKEND_WS_URL = process.env.MINIMIND_BACKEND_WS_URL || "ws://127.0.0.1:8011/ws/realtime";
const LLAMA_HTTP_BASE = process.env.MINIMIND_LLM_BASE_URL || "http://127.0.0.1:8080";
const MEMORY_ENABLED = !/^(0|false|no)$/i.test(process.env.MINIMIND_MEMORY_ENABLED || "0");
const MEMORY_EMBEDDER_HTTP_BASE = process.env.MINIMIND_MEMORY_EMBEDDER_BASE_URL || "http://127.0.0.1:8081";
const DEV_REPO_ROOT = path.resolve(__dirname, "..");
const PACKAGED_BUNDLE_ROOT = path.join(process.resourcesPath, "app-bundle");
const PACKAGED_APP_ROOT = path.dirname(process.execPath);
const REPO_ROOT = app.isPackaged ? PACKAGED_BUNDLE_ROOT : DEV_REPO_ROOT;
const APP_ROOT = app.isPackaged ? PACKAGED_APP_ROOT : DEV_REPO_ROOT;
const ASSETS_ROOT = process.env.MINIMIND_ASSETS_ROOT || path.join(APP_ROOT, "assets");
const LLM_MODULE_ROOT = path.join(REPO_ROOT, "llm-module");
const DEFAULT_TTS_MODULE_SRC = path.join(REPO_ROOT, "voxcpm-tts-streaming-module", "src");
const DEFAULT_LLAMA_SERVER_PATH = path.join(ASSETS_ROOT, "llama-bin", process.platform === "win32" ? "llama-server.exe" : "llama-server");
const DEFAULT_LLM_MODELS_ROOT = path.join(ASSETS_ROOT, "llm");
const DEFAULT_MEMORY_MODELS_ROOT = path.join(ASSETS_ROOT, "embedding");
const DEFAULT_ASR_MODEL_PATH = path.join(ASSETS_ROOT, "asr", "SenseVoiceSmall");
const DEFAULT_VAD_MODEL_PATH = path.join(REPO_ROOT, "model", "vad", "silero_vad.onnx");
const DEFAULT_TTS_MODEL_PATH = path.join(ASSETS_ROOT, "tts", "openbmb__VoxCPM1.5");
const configuredStartTimeoutMs = Number(process.env.MINIMIND_SERVICE_START_TIMEOUT_MS || 180000);
const DEFAULT_START_TIMEOUT_MS = Number.isFinite(configuredStartTimeoutMs) ? configuredStartTimeoutMs : 180000;
const WINDOW_ICON_PATH = path.join(__dirname, "build", "icon.ico");

const CONFIG_DIR = path.join(app.getPath("userData"), "config");
const CONFIG_FILE = path.join(CONFIG_DIR, "app-config.json");

const LORA_ROOT = process.env.MINIMIND_TTS_LORA_ROOT || path.join(ASSETS_ROOT, "lora");

const DEFAULT_TTS_CONFIG = {
  lora_selection: null,
  prompt_audio_path: null,
  prompt_text: null,
  cfg_value: 2.0,
  inference_timesteps: 10,
  seed: -1
};

function ensureConfigDir() {
  if (!fs.existsSync(CONFIG_DIR)) {
    fs.mkdirSync(CONFIG_DIR, { recursive: true });
  }
}

function loadConfig() {
  try {
    ensureConfigDir();
    if (fs.existsSync(CONFIG_FILE)) {
      const data = fs.readFileSync(CONFIG_FILE, "utf8");
      return JSON.parse(data);
    }
  } catch (error) {
    console.error("Failed to load config:", error);
  }
  return { llm: { modelPath: null }, tts: { ...DEFAULT_TTS_CONFIG } };
}

function saveConfig(config) {
  try {
    ensureConfigDir();
    fs.writeFileSync(CONFIG_FILE, JSON.stringify(config, null, 2), "utf8");
    return true;
  } catch (error) {
    console.error("Failed to save config:", error);
    return false;
  }
}

function loadLlmConfig() {
  const config = loadConfig();
  return config.llm || { modelPath: null };
}

function saveLlmConfig(llmConfig) {
  const config = loadConfig();
  config.llm = llmConfig;
  return saveConfig(config);
}

function loadTtsConfig() {
  const config = loadConfig();
  return config.tts || { ...DEFAULT_TTS_CONFIG };
}

function saveTtsConfig(ttsConfig) {
  const config = loadConfig();
  config.tts = ttsConfig;
  return saveConfig(config);
}

function pathExists(targetPath, kind = "any") {
  try {
    const stats = fs.statSync(targetPath);
    if (kind === "file") {
      return stats.isFile();
    }
    if (kind === "directory") {
      return stats.isDirectory();
    }
    return true;
  } catch (_) {
    return false;
  }
}

function listGgufModels(rootDir) {
  if (!pathExists(rootDir, "directory")) {
    return [];
  }

  const matches = [];
  const stack = [rootDir];

  while (stack.length) {
    const currentDir = stack.pop();
    let entries = [];
    try {
      entries = fs.readdirSync(currentDir, { withFileTypes: true });
    } catch (_) {
      continue;
    }

    for (const entry of entries) {
      const fullPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        stack.push(fullPath);
        continue;
      }
      if (entry.isFile() && entry.name.toLowerCase().endsWith(".gguf")) {
        matches.push(fullPath);
      }
    }
  }

  return matches.sort((a, b) => a.localeCompare(b));
}

function resolveLlmModelSelection() {
  const config = loadLlmConfig();
  const configuredPath = config.modelPath || null;
  const envPath = process.env.MINIMIND_LLM_MODEL || null;
  const ggufCandidates = listGgufModels(DEFAULT_LLM_MODELS_ROOT);

  if (configuredPath && pathExists(configuredPath, "file")) {
    return {
      configuredPath,
      resolvedPath: configuredPath,
      source: "config",
      candidates: ggufCandidates
    };
  }

  if (envPath && pathExists(envPath, "file")) {
    return {
      configuredPath,
      resolvedPath: envPath,
      source: "env",
      candidates: ggufCandidates
    };
  }

  if (ggufCandidates.length === 1) {
    return {
      configuredPath,
      resolvedPath: ggufCandidates[0],
      source: "assets.auto",
      candidates: ggufCandidates
    };
  }

  return {
    configuredPath,
    resolvedPath: null,
    source: ggufCandidates.length > 1 ? "assets.multiple" : "missing",
    candidates: ggufCandidates
  };
}

function resolveMemoryModelSelection() {
  const envPath = process.env.MINIMIND_MEMORY_EMBEDDER_MODEL_PATH || null;
  const ggufCandidates = listGgufModels(DEFAULT_MEMORY_MODELS_ROOT);

  if (envPath && pathExists(envPath, "file")) {
    return {
      resolvedPath: envPath,
      source: "env",
      candidates: ggufCandidates
    };
  }

  if (ggufCandidates.length === 1) {
    return {
      resolvedPath: ggufCandidates[0],
      source: "assets.auto",
      candidates: ggufCandidates
    };
  }

  return {
    resolvedPath: null,
    source: ggufCandidates.length > 1 ? "assets.multiple" : "missing",
    candidates: ggufCandidates
  };
}

function getResourceStatus() {
  const llmSelection = resolveLlmModelSelection();
  const memorySelection = resolveMemoryModelSelection();
  const llamaServerPath = process.env.MINIMIND_LLAMA_SERVER || DEFAULT_LLAMA_SERVER_PATH;
  const asrModelPath = process.env.MINIMIND_ASR_MODEL_PATH || DEFAULT_ASR_MODEL_PATH;
  const vadModelPath = process.env.MINIMIND_VAD_MODEL_PATH || DEFAULT_VAD_MODEL_PATH;
  const ttsModelPath = process.env.MINIMIND_TTS_MODEL_PATH || DEFAULT_TTS_MODEL_PATH;

  return {
    appRoot: APP_ROOT,
    assetsRoot: ASSETS_ROOT,
    llama: {
      path: llamaServerPath,
      exists: pathExists(llamaServerPath, "file")
    },
    llm: {
      directory: DEFAULT_LLM_MODELS_ROOT,
      configuredPath: llmSelection.configuredPath,
      resolvedPath: llmSelection.resolvedPath,
      exists: Boolean(llmSelection.resolvedPath && pathExists(llmSelection.resolvedPath, "file")),
      source: llmSelection.source,
      candidateCount: llmSelection.candidates.length
    },
    memory: {
      enabled: MEMORY_ENABLED,
      directory: DEFAULT_MEMORY_MODELS_ROOT,
      embedderHttpBase: MEMORY_EMBEDDER_HTTP_BASE,
      resolvedPath: memorySelection.resolvedPath,
      exists: !MEMORY_ENABLED || Boolean(memorySelection.resolvedPath && pathExists(memorySelection.resolvedPath, "file")),
      source: memorySelection.source,
      candidateCount: memorySelection.candidates.length
    },
    asr: {
      path: asrModelPath,
      exists: pathExists(asrModelPath, "directory")
    },
    vad: {
      path: vadModelPath,
      exists: pathExists(vadModelPath, "file")
    },
    tts: {
      path: ttsModelPath,
      exists: pathExists(ttsModelPath, "directory")
    },
    lora: {
      path: LORA_ROOT,
      exists: pathExists(LORA_ROOT, "directory")
    },
    summary: {
      llmReady: pathExists(llamaServerPath, "file") && Boolean(llmSelection.resolvedPath && pathExists(llmSelection.resolvedPath, "file")),
      memoryReady: !MEMORY_ENABLED || Boolean(memorySelection.resolvedPath && pathExists(memorySelection.resolvedPath, "file")),
      asrReady: pathExists(asrModelPath, "directory"),
      vadReady: pathExists(vadModelPath, "file"),
      ttsReady: pathExists(ttsModelPath, "directory")
    }
  };
}

function getPathInfo(targetPath) {
  if (!targetPath) {
    return {
      exists: false,
      type: null,
      path: null,
    };
  }

  try {
    const resolvedPath = path.resolve(String(targetPath));
    const stats = fs.statSync(resolvedPath);
    return {
      exists: true,
      type: stats.isDirectory() ? "directory" : stats.isFile() ? "file" : "other",
      path: resolvedPath,
      sizeBytes: stats.size,
    };
  } catch (error) {
    return {
      exists: false,
      type: null,
      path: path.resolve(String(targetPath)),
      error: error.message,
    };
  }
}

function scanLoraDirectory() {
  const loraRoot = path.resolve(LORA_ROOT);
  if (!fs.existsSync(loraRoot)) {
    return [];
  }

  const checkpoints = [];

  function scanDir(dir, relativePath = "") {
    try {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        if (!entry.isDirectory()) continue;

        const fullPath = path.join(dir, entry.name);
        const relPath = relativePath ? `${relativePath}/${entry.name}` : entry.name;

        const hasSafetensors = fs.existsSync(path.join(fullPath, "lora_weights.safetensors"));
        const hasCkpt = fs.existsSync(path.join(fullPath, "lora_weights.ckpt"));

        if (hasSafetensors || hasCkpt) {
          let baseModel = null;
          const configPath = path.join(fullPath, "lora_config.json");
          if (fs.existsSync(configPath)) {
            try {
              const configData = JSON.parse(fs.readFileSync(configPath, "utf8"));
              baseModel = configData.base_model || null;
            } catch (e) {
              // Ignore config read errors
            }
          }
          checkpoints.push({
            path: relPath,
            label: relPath,
            base_model: baseModel
          });
        }

        scanDir(fullPath, relPath);
      }
    } catch (e) {
      // Ignore directory read errors
    }
  }

  scanDir(loraRoot);
  return checkpoints.sort((a, b) => (b.path || "").localeCompare(a.path || ""));
}

let mainWindow = null;
let serviceStartPromise = null;
let serviceStopPromise = null;

const managedServices = {
  llama: null,
  memory: null,
  realtime: null
};
const serviceLogFiles = {
  llama: null,
  memory: null,
  realtime: null
};
const serviceLogStreams = {
  llama: null,
  memory: null,
  realtime: null
};
const serviceLaunchInfo = {
  llama: null,
  memory: null,
  realtime: null
};
const serviceExits = {
  llama: null,
  memory: null,
  realtime: null
};
const serviceLogs = {
  llama: [],
  memory: [],
  realtime: []
};
let runtimeLoggingEnabled = false;

function parseHttpEndpoint(url, fallbackHost, fallbackPort) {
  try {
    const parsed = new URL(url);
    return {
      host: parsed.hostname || fallbackHost,
      port: Number(parsed.port || fallbackPort),
      origin: parsed.origin
    };
  } catch (_) {
    return { host: fallbackHost, port: fallbackPort, origin: `http://${fallbackHost}:${fallbackPort}` };
  }
}

const llamaEndpoint = parseHttpEndpoint(LLAMA_HTTP_BASE, "127.0.0.1", 8080);
const memoryEndpoint = parseHttpEndpoint(MEMORY_EMBEDDER_HTTP_BASE, "127.0.0.1", 8081);
const backendEndpoint = parseHttpEndpoint(BACKEND_HTTP_BASE, "127.0.0.1", 8011);

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function getServiceLogDir() {
  const logDir = path.join(app.getPath("userData"), "service-logs");
  if (!fs.existsSync(logDir)) {
    fs.mkdirSync(logDir, { recursive: true });
  }
  return logDir;
}

function getTurnCaptureDir() {
  const captureDir = path.join(getServiceLogDir(), "session-captures");
  if (!fs.existsSync(captureDir)) {
    fs.mkdirSync(captureDir, { recursive: true });
  }
  return captureDir;
}

function makeLogTimestamp() {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

function closeServiceLog(name) {
  if (serviceLogStreams[name]) {
    serviceLogStreams[name].end();
    serviceLogStreams[name] = null;
  }
}

function openServiceLog(name, command, args, cwd) {
  closeServiceLog(name);
  const logFile = path.join(getServiceLogDir(), `${makeLogTimestamp()}-${name}.log`);
  const stream = fs.createWriteStream(logFile, { flags: "a", encoding: "utf8" });
  serviceLogFiles[name] = logFile;
  serviceLogStreams[name] = stream;
  stream.write(`[${new Date().toISOString()}] ${name} start\n`);
  stream.write(`cwd: ${cwd}\n`);
  stream.write(`command: ${command} ${args.join(" ")}\n\n`);
  return logFile;
}

function ensureServiceLog(name) {
  if (!runtimeLoggingEnabled) {
    return null;
  }
  if (serviceLogStreams[name]) {
    return serviceLogFiles[name];
  }
  const launchInfo = serviceLaunchInfo[name];
  if (!launchInfo) {
    return null;
  }
  return openServiceLog(name, launchInfo.command, launchInfo.args, launchInfo.cwd);
}

function writeServiceLog(name, level, text) {
  if (!runtimeLoggingEnabled || !text) {
    return;
  }
  ensureServiceLog(name);
  const stream = serviceLogStreams[name];
  if (!stream) {
    return;
  }
  for (const line of text.replace(/\r\n/g, "\n").split("\n")) {
    if (line) {
      stream.write(`[${new Date().toISOString()}] [${level}] ${line}\n`);
    }
  }
}

function sendServiceEvent(type, payload = {}) {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return;
  }
  mainWindow.webContents.send("voice-services:event", {
    type,
    at: new Date().toISOString(),
    ...payload
  });
}

function getVoiceLoggingState() {
  return {
    enabled: runtimeLoggingEnabled,
    logsDir: getServiceLogDir(),
    turnCaptureDir: getTurnCaptureDir(),
  };
}

function setVoiceLoggingEnabled(enabled) {
  runtimeLoggingEnabled = Boolean(enabled);
  if (!runtimeLoggingEnabled) {
    closeServiceLog("llama");
    closeServiceLog("memory");
    closeServiceLog("realtime");
  } else {
    if (isChildRunning(managedServices.llama)) {
      ensureServiceLog("llama");
    }
    if (isChildRunning(managedServices.memory)) {
      ensureServiceLog("memory");
    }
    if (isChildRunning(managedServices.realtime)) {
      ensureServiceLog("realtime");
    }
  }
  sendServiceEvent("logging.capture_changed", getVoiceLoggingState());
  return getVoiceLoggingState();
}

function resolvePythonExecutable() {
  if (process.env.MINIMIND_PYTHON) {
    return process.env.MINIMIND_PYTHON;
  }

  const bundledPython = process.platform === "win32"
    ? path.join(REPO_ROOT, "runtime", "python", "python.exe")
    : path.join(REPO_ROOT, "runtime", "python", "bin", "python3");
  if (fs.existsSync(bundledPython)) {
    return bundledPython;
  }

  if (process.env.CONDA_PREFIX) {
    const condaPython = path.join(process.env.CONDA_PREFIX, process.platform === "win32" ? "python.exe" : "bin/python");
    if (fs.existsSync(condaPython)) {
      return condaPython;
    }
  }

  return process.env.PYTHON || "python";
}

function checkMem0RuntimeReady(pythonExecutable) {
  if (!MEMORY_ENABLED) {
    return;
  }

  const result = spawnSync(
    pythonExecutable,
    [
      "-X",
      "utf8",
      "-c",
      "from mem0 import Memory; import openai, posthog, qdrant_client, sqlalchemy; print(Memory.__name__)"
    ],
    {
      cwd: REPO_ROOT,
      env: {
        ...process.env,
        PYTHONUTF8: "1",
        PYTHONIOENCODING: "utf-8",
      },
      windowsHide: true,
      encoding: "utf8",
    }
  );

  if (result.error) {
    throw new Error(`Memory dependency check failed for ${pythonExecutable}: ${result.error.message}`);
  }

  if (result.status !== 0) {
    const detail = String(result.stderr || result.stdout || `exit code ${result.status}`).trim();
    throw new Error(
      `Memory is enabled but mem0 dependencies are unavailable in ${pythonExecutable}. ` +
      `Run: ${pythonExecutable} -m pip install -r requirements-mem0.txt. ` +
      `Details: ${detail.slice(-500)}`
    );
  }
}

function isChildRunning(child) {
  return Boolean(child && child.exitCode === null && !child.killed);
}

function attachProcessEvents(name, child) {
  const pipeLog = (stream, level) => {
    stream.on("data", (chunk) => {
      const rawText = chunk.toString("utf8");
      writeServiceLog(name, level, rawText);
      const text = rawText.trim();
      if (text) {
        serviceLogs[name].push(`[${level}] ${text}`);
        serviceLogs[name] = serviceLogs[name].slice(-30);
        if (runtimeLoggingEnabled) {
          sendServiceEvent("process.output", {
            service: name,
            level,
            text: text.slice(-4000),
            logFile: serviceLogFiles[name]
          });
        }
      }
    });
  };

  pipeLog(child.stdout, "stdout");
  pipeLog(child.stderr, "stderr");

  child.on("exit", (code, signal) => {
    serviceExits[name] = {
      code,
      signal,
      logs: [...serviceLogs[name]]
    };
    sendServiceEvent("process.exit", { service: name, code, signal });
    closeServiceLog(name);
    if (managedServices[name] === child) {
      managedServices[name] = null;
    }
  });

  child.on("error", (error) => {
    serviceExits[name] = {
      code: null,
      signal: null,
      error: error.message,
      logs: [...serviceLogs[name]]
    };
    sendServiceEvent("process.error", { service: name, message: error.message });
    closeServiceLog(name);
  });
}

async function fetchJson(url, timeoutMs = 3000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) {
      return { ok: false, status: response.status, data: null };
    }
    return { ok: true, status: response.status, data: await response.json() };
  } catch (error) {
    return { ok: false, status: 0, data: null, error: error.message };
  } finally {
    clearTimeout(timeout);
  }
}

async function postJson(url, payload, timeoutMs = DEFAULT_START_TIMEOUT_MS) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload ?? {}),
      signal: controller.signal
    });
    let data = null;
    try {
      data = await response.json();
    } catch (_) {
      data = null;
    }
    return { ok: response.ok, status: response.status, data };
  } catch (error) {
    return { ok: false, status: 0, data: null, error: error.message };
  } finally {
    clearTimeout(timeout);
  }
}

async function checkLlamaReady() {
  const result = await fetchJson(`${llamaEndpoint.origin}/v1/models`);
  return result.ok ? result : null;
}

async function checkMemoryReady() {
  if (!MEMORY_ENABLED) {
    return { ok: true, status: 200, data: { disabled: true } };
  }
  const result = await fetchJson(`${memoryEndpoint.origin}/v1/models`);
  return result.ok ? result : null;
}

async function checkRealtimeReady() {
  const result = await fetchJson(`${backendEndpoint.origin}/health`);
  if (!result.ok) {
    return null;
  }
  if (result.data && result.data.upstream_ok === false) {
    return null;
  }
  return result;
}

async function warmupRealtimeModels(ttsOptions = null) {
  sendServiceEvent("warmup.begin", { service: "realtime", stage: "models", url: `${backendEndpoint.origin}/warmup` });
  const result = await postJson(`${backendEndpoint.origin}/warmup`, {
    force: false,
    tts_options: ttsOptions || undefined,
  });
  if (!result.ok || !result.data || result.data.ok === false) {
    const detail = result.data && (result.data.detail || result.data.error);
    const errorMessage = detail || result.error || `warmup failed with status ${result.status}`;
    sendServiceEvent("warmup.error", { service: "realtime", stage: "models", message: errorMessage });
    throw new Error(errorMessage);
  }
  sendServiceEvent("warmup.completed", {
    service: "realtime",
    stage: "models",
    cached: Boolean(result.data.cached),
    asr_model_load_ms: result.data.asr_model_load_ms,
    asr_inference_ms: result.data.asr_inference_ms,
    tts_model_load_ms: result.data.tts_model_load_ms,
    tts_inference_ms: result.data.tts_inference_ms,
    total_ms: result.data.total_ms,
  });
  return result.data;
}

function formatRecentServiceLogs(name) {
  const logs = serviceLogs[name] || [];
  if (!logs.length) {
    return "";
  }
  return ` Recent ${name} logs: ${logs.slice(-8).join(" | ")}`;
}

async function waitForReady(service, name, checkReady, childGetter, timeoutMs = DEFAULT_START_TIMEOUT_MS) {
  const startedAt = Date.now();
  let lastError = null;

  while (Date.now() - startedAt < timeoutMs) {
    const exit = serviceExits[name];
    if (exit) {
      const exitDetail = exit.error || `code ${exit.code}${exit.signal ? `, signal ${exit.signal}` : ""}`;
      throw new Error(`${service} exited early with ${exitDetail}.${formatRecentServiceLogs(name)}`);
    }

    const child = childGetter();
    if (child && child.exitCode !== null) {
      throw new Error(`${service} exited early with code ${child.exitCode}.${formatRecentServiceLogs(name)}`);
    }

    const result = await checkReady();
    if (result) {
      return result;
    }

    lastError = `${service} is not ready yet`;
    await delay(1000);
  }

  throw new Error(`${lastError || service} after ${Math.round(timeoutMs / 1000)}s`);
}

function spawnManagedService(name, command, args, cwd) {
  serviceExits[name] = null;
  serviceLogs[name] = [];
  serviceLaunchInfo[name] = { command, args: [...args], cwd };
  const logFile = runtimeLoggingEnabled ? openServiceLog(name, command, args, cwd) : null;
  sendServiceEvent("process.start", { service: name, command, args, cwd, logFile });
  const childEnv = {
    ...process.env,
    PYTHONUTF8: "1",
    PYTHONIOENCODING: "utf-8",
  };
  const child = spawn(command, args, {
    cwd,
    env: childEnv,
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"]
  });
  managedServices[name] = child;
  attachProcessEvents(name, child);
  return child;
}

function buildPythonInvocation(pythonExecutable, moduleArgs) {
  return {
    command: pythonExecutable,
    args: ["-X", "utf8", ...moduleArgs],
  };
}

function buildLlamaArgs(pythonExecutable) {
  const args = ["scripts/start_llama_server.py", "--host", llamaEndpoint.host, "--port", String(llamaEndpoint.port)];
  const llamaServerPath = process.env.MINIMIND_LLAMA_SERVER || DEFAULT_LLAMA_SERVER_PATH;
  if (!pathExists(llamaServerPath, "file")) {
    throw new Error(`llama-server executable not found: ${llamaServerPath}`);
  }
  args.push("--server", llamaServerPath);

  const llmSelection = resolveLlmModelSelection();
  const modelPath = llmSelection.resolvedPath;
  if (modelPath) {
    args.push("--model", modelPath);
  } else {
    throw new Error(`No GGUF model is available. Put one in ${DEFAULT_LLM_MODELS_ROOT} or choose a local GGUF file in Settings.`);
  }

  const memoryInferEnabled = !/^(0|false|no)$/i.test(process.env.MINIMIND_MEMORY_INFER || "1");
  const defaultLlamaCtxSize = MEMORY_ENABLED && memoryInferEnabled ? "16384" : "8192";
  args.push("--ctx-size", process.env.MINIMIND_LLAMA_CTX_SIZE || defaultLlamaCtxSize);
  args.push("--n-gpu-layers", process.env.MINIMIND_LLAMA_N_GPU_LAYERS || "99");

  return buildPythonInvocation(pythonExecutable, args);
}

function buildMemoryEmbeddingArgs(pythonExecutable) {
  const args = ["scripts/start_llama_server.py", "--host", memoryEndpoint.host, "--port", String(memoryEndpoint.port)];
  const llamaServerPath = process.env.MINIMIND_LLAMA_SERVER || DEFAULT_LLAMA_SERVER_PATH;
  if (!pathExists(llamaServerPath, "file")) {
    throw new Error(`llama-server executable not found: ${llamaServerPath}`);
  }
  args.push("--server", llamaServerPath);

  const memorySelection = resolveMemoryModelSelection();
  const modelPath = memorySelection.resolvedPath;
  if (modelPath) {
    args.push("--model", modelPath);
  } else {
    throw new Error(`Memory is enabled but no embedding GGUF model is available. Set MINIMIND_MEMORY_EMBEDDER_MODEL_PATH or put one GGUF in ${DEFAULT_MEMORY_MODELS_ROOT}.`);
  }

  args.push("--ctx-size", process.env.MINIMIND_MEMORY_CTX_SIZE || "512");
  args.push("--n-gpu-layers", process.env.MINIMIND_MEMORY_N_GPU_LAYERS || "99");
  args.push("--extra", "--embedding", "--pooling", process.env.MINIMIND_MEMORY_POOLING || "mean");

  return buildPythonInvocation(pythonExecutable, args);
}

function buildRealtimeArgs(pythonExecutable) {
  const asrModelPath = process.env.MINIMIND_ASR_MODEL_PATH || DEFAULT_ASR_MODEL_PATH;
  if (!pathExists(asrModelPath, "directory")) {
    throw new Error(`ASR model directory not found: ${asrModelPath}`);
  }
  const vadModelPath = process.env.MINIMIND_VAD_MODEL_PATH || DEFAULT_VAD_MODEL_PATH;
  if (!pathExists(vadModelPath, "file")) {
    throw new Error(`VAD model file not found: ${vadModelPath}`);
  }

  const args = [
    "-m",
    "realtime.app",
    "--host",
    backendEndpoint.host,
    "--port",
    String(backendEndpoint.port),
    "--asr-model-path",
    asrModelPath,
    "--asr-device",
    process.env.MINIMIND_ASR_DEVICE || "cuda",
    "--vad-path",
    vadModelPath,
    "--llm-base-url",
    llamaEndpoint.origin
  ];

  const disableTts = /^(1|true|yes)$/i.test(process.env.MINIMIND_DISABLE_TTS || "");
  const requestedTtsModelPath = process.env.MINIMIND_TTS_MODEL_PATH || DEFAULT_TTS_MODEL_PATH;
  const hasBundledTtsStack = fs.existsSync(DEFAULT_TTS_MODULE_SRC) && fs.existsSync(requestedTtsModelPath);
  const enableTts = /^(1|true|yes)$/i.test(process.env.MINIMIND_ENABLE_TTS || "") || hasBundledTtsStack;

  if (disableTts || !enableTts) {
    args.push("--disable-tts");
  } else {
    args.push("--enable-tts");
    if (requestedTtsModelPath && pathExists(requestedTtsModelPath, "directory")) {
      args.push("--tts-model-path", requestedTtsModelPath);
    }
    if (pathExists(LORA_ROOT, "directory")) {
      args.push("--tts-lora-root", LORA_ROOT);
    }
  }
  args.push("--debug-output-dir", getTurnCaptureDir());
  if (/^(1|true|yes)$/i.test(process.env.MINIMIND_DEBUG_TURNS || "")) {
    args.push("--debug-turns");
  }
  if (process.env.MINIMIND_DEBUG_OUTPUT_DIR) {
    args.push("--debug-output-dir", process.env.MINIMIND_DEBUG_OUTPUT_DIR);
  }
  if (MEMORY_ENABLED) {
    args.push("--enable-memory");
  }

  return buildPythonInvocation(pythonExecutable, args);
}

async function startVoiceServices(ttsOptions = null) {
  if (serviceStartPromise) {
    return serviceStartPromise;
  }

  serviceStartPromise = (async () => {
    try {
      const pythonExecutable = resolvePythonExecutable();
      checkMem0RuntimeReady(pythonExecutable);
      sendServiceEvent("start.begin", { python: pythonExecutable });

      let llamaReady = await checkLlamaReady();
      if (llamaReady) {
        sendServiceEvent("service.ready", { service: "llama", mode: "external", url: llamaEndpoint.origin });
      } else {
        if (!isChildRunning(managedServices.llama)) {
          const { command, args } = buildLlamaArgs(pythonExecutable);
          spawnManagedService("llama", command, args, LLM_MODULE_ROOT);
        }
        llamaReady = await waitForReady("llama-server", "llama", checkLlamaReady, () => managedServices.llama);
        sendServiceEvent("service.ready", { service: "llama", mode: "managed", url: llamaEndpoint.origin });
      }

      if (MEMORY_ENABLED) {
        let memoryReady = await checkMemoryReady();
        if (memoryReady) {
          sendServiceEvent("service.ready", { service: "memory", mode: "external", url: memoryEndpoint.origin });
        } else {
          if (!isChildRunning(managedServices.memory)) {
            const { command, args } = buildMemoryEmbeddingArgs(pythonExecutable);
            spawnManagedService("memory", command, args, LLM_MODULE_ROOT);
          }
          memoryReady = await waitForReady("memory embedding server", "memory", checkMemoryReady, () => managedServices.memory);
          sendServiceEvent("service.ready", { service: "memory", mode: "managed", url: memoryEndpoint.origin });
        }
      }

      let realtimeReady = await checkRealtimeReady();
      if (realtimeReady) {
        sendServiceEvent("service.ready", { service: "realtime", mode: "external", url: backendEndpoint.origin });
      } else {
        if (!isChildRunning(managedServices.realtime)) {
          const { command, args } = buildRealtimeArgs(pythonExecutable);
          spawnManagedService("realtime", command, args, REPO_ROOT);
        }
        realtimeReady = await waitForReady("realtime backend", "realtime", checkRealtimeReady, () => managedServices.realtime);
        sendServiceEvent("service.ready", { service: "realtime", mode: "managed", url: backendEndpoint.origin });
      }

      await warmupRealtimeModels(ttsOptions);

      const status = getVoiceServicesStatus();
      sendServiceEvent("start.completed", status);
      return status;
    } catch (error) {
      sendServiceEvent("start.error", { message: error.message });
      await stopVoiceServices().catch((stopError) => sendServiceEvent("stop.error", { message: stopError.message }));
      throw error;
    }
  })();

  try {
    return await serviceStartPromise;
  } finally {
    serviceStartPromise = null;
  }
}

function waitForChildExit(child, timeoutMs = 10000) {
  if (!child || child.exitCode !== null) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, timeoutMs);
    child.once("exit", () => {
      clearTimeout(timer);
      resolve();
    });
  });
}

async function killProcessTree(name, child) {
  if (!isChildRunning(child)) {
    return;
  }

  sendServiceEvent("process.stop", { service: name, pid: child.pid });
  if (process.platform === "win32") {
    const killer = spawn("taskkill", ["/PID", String(child.pid), "/T", "/F"], {
      windowsHide: true,
      stdio: "ignore"
    });
    await new Promise((resolve) => killer.once("exit", resolve));
  } else {
    child.kill("SIGTERM");
  }

  await waitForChildExit(child);
  closeServiceLog(name);
}

async function stopVoiceServices() {
  if (serviceStopPromise) {
    return serviceStopPromise;
  }

  serviceStopPromise = (async () => {
    sendServiceEvent("stop.begin");
    await killProcessTree("realtime", managedServices.realtime);
    managedServices.realtime = null;
    await killProcessTree("memory", managedServices.memory);
    managedServices.memory = null;
    await killProcessTree("llama", managedServices.llama);
    managedServices.llama = null;
    const status = getVoiceServicesStatus();
    sendServiceEvent("stop.completed", status);
    return status;
  })();

  try {
    return await serviceStopPromise;
  } finally {
    serviceStopPromise = null;
  }
}

function getVoiceServicesStatus() {
  return {
    backend: {
      httpBase: BACKEND_HTTP_BASE,
      wsUrl: BACKEND_WS_URL
    },
    llm: {
      httpBase: llamaEndpoint.origin,
      managed: isChildRunning(managedServices.llama),
      pid: managedServices.llama ? managedServices.llama.pid : null,
      logFile: serviceLogFiles.llama
    },
    memory: {
      enabled: MEMORY_ENABLED,
      httpBase: memoryEndpoint.origin,
      managed: isChildRunning(managedServices.memory),
      pid: managedServices.memory ? managedServices.memory.pid : null,
      logFile: serviceLogFiles.memory
    },
    realtime: {
      httpBase: backendEndpoint.origin,
      managed: isChildRunning(managedServices.realtime),
      pid: managedServices.realtime ? managedServices.realtime.pid : null,
      logFile: serviceLogFiles.realtime
    },
    logs: {
      dir: getServiceLogDir()
    }
  };
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1180,
    minHeight: 760,
    frame: false,
    titleBarStyle: "hidden",
    backgroundColor: "#121417",
    icon: pathExists(WINDOW_ICON_PATH, "file") ? WINDOW_ICON_PATH : undefined,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  });

  mainWindow.once("ready-to-show", () => {
    mainWindow.show();
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });

  mainWindow.loadFile(path.join(__dirname, "renderer", "index.html"));
}

app.whenReady().then(() => {
  ipcMain.handle("window:minimize", () => {
    if (mainWindow) {
      mainWindow.minimize();
    }
  });

  ipcMain.handle("window:toggle-maximize", () => {
    if (!mainWindow) {
      return false;
    }
    if (mainWindow.isMaximized()) {
      mainWindow.unmaximize();
      return false;
    }
    mainWindow.maximize();
    return true;
  });

  ipcMain.handle("window:close", () => {
    if (mainWindow) {
      mainWindow.close();
    }
  });

  ipcMain.handle("window:is-maximized", () => {
    return mainWindow ? mainWindow.isMaximized() : false;
  });

  ipcMain.handle("app:get-info", () => {
    return {
      name: app.getName(),
      version: app.getVersion()
    };
  });

  ipcMain.handle("backend:get-config", () => {
    return {
      httpBase: BACKEND_HTTP_BASE,
      wsUrl: BACKEND_WS_URL
    };
  });
  ipcMain.handle("assets:get-resource-status", () => getResourceStatus());
  ipcMain.handle("path:get-info", (_event, targetPath) => getPathInfo(targetPath));

  ipcMain.handle("voice-services:start", (_event, ttsOptions) => startVoiceServices(ttsOptions));
  ipcMain.handle("voice-services:stop", () => stopVoiceServices());
  ipcMain.handle("voice-services:get-status", () => getVoiceServicesStatus());
  ipcMain.handle("voice-logging:get-state", () => getVoiceLoggingState());
  ipcMain.handle("voice-logging:set-enabled", (_event, enabled) => setVoiceLoggingEnabled(enabled));
  ipcMain.handle("voice-services:open-log-folder", () => shell.openPath(getServiceLogDir()));

  ipcMain.handle("llm:show-model-file-dialog", async () => {
    if (!mainWindow) {
      return { canceled: true, filePaths: [] };
    }
    const result = await dialog.showOpenDialog(mainWindow, {
      title: "选择 LLM 模型文件",
      filters: [
        { name: "GGUF 模型", extensions: ["gguf"] },
        { name: "所有文件", extensions: ["*"] }
      ],
      properties: ["openFile"]
    });
    return result;
  });

  ipcMain.handle("llm:get-model-path", () => {
    const config = loadLlmConfig();
    return config.modelPath || null;
  });

  ipcMain.handle("llm:set-model-path", (_event, modelPath) => {
    const config = loadLlmConfig();
    config.modelPath = modelPath || null;
    return saveLlmConfig(config);
  });

  ipcMain.handle("llm:get-model-info", (_event, modelPath) => {
    if (!modelPath || !fs.existsSync(modelPath)) {
      return { exists: false, name: null, size: null };
    }
    try {
      const stats = fs.statSync(modelPath);
      const name = path.basename(modelPath);
      const sizeMB = Math.round(stats.size / (1024 * 1024));
      const sizeGB = (stats.size / (1024 * 1024 * 1024)).toFixed(2);
      return {
        exists: true,
        name,
        size: sizeMB >= 1024 ? `${sizeGB} GB` : `${sizeMB} MB`,
        sizeBytes: stats.size
      };
    } catch (error) {
      return { exists: false, name: null, size: null, error: error.message };
    }
  });

  ipcMain.handle("lora:scan-local", () => {
    try {
      return { models: scanLoraDirectory() };
    } catch (error) {
      console.error("Failed to scan LoRA directory:", error);
      return { models: [], error: error.message };
    }
  });

  ipcMain.handle("tts:show-audio-file-dialog", async () => {
    if (!mainWindow) {
      return { canceled: true, filePaths: [] };
    }
    const result = await dialog.showOpenDialog(mainWindow, {
      title: "选择参考音频文件",
      filters: [
        { name: "音频文件", extensions: ["wav", "mp3", "flac", "ogg", "m4a", "aac"] },
        { name: "所有文件", extensions: ["*"] }
      ],
      properties: ["openFile"]
    });
    return result;
  });

  ipcMain.handle("tts:get-config", () => {
    return loadTtsConfig();
  });

  ipcMain.handle("tts:save-config", (_event, ttsConfig) => {
    return saveTtsConfig(ttsConfig);
  });

  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

let quitAfterCleanup = false;

app.on("before-quit", (event) => {
  if (quitAfterCleanup) {
    return;
  }
  event.preventDefault();
  quitAfterCleanup = true;
  stopVoiceServices()
    .catch((error) => sendServiceEvent("stop.error", { message: error.message }))
    .finally(() => app.exit(0));
});
