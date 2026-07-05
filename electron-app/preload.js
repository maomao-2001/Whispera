const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("desktopWindow", {
  minimize: () => ipcRenderer.invoke("window:minimize"),
  toggleMaximize: () => ipcRenderer.invoke("window:toggle-maximize"),
  close: () => ipcRenderer.invoke("window:close"),
  isMaximized: () => ipcRenderer.invoke("window:is-maximized")
});

contextBridge.exposeInMainWorld("desktopApp", {
  getInfo: () => ipcRenderer.invoke("app:get-info"),
  getBackendConfig: () => ipcRenderer.invoke("backend:get-config"),
  getResourceStatus: () => ipcRenderer.invoke("assets:get-resource-status"),
  getPathInfo: (targetPath) => ipcRenderer.invoke("path:get-info", targetPath),
  startVoiceServices: (ttsOptions) => ipcRenderer.invoke("voice-services:start", ttsOptions),
  stopVoiceServices: () => ipcRenderer.invoke("voice-services:stop"),
  getVoiceServicesStatus: () => ipcRenderer.invoke("voice-services:get-status"),
  getVoiceLoggingState: () => ipcRenderer.invoke("voice-logging:get-state"),
  setVoiceLoggingEnabled: (enabled) => ipcRenderer.invoke("voice-logging:set-enabled", enabled),
  openVoiceServicesLogFolder: () => ipcRenderer.invoke("voice-services:open-log-folder"),
  onVoiceServicesEvent: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("voice-services:event", listener);
    return () => ipcRenderer.removeListener("voice-services:event", listener);
  },
  showModelFileDialog: () => ipcRenderer.invoke("llm:show-model-file-dialog"),
  getLlmModelPath: () => ipcRenderer.invoke("llm:get-model-path"),
  setLlmModelPath: (modelPath) => ipcRenderer.invoke("llm:set-model-path", modelPath),
  getLlmModelInfo: (modelPath) => ipcRenderer.invoke("llm:get-model-info", modelPath),
  scanLoraLocal: () => ipcRenderer.invoke("lora:scan-local"),
  showAudioFileDialog: () => ipcRenderer.invoke("tts:show-audio-file-dialog"),
  getTtsConfig: () => ipcRenderer.invoke("tts:get-config"),
  saveTtsConfig: (config) => ipcRenderer.invoke("tts:save-config", config)
});
