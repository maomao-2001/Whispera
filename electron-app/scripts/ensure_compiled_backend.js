const fs = require("fs");
const path = require("path");

const repoRoot = path.resolve(__dirname, "..", "..");
const bundleRoot = path.join(repoRoot, "build", "compiled-backend");

const requiredChecks = [
  {
    label: "realtime wrapper",
    file: path.join(bundleRoot, "realtime", "app.py"),
  },
  {
    label: "llama launcher",
    file: path.join(bundleRoot, "llm-module", "scripts", "start_llama_server.py"),
  },
  {
    label: "realtime compiled module",
    dir: path.join(bundleRoot, "realtime"),
  },
  {
    label: "llm_module compiled module",
    dir: path.join(bundleRoot, "llm-module", "src", "llm_module"),
  },
  {
    label: "voxcpm source module",
    file: path.join(bundleRoot, "voxcpm-tts-streaming-module", "src", "voxcpm", "__init__.py"),
  },
];

const forbiddenCompiledDirs = [
  {
    label: "voxcpm source-only tree",
    dir: path.join(bundleRoot, "voxcpm-tts-streaming-module", "src", "voxcpm"),
  },
];

function hasCompiledModule(dir) {
  if (!fs.existsSync(dir)) {
    return false;
  }

  const stack = [dir];
  while (stack.length) {
    const current = stack.pop();
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const fullPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(fullPath);
        continue;
      }
      if (/\.(pyd|so)$/i.test(entry.name)) {
        return true;
      }
    }
  }
  return false;
}

const missing = [];
for (const check of requiredChecks) {
  if (check.file && !fs.existsSync(check.file)) {
    missing.push(`${check.label}: ${check.file}`);
    continue;
  }
  if (check.dir && !hasCompiledModule(check.dir)) {
    missing.push(`${check.label}: no compiled module found under ${check.dir}`);
  }
}

for (const check of forbiddenCompiledDirs) {
  if (hasCompiledModule(check.dir)) {
    missing.push(`${check.label}: unexpected compiled module found under ${check.dir}`);
  }
}

if (missing.length) {
  console.error("Compiled backend bundle is missing or incomplete.");
  console.error("Run `powershell -ExecutionPolicy Bypass -File .\\scripts\\build_compiled_backend.ps1` first.");
  for (const item of missing) {
    console.error(`- ${item}`);
  }
  process.exit(1);
}

console.log(`Compiled backend bundle found at ${bundleRoot}`);
