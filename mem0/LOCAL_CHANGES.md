# Local Notes

This directory vendors the upstream `mem0` project into current project.

- Upstream repository: `https://github.com/mem0ai/mem0`
- Imported from commit: `31cec11a790868f88c9acafb8b70eb25071f2150`

## Local changes

Only the following file is intentionally changed relative to upstream:

- `mem0/__init__.py`
  - Wrap `importlib.metadata.version("mem0ai")` in a `try/except`
  - Fall back to `"0.0.0-local"` when package metadata is unavailable

## Update note

When syncing a newer upstream version, check whether `mem0/__init__.py`
changed upstream before reapplying the local fallback.

## Converting to a vendored directory

Use this order when removing the nested `mem0/.git` repository and keeping
`mem0/` as part of the main repo:

1. Confirm local changes inside `mem0/` are the ones you want to keep.
2. Record the upstream commit and local patch notes in this file.
3. Optionally back up `mem0/.git` to `.nested-git-backups/mem0.git/`.
4. Delete `mem0/.git`.
5. Run `git status -- mem0` from the main repo and confirm `mem0/` now shows
   up as normal project files.
6. Add and commit `mem0/` from the main repo.

Suggested PowerShell commands:

```powershell
Copy-Item mem0\.git .nested-git-backups\mem0.git -Recurse
Remove-Item mem0\.git -Recurse -Force
git status -- mem0
git add mem0
```
