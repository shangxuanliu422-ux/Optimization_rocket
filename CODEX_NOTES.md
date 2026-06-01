# Codex Notes

This project has a few local-environment quirks that future Codex sessions should follow.

- Prefer the project virtual environment for Python:
  `C:\Users\james\Desktop\Python_project\Optimization_rocket\.venv\Scripts\python.exe`
- Do not rely on the system Python. It may miss packages such as `matplotlib`.
- The user normally uses Windows `cmd`, not PowerShell. Prefer `cmd`-style commands in explanations and examples.
- Ordinary PowerShell shell calls may fail with `windows sandbox: spawn setup refresh`.
  If that happens, use the Node REPL MCP or the project `.venv` Python instead of repeatedly retrying plain shell.
- Be careful with Chinese text in shell output. PowerShell encoding may display Chinese comments as mojibake. Prefer explicit UTF-8 reads or Python/Node file reads when inspecting Chinese text.
- Do not modify old analysis scripts unless the user explicitly asks. Prefer adding new files for new experiments.

User normally uses Windows cmd, not PowerShell. Prefer cmd-style commands in explanations and examples.
