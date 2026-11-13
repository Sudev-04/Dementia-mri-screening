# VS Code setup

These are ready-made VS Code tasks, debug configurations and settings for this project.
They are kept in `vscode-setup/` instead of `.vscode/` because the assistant that built the
project is not allowed to write into `.vscode` (task files can run commands, so that folder
is protected). Read them, then turn them on yourself:

    PowerShell:   Copy-Item vscode-setup\*.json .vscode\ -Force
    macOS/Linux:  mkdir -p .vscode && cp vscode-setup/*.json .vscode/

Then use **Terminal > Run Task**:

1. Set up (creates `.venv`, installs the app and test packages)
2. Add demo accounts
3. Run app (with the synthetic demo MRI model)

After task 1, pick the `.venv` interpreter when VS Code asks (or **Python: Select Interpreter**).
`launch.json` adds two debug targets: the app server, and a short training run on synthetic data.
