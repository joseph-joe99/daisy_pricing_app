"""Native "open file" dialog for the path boxes in the app.

The dialog opens on the computer that runs Daisy (the server), not in the browser, so it is only useful when the app
runs on the machine whose files you want to pick. It blocks until the user chooses or cancels, so call it from a
worker thread, not from the event loop.
"""

from __future__ import annotations

import base64
import subprocess
import sys

# Tk fallback for Linux and any other platform: prints the chosen path, or nothing if the dialog was cancelled.
_TK_SCRIPT = """
import sys
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
path = filedialog.askopenfilename(title=sys.argv[1], filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
root.destroy()
if path:
    print(path)
"""

# Windows: the standard Open dialog through PowerShell (no extra packages needed).
_POWERSHELL_SCRIPT = """
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = '{title}'
$dialog.Filter = 'CSV files (*.csv)|*.csv|All files (*.*)|*.*'
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{ Write-Output $dialog.FileName }}
"""


def _quote_applescript(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def pick_file(title: str = "Choose a file") -> str | None:
    """Show a native open-file dialog and return the chosen path, or None if the user cancelled or no dialog could open."""
    try:
        if sys.platform == "darwin":
            prompt = _quote_applescript(title)
            script = f'POSIX path of (choose file with prompt "{prompt}")'
            command = ["osascript", "-e", script]
        elif sys.platform.startswith("win"):
            # the script goes in as base64 so quotes and line breaks in it cannot be mangled on the command line
            script = _POWERSHELL_SCRIPT.format(title=title.replace("'", "''"))
            encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
            command = ["powershell", "-NoProfile", "-STA", "-EncodedCommand", encoded]
        else:
            command = [sys.executable, "-c", _TK_SCRIPT, title]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),  # no console window flashes on Windows
        )
    except OSError:
        return None
    # osascript exits non-zero when the user cancels; the other dialogs print nothing when cancelled
    path = result.stdout.strip()
    return path or None
