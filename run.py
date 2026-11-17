"""Start Sports Helpers on this computer: check everything, then open the browser.

    python run.py                 # http://127.0.0.1:8000 (or the next free port)
    python run.py --port 9000
    python run.py --no-browser

Checks, in order: the Python version, FFmpeg (ffmpeg + ffprobe), and the Python
packages, which it offers to install into this interpreter if any are missing.
On Windows you can double-click start.bat instead; on macOS/Linux run ./start.sh.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGES = {   # import name -> pip name
    "fastapi": "fastapi", "uvicorn": "uvicorn", "multipart": "python-multipart",
    "cv2": "opencv-python-headless", "numpy": "numpy", "matplotlib": "matplotlib",
    "openai": "openai", "dotenv": "python-dotenv",
}


def fail(message: str) -> None:
    print(f"\n[!] {message}\n")
    if platform.system() == "Windows" and not sys.stdin.isatty():
        input("Press Enter to close…")
    sys.exit(1)


def check_python() -> None:
    if sys.version_info < (3, 11):
        fail(f"Python 3.11 or newer is needed; this is {platform.python_version()}. "
             "Get it from https://www.python.org/downloads/")
    print(f"[ok] Python {platform.python_version()}")


def check_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if not shutil.which(tool)]
    if not missing:
        print("[ok] FFmpeg found")
        return
    hint = {
        "Windows": "Open PowerShell and run:  winget install Gyan.FFmpeg\n"
                   "    then close and reopen this window so PATH picks it up.",
        "Darwin": "Run:  brew install ffmpeg   (Homebrew: https://brew.sh)",
    }.get(platform.system(), "Run:  sudo apt install ffmpeg   (or your distribution's package)")
    fail(f"{' and '.join(missing)} not found on PATH.\n    {hint}")


def check_packages(auto_yes: bool) -> None:
    missing = [pip for mod, pip in PACKAGES.items() if importlib.util.find_spec(mod) is None]
    if not missing:
        print("[ok] Python packages installed")
        return
    print(f"[..] Missing Python packages: {', '.join(missing)}")
    if not auto_yes:
        answer = input("     Install them now with pip? [Y/n] ").strip().lower()
        if answer not in ("", "y", "yes"):
            fail(f"Install them with:  {sys.executable} -m pip install -r requirements.txt")
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")]
    print("     " + " ".join(cmd))
    if subprocess.call(cmd) != 0:
        fail("pip could not install the requirements; see the messages above.")
    print("[ok] Python packages installed")


def free_port(host: str, start: int) -> int:
    for port in range(start, start + 20):
        with socket.socket() as s:
            if s.connect_ex((host, port)) != 0:
                return port
    fail(f"No free port between {start} and {start + 19}.")
    return start


def open_when_ready(url: str) -> None:
    for _ in range(60):
        try:
            urllib.request.urlopen(url + "api/config", timeout=1)
            webbrowser.open(url)
            return
        except OSError:
            time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Sports Helpers locally.")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("-y", "--yes", action="store_true", help="install missing packages without asking")
    args = parser.parse_args()

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    check_python()
    check_ffmpeg()
    check_packages(args.yes)

    port = free_port(args.host, args.port)
    if port != args.port:
        print(f"[..] Port {args.port} is busy; using {port}")
    url = f"http://{args.host}:{port}/"
    from webapp import jobs

    mode = jobs.mode()
    print(f"\nSports Helpers is starting at {url}  ({mode} mode)")
    if mode == "fixture":
        print("  No VLMRUN_API_KEY: only the synthetic sample clips can be analyzed.")
        print("  To analyze your own videos, copy .env.example to .env and add a key.")
    print("  Press Ctrl+C to stop.\n")
    if not args.no_browser:
        threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()

    import uvicorn

    from webapp.server import app

    uvicorn.run(app, host=args.host, port=port, log_level="warning")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
