"""One command to run the whole auditor:  python run.py

Checks dependencies, starts Flask on port 5050, opens the browser at the
setup wizard, and prints the URL. When a run finishes, the headline
numbers print to this terminal (see app.py) so the whole flow is
screen-recordable. Fully functional with zero API keys — only the
plain-english strategy path and the coach need one.
"""

from __future__ import annotations

import sys
import threading
import webbrowser

REQUIRED = ["pandas", "numpy", "yfinance", "requests", "flask"]


def check_deps():
    missing = []
    for name in REQUIRED:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        sys.exit("Missing dependencies: " + ", ".join(missing)
                 + "\nInstall them with:  pip install -r requirements.txt")


def main():
    check_deps()
    from app import PORT, app
    import ai_client

    url = f"http://127.0.0.1:{PORT}/"
    print("=" * 72)
    print("TRADE AUDITOR")
    print(f"  wizard:    {url}")
    print(f"  dashboard: {url}dashboard   (after a run)")
    print("  AI paths:  " + ("enabled" if ai_client.is_configured()
                             else "no key set — presets, manual rules, engine and "
                                  "dashboard all run fine; coaching stays greyed out"))
    print("=" * 72)

    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=PORT, debug=False)


if __name__ == "__main__":
    main()
