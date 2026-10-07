"""Serve the browser demo at http://localhost:8000/web/

Usage:
    python serve_web.py [--port 8000]
The project root is served so the page can load ../models/hand_landmarker.task.
Webcam access in the browser requires localhost (or https).
"""
import argparse
import functools
import http.server
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as httpd:
        url = f"http://localhost:{args.port}/web/"
        print(f"Serving {ROOT} -> {url}  (Ctrl+C to stop)")
        if not args.no_browser:
            webbrowser.open(url)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
