"""HTTP measurement for libraries without a server, keeping model work on the loading thread.

The client sends the same System One JSON used by native servers. A tiny stdlib server calls the existing
adapter once and sends its normalized answers back. It adds HTTP costs, not a scheduler or a different batch
policy. Results explicitly distinguish this benchmark wrapper from a runtime's native server.
"""

from __future__ import annotations

import json
import socketserver
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer


class LocalServer(HTTPServer):
    """HTTPServer without its lookup of the host's fully qualified name on bind, a reverse DNS lookup that can stall
    for seconds (it hung on GitHub's macOS runners); nothing here reads `server_name`."""

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = str(self.server_address[0]), int(self.server_address[1])


def run_http(args, decide, meta, *, memory, fits, run):
    result, failed = [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/memory":
                    value = memory() if memory else None
                elif self.path == "/fits":
                    value = fits(body["state"], body["nq"])
                elif self.path == "/v1/systemone":
                    value = decide({"systemone": body})
                else:
                    self.send_error(404)
                    return
                payload = json.dumps(value).encode()
                self.send_response(200)
            except Exception as exc:
                payload = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode()
                self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = LocalServer(("127.0.0.1", 0), Handler)
    base = f"http://127.0.0.1:{server.server_port}"

    def post(path, body):
        req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=1800) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(exc.read().decode(errors="replace")) from exc

    def client():
        try:
            result.append(
                run(
                    args,
                    lambda req: post("/v1/systemone", req["systemone"]),
                    {**meta, "transport": "benchmark HTTP wrapper"},
                    memory=(lambda: post("/memory", {})) if memory else None,
                    fits=(lambda state, nq: post("/fits", {"state": state, "nq": nq})) if fits else None,
                )
            )
        except BaseException as exc:
            failed.append(exc)
        finally:
            server.shutdown()

    worker = threading.Thread(target=client, name="benchmark-http-client")
    worker.start()
    try:
        server.serve_forever(poll_interval=0.05)
    finally:
        server.server_close()
        worker.join()
    if failed:
        raise failed[0]
    return result[0]
