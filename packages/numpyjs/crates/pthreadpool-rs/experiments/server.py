#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Simple HTTP server with COOP/COEP headers for SharedArrayBuffer support."""

import http.server
import socketserver
import os

PORT = 8765

class COOPCOEPHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Serving on http://localhost:{PORT} with COOP/COEP headers")
print(f"Test at: http://localhost:{PORT}/test_dispatch.html")

with socketserver.TCPServer(("", PORT), COOPCOEPHandler) as httpd:
    httpd.serve_forever()
