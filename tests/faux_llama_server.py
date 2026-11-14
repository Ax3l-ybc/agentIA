#!/usr/bin/env python3
"""Faux llama-server pour les tests : plante sur un message vocal, sauf avec --no-mmproj-offload.

Imite la Jetson dont la puce graphique manque de mémoire pour le module audio.
"""

import http.server
import json
import os
import sys

args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
audio_hors_gpu = "--no-mmproj-offload" in args


class Gestion(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        corps = self.rfile.read(int(self.headers["Content-Length"])).decode()
        if "input_audio" in corps and not audio_hors_gpu:
            print("ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory", flush=True)
            os._exit(1)
        reponse = json.dumps({"choices": [{"message": {"role": "assistant", "content": "prêt"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(reponse)

    def log_message(self, *a):
        pass


http.server.HTTPServer(("127.0.0.1", port), Gestion).serve_forever()
