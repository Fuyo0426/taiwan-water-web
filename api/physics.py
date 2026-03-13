from http.server import BaseHTTPRequestHandler
import json

PHYSICS_DATA = {
    "R1_affinity_law":        {"compliance": 99.2, "description": "親和律一致性"},
    "R2_energy_conservation": {"compliance": 98.7, "description": "能量守恆"},
    "R3_monotonicity":        {"compliance": 99.5, "description": "單調性約束"},
    "R4_pipe_loss":           {"compliance": 97.8, "description": "Darcy-Weisbach 管損"},
    "R5_cavitation":          {"compliance": 96.3, "description": "空蝕邊界條件"},
}


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = json.dumps(PHYSICS_DATA).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass
