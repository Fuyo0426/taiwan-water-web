from http.server import BaseHTTPRequestHandler
import json


def _mock_pareto():
    solutions = []
    for i in range(10):
        cost = 12000 + i * 800
        energy = 4800 + i * 250
        compliance = 99.9 - i * 0.15
        solutions.append({
            "id": i + 1,
            "cost_ntd": cost,
            "energy_kwh": energy,
            "compliance_rate": round(compliance, 1),
            "co2_kg": round(energy * 0.494),
        })
    return solutions


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        data = {"fallback": _mock_pareto()}
        payload = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass
