from http.server import BaseHTTPRequestHandler
import json
import random
from datetime import datetime, timedelta


def _mock_history():
    records = []
    base = datetime(2025, 3, 1)
    for i in range(30):
        d = base + timedelta(days=i)
        savings = round(random.uniform(5.0, 6.0), 2)
        records.append({
            "file": f"schedule_{d.strftime('%Y-%m-%d')}.json",
            "date": d.strftime("%Y-%m-%d"),
            "cost_ntd": random.randint(13000, 16000),
            "energy_kwh": random.randint(4500, 5500),
            "savings_pct": savings,
            "compliance_rate": round(random.uniform(98.5, 99.9), 1),
        })
    return list(reversed(records))


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        records = _mock_history()
        payload = json.dumps({"count": len(records), "records": records}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass
