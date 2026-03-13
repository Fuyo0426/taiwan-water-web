from http.server import BaseHTTPRequestHandler
import json

PUMP_SPECS = {
    "P1": {"hp": 200, "type": "vertical_centrifugal", "poles": 4,
           "sync_rpm": 1800, "rated_hz": 60.0, "rated_flow_cmd": 72000,
           "rated_power_kw": 130.0, "bep_hz": 55.0, "bep_efficiency": 0.855,
           "max_hz": 58.5, "min_hz": 40.0},
    "P2": {"hp": 100, "type": "vertical_centrifugal", "poles": 4,
           "sync_rpm": 1800, "rated_hz": 60.0, "rated_flow_cmd": 38000,
           "rated_power_kw": 65.0, "bep_hz": 54.0, "bep_efficiency": 0.842,
           "max_hz": 58.5, "min_hz": 40.0},
    "P3": {"hp": 100, "type": "vertical_centrifugal", "poles": 4,
           "sync_rpm": 1800, "rated_hz": 60.0, "rated_flow_cmd": 36000,
           "rated_power_kw": 63.0, "bep_hz": 54.0, "bep_efficiency": 0.838,
           "max_hz": 58.5, "min_hz": 40.0},
    "P4": {"hp": 100, "type": "vertical_centrifugal", "poles": 4,
           "sync_rpm": 1800, "rated_hz": 60.0, "rated_flow_cmd": 34000,
           "rated_power_kw": 61.0, "bep_hz": 54.0, "bep_efficiency": 0.835,
           "max_hz": 58.5, "min_hz": 40.0},
    "P5": {"hp": 200, "type": "vertical_centrifugal", "poles": 4,
           "sync_rpm": 1800, "rated_hz": 60.0, "rated_flow_cmd": 68000,
           "rated_power_kw": 125.0, "bep_hz": 51.5, "bep_efficiency": 0.848,
           "max_hz": 58.5, "min_hz": 40.0},
}


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = json.dumps(PUMP_SPECS).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass
