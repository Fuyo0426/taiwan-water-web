from http.server import BaseHTTPRequestHandler
import json
import random
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse, parse_qs

TW_TZ = timezone(timedelta(hours=8))


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        date = params.get("date", [datetime.now(TW_TZ).strftime("%Y-%m-%d")])[0]

        base_hz = [44, 44, 44, 44, 44, 44, 44, 44, 44,
                   46, 46, 54, 54, 54, 54, 54, 54, 46,
                   46, 46, 46, 46, 44, 44]
        schedule = []
        for h in range(24):
            hz = base_hz[h]
            if (h >= 10 and h < 12) or (h >= 13 and h < 17):
                rate = 4.02
            elif (h >= 9 and h < 10) or (h >= 17 and h < 22):
                rate = 2.36
            else:
                rate = 1.24
            pumps = {"P1": hz, "P2": hz - 2, "P3": hz - 2, "P4": hz - 4, "P5": 0.0}
            active = sum(v for v in pumps.values() if v > 0)
            power = round(active * 2.1, 1)
            flow = round(active * 145)
            row = {"hour": h, "rate": rate, "power_kw": power,
                   "flow_cmd": flow * 24, "cost_ntd": round(power * rate, 2),
                   "pool_level": round(2.1 + 0.3 * (h % 6) / 6, 2),
                   "is_precharge": h in (7, 8, 9)}
            row.update(pumps)
            schedule.append(row)

        savings_pct = round(random.uniform(5.0, 6.0), 2)
        result = {"date": date, "savings_pct": savings_pct, "schedule": schedule}

        payload = json.dumps(result).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass
