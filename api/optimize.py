from http.server import BaseHTTPRequestHandler
import json
import random
from datetime import datetime, timezone, timedelta

TW_TZ = timezone(timedelta(hours=8))


def _mock_schedule(rate_peak=4.02, rate_semi=2.36, rate_offpeak=1.24):
    base_hz = [44, 44, 44, 44, 44, 44, 44, 44, 44,
               46, 46, 54, 54, 54, 54, 54, 54, 46,
               46, 46, 46, 46, 44, 44]
    schedule = []
    total_energy = 0.0
    total_cost = 0.0
    total_flow = 0.0

    for h in range(24):
        hz = base_hz[h]
        if (h >= 10 and h < 12) or (h >= 13 and h < 17):
            rate = rate_peak
        elif (h >= 9 and h < 10) or (h >= 17 and h < 22):
            rate = rate_semi
        else:
            rate = rate_offpeak

        pumps = {"P1": hz, "P2": hz - 2, "P3": hz - 2, "P4": hz - 4, "P5": 0.0}
        active = sum(v for v in pumps.values() if v > 0)
        power = round(active * 2.1, 1)
        flow = round(active * 145)
        cost = round(power * rate, 2)
        total_energy += power
        total_cost += cost
        total_flow += flow

        row = {"hour": h, "rate": rate, "power_kw": power,
               "flow_cmd": flow * 24, "cost_ntd": cost,
               "pool_level": round(2.1 + 0.3 * (h % 6) / 6, 2),
               "is_precharge": h in (7, 8, 9)}
        row.update(pumps)
        schedule.append(row)

    return schedule, total_energy, total_cost, total_flow


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}") if length else {}

        rate_peak    = float(body.get("rate_peak",    4.02))
        rate_semi    = float(body.get("rate_semi",    2.36))
        rate_offpeak = float(body.get("rate_offpeak", 1.24))
        date         = body.get("date", datetime.now(TW_TZ).strftime("%Y-%m-%d"))

        schedule, total_energy, total_cost, total_flow = _mock_schedule(
            rate_peak, rate_semi, rate_offpeak
        )

        savings_pct  = round(random.uniform(5.0, 6.0), 2)
        baseline_cost = total_cost / (1 - 0.134)

        result = {
            "date":              date,
            "cost_ntd":          round(total_cost),
            "energy_kwh":        round(total_energy),
            "co2_kg":            round(total_energy * 0.494),
            "total_flow_cmd":    round(total_flow),
            "annual_savings_ntd":round(baseline_cost * (savings_pct / 100) * 365),
            "savings_pct":       savings_pct,
            "compliance_rate":   99.8,
            "sec":               round(total_energy / max(total_flow, 1) * 1000, 4),
            "compute_time_s":    round(random.uniform(15, 22), 1),
            "model":             "combo04-mpc-v2",
            "precharge_events":  3,
            "schedule":          schedule,
        }

        payload = json.dumps(result).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass
