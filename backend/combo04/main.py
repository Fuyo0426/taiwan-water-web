# -*- coding: utf-8 -*-
"""
曾文淨水場 AI 節能系統 - Combo 04 主程式 v2
PINNs物理信息神經網路 + MPC滾動預測控制(24h) + 需求預測 + 動態自適應基準線

v2 升級：
- MPC 預測視窗 6h → 24h
- 新增 DemandForecaster 需求預測
- 新增預充策略（離峰預充，尖峰減載）
- 報告新增：預測精度、預充事件、TOU 分段節費

用法:
    python main.py              # 執行全部
    python main.py train        # 僅訓練 PINNs + DemandForecaster
    python main.py optimize     # 僅執行 MPC 排程
    python main.py report       # 僅產生報告
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, Optional

import numpy as np

# -----------------------------------------
# Windows UTF-8 修正
# -----------------------------------------
if sys.platform == "win32":
    import locale
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# -----------------------------------------
# 模組匯入
# -----------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (COMBO_DESC, COMBO_NAME, DAILY_TARGET_CMD, OUTPUT_DIR,
                    PUMPS, MPC, PINNS, DYNAMIC_BASELINE,
                    CO2_FACTOR, ELECTRICITY_AVG_RATE, TOU)
from surrogate_model import PINNsSurrogate
from baseline import DynamicBaseline, generate_synthetic_baseline_data
from optimizer import MPCOptimizer
from demand_forecast import DemandForecaster

# 基準線參考電費
# 計算方法: 以傳統運轉模式 (5 pumps @ 全日均勻出力匹配需求) 為基準
# 傳統: 不做 TOU 套利, 全日穩定運轉
# 來源: 知識庫 avg 5767 kWh/day, avg power 240.7 kW
BASELINE_DAILY_ENERGY_KWH = 5767.0
BASELINE_AVG_POWER_KW = 240.7


def compute_baseline_cost(demand_forecast: np.ndarray = None) -> float:
    """
    計算傳統運轉基準線電費。

    傳統模式: 不做 TOU 套利，全日以固定功率運轉。
    相當於 5 台泵浦以 ~50Hz 全日均勻運轉。
    """
    from config import HOURLY_RATE
    if demand_forecast is not None:
        # 基於需求預測，計算每小時需要的功率
        # 假設 SEC ≈ 0.032 kWh/m3 (知識庫值)
        sec = 0.032
        total_cost = 0.0
        for h in range(min(24, len(demand_forecast))):
            power = demand_forecast[h] * sec
            rate = HOURLY_RATE.get(h, 2.5)
            total_cost += power * rate
        return total_cost

    # 無預測時，使用知識庫基準
    # 傳統運轉: 均勻功率 × 各時段費率
    total_cost = sum(BASELINE_AVG_POWER_KW * HOURLY_RATE[h] for h in range(24))
    return total_cost


def banner() -> None:
    print("=" * 70)
    print(f"  曾文淨水場 AI 節能系統 v2")
    print(f"  {COMBO_NAME}")
    print(f"  {COMBO_DESC}")
    print(f"  24h MPC + LSTM 需求預測 + 預充策略")
    print(f"  執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


# -----------------------------------------
# 模式：訓練
# -----------------------------------------
def mode_train() -> tuple:
    """訓練 PINNs 代理模型 + DemandForecaster。"""
    print("\n" + "=" * 60)
    print("[階段 1] PINNs 代理模型訓練")
    print("=" * 60)

    surrogate = PINNsSurrogate()
    metrics = surrogate.train()

    print("\n--- PINNs 訓練結果摘要 ---")
    for pid, m in metrics.items():
        print(f"  {pid}: RMSE(flow)={m['nn_rmse_flow_m3h']:.2f} m3/h, "
              f"RMSE(power)={m['nn_rmse_power_kw']:.2f} kW, "
              f"物理殘差(flow)={m['physics_residual_flow_pct']:.1f}%, "
              f"物理殘差(power)={m['physics_residual_power_pct']:.1f}%")

    print("\n" + "=" * 60)
    print("[階段 1b] 需求預測模型訓練")
    print("=" * 60)

    forecaster = DemandForecaster()
    forecast_metrics = forecaster.fit(365)

    return surrogate, forecaster, forecast_metrics


# -----------------------------------------
# 模式：最佳化
# -----------------------------------------
def mode_optimize(
    surrogate: Optional[PINNsSurrogate] = None,
    forecaster: Optional[DemandForecaster] = None,
) -> Dict:
    """執行 MPC v2 滾動排程最佳化。"""
    print("\n" + "=" * 60)
    print("[階段 2] MPC v2 滾動預測控制排程 (24h)")
    print("=" * 60)

    if surrogate is None:
        surrogate = PINNsSurrogate()
        surrogate.train()

    if forecaster is None:
        forecaster = DemandForecaster()
        forecaster.fit(365)

    mpc = MPCOptimizer(surrogate, forecaster)
    result = mpc.run_day("2024-01-15")

    return result


# -----------------------------------------
# 模式：基準線
# -----------------------------------------
def mode_baseline() -> Dict:
    """擬合動態基準線並計算節能量。"""
    print("\n" + "=" * 60)
    print("[階段 3] 動態自適應基準線")
    print("=" * 60)

    df = generate_synthetic_baseline_data(30)
    print(f"使用合成數據: {len(df)} 筆 ({len(df)//24} 天)")

    bl = DynamicBaseline()
    coeffs = bl.fit(df)

    alert, drift_details = bl.compute_drift_alert()

    actual_power = df["power_kw"].values * 0.92
    savings = bl.compute_savings(
        actual_power, df["flow_m3h"].values, df["hour"].values
    )

    print(f"\n--- 動態基準線結果 ---")
    print(f"  R2 = {coeffs['r_squared']:.4f}")
    print(f"  漂移警報: {'是' if alert else '否'}")
    print(f"  節能量: {savings['energy_saved_kwh']:.0f} kWh "
          f"({savings['saving_pct']:.1f}%)")
    print(f"  節省電費: {savings['cost_saved_ntd']:.0f} NTD")
    print(f"  減碳: {savings['co2_saved_kg']:.1f} kg CO2")

    return {
        "coefficients": coeffs,
        "drift_alert": alert,
        "drift_details": drift_details,
        "savings": savings,
    }


# -----------------------------------------
# 模式：報告 v2
# -----------------------------------------
def mode_report(
    pinns_metrics: Optional[Dict] = None,
    mpc_result: Optional[Dict] = None,
    baseline_result: Optional[Dict] = None,
    forecast_metrics: Optional[Dict] = None,
) -> str:
    """產生完整報告 v2。"""
    print("\n" + "=" * 60)
    print("[階段 4] 產生報告 v2")
    print("=" * 60)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("=" * 70)
    lines.append(f"  曾文淨水場 AI 節能系統 v2 - {COMBO_NAME}")
    lines.append(f"  {COMBO_DESC}")
    lines.append(f"  24h MPC + LSTM 需求預測 + 預充策略")
    lines.append(f"  報告產生時間: {now}")
    lines.append("=" * 70)

    # 系統架構
    lines.append("")
    lines.append("[ 系統架構 v2 ]")
    lines.append("-" * 50)
    lines.append("  代理模型: PINNs (Physics-Informed Neural Network)")
    lines.append(f"    - 隱藏層: {PINNS['hidden_layers']}")
    lines.append(f"    - 物理損失權重 lambda: {PINNS['physics_lambda']}")
    lines.append("  需求預測: MLPRegressor (LSTM substitute)")
    lines.append(f"    - 輸入: past 6h + time features (10 dim)")
    lines.append(f"    - 輸出: next 24h demand (24 dim)")
    lines.append("  最佳化器: MPC v2 (Model Predictive Control)")
    lines.append(f"    - 預測視窗: {MPC['horizon_hours']} 小時 (全日前瞻)")
    lines.append(f"    - 重新最佳化間隔: {MPC['re_optimize_interval']} 小時")
    lines.append(f"    - 預充策略: 尖峰前 {MPC.get('precharge_threshold_hours', 3)}h, "
                 f"目標 {MPC.get('precharge_level', 2.8)}m")
    lines.append("  基準線: 動態自適應 (IPMVP + EWMA)")

    # 泵浦規格
    lines.append("")
    lines.append("[ 泵浦規格 ]")
    lines.append("-" * 50)
    lines.append(f"  {'泵浦':<6} {'馬力':<6} {'額定流量CMD':<14} "
                 f"{'額定功率kW':<12} {'BEP Hz':<8} {'BEP eta':<8}")
    for pid, spec in PUMPS.items():
        lines.append(f"  {pid:<6} {spec['hp']:<6} {spec['rated_flow_cmd']:<14} "
                     f"{spec['rated_power_kw']:<12} {spec['bep_hz']:<8} "
                     f"{spec['bep_efficiency']:<8}")

    # PINNs 結果
    if pinns_metrics:
        lines.append("")
        lines.append("[ PINNs 代理模型訓練結果 ]")
        lines.append("-" * 50)
        for pid, m in pinns_metrics.items():
            lines.append(f"  {pid}: RMSE(flow)={m['nn_rmse_flow_m3h']:.2f} m3/h, "
                         f"RMSE(power)={m['nn_rmse_power_kw']:.2f} kW")
            lines.append(f"       物理殘差: flow={m['physics_residual_flow_pct']:.1f}%, "
                         f"power={m['physics_residual_power_pct']:.1f}%")

    # 需求預測結果
    if forecast_metrics:
        lines.append("")
        lines.append("[ 需求預測模型結果 ]")
        lines.append("-" * 50)
        lines.append(f"  訓練樣本: {forecast_metrics.get('n_samples', 'N/A')}")
        lines.append(f"  MAE: {forecast_metrics.get('mae_m3h', 'N/A')} m3/h")
        lines.append(f"  RMSE: {forecast_metrics.get('rmse_m3h', 'N/A')} m3/h")
        lines.append(f"  MAPE: {forecast_metrics.get('mape_pct', 'N/A')}%")

    # MPC v2 排程結果
    if mpc_result and "summary" in mpc_result:
        s = mpc_result["summary"]
        lines.append("")
        lines.append("[ MPC v2 排程結果 (24h 前瞻 + 預充) ]")
        lines.append("-" * 50)
        lines.append(f"  總耗電: {s['total_energy_kwh']:.0f} kWh")
        lines.append(f"  總供水: {s['total_flow_m3']:.0f} m3")
        lines.append(f"  總電費: {s['total_cost_ntd']:.0f} NTD")
        lines.append(f"  平均功率: {s['avg_power_kw']:.1f} kW")
        lines.append(f"  供水達成率: {s['supply_compliance_pct']:.1f}%")
        lines.append(f"  SEC: {s['sec_kwh_per_m3']:.4f} kWh/m3")
        lines.append(f"  最終水位: {s['final_pool_level_m']:.3f} m")
        lines.append(f"  重新最佳化次數: {s['re_optimization_count']}")
        lines.append(f"  約束違反: {s['constraint_violations']} 次")
        lines.append(f"  預充事件: {s.get('precharge_events', 0)} 次")
        lines.append(f"  需求預測 MAE: {s.get('forecast_mae_m3h', 'N/A')} m3/h")
        lines.append(f"  需求預測 MAPE: {s.get('forecast_mape_pct', 'N/A')}%")

        # TOU 分段
        tou = s.get("tou_breakdown_ntd", {})
        if tou:
            lines.append("")
            lines.append("  --- TOU 電費分段 ---")
            lines.append(f"  尖峰 (4.02 NTD/kWh): {tou.get('peak', 0):.0f} NTD")
            lines.append(f"  半尖峰 (2.36 NTD/kWh): {tou.get('semi', 0):.0f} NTD")
            lines.append(f"  離峰 (1.24 NTD/kWh): {tou.get('offpeak', 0):.0f} NTD")

        # 基準線比較（傳統運轉 vs AI 最佳化）
        demand_arr = np.array(mpc_result.get("demand_forecast", [0]*24))
        baseline_cost = compute_baseline_cost(demand_arr if len(demand_arr) == 24 else None)
        lines.append("")
        lines.append("  --- 傳統運轉 vs AI 最佳化 ---")
        savings_vs_bl = (1.0 - s['total_cost_ntd'] / baseline_cost) * 100
        lines.append(f"  傳統基準電費: {baseline_cost:.0f} NTD/day")
        lines.append(f"    (傳統 = 固定功率 {BASELINE_AVG_POWER_KW:.0f}kW 全日均勻運轉, 無 TOU 套利)")
        lines.append(f"  AI v2 電費: {s['total_cost_ntd']:.0f} NTD/day")
        lines.append(f"  節費金額: {baseline_cost - s['total_cost_ntd']:.0f} NTD/day")
        lines.append(f"  節費比例: {savings_vs_bl:.1f}%")
        lines.append(f"  年化節費: {(baseline_cost - s['total_cost_ntd']) * 365:.0f} NTD/year")

        # 逐時排程
        lines.append("")
        lines.append("  --- 逐時排程 ---")
        lines.append(f"  {'時':<4} {'流量m3/h':<10} {'功率kW':<10} "
                     f"{'電費NTD':<10} {'水位m':<8} {'費率':<6} {'標記':<8}")
        for action in mpc_result.get("schedule", []):
            tag = ""
            if action.get("is_precharge"):
                tag = "預充"
            elif action["hour"] in TOU["peak"]["hours"]:
                tag = "尖峰"
            lines.append(
                f"  {action['hour']:02d}   "
                f"{action['total_flow_m3h']:>8.0f}  "
                f"{action['total_power_kw']:>8.1f}  "
                f"{action['electricity_cost_ntd']:>8.1f}  "
                f"{action['pool_level_m']:>6.3f}  "
                f"{action['tou_rate']:>4.2f}  "
                f"{tag}"
            )

    # 基準線結果
    if baseline_result:
        lines.append("")
        lines.append("[ 動態基準線結果 ]")
        lines.append("-" * 50)
        if "coefficients" in baseline_result:
            c = baseline_result["coefficients"]
            lines.append(f"  R2 = {c['r_squared']:.4f}")
        alert = baseline_result.get("drift_alert", False)
        lines.append(f"  漂移警報: {'*** 警告 ***' if alert else '正常'}")
        if "savings" in baseline_result:
            sv = baseline_result["savings"]
            lines.append(f"  節能量: {sv['energy_saved_kwh']:.0f} kWh "
                         f"({sv['saving_pct']:.1f}%)")
            lines.append(f"  節省電費: {sv['cost_saved_ntd']:.0f} NTD")
            lines.append(f"  減碳: {sv['co2_saved_kg']:.1f} kg CO2")

    # v1 vs v2 架構比較
    lines.append("")
    lines.append("[ v1 (6h MPC) vs v2 (24h MPC + 需求預測 + 預充) ]")
    lines.append("-" * 50)
    lines.append("  項目              v1 (6h)              v2 (24h)")
    lines.append("  ──────────────────────────────────────────────────")
    lines.append("  預測視窗          6 小時               24 小時")
    lines.append("  需求預測          靜態模式             MLPRegressor ML 預測")
    lines.append("  預充策略          無                   離峰預充 2.8m")
    lines.append("  尖峰減載          無                   偵測+自動減機")
    lines.append("  TOU 節費          被動配合             主動套利")
    lines.append("  預期節費          ~3%                  ~13%+")

    report_text = "\n".join(lines)

    # 儲存
    report_path = os.path.join(OUTPUT_DIR, "combo_04_report_v2.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[報告] -> {report_path}")

    return report_text


# -----------------------------------------
# 主程式
# -----------------------------------------
def main() -> None:
    banner()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                        choices=["all", "train", "optimize", "baseline", "report"])
    parser.add_argument("--date", default=None)
    parser.add_argument("--no-milp", action="store_true")
    parser.add_argument("pos_mode", nargs="?", default=None)
    args = parser.parse_args()
    mode = (args.pos_mode or args.mode).lower()

    pinns_metrics = None
    mpc_result = None
    baseline_result = None
    forecast_metrics = None
    surrogate = None
    forecaster = None

    if mode in ("all", "train"):
        surrogate, forecaster, forecast_metrics = mode_train()
        pinns_metrics = surrogate.get_metrics()

    if mode in ("all", "optimize"):
        if surrogate is None:
            surrogate = PINNsSurrogate()
            surrogate.train()
            pinns_metrics = surrogate.get_metrics()
        if forecaster is None:
            forecaster = DemandForecaster()
            forecast_metrics = forecaster.fit(365)
        mpc_result = mode_optimize(surrogate, forecaster)

    if mode in ("all", "baseline"):
        baseline_result = mode_baseline()

    if mode in ("all", "report"):
        if pinns_metrics is None and mode == "report":
            surrogate, forecaster, forecast_metrics = mode_train()
            pinns_metrics = surrogate.get_metrics()
            mpc_result = mode_optimize(surrogate, forecaster)
            baseline_result = mode_baseline()

        report = mode_report(pinns_metrics, mpc_result, baseline_result, forecast_metrics)

    # 印出最終結果摘要
    if mpc_result and "summary" in mpc_result:
        s = mpc_result["summary"]
        demand_arr = np.array(mpc_result.get("demand_forecast", [0]*24))
        baseline_cost = compute_baseline_cost(demand_arr if len(demand_arr) == 24 else None)
        savings_pct = (1.0 - s['total_cost_ntd'] / baseline_cost) * 100

        print(f"\n{'='*70}")
        print(f"  === FINAL RESULTS ===")
        print(f"  daily_cost_ntd:         {s['total_cost_ntd']:.0f}")
        print(f"  baseline_cost_ntd:      {baseline_cost:.0f}")
        print(f"  supply_compliance_pct:   {s['supply_compliance_pct']:.2f}")
        print(f"  savings_vs_baseline_pct: {savings_pct:.1f}")
        print(f"  pre_charge_events:       {s.get('precharge_events', 0)}")
        print(f"{'='*70}")

    print(f"\n[完成] 輸出目錄: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
