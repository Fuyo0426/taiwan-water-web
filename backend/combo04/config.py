# -*- coding: utf-8 -*-
"""
曾文淨水場 AI 節能系統 - 全域設定
台灣自來水股份有限公司 × 成大博士研究

Combo 04: PINNs物理信息神經網路 + MPC滾動預測控制 + 動態自適應基準線
"""

import os

# ─────────────────────────────────────────────
# Combo 識別
# ─────────────────────────────────────────────
COMBO_NAME = "combo_04_PINNs-MPC-DynamicBL"
COMBO_DESC = "PINNs物理信息神經網路 + MPC滾動預測控制 + 動態自適應基準線"

# ─────────────────────────────────────────────
# 路徑設定
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.environ.get("DATA_DIR",   os.path.join(BASE_DIR, "data"))
FLOW_DIR   = os.path.join(DATA_DIR, "2_flow")
ALL_DIR    = os.path.join(DATA_DIR, "1_all")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(BASE_DIR, "output"))

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "schedules"), exist_ok=True)

# ─────────────────────────────────────────────
# 抽水機基本規格（曾文淨水場）
# ─────────────────────────────────────────────
PUMPS = {
    "P1": {"hp": 200, "type": "vertical_centrifugal", "poles": 4,
           "sync_rpm": 1800, "rated_hz": 60.0,
           "rated_flow_cmd": 72000,   # CMD（立方米/天）@ 60Hz
           "rated_power_kw": 130.0,   # kW @ 60Hz BEP
           "bep_hz": 55.0,            # 最佳效率頻率
           "bep_efficiency": 0.855,   # η_total @ BEP
           "max_hz": 58.5,            # 禁區上限（超過功耗陡增）
           "min_hz": 40.0},
    "P2": {"hp": 100, "type": "vertical_centrifugal", "poles": 4,
           "sync_rpm": 1800, "rated_hz": 60.0,
           "rated_flow_cmd": 38000,
           "rated_power_kw": 65.0,
           "bep_hz": 54.0,
           "bep_efficiency": 0.842,
           "max_hz": 58.5,
           "min_hz": 40.0},
    "P3": {"hp": 100, "type": "vertical_centrifugal", "poles": 4,
           "sync_rpm": 1800, "rated_hz": 60.0,
           "rated_flow_cmd": 36000,
           "rated_power_kw": 63.0,
           "bep_hz": 54.0,
           "bep_efficiency": 0.838,
           "max_hz": 58.5,
           "min_hz": 40.0},
    "P4": {"hp": 100, "type": "vertical_centrifugal", "poles": 4,
           "sync_rpm": 1800, "rated_hz": 60.0,
           "rated_flow_cmd": 34000,
           "rated_power_kw": 61.0,
           "bep_hz": 54.0,
           "bep_efficiency": 0.835,
           "max_hz": 58.5,
           "min_hz": 40.0},
    "P5": {"hp": 200, "type": "vertical_centrifugal", "poles": 4,
           "sync_rpm": 1800, "rated_hz": 60.0,
           "rated_flow_cmd": 68000,
           "rated_power_kw": 125.0,
           "bep_hz": 51.5,
           "bep_efficiency": 0.848,
           "max_hz": 58.5,
           "min_hz": 40.0},
}

# ─────────────────────────────────────────────
# 台電時間電價（TOU）- 台灣工業用電
# 單位：NTD/kWh
# ─────────────────────────────────────────────
TOU = {
    "peak":     {"rate": 4.02, "hours": list(range(9, 12)) + list(range(13, 17))},   # 尖峰
    "semi":     {"rate": 2.36, "hours": list(range(7, 9)) + list(range(12, 13)) + list(range(17, 22))},  # 半尖峰
    "offpeak":  {"rate": 1.24, "hours": list(range(22, 24)) + list(range(0, 7))},    # 離峰
}
# 建立逐小時費率查詢表
HOURLY_RATE = {}
for period, info in TOU.items():
    for h in info["hours"]:
        HOURLY_RATE[h] = info["rate"]

CONTRACT_CAPACITY_KW = 600.0   # 契約容量（超約罰3倍）
OVERCONTRACT_PENALTY  = 3.0    # 超約倍率

# ─────────────────────────────────────────────
# 水位約束（配水池）
# ─────────────────────────────────────────────
POOL_LEVEL_MIN = 1.5    # 最低安全水位 (M)
POOL_LEVEL_MAX = 3.0    # 最高水位 (M)
POOL_LEVEL_INIT = 2.0   # 初始水位 (M)
POOL_AREA_M2    = 5000  # 配水池面積（估算，m²）

# ─────────────────────────────────────────────
# 每日目標產水量（CMD）
# 依知識庫：平均日耗電 5767 kWh，平均功率 240.7 kW
# ─────────────────────────────────────────────
DAILY_TARGET_CMD = 180000   # 立方米/天（目標供水量）
TARGET_COMPLIANCE_RATE = 0.998  # >=99.8% 供水達成率

# ─────────────────────────────────────────────
# IPMVP 節能驗證參數
# ─────────────────────────────────────────────
CO2_FACTOR = 0.494  # kg CO2/kWh（台灣電力排放係數）
ELECTRICITY_AVG_RATE = 2.5  # NTD/kWh（平均費率，含基本電費分攤）

# ─────────────────────────────────────────────
# PINNs 模型參數
# ─────────────────────────────────────────────
PINNS = {
    "hidden_layers": [64, 128, 128, 64],
    "activation": "relu",
    "epochs": 500,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "val_split": 0.2,
    "random_seed": 42,
    "physics_lambda": 0.4,         # 物理損失權重（v3.2 調參：0.1→0.4，強化物理一致性）
    "lambda_pipe_loss": 0.25,      # R4: Darcy-Weisbach 管損一致性權重（v3.2 調參：0.05→0.25）
    "lambda_cavitation": 0.15,     # R5: 空蝕邊界條件權重（v3.2 調參：0.03→0.15）
    "k_pipe_loss": 0.05,           # 管損係數（額定流量下 5% 效率損失）
    "training_points": 3000,       # 每泵訓練點數
    "bep_dense_ratio": 0.3,        # BEP附近密集取樣比例
    "affinity_law_residual": True,  # 啟用親和律殘差
}

# ─────────────────────────────────────────────
# MPC 控制參數
# ─────────────────────────────────────────────
MPC = {
    "horizon_hours": 24,           # 預測視窗 24 小時（v2: 全日前瞻）
    "re_optimize_interval": 1,     # 每 1 小時重新最佳化
    "demand_forecast_hours": 24,   # 需求預測時距
    "precharge_threshold_hours": 3, # 尖峰前 3 小時啟動預充
    "precharge_level": 2.8,        # 預充目標水位 (m)
    "time_steps": 24,              # 24小時
    "step_minutes": 60,            # 每步 60 分鐘
    "min_run_minutes": 30,         # 最短連續運轉 30 分鐘
    "max_starts_per_day": 4,       # 每日最大啟動次數
    "freq_steps": [40, 44, 48, 50, 52, 54, 56, 58],  # 頻率離散化
    "pool_weight": 50.0,           # 水位偏離懲罰權重（提高，強制守住水位）
    "demand_penalty": 5000.0,      # 供水不足懲罰（大幅提高，確保達成率≥99.8%）
}

# ─────────────────────────────────────────────
# 動態基準線參數
# ─────────────────────────────────────────────
DYNAMIC_BASELINE = {
    "rolling_window_days": 30,     # 長期趨勢滾動窗口
    "ewma_span_days": 7,           # EWMA 短期調整跨度
    "drift_alert_threshold": 0.05, # SEC 漂移警報閾值 (5%)
    "min_data_points": 14,         # 最少需要 14 天數據才啟動
}
