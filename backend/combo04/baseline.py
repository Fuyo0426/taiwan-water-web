# -*- coding: utf-8 -*-
"""
動態自適應基準線模型
Combo 04: 雙層基準線 = 長期 IPMVP 迴歸 + 短期 EWMA 修正

vs 傳統 IPMVP 靜態基準線的優勢：
- 季節性自動追蹤（Fourier 特徵 sin/cos）
- 30 天滾動窗口避免老舊數據拖累
- EWMA 即時修正近期操作變化（如泵浦劣化、管路阻力增加）
- SEC 漂移告警：自動偵測效率衰退
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

try:
    from config import (DAILY_TARGET_CMD, DYNAMIC_BASELINE, OUTPUT_DIR,
                        PUMPS, ELECTRICITY_AVG_RATE, CO2_FACTOR)
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import (DAILY_TARGET_CMD, DYNAMIC_BASELINE, OUTPUT_DIR,
                        PUMPS, ELECTRICITY_AVG_RATE, CO2_FACTOR)


class DynamicBaseline:
    """
    雙層動態自適應基準線。

    第 1 層：長期趨勢 (IPMVP 迴歸，30 天滾動窗口)
        E_long = alpha * Q + beta * sin(2*pi*h/24) + gamma * cos(2*pi*h/24) + delta

    第 2 層：短期修正 (EWMA, span=7 天)
        E_adjusted = E_long + ewma_residual
    """

    def __init__(self) -> None:
        self.lr_model: Optional[LinearRegression] = None
        self.lr_coeffs: Dict[str, float] = {}

        # EWMA 狀態
        self.ewma_residual: float = 0.0
        self.ewma_alpha: float = 2.0 / (DYNAMIC_BASELINE["ewma_span_days"] + 1)

        # SEC 追蹤
        self.sec_history: List[float] = []
        self.sec_long_term_mean: float = 0.0
        self.sec_long_term_std: float = 0.0

        # 數據窗口
        self.rolling_window_days: int = DYNAMIC_BASELINE["rolling_window_days"]
        self.history_buffer: Optional[pd.DataFrame] = None

        self.fitted: bool = False

    # ─────────────────────────────────────────
    # 特徵工程
    # ─────────────────────────────────────────
    @staticmethod
    def _build_features(
        flow_m3h: np.ndarray, hour: np.ndarray
    ) -> np.ndarray:
        """
        建立迴歸特徵矩陣。
        [flow, sin(2*pi*h/24), cos(2*pi*h/24)]
        """
        flow = np.asarray(flow_m3h, dtype=float).reshape(-1)
        h = np.asarray(hour, dtype=float).reshape(-1)

        sin_h = np.sin(2 * np.pi * h / 24.0)
        cos_h = np.cos(2 * np.pi * h / 24.0)

        return np.column_stack([flow, sin_h, cos_h])

    # ─────────────────────────────────────────
    # 擬合
    # ─────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        在完整歷史數據上擬合基準線模型。

        df 需包含欄位：
            - flow_m3h: 流量 (m3/h)
            - power_kw: 總功率 (kW)
            - hour: 小時 (0-23)
            - date: 日期（用於滾動窗口）

        回傳:
            迴歸係數 dict
        """
        df = df.copy()
        if len(df) < DYNAMIC_BASELINE["min_data_points"] * 24:
            print(f"[基準線] 警告：數據量不足 "
                  f"({len(df)} 筆 < {DYNAMIC_BASELINE['min_data_points']*24} 筆)")

        # 取最近 N 天
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            cutoff = df["date"].max() - pd.Timedelta(days=self.rolling_window_days)
            df_window = df[df["date"] >= cutoff].copy()
        else:
            n_keep = self.rolling_window_days * 24
            df_window = df.tail(n_keep).copy()

        X = self._build_features(df_window["flow_m3h"].values,
                                 df_window["hour"].values)
        y = df_window["power_kw"].values

        self.lr_model = LinearRegression()
        self.lr_model.fit(X, y)

        self.lr_coeffs = {
            "alpha_flow": round(float(self.lr_model.coef_[0]), 6),
            "beta_sin": round(float(self.lr_model.coef_[1]), 4),
            "gamma_cos": round(float(self.lr_model.coef_[2]), 4),
            "delta_intercept": round(float(self.lr_model.intercept_), 4),
            "r_squared": round(float(self.lr_model.score(X, y)), 4),
        }

        # 初始化 EWMA 殘差
        y_pred = self.lr_model.predict(X)
        residuals = y - y_pred
        self.ewma_residual = float(np.mean(residuals[-24:]))  # 最近一天均值

        # 初始化 SEC 追蹤
        self.sec_history = []
        sec_values = self.compute_sec(df_window)
        if sec_values is not None and len(sec_values) > 0:
            self.sec_history = sec_values.tolist()
            self.sec_long_term_mean = float(np.mean(sec_values))
            self.sec_long_term_std = float(np.std(sec_values))

        self.history_buffer = df_window.copy()
        self.fitted = True

        print(f"[基準線] 擬合完成 (R²={self.lr_coeffs['r_squared']:.4f})")
        print(f"  alpha={self.lr_coeffs['alpha_flow']:.6f}, "
              f"beta={self.lr_coeffs['beta_sin']:.4f}, "
              f"gamma={self.lr_coeffs['gamma_cos']:.4f}, "
              f"delta={self.lr_coeffs['delta_intercept']:.4f}")
        print(f"  EWMA 殘差初始值: {self.ewma_residual:.2f} kW")

        self._save()
        return self.lr_coeffs

    # ─────────────────────────────────────────
    # 線上更新
    # ─────────────────────────────────────────
    def update(self, new_day_data: pd.DataFrame) -> None:
        """
        用新一天的觀測數據更新 EWMA 短期修正。

        new_day_data 需包含：flow_m3h, power_kw, hour
        """
        if not self.fitted:
            raise RuntimeError("模型尚未擬合，請先呼叫 fit()")

        X = self._build_features(new_day_data["flow_m3h"].values,
                                 new_day_data["hour"].values)
        y_actual = new_day_data["power_kw"].values
        y_pred = self.lr_model.predict(X)

        # 逐點更新 EWMA 殘差
        for actual, pred in zip(y_actual, y_pred):
            residual = actual - pred
            self.ewma_residual = (self.ewma_alpha * residual +
                                  (1 - self.ewma_alpha) * self.ewma_residual)

        # 更新 SEC 歷史
        daily_energy = float(np.sum(y_actual))  # kWh (假設每筆 1 小時)
        daily_flow = float(np.sum(new_day_data["flow_m3h"].values))  # m3
        if daily_flow > 0:
            daily_sec = daily_energy / daily_flow
            self.sec_history.append(daily_sec)

        print(f"[基準線] 線上更新完成, EWMA 殘差={self.ewma_residual:.2f} kW")

    # ─────────────────────────────────────────
    # 預測
    # ─────────────────────────────────────────
    def predict_baseline(self, flow_m3h: float, hour: int) -> float:
        """
        預測基準線功率 (kW)。

        E_adjusted = E_long + ewma_residual

        Args:
            flow_m3h: 流量 (m3/h)
            hour: 小時 (0-23)

        Returns:
            基準線功率 (kW)
        """
        if not self.fitted:
            raise RuntimeError("模型尚未擬合，請先呼叫 fit()")

        X = self._build_features(
            np.array([flow_m3h]), np.array([hour])
        )
        e_long = float(self.lr_model.predict(X)[0])
        e_adjusted = e_long + self.ewma_residual

        return max(e_adjusted, 0.0)

    def predict_baseline_batch(
        self, flows: np.ndarray, hours: np.ndarray
    ) -> np.ndarray:
        """批次預測基準線功率。"""
        if not self.fitted:
            raise RuntimeError("模型尚未擬合，請先呼叫 fit()")

        X = self._build_features(flows, hours)
        e_long = self.lr_model.predict(X)
        e_adjusted = e_long + self.ewma_residual
        return np.maximum(e_adjusted, 0.0)

    # ─────────────────────────────────────────
    # SEC 計算
    # ─────────────────────────────────────────
    def compute_sec(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        計算 SEC (Specific Energy Consumption) 時間序列。
        SEC = kWh / m3（每立方米水的耗電量）

        按日聚合計算。
        """
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            daily = df.groupby(df["date"].dt.date).agg(
                total_energy=("power_kw", "sum"),      # kWh（每筆 1 小時）
                total_flow=("flow_m3h", "sum"),         # m3
            )
        else:
            # 假設按 24 小時為一天分組
            n_days = len(df) // 24
            if n_days == 0:
                return None
            secs = []
            for d in range(n_days):
                chunk = df.iloc[d * 24: (d + 1) * 24]
                energy = chunk["power_kw"].sum()
                flow = chunk["flow_m3h"].sum()
                if flow > 0:
                    secs.append(energy / flow)
            return np.array(secs)

        valid = daily[daily["total_flow"] > 0]
        sec_values = (valid["total_energy"] / valid["total_flow"]).values
        return sec_values

    # ─────────────────────────────────────────
    # 漂移警報
    # ─────────────────────────────────────────
    def compute_drift_alert(self) -> Tuple[bool, Dict[str, float]]:
        """
        檢測 SEC 是否已漂移超過長期趨勢的 5%。

        漂移代表泵浦效率衰退，應觸發維護警報。

        Returns:
            (is_alert, details)
        """
        threshold = DYNAMIC_BASELINE["drift_alert_threshold"]

        if len(self.sec_history) < 7:
            return False, {"reason": "數據不足，至少需要 7 天"}

        recent_sec = np.mean(self.sec_history[-7:])
        long_term_sec = self.sec_long_term_mean

        if long_term_sec == 0:
            return False, {"reason": "長期 SEC 均值為 0"}

        drift_pct = (recent_sec - long_term_sec) / long_term_sec

        details = {
            "recent_sec_7d": round(float(recent_sec), 6),
            "long_term_sec": round(float(long_term_sec), 6),
            "drift_pct": round(float(drift_pct * 100), 2),
            "threshold_pct": round(float(threshold * 100), 1),
        }

        is_alert = drift_pct > threshold
        if is_alert:
            print(f"[基準線] *** 漂移警報 *** SEC 上升 {drift_pct*100:.1f}% "
                  f"(閾值 {threshold*100:.0f}%)")
            print(f"  近 7 日 SEC={recent_sec:.4f}, 長期 SEC={long_term_sec:.4f}")
            print(f"  建議：檢查泵浦效率、管路阻力、葉輪磨損")

        return is_alert, details

    # ─────────────────────────────────────────
    # 節能量計算
    # ─────────────────────────────────────────
    def compute_savings(
        self, actual_power: np.ndarray, flows: np.ndarray, hours: np.ndarray
    ) -> Dict[str, float]:
        """
        計算節能量 = 基準線預測 - 實際耗電。

        Returns:
            {energy_saved_kwh, cost_saved_ntd, co2_saved_kg,
             saving_pct, baseline_total, actual_total}
        """
        baseline = self.predict_baseline_batch(flows, hours)
        baseline_total = float(np.sum(baseline))
        actual_total = float(np.sum(actual_power))
        saved = baseline_total - actual_total

        return {
            "baseline_total_kwh": round(baseline_total, 1),
            "actual_total_kwh": round(actual_total, 1),
            "energy_saved_kwh": round(saved, 1),
            "saving_pct": round(saved / (baseline_total + 1e-8) * 100, 2),
            "cost_saved_ntd": round(saved * ELECTRICITY_AVG_RATE, 0),
            "co2_saved_kg": round(saved * CO2_FACTOR, 1),
        }

    # ─────────────────────────────────────────
    # 儲存/載入
    # ─────────────────────────────────────────
    def _save(self) -> None:
        model_dir = os.path.join(OUTPUT_DIR, "models")
        os.makedirs(model_dir, exist_ok=True)

        state = {
            "model_type": "dynamic_baseline_ipmvp_ewma",
            "combo": "combo_04_PINNs-MPC-DynamicBL",
            "coefficients": self.lr_coeffs,
            "ewma_residual": round(self.ewma_residual, 4),
            "ewma_alpha": round(self.ewma_alpha, 4),
            "rolling_window_days": self.rolling_window_days,
            "sec_long_term_mean": round(self.sec_long_term_mean, 6),
            "sec_long_term_std": round(self.sec_long_term_std, 6),
            "sec_history_length": len(self.sec_history),
            "drift_threshold": DYNAMIC_BASELINE["drift_alert_threshold"],
            "timestamp": datetime.now().isoformat(),
        }

        path = os.path.join(model_dir, "dynamic_baseline.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        print(f"[基準線] 模型狀態 → {path}")

    def load(self) -> bool:
        """載入已儲存的基準線狀態（僅元資訊，迴歸模型需重新 fit）。"""
        path = os.path.join(OUTPUT_DIR, "models", "dynamic_baseline.json")
        if not os.path.exists(path):
            return False

        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        self.ewma_residual = state.get("ewma_residual", 0.0)
        self.sec_long_term_mean = state.get("sec_long_term_mean", 0.0)
        self.sec_long_term_std = state.get("sec_long_term_std", 0.0)
        print(f"[基準線] 已載入狀態：EWMA={self.ewma_residual:.4f}")
        return True


# ─────────────────────────────────────────────
# 合成測試數據生成（無 SCADA 時使用）
# ─────────────────────────────────────────────
def generate_synthetic_baseline_data(n_days: int = 30) -> pd.DataFrame:
    """
    生成合成歷史數據，用於基準線擬合測試。
    模擬曾文淨水場典型運轉模式。
    """
    rng = np.random.RandomState(42)
    records = []

    for day in range(n_days):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)
        for hour in range(24):
            # 日間需水量較高
            if 6 <= hour <= 22:
                base_flow = 7500 + rng.normal(0, 300)
            else:
                base_flow = 5000 + rng.normal(0, 200)

            # 季節性趨勢
            seasonal = 500 * np.sin(2 * np.pi * day / 365)
            flow_m3h = max(base_flow + seasonal, 2000)

            # 功率（SEC 約 0.032 kWh/m3 + 噪聲）
            sec = 0.032 + rng.normal(0, 0.002)
            power_kw = flow_m3h * sec + rng.normal(0, 5)
            power_kw = max(power_kw, 50)

            records.append({
                "date": date,
                "hour": hour,
                "flow_m3h": round(flow_m3h, 1),
                "power_kw": round(power_kw, 1),
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("=" * 60)
    print("動態自適應基準線測試")
    print("=" * 60)

    # 生成合成數據
    df = generate_synthetic_baseline_data(30)
    print(f"合成數據: {len(df)} 筆 ({len(df)//24} 天)")

    # 擬合
    bl = DynamicBaseline()
    coeffs = bl.fit(df)

    # 預測測試
    print("\n預測測試:")
    for hour in [3, 10, 15, 22]:
        pred = bl.predict_baseline(7000.0, hour)
        print(f"  flow=7000 m3/h, hour={hour} → baseline={pred:.1f} kW")

    # 漂移檢測
    alert, details = bl.compute_drift_alert()
    print(f"\n漂移檢測: alert={alert}, details={details}")

    # 線上更新
    new_day = df.tail(24).copy()
    bl.update(new_day)
