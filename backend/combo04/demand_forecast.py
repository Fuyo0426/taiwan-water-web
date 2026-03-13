# -*- coding: utf-8 -*-
"""
LSTM-like 需求預測模型（MLPRegressor 實作）
Combo 04 v2: 24 小時需求預測，支援 MPC 長期前瞻

輸入特徵: 過去 6 小時流量 + hour-of-day + day-of-week
輸出: 未來 24 小時逐時需求預測 (m3/h)

使用 sklearn MLPRegressor 替代 LSTM，以 sliding window 方式捕捉時序特徵。
"""

import json
import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

try:
    from config import DAILY_TARGET_CMD, OUTPUT_DIR
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import DAILY_TARGET_CMD, OUTPUT_DIR


class DemandForecaster:
    """
    24 小時需求預測器。

    使用 MLPRegressor 學習時序需求模式：
    - 輸入: [past_6h_flows (6), hour_sin, hour_cos, dow_sin, dow_cos] = 10 features
    - 輸出: [next_24h_demands] = 24 values

    訓練數據: 365 天合成需求模式（含日間波動、週期性、噪聲）
    """

    LOOKBACK = 6   # 過去 6 小時
    HORIZON = 24   # 預測 24 小時

    def __init__(self) -> None:
        self.model: Optional[MLPRegressor] = None
        self.scaler_x: Optional[StandardScaler] = None
        self.scaler_y: Optional[StandardScaler] = None
        self.fitted: bool = False
        self._base_demand: float = DAILY_TARGET_CMD / 24.0  # 7500 m3/h

    # -----------------------------------------
    # 需求模式定義
    # -----------------------------------------
    @staticmethod
    def _hourly_factor(hour: int) -> float:
        """
        逐時需求因子。

        Morning peak (7-9h): +20%
        Evening peak (17-21h): +15%
        Night valley (0-6h): -15%
        Daytime normal (10-16h): +5%
        """
        if 7 <= hour <= 9:
            return 1.20
        elif 17 <= hour <= 21:
            return 1.15
        elif 0 <= hour <= 6:
            return 0.85
        elif 22 <= hour <= 23:
            return 0.90
        else:  # 10-16
            return 1.05

    def _generate_synthetic_demand(
        self, n_days: int = 365, seed: int = 42
    ) -> np.ndarray:
        """
        生成合成需求時序資料。

        Returns:
            shape (n_days * 24,) 的逐時需求 m3/h
        """
        rng = np.random.RandomState(seed)
        demands = []

        for day in range(n_days):
            for hour in range(24):
                base = self._base_demand
                factor = self._hourly_factor(hour)

                # 星期效應（週末稍低）
                dow = day % 7
                if dow >= 5:  # 週六日
                    factor *= 0.95

                # 季節微調（夏季略高）
                seasonal = 1.0 + 0.03 * np.sin(2 * np.pi * day / 365)

                # 高斯噪聲 sigma=5%
                noise = 1.0 + rng.normal(0, 0.05)

                demand = base * factor * seasonal * noise
                demands.append(max(demand, 1000.0))

        return np.array(demands)

    # -----------------------------------------
    # 特徵工程
    # -----------------------------------------
    def _build_features(
        self, demands: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        從連續需求時序建立 sliding window 訓練資料。

        X features (10):
          - past 6h flows (6 values, normalized by base_demand)
          - sin(2*pi*hour/24), cos(2*pi*hour/24)
          - sin(2*pi*dow/7), cos(2*pi*dow/7)

        Y targets (24):
          - next 24h demands (normalized by base_demand)
        """
        n_total = len(demands)
        X_list = []
        Y_list = []

        for i in range(self.LOOKBACK, n_total - self.HORIZON):
            # 過去 6 小時
            past = demands[i - self.LOOKBACK: i] / self._base_demand

            # 時間特徵
            hour = i % 24
            day_idx = i // 24
            dow = day_idx % 7

            hour_sin = np.sin(2 * np.pi * hour / 24.0)
            hour_cos = np.cos(2 * np.pi * hour / 24.0)
            dow_sin = np.sin(2 * np.pi * dow / 7.0)
            dow_cos = np.cos(2 * np.pi * dow / 7.0)

            x = np.concatenate([past, [hour_sin, hour_cos, dow_sin, dow_cos]])
            y = demands[i: i + self.HORIZON] / self._base_demand

            X_list.append(x)
            Y_list.append(y)

        return np.array(X_list), np.array(Y_list)

    # -----------------------------------------
    # 訓練
    # -----------------------------------------
    def fit(self, n_days: int = 365, seed: int = 42) -> dict:
        """
        生成合成數據並訓練 MLPRegressor。

        Returns:
            訓練指標 dict
        """
        print(f"[需求預測] 生成 {n_days} 天合成需求資料...")
        demands = self._generate_synthetic_demand(n_days, seed)
        print(f"  資料量: {len(demands)} 小時, "
              f"平均需求: {demands.mean():.0f} m3/h, "
              f"標準差: {demands.std():.0f} m3/h")

        X, Y = self._build_features(demands)
        print(f"  訓練樣本: {len(X)}, 特徵數: {X.shape[1]}, 輸出維度: {Y.shape[1]}")

        # 標準化
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        X_scaled = self.scaler_x.fit_transform(X)
        Y_scaled = self.scaler_y.fit_transform(Y)

        # 訓練
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 128, 64),
            activation="relu",
            max_iter=500,
            batch_size=256,
            learning_rate_init=1e-3,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            verbose=False,
        )

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_scaled, Y_scaled)

        self.fitted = True

        # 評估
        Y_pred_scaled = self.model.predict(X_scaled)
        Y_pred = self.scaler_y.inverse_transform(Y_pred_scaled) * self._base_demand
        Y_actual = Y * self._base_demand

        mae = np.mean(np.abs(Y_pred - Y_actual))
        rmse = np.sqrt(np.mean((Y_pred - Y_actual) ** 2))
        mape = np.mean(np.abs(Y_pred - Y_actual) / (Y_actual + 1e-8)) * 100

        metrics = {
            "n_samples": int(len(X)),
            "n_iterations": int(self.model.n_iter_),
            "mae_m3h": round(float(mae), 1),
            "rmse_m3h": round(float(rmse), 1),
            "mape_pct": round(float(mape), 2),
        }

        print(f"[需求預測] 訓練完成")
        print(f"  MAE: {mae:.1f} m3/h")
        print(f"  RMSE: {rmse:.1f} m3/h")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  迭代次數: {self.model.n_iter_}")

        # 儲存模型
        self._save()

        return metrics

    # -----------------------------------------
    # 預測
    # -----------------------------------------
    def predict_24h(
        self, current_hour: int, recent_flows: Optional[List[float]] = None,
        day_of_week: int = 0,
    ) -> np.ndarray:
        """
        預測未來 24 小時的逐時需求。

        Args:
            current_hour: 當前小時 (0-23)
            recent_flows: 最近 6 小時實際流量 (m3/h)。
                         若 None，使用基於模式的估算。
            day_of_week: 星期幾 (0=Mon, 6=Sun)

        Returns:
            shape (24,) 的需求預測 m3/h
        """
        if not self.fitted:
            # 未訓練時，回退到靜態模式
            return self._fallback_pattern(current_hour)

        # 構建輸入
        if recent_flows is None or len(recent_flows) < self.LOOKBACK:
            # 用模式估算填充
            recent = []
            for offset in range(self.LOOKBACK, 0, -1):
                h = (current_hour - offset) % 24
                recent.append(self._base_demand * self._hourly_factor(h))
            recent_flows = recent

        past = np.array(recent_flows[-self.LOOKBACK:]) / self._base_demand

        hour_sin = np.sin(2 * np.pi * current_hour / 24.0)
        hour_cos = np.cos(2 * np.pi * current_hour / 24.0)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7.0)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7.0)

        x = np.concatenate([past, [hour_sin, hour_cos, dow_sin, dow_cos]])
        x = x.reshape(1, -1)

        x_scaled = self.scaler_x.transform(x)
        y_scaled = self.model.predict(x_scaled)
        y = self.scaler_y.inverse_transform(y_scaled)[0] * self._base_demand

        # 確保合理範圍
        y = np.clip(y, self._base_demand * 0.5, self._base_demand * 1.8)

        return y

    def _fallback_pattern(self, current_hour: int) -> np.ndarray:
        """未訓練時的回退需求模式。"""
        result = np.zeros(24)
        for i in range(24):
            h = (current_hour + i) % 24
            result[i] = self._base_demand * self._hourly_factor(h)
        return result

    # -----------------------------------------
    # 尖峰偵測
    # -----------------------------------------
    def detect_peaks(
        self, forecast: np.ndarray, threshold_factor: float = 1.10
    ) -> List[int]:
        """
        從 24h 預測中偵測尖峰時段。

        Args:
            forecast: 24h 需求預測
            threshold_factor: 超過平均需求此倍數即為尖峰

        Returns:
            尖峰小時索引列表 (relative to forecast start)
        """
        avg = np.mean(forecast)
        peaks = [i for i, d in enumerate(forecast) if d > avg * threshold_factor]
        return peaks

    # -----------------------------------------
    # 儲存/載入
    # -----------------------------------------
    def _save(self) -> None:
        model_dir = os.path.join(OUTPUT_DIR, "models")
        os.makedirs(model_dir, exist_ok=True)

        state = {
            "model": self.model,
            "scaler_x": self.scaler_x,
            "scaler_y": self.scaler_y,
            "base_demand": self._base_demand,
        }

        path = os.path.join(model_dir, "demand_forecast.pkl")
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"[需求預測] 模型 -> {path}")

    def load(self) -> bool:
        path = os.path.join(OUTPUT_DIR, "models", "demand_forecast.pkl")
        if not os.path.exists(path):
            return False

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.model = state["model"]
        self.scaler_x = state["scaler_x"]
        self.scaler_y = state["scaler_y"]
        self._base_demand = state["base_demand"]
        self.fitted = True
        print(f"[需求預測] 已載入模型: {path}")
        return True


# -----------------------------------------
# 直接執行測試
# -----------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("需求預測模型測試")
    print("=" * 60)

    forecaster = DemandForecaster()
    metrics = forecaster.fit(365)

    print("\n--- 預測測試 ---")
    for hour in [0, 6, 12, 18]:
        forecast = forecaster.predict_24h(hour)
        peaks = forecaster.detect_peaks(forecast)
        print(f"  hour={hour:02d}: "
              f"mean={forecast.mean():.0f}, "
              f"max={forecast.max():.0f}, "
              f"min={forecast.min():.0f}, "
              f"peaks={peaks}")
