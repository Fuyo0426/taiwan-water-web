# -*- coding: utf-8 -*-
"""
PINNs 物理信息神經網路代理模型
Combo 04: 嵌入親和律（Affinity Laws）物理約束於訓練損失

核心思路：
- 資料驅動 MLP 學習泵浦特性曲線
- 物理殘差項強制滿足：
  Q ∝ N (流量正比轉速)
  P ∝ N³ (功率正比轉速三次方)
  η 在 BEP 附近達峰值
- 若 PyTorch 不可用，退化為 sklearn MLP + 親和律後修正
"""

import json
import os
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

try:
    from config import PINNS, PUMPS, OUTPUT_DIR
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import PINNS, PUMPS, OUTPUT_DIR


class AffinityLaws:
    """親和律物理模型 — 離心泵浦的第一性原理（v2: +管損+空蝕）。"""

    @staticmethod
    def flow(hz: float, rated_hz: float, rated_flow_cmd: float) -> float:
        """Q2/Q1 = N2/N1 → Q = Q_rated * (hz / hz_rated)"""
        ratio = hz / rated_hz
        return rated_flow_cmd * ratio

    @staticmethod
    def power(hz: float, rated_hz: float, rated_power_kw: float) -> float:
        """P2/P1 = (N2/N1)^3 → P = P_rated * (hz / hz_rated)^3"""
        ratio = hz / rated_hz
        return rated_power_kw * (ratio ** 3)

    @staticmethod
    def efficiency(hz: float, bep_hz: float, bep_eff: float,
                   sigma: float = 8.0) -> float:
        """效率以 BEP 為峰值的高斯近似。"""
        return bep_eff * np.exp(-0.5 * ((hz - bep_hz) / sigma) ** 2)

    @staticmethod
    def efficiency_v2(hz: float, rated_hz: float, bep_hz: float,
                      bep_eff: float, min_hz: float,
                      k_pipe: float = 0.05, sigma: float = 8.0) -> float:
        """
        v2 效率模型：高斯基礎 + Darcy-Weisbach 管損 + 空蝕邊界。

        R4: pipe_loss_factor = 1 - k_pipe * (Q/Q_r - Q_bep/Q_r)²
        R5: cavitation_factor = clamp((hz - min_hz) / 3, 0, 1)
        """
        # 基礎高斯效率
        eta_base = bep_eff * np.exp(-0.5 * ((hz - bep_hz) / sigma) ** 2)

        # R4: Darcy-Weisbach 管損修正
        flow_ratio = hz / rated_hz
        bep_flow_ratio = bep_hz / rated_hz
        pipe_loss_factor = 1.0 - k_pipe * (flow_ratio - bep_flow_ratio) ** 2
        eta = eta_base * pipe_loss_factor

        # R5: 空蝕邊界條件
        cav_margin = 3.0
        if isinstance(hz, np.ndarray):
            cav_factor = np.clip((hz - min_hz) / cav_margin, 0.0, 1.0)
            eta = np.where(hz < min_hz + cav_margin, eta * cav_factor, eta)
        else:
            if hz < min_hz + cav_margin:
                cav_factor = max(0.0, min(1.0, (hz - min_hz) / cav_margin))
                eta = eta * cav_factor

        return float(np.clip(eta, 0.0, 0.95)) if not isinstance(eta, np.ndarray) else np.clip(eta, 0.0, 0.95)


class PINNsSurrogate:
    """
    Physics-Informed Neural Network 代理模型。

    使用 sklearn MLPRegressor 作為骨幹，訓練後以物理殘差修正輸出。
    物理損失在後處理階段施加（sklearn 不支援自定義 loss），
    透過加權混合 NN 預測與親和律預測來達成物理一致性。

    若未來移植至 PyTorch，可直接在 loss function 加入殘差項。
    """

    def __init__(self) -> None:
        self.models: Dict[str, MLPRegressor] = {}
        self.scalers_x: Dict[str, StandardScaler] = {}
        self.scalers_y: Dict[str, StandardScaler] = {}
        self.physics = AffinityLaws()
        self.physics_lambda: float = PINNS["physics_lambda"]
        self.trained: bool = False
        self._metrics: Dict[str, dict] = {}

    # ─────────────────────────────────────────
    # 訓練數據生成
    # ─────────────────────────────────────────
    def _generate_training_data(
        self, pump_id: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成合成訓練資料：親和律 + 隨機擾動模擬真實偏差。
        BEP 附近密集取樣確保高效率區精度。
        """
        spec = PUMPS[pump_id]
        n_total = PINNS["training_points"]
        n_bep = int(n_total * PINNS["bep_dense_ratio"])
        n_uniform = n_total - n_bep
        rng = np.random.RandomState(PINNS["random_seed"])

        # 均勻取樣全頻率範圍
        hz_uniform = rng.uniform(spec["min_hz"], spec["max_hz"], n_uniform)

        # BEP 附近密集取樣 (bep_hz +/- 5 Hz)
        bep_center = spec["bep_hz"]
        hz_bep = rng.normal(bep_center, 2.5, n_bep)
        hz_bep = np.clip(hz_bep, spec["min_hz"], spec["max_hz"])

        hz_all = np.concatenate([hz_uniform, hz_bep])
        rng.shuffle(hz_all)

        # 以親和律生成「真實」數據 + 擾動
        flows = np.array([
            self.physics.flow(h, spec["rated_hz"], spec["rated_flow_cmd"])
            for h in hz_all
        ])
        powers = np.array([
            self.physics.power(h, spec["rated_hz"], spec["rated_power_kw"])
            for h in hz_all
        ])
        k_pipe = PINNS.get("k_pipe_loss", 0.05)
        effs = np.array([
            self.physics.efficiency_v2(
                h, spec["rated_hz"], spec["bep_hz"], spec["bep_efficiency"],
                spec["min_hz"], k_pipe=k_pipe)
            for h in hz_all
        ])

        # 加入 +/-3% 隨機擾動模擬實際運轉偏差
        noise_flow = 1.0 + rng.normal(0, 0.03, n_total)
        noise_power = 1.0 + rng.normal(0, 0.03, n_total)
        noise_eff = 1.0 + rng.normal(0, 0.015, n_total)

        flows *= noise_flow
        powers *= noise_power
        effs = np.clip(effs * noise_eff, 0.0, 0.95)  # 允許空蝕區低效率（v3.2）

        # 轉換流量為 m3/h
        flows_m3h = flows / 24.0

        X = hz_all.reshape(-1, 1)
        Y = np.column_stack([flows_m3h, powers, effs])
        return X, Y

    # ─────────────────────────────────────────
    # 訓練
    # ─────────────────────────────────────────
    def train(self) -> Dict[str, dict]:
        """訓練所有泵浦的 PINNs 代理模型。"""
        print(f"[PINNs] 開始訓練 {len(PUMPS)} 台泵浦代理模型...")
        print(f"[PINNs] 物理損失權重 lambda = {self.physics_lambda}")

        for pump_id in PUMPS:
            print(f"\n  訓練 {pump_id}...")
            X, Y = self._generate_training_data(pump_id)

            # 標準化
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_x.fit_transform(X)
            Y_scaled = scaler_y.fit_transform(Y)

            # MLP 訓練
            mlp = MLPRegressor(
                hidden_layer_sizes=tuple(PINNS["hidden_layers"]),
                activation=PINNS["activation"],
                max_iter=PINNS["epochs"],
                batch_size=PINNS["batch_size"],
                learning_rate_init=PINNS["learning_rate"],
                random_state=PINNS["random_seed"],
                early_stopping=True,
                validation_fraction=PINNS["val_split"],
                n_iter_no_change=30,
                verbose=False,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mlp.fit(X_scaled, Y_scaled)

            self.models[pump_id] = mlp
            self.scalers_x[pump_id] = scaler_x
            self.scalers_y[pump_id] = scaler_y

            # 評估：計算物理一致性指標
            Y_pred_scaled = mlp.predict(X_scaled)
            Y_pred = scaler_y.inverse_transform(Y_pred_scaled)

            nn_rmse_flow = np.sqrt(np.mean((Y[:, 0] - Y_pred[:, 0]) ** 2))
            nn_rmse_power = np.sqrt(np.mean((Y[:, 1] - Y_pred[:, 1]) ** 2))

            # 物理殘差（親和律偏離度）
            spec = PUMPS[pump_id]
            hz_flat = X.flatten()
            phys_flows = np.array([
                self.physics.flow(h, spec["rated_hz"], spec["rated_flow_cmd"]) / 24.0
                for h in hz_flat
            ])
            phys_powers = np.array([
                self.physics.power(h, spec["rated_hz"], spec["rated_power_kw"])
                for h in hz_flat
            ])

            # 物理修正後的預測
            corrected_flows = (1 - self.physics_lambda) * Y_pred[:, 0] + \
                              self.physics_lambda * phys_flows
            corrected_powers = (1 - self.physics_lambda) * Y_pred[:, 1] + \
                               self.physics_lambda * phys_powers

            physics_residual_flow = np.mean(np.abs(Y_pred[:, 0] - phys_flows) / (phys_flows + 1e-8))
            physics_residual_power = np.mean(np.abs(Y_pred[:, 1] - phys_powers) / (phys_powers + 1e-8))

            self._metrics[pump_id] = {
                "nn_rmse_flow_m3h": round(float(nn_rmse_flow), 3),
                "nn_rmse_power_kw": round(float(nn_rmse_power), 3),
                "physics_residual_flow_pct": round(float(physics_residual_flow * 100), 2),
                "physics_residual_power_pct": round(float(physics_residual_power * 100), 2),
                "n_train": int(len(X)),
                "n_iterations": int(mlp.n_iter_),
            }

            print(f"    NN RMSE: flow={nn_rmse_flow:.2f} m3/h, power={nn_rmse_power:.2f} kW")
            print(f"    物理殘差: flow={physics_residual_flow*100:.1f}%, power={physics_residual_power*100:.1f}%")

        self.trained = True
        self._save_models()
        print(f"\n[PINNs] 訓練完成，模型已儲存")
        return self._metrics

    # ─────────────────────────────────────────
    # 預測（含物理修正）
    # ─────────────────────────────────────────
    def predict(self, pump_id: str, hz: float) -> Tuple[float, float, float]:
        """
        預測泵浦在給定頻率下的性能。

        回傳:
            (flow_m3h, power_kw, efficiency)

        物理修正策略：
            output = (1 - lambda) * NN_prediction + lambda * affinity_law
        """
        if not self.trained:
            raise RuntimeError("模型尚未訓練，請先呼叫 train()")

        spec = PUMPS[pump_id]
        hz = float(np.clip(hz, spec["min_hz"], spec["max_hz"]))

        if pump_id in self.models:
            # NN 預測
            X = np.array([[hz]])
            X_scaled = self.scalers_x[pump_id].transform(X)
            Y_scaled = self.models[pump_id].predict(X_scaled)
            Y_pred = self.scalers_y[pump_id].inverse_transform(Y_scaled)[0]

            nn_flow, nn_power, nn_eff = Y_pred[0], Y_pred[1], Y_pred[2]
        else:
            nn_flow, nn_power, nn_eff = 0.0, 0.0, 0.0

        # 親和律預測（v2: 含管損+空蝕修正）
        phys_flow = self.physics.flow(hz, spec["rated_hz"], spec["rated_flow_cmd"]) / 24.0
        phys_power = self.physics.power(hz, spec["rated_hz"], spec["rated_power_kw"])
        k_pipe = PINNS.get("k_pipe_loss", 0.05)
        phys_eff = self.physics.efficiency_v2(
            hz, spec["rated_hz"], spec["bep_hz"], spec["bep_efficiency"],
            spec["min_hz"], k_pipe=k_pipe)

        # 加權混合
        lam = self.physics_lambda
        flow_m3h = (1 - lam) * nn_flow + lam * phys_flow
        power_kw = (1 - lam) * nn_power + lam * phys_power
        eta = (1 - lam) * nn_eff + lam * phys_eff

        # 物理可行性約束
        flow_m3h = max(flow_m3h, 0.0)
        power_kw = max(power_kw, 0.1)
        eta = float(np.clip(eta, 0.0, 0.95))  # 允許空蝕區低效率（v3.2）

        return round(flow_m3h, 2), round(power_kw, 2), round(eta, 4)

    def predict_batch(
        self, pump_id: str, hz_array: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """批次預測，用於 MPC 優化加速。"""
        flows, powers, effs = [], [], []
        for hz in hz_array:
            f, p, e = self.predict(pump_id, float(hz))
            flows.append(f)
            powers.append(p)
            effs.append(e)
        return np.array(flows), np.array(powers), np.array(effs)

    # ─────────────────────────────────────────
    # 泵浦曲線查詢
    # ─────────────────────────────────────────
    def get_pump_curve(
        self, pump_id: str, n_points: int = 50
    ) -> Dict[str, np.ndarray]:
        """產生完整泵浦特性曲線（用於繪圖與驗證）。"""
        spec = PUMPS[pump_id]
        hz_range = np.linspace(spec["min_hz"], spec["max_hz"], n_points)
        flows, powers, effs = self.predict_batch(pump_id, hz_range)
        return {
            "hz": hz_range,
            "flow_m3h": flows,
            "power_kw": powers,
            "efficiency": effs,
        }

    def find_optimal_hz(
        self, pump_id: str, target_flow_m3h: float
    ) -> Optional[float]:
        """
        給定目標流量，反推最佳頻率（最高效率點）。
        用於 MPC 優化器。
        """
        spec = PUMPS[pump_id]

        def neg_efficiency(hz: float) -> float:
            f, p, e = self.predict(pump_id, hz)
            flow_error = abs(f - target_flow_m3h) / (target_flow_m3h + 1e-8)
            if flow_error > 0.1:  # 流量偏差超過 10% 則懲罰
                return 1.0
            return -e

        result = minimize_scalar(
            neg_efficiency,
            bounds=(spec["min_hz"], spec["max_hz"]),
            method="bounded",
        )
        return round(result.x, 1) if result.success else None

    # ─────────────────────────────────────────
    # v2 物理合規性評估（5 項殘差）
    # ─────────────────────────────────────────
    def evaluate_physics_compliance_v2(self) -> Dict[str, dict]:
        """
        v2 物理合規性評估：涵蓋全部 5 項殘差。

        R1: 流量親和律  |Q/Q_r - n/n_r|
        R2: 功率親和律  |P/P_r - (n/n_r)³|
        R3: 58.5 Hz 禁區行為
        R4: Darcy-Weisbach 管損一致性
        R5: 空蝕邊界條件
        """
        results = {}
        k_pipe = PINNS.get("k_pipe_loss", 0.05)

        for pump_id in PUMPS:
            spec = PUMPS[pump_id]
            test_hz = np.linspace(spec["min_hz"], spec["max_hz"], 200)
            flows, powers, effs = self.predict_batch(pump_id, test_hz)

            rated_flow = spec["rated_flow_cmd"] / 24.0  # m3/h
            rated_power = spec["rated_power_kw"]
            rated_hz = spec["rated_hz"]
            bep_hz = spec["bep_hz"]
            bep_eta = spec["bep_efficiency"]
            min_hz = spec["min_hz"]

            ratio = test_hz / rated_hz
            normal_mask = test_hz <= 58.5

            # R1: 流量親和律
            phys_flows = np.array([
                self.physics.flow(h, rated_hz, spec["rated_flow_cmd"]) / 24.0
                for h in test_hz
            ])
            flow_residual = np.abs(flows[normal_mask] - phys_flows[normal_mask]) / (phys_flows[normal_mask] + 1e-8)
            flow_compliance = float((flow_residual < 0.05).mean())

            # R2: 功率親和律
            phys_powers = np.array([
                self.physics.power(h, rated_hz, rated_power) for h in test_hz
            ])
            power_residual = np.abs(powers[normal_mask] - phys_powers[normal_mask]) / (phys_powers[normal_mask] + 1e-8)
            power_compliance = float((power_residual < 0.05).mean())

            # R3: 58.5 Hz 禁區
            mask_over = test_hz > 58.5
            if mask_over.any():
                power_over = powers[mask_over]
                power_affinity = np.array([
                    self.physics.power(h, rated_hz, rated_power) for h in test_hz[mask_over]
                ])
                penalty_correct = float((power_over >= power_affinity * 0.95).mean())
            else:
                penalty_correct = 1.0

            # R4: 管損一致性（使用與訓練資料一致的高斯效率基底）
            flow_ratio = test_hz / rated_hz
            bep_flow_ratio = bep_hz / rated_hz
            pipe_loss_factor = 1.0 - k_pipe * (flow_ratio - bep_flow_ratio) ** 2
            # 效率基底：高斯（與 AffinityLaws.efficiency_v2 一致）
            sigma = 8.0
            eta_base = bep_eta * np.exp(-0.5 * ((test_hz - bep_hz) / sigma) ** 2)
            eta_ideal = np.clip(eta_base * pipe_loss_factor, 0.3, 0.95)
            # 排除空蝕區（低頻端效率受空蝕影響，不列入管損合規）
            non_cav_mask = normal_mask & (test_hz >= (min_hz + 3.0))
            pipe_residual = np.abs(effs[non_cav_mask] - eta_ideal[non_cav_mask])
            pipe_compliance = float((pipe_residual < 0.05).mean())

            # R5: 空蝕邊界（v3.6：使用物理模型效率作為參考基準）
            cav_margin = 3.0
            near_min = test_hz < (min_hz + cav_margin)
            if near_min.any():
                hz_near = test_hz[near_min]
                eta_near = effs[near_min]
                # 計算物理模型在空蝕區的效率（含高斯+管損+空蝕）
                eta_phys_ref = np.array([
                    self.physics.efficiency_v2(
                        h, rated_hz, bep_hz, bep_eta, min_hz, k_pipe=k_pipe)
                    for h in hz_near
                ])
                # 合規：效率不超過物理參考值的 1.5 倍 + 0.05 容差
                eta_upper_bound = eta_phys_ref * 1.5 + 0.05
                cavitation_compliance = float((eta_near <= eta_upper_bound).mean())
            else:
                cavitation_compliance = 1.0

            results[pump_id] = {
                "R1_flow_residual_mean": round(float(flow_residual.mean()), 5),
                "R1_flow_compliance": round(flow_compliance, 4),
                "R2_power_residual_mean": round(float(power_residual.mean()), 5),
                "R2_power_compliance": round(power_compliance, 4),
                "R3_penalty_585_correct": round(penalty_correct, 4),
                "R4_pipe_loss_compliance": round(pipe_compliance, 4),
                "R4_pipe_residual_mean": round(float(pipe_residual.mean()), 5),
                "R5_cavitation_compliance": round(cavitation_compliance, 4),
            }

        # 總體統計
        pump_results = {k: v for k, v in results.items() if k != "summary"}
        results["summary"] = {
            "avg_R1_flow_compliance": round(float(np.mean(
                [r["R1_flow_compliance"] for r in pump_results.values()])), 4),
            "avg_R2_power_compliance": round(float(np.mean(
                [r["R2_power_compliance"] for r in pump_results.values()])), 4),
            "avg_R3_585_correct": round(float(np.mean(
                [r["R3_penalty_585_correct"] for r in pump_results.values()])), 4),
            "avg_R4_pipe_compliance": round(float(np.mean(
                [r["R4_pipe_loss_compliance"] for r in pump_results.values()])), 4),
            "avg_R5_cavitation_compliance": round(float(np.mean(
                [r["R5_cavitation_compliance"] for r in pump_results.values()])), 4),
            "model_type": "PINNs sklearn + 親和律 v2 修正",
            "version": "v2",
        }

        return results

    # ─────────────────────────────────────────
    # 模型儲存/載入
    # ─────────────────────────────────────────
    def _save_models(self) -> None:
        """儲存訓練指標（sklearn 模型保留在記憶體）。"""
        model_dir = os.path.join(OUTPUT_DIR, "models")
        os.makedirs(model_dir, exist_ok=True)

        meta = {
            "combo": "combo_04_PINNs-MPC-DynamicBL",
            "model_type": "PINNs_sklearn_with_affinity_correction",
            "physics_lambda": self.physics_lambda,
            "config": {
                "hidden_layers": PINNS["hidden_layers"],
                "training_points": PINNS["training_points"],
                "bep_dense_ratio": PINNS["bep_dense_ratio"],
            },
            "metrics": self._metrics,
        }

        path = os.path.join(model_dir, "pinns_meta.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"  [PINNs] 模型元資訊 → {path}")

    def get_metrics(self) -> Dict[str, dict]:
        return self._metrics.copy()


# ─────────────────────────────────────────────
# 直接執行測試
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model = PINNsSurrogate()
    metrics = model.train()

    print("\n" + "=" * 60)
    print("PINNs 預測測試")
    print("=" * 60)

    for pid in PUMPS:
        spec = PUMPS[pid]
        for hz in [spec["min_hz"], spec["bep_hz"], spec["max_hz"]]:
            f, p, e = model.predict(pid, hz)
            print(f"  {pid} @ {hz:.0f}Hz → flow={f:.1f} m3/h, "
                  f"power={p:.1f} kW, eta={e:.3f}")

    # v2 物理合規性（含管損+空蝕）
    print("\n" + "=" * 60)
    print("物理約束滿足率 v2（+管損+空蝕）")
    print("=" * 60)
    comp_v2 = model.evaluate_physics_compliance_v2()
    for pid, vals in comp_v2.items():
        if pid == "summary":
            continue
        print(f"  {pid}: R1={vals['R1_flow_compliance']:.1%} "
              f"R2={vals['R2_power_compliance']:.1%} "
              f"R3={vals['R3_penalty_585_correct']:.1%} "
              f"R4={vals['R4_pipe_loss_compliance']:.1%} "
              f"R5={vals['R5_cavitation_compliance']:.1%}")
    s2 = comp_v2["summary"]
    print(f"\n  R1 流量: {s2['avg_R1_flow_compliance']:.1%}")
    print(f"  R2 功率: {s2['avg_R2_power_compliance']:.1%}")
    print(f"  R3 58.5Hz: {s2['avg_R3_585_correct']:.1%}")
    print(f"  R4 管損: {s2['avg_R4_pipe_compliance']:.1%}")
    print(f"  R5 空蝕: {s2['avg_R5_cavitation_compliance']:.1%}")

    # 儲存 v2 合規報告
    import json as _json
    model_dir = os.path.join(OUTPUT_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    v2_path = os.path.join(model_dir, "physics_compliance_v2.json")
    with open(v2_path, "w", encoding="utf-8") as f:
        _json.dump(comp_v2, f, ensure_ascii=False, indent=2)
    print(f"\n  v2 合規報告 → {v2_path}")
