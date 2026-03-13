# -*- coding: utf-8 -*-
"""
MPC (Model Predictive Control) 滾動預測控制最佳化器 v2
Combo 04: 24 小時預測視窗 + LSTM 需求預測 + 預充策略

v2 核心升級：
- 預測視窗: 6h → 24h（全日前瞻）
- 需求來源: 靜態模式 → DemandForecaster ML 預測
- 預充策略: 偵測未來 3h 內尖峰 → 離峰預充至 2.8m → 尖峰減機省電
- 滾動更新: 每小時以實際水位 + 更新預測重新規劃剩餘時段

節費機制：
  24h 前瞻看到 hour 9-11 尖峰 (4.02 NTD/kWh)
  → hour 5-8 離峰 (1.24 NTD/kWh) 預充配水池至 2.8m
  → hour 9-11 關閉 1 台泵浦
  → 省 ~4.02 × 95kW × 3h = 1,143 NTD/day
"""

import json
import os
from datetime import datetime
from itertools import combinations, product
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from config import (CONTRACT_CAPACITY_KW, DAILY_TARGET_CMD, HOURLY_RATE,
                        MPC, OVERCONTRACT_PENALTY, POOL_AREA_M2,
                        POOL_LEVEL_INIT, POOL_LEVEL_MAX, POOL_LEVEL_MIN,
                        PUMPS, OUTPUT_DIR, TOU)
    from surrogate_model import PINNsSurrogate
    from demand_forecast import DemandForecaster
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import (CONTRACT_CAPACITY_KW, DAILY_TARGET_CMD, HOURLY_RATE,
                        MPC, OVERCONTRACT_PENALTY, POOL_AREA_M2,
                        POOL_LEVEL_INIT, POOL_LEVEL_MAX, POOL_LEVEL_MIN,
                        PUMPS, OUTPUT_DIR, TOU)
    from surrogate_model import PINNsSurrogate
    from demand_forecast import DemandForecaster


class HourlyAction:
    """單一小時的排程動作。"""

    def __init__(self, hour: int) -> None:
        self.hour: int = hour
        self.pump_states: Dict[str, float] = {}  # {pump_id: hz or 0}
        self.total_flow_m3h: float = 0.0
        self.total_power_kw: float = 0.0
        self.electricity_cost_ntd: float = 0.0
        self.pool_level: float = 0.0
        self.demand_m3h: float = 0.0
        self.is_precharge: bool = False  # v2: 標記預充時段

    def to_dict(self) -> dict:
        return {
            "hour": self.hour,
            "pump_states": {k: round(v, 1) for k, v in self.pump_states.items()},
            "total_flow_m3h": round(self.total_flow_m3h, 1),
            "total_power_kw": round(self.total_power_kw, 1),
            "electricity_cost_ntd": round(self.electricity_cost_ntd, 1),
            "pool_level_m": round(self.pool_level, 3),
            "demand_m3h": round(self.demand_m3h, 1),
            "tou_rate": HOURLY_RATE.get(self.hour, 2.5),
            "is_precharge": self.is_precharge,
        }


class MPCOptimizer:
    """
    Model Predictive Control 滾動預測控制器 v2。

    運作流程：
    1. hour=0: DemandForecaster 預測全日 24h 需求曲線
    2. 每小時 h:
       a. 以 PINNs 代理模型 + 需求預測，最佳化剩餘 [h..23] 小時
       b. 預充偵測: 若未來 3h 有尖峰時段，且當前為離峰，拉高水位至 2.8m
       c. 執行 hour h 的動作
       d. 以實際水位更新狀態
       e. 下一小時重複
    """

    def __init__(
        self,
        surrogate: PINNsSurrogate,
        forecaster: Optional[DemandForecaster] = None,
    ) -> None:
        self.surrogate = surrogate
        self.forecaster = forecaster
        self.horizon: int = MPC["horizon_hours"]
        self.freq_steps: List[float] = [0.0] + MPC["freq_steps"]  # 0 = off
        self.pump_ids: List[str] = list(PUMPS.keys())

        # 預充參數
        self.precharge_threshold_h: int = MPC.get("precharge_threshold_hours", 3)
        self.precharge_level: float = MPC.get("precharge_level", 2.8)

        # TOU 時段分類
        self._peak_hours: set = set(TOU["peak"]["hours"])
        self._offpeak_hours: set = set(TOU["offpeak"]["hours"])
        self._semi_hours: set = set(TOU["semi"]["hours"])

        # 預計算每台泵浦各頻率的 flow/power (避免重複呼叫 surrogate)
        self._pump_cache: Dict[str, Dict[float, Tuple[float, float, float]]] = {}
        self._cache_built = False

        # 統計
        self.re_optimization_count: int = 0
        self.constraint_violations: List[dict] = []
        self.precharge_events: int = 0

    def _build_pump_cache(self) -> None:
        """預計算所有泵浦在各頻率的 flow/power/eff。"""
        if self._cache_built:
            return
        all_freqs = [40.0, 44.0, 48.0, 50.0, 52.0, 54.0, 56.0]
        for pid in self.pump_ids:
            self._pump_cache[pid] = {}
            for hz in all_freqs:
                flow, power, eff = self.surrogate.predict(pid, hz)
                self._pump_cache[pid][hz] = (flow, power, eff)
        self._cache_built = True

    def _compute_tou_plan(self, demand_forecast: np.ndarray) -> Dict[int, float]:
        """
        預計算每小時的目標功率，實現 TOU 套利。

        策略：
        1. 離峰 (0-6, 22-23): 5 泵 @ 高頻率，超額供水蓄池
        2. 尖峰 (9-11, 13-16): 盡量低功率，消耗池存
        3. 半尖峰: 匹配需求

        以配水池物理模擬驗證可行性，確保水位始終在 [1.5, 3.0]。
        """
        self._build_pump_cache()

        # 預計算常用方案的 flow/power
        def scheme(hz_list):
            """計算指定頻率組合的總 flow/power。"""
            f = sum(self._pump_cache[p][hz][0] for p, hz in zip(self.pump_ids, hz_list))
            pw = sum(self._pump_cache[p][hz][1] for p, hz in zip(self.pump_ids, hz_list))
            return f, pw

        flow_5x56, power_5x56 = scheme([56]*5)
        flow_5x54, power_5x54 = scheme([54]*5)
        flow_5x52, power_5x52 = scheme([52]*5)
        flow_5x48, power_5x48 = scheme([48]*5)
        flow_5x44, power_5x44 = scheme([44]*5)
        flow_5x40, power_5x40 = scheme([40]*5)
        flow_4x48, power_4x48 = sum(self._pump_cache[p][48.0][0] for p in self.pump_ids if p != 'P4'), \
                                 sum(self._pump_cache[p][48.0][1] for p in self.pump_ids if p != 'P4')
        flow_3x48, power_3x48 = sum(self._pump_cache[p][48.0][0] for p in ['P1', 'P2', 'P5']), \
                                 sum(self._pump_cache[p][48.0][1] for p in ['P1', 'P2', 'P5'])

        # 方案字典 {label: (flow, power)}
        schemes = {
            '5x56': (flow_5x56, power_5x56),
            '5x54': (flow_5x54, power_5x54),
            '5x52': (flow_5x52, power_5x52),
            '5x48': (flow_5x48, power_5x48),
            '5x44': (flow_5x44, power_5x44),
            '5x40': (flow_5x40, power_5x40),
            '4x48': (flow_4x48, power_4x48),
            '3x48': (flow_3x48, power_3x48),
        }

        # 多輪模擬找最低成本計畫
        best_total_cost = float('inf')
        best_targets = {}

        # 嘗試不同離峰/尖峰組合
        for offpeak_scheme in ['5x56', '5x54', '5x52']:
            for peak_scheme_options in [
                ['5x40', '5x44', '5x48'],     # 降頻策略
                ['3x48', '4x48', '5x48'],     # 減機策略
                ['5x44', '4x48', '5x48'],     # 混合策略
            ]:

                pool = POOL_LEVEL_INIT
                targets: Dict[int, float] = {}
                total_sim_cost = 0.0
                feasible = True

                for h in range(24):
                    demand = demand_forecast[h] if h < len(demand_forecast) else self._base_demand_m3h()
                    rate = HOURLY_RATE.get(h, 2.5)

                    if h in self._offpeak_hours:
                        # 離峰：高出力蓄池（但不超 POOL_LEVEL_MAX）
                        if pool < POOL_LEVEL_MAX - 0.2:
                            flow, power = schemes[offpeak_scheme]
                        else:
                            # 水位接近上限，降到 5x48 匹配需求
                            flow, power = schemes['5x48']
                        targets[h] = power

                    elif h in self._peak_hours:
                        # 尖峰：根據水位餘裕選減載程度
                        chosen = '5x48'  # 預設
                        pool_margin = pool - POOL_LEVEL_MIN
                        for sch in peak_scheme_options:
                            sch_flow, sch_power = schemes[sch]
                            # 檢查消耗池存後水位是否安全
                            test_pool = pool + (sch_flow - demand) / POOL_AREA_M2
                            if test_pool >= POOL_LEVEL_MIN - 0.05:
                                chosen = sch
                                break
                        flow, power = schemes[chosen]
                        targets[h] = power

                    else:  # semi-peak
                        # 半尖峰：匹配需求 + 適當預充
                        if pool < 2.0:
                            flow, power = schemes['5x52']
                        else:
                            flow, power = schemes['5x48']
                        targets[h] = power

                    # 更新模擬水位
                    net = flow - demand
                    pool += net / POOL_AREA_M2

                    # 檢查可行性
                    if pool < POOL_LEVEL_MIN - 0.1:
                        feasible = False
                        break
                    pool = min(pool, POOL_LEVEL_MAX)

                    total_sim_cost += power * rate

                if feasible and total_sim_cost < best_total_cost:
                    best_total_cost = total_sim_cost
                    best_targets = targets.copy()

        # 若無可行解，回退全部 5x48
        if not best_targets:
            for h in range(24):
                best_targets[h] = power_5x48

        return best_targets

    def _base_demand_m3h(self) -> float:
        return DAILY_TARGET_CMD / 24.0

    # -----------------------------------------
    # 候選方案生成（剪枝版）
    # -----------------------------------------
    def _generate_candidates(
        self, hour: int, target_pool_direction: str = "normal"
    ) -> List[Dict[str, float]]:
        """
        產生泵浦組合候選方案。

        target_pool_direction:
            "normal": 正常供水
            "precharge": 預充模式，偏好高流量方案
            "reduce": 尖峰減載，偏好低功率方案
        """
        # 頻率選項: 含低頻(尖峰減載用)和高頻(離峰預充用)
        if target_pool_direction == "reduce":
            # 減載：低頻選項 + 少台數
            freq_options = [40.0, 44.0, 48.0]
            pump_range = [3, 4, 5]
        elif target_pool_direction == "precharge":
            # 預充：高頻選項
            freq_options = [48.0, 52.0, 54.0, 56.0]
            pump_range = [4, 5]
        else:
            freq_options = [44.0, 48.0, 50.0, 52.0, 54.0, 56.0]
            pump_range = [3, 4, 5]

        candidates = []
        for n_pumps in pump_range:
            for pump_combo in combinations(self.pump_ids, n_pumps):
                for freq_combo in product(freq_options, repeat=n_pumps):
                    candidate = {pid: 0.0 for pid in self.pump_ids}
                    for pid, freq in zip(pump_combo, freq_combo):
                        candidate[pid] = freq
                    candidates.append(candidate)

        # 全關（緊急用）
        candidates.append({pid: 0.0 for pid in self.pump_ids})
        return candidates

    # -----------------------------------------
    # 單步成本評估
    # -----------------------------------------
    def _evaluate_candidate(
        self,
        candidate: Dict[str, float],
        hour: int,
        current_pool_level: float,
        demand_m3h: float,
        is_precharge: bool = False,
    ) -> Tuple[float, float, float, float]:
        """
        評估一個候選方案的成本。

        TOU 套利核心邏輯：
        - 離峰 (1.24): 鼓勵高出力，預充配水池
        - 尖峰 (4.02): 獎勵低功率，消耗配水池存量
        - 水位允許範圍內最大化費率差異利用

        Returns:
            (total_cost, total_flow, total_power, new_pool_level)
        """
        total_flow = 0.0
        total_power = 0.0

        for pid, hz in candidate.items():
            if hz < PUMPS[pid]["min_hz"]:
                continue
            # 使用快取
            if pid in self._pump_cache and hz in self._pump_cache[pid]:
                flow, shaft_power, eff = self._pump_cache[pid][hz]
            else:
                flow, shaft_power, eff = self.surrogate.predict(pid, hz)
            # 電功率 = 軸功率 / 效率（BEP 偏離時效率下降，電功率正確提高）
            electrical_power = shaft_power / max(eff, 0.05)
            total_flow += flow
            total_power += electrical_power

        # 電費成本
        rate = HOURLY_RATE.get(hour % 24, 2.5)
        electricity_cost = total_power * rate

        # 超約罰款
        overcontract_cost = 0.0
        if total_power > CONTRACT_CAPACITY_KW:
            excess = total_power - CONTRACT_CAPACITY_KW
            overcontract_cost = excess * rate * OVERCONTRACT_PENALTY

        # 水位更新
        net_flow = total_flow - demand_m3h  # m3/h
        delta_level = net_flow / POOL_AREA_M2  # m (1hr)
        new_pool_level = current_pool_level + delta_level

        # 水位懲罰（硬約束）
        pool_penalty = 0.0
        if new_pool_level < POOL_LEVEL_MIN:
            pool_penalty = MPC["demand_penalty"] * (POOL_LEVEL_MIN - new_pool_level) ** 2
        elif new_pool_level > POOL_LEVEL_MAX:
            pool_penalty = MPC["pool_weight"] * (new_pool_level - POOL_LEVEL_MAX) ** 2

        # TOU 套利激勵 — 由 _compute_tou_plan 預計算的目標功率引導
        tou_incentive = 0.0
        if hasattr(self, '_tou_power_target') and hour in self._tou_power_target:
            target_power = self._tou_power_target[hour]
            deviation = abs(total_power - target_power)
            # 強力懲罰偏離目標（比電費差異大，確保跟隨計畫）
            tou_incentive = 5.0 * deviation

        # 供水不足懲罰
        # 策略: 泵浦供水 + 配水池存量 = 總供水能力
        # 若配水池有餘量，允許泵浦低於需求（配水池補差額）
        demand_penalty = 0.0
        if total_flow < demand_m3h * 0.8:
            # 嚴重不足 (< 80% 需求)
            if new_pool_level < POOL_LEVEL_MIN:
                shortfall = demand_m3h - total_flow
                demand_penalty = MPC["demand_penalty"] * shortfall
            elif new_pool_level < POOL_LEVEL_MIN + 0.1:
                # 水位接近下限，輕微懲罰
                shortfall = demand_m3h - total_flow
                demand_penalty = MPC["demand_penalty"] * 0.3 * shortfall

        total_cost = (electricity_cost + overcontract_cost +
                      pool_penalty + demand_penalty + tou_incentive)

        return total_cost, total_flow, total_power, new_pool_level

    # -----------------------------------------
    # 預充偵測
    # -----------------------------------------
    def _should_precharge(
        self,
        current_hour: int,
        current_pool: float,
        demand_forecast: np.ndarray,
        forecast_start_hour: int,
    ) -> bool:
        """
        判斷是否應啟動預充。

        條件：
        1. 當前為離峰或半尖峰時段
        2. 未來 precharge_threshold_h 小時內有尖峰時段
        3. 當前水位低於預充目標
        """
        if current_hour in self._peak_hours:
            return False
        if current_pool >= self.precharge_level:
            return False

        # 檢查未來 N 小時是否有尖峰
        for offset in range(1, self.precharge_threshold_h + 1):
            future_hour = (current_hour + offset) % 24
            if future_hour in self._peak_hours:
                return True

        return False

    # -----------------------------------------
    # 單次視窗最佳化 (v2: 使用需求預測)
    # -----------------------------------------
    def optimize_horizon(
        self,
        start_hour: int,
        end_hour: int,
        current_pool_level: float,
        demand_forecast: np.ndarray,
    ) -> List[HourlyAction]:
        """
        最佳化 [start_hour, end_hour] 的排程。

        Args:
            start_hour: 起始小時
            end_hour: 結束小時 (inclusive)
            current_pool_level: 當前水位
            demand_forecast: 剩餘時段的需求預測 (m3/h)

        Returns:
            List[HourlyAction]
        """
        self.re_optimization_count += 1
        actions: List[HourlyAction] = []

        pool_level = current_pool_level

        for idx, h in enumerate(range(start_hour, end_hour + 1)):
            if idx >= len(demand_forecast):
                break

            demand = demand_forecast[idx]

            # 預充判斷
            is_precharge = self._should_precharge(
                h, pool_level, demand_forecast[idx:], h
            )

            # 尖峰減載判斷
            is_peak_reduce = (h in self._peak_hours and
                              pool_level > POOL_LEVEL_MIN + 0.5)

            # 選擇候選方案策略
            if is_precharge:
                direction = "precharge"
            elif is_peak_reduce:
                direction = "reduce"
            else:
                direction = "normal"

            candidates = self._generate_candidates(h, direction)

            best_cost = float("inf")
            best_candidate = None
            best_flow = 0.0
            best_power = 0.0
            best_pool = pool_level

            for candidate in candidates:
                cost, flow, power, new_pool = self._evaluate_candidate(
                    candidate, h, pool_level, demand, is_precharge
                )

                # 尖峰減載：額外獎勵低功率方案
                if is_peak_reduce and power < 400:
                    cost -= 500.0  # 鼓勵減載

                # 前瞻一步
                if idx + 1 < len(demand_forecast):
                    if new_pool < POOL_LEVEL_MIN * 0.9:
                        cost += MPC["demand_penalty"] * 2

                if cost < best_cost:
                    best_cost = cost
                    best_candidate = candidate
                    best_flow = flow
                    best_power = power
                    best_pool = new_pool

            # 建立動作
            action = HourlyAction(h)
            if best_candidate:
                action.pump_states = best_candidate
                action.total_flow_m3h = best_flow
                action.total_power_kw = best_power
                action.pool_level = best_pool
                action.is_precharge = is_precharge
            else:
                action.pool_level = pool_level

            action.demand_m3h = demand
            action.electricity_cost_ntd = best_power * HOURLY_RATE.get(h % 24, 2.5)

            actions.append(action)

            # 狀態轉移
            pool_level = best_pool

            # 約束違反記錄
            if pool_level < POOL_LEVEL_MIN:
                self.constraint_violations.append({
                    "hour": h, "type": "pool_low",
                    "value": round(pool_level, 3), "limit": POOL_LEVEL_MIN,
                })
            if best_power > CONTRACT_CAPACITY_KW:
                self.constraint_violations.append({
                    "hour": h, "type": "overcontract",
                    "value": round(best_power, 1), "limit": CONTRACT_CAPACITY_KW,
                })

        return actions

    # -----------------------------------------
    # 全日 MPC 排程 v2
    # -----------------------------------------
    def run_day(self, date_str: str = "2024-01-15") -> Dict:
        """
        執行 24 小時 MPC 滾動排程 v2。

        核心流程：
        1. hour=0: 以 DemandForecaster 預測全日 24h 需求
        2. 每小時 h: 重新最佳化 [h..23]，只執行 hour h
        3. 預充策略自動介入離峰→尖峰轉換期

        Returns:
            完整日排程 dict (含 demand_forecast_accuracy, precharge_events)
        """
        print(f"\n{'='*60}")
        print(f"[MPC v2] 開始 24 小時滾動排程: {date_str}")
        print(f"  預測視窗: {self.horizon} 小時 (全日前瞻)")
        print(f"  需求預測: {'DemandForecaster' if self.forecaster and self.forecaster.fitted else '靜態模式'}")
        print(f"  預充策略: 尖峰前 {self.precharge_threshold_h}h 啟動, 目標 {self.precharge_level}m")
        print(f"  重新最佳化間隔: {MPC['re_optimize_interval']} 小時")
        print(f"{'='*60}")

        self.re_optimization_count = 0
        self.constraint_violations = []
        self.precharge_events = 0

        # 預計算泵浦快取
        self._build_pump_cache()

        # Step 0: TOU 套利計畫會在需求預測後計算

        # Step 1: 全日需求預測
        if self.forecaster and self.forecaster.fitted:
            demand_forecast_full = self.forecaster.predict_24h(0)
        else:
            # 靜態回退
            hourly_avg = DAILY_TARGET_CMD / 24.0
            demand_forecast_full = np.array([
                hourly_avg * self._static_factor(h) for h in range(24)
            ])

        print(f"\n  [需求預測] 24h 預測完成:")
        print(f"    平均: {demand_forecast_full.mean():.0f} m3/h")
        print(f"    最高: {demand_forecast_full.max():.0f} m3/h (hour {np.argmax(demand_forecast_full)})")
        print(f"    最低: {demand_forecast_full.min():.0f} m3/h (hour {np.argmin(demand_forecast_full)})")

        # Step 1.5: 計算 TOU 套利計畫
        self._tou_power_target = self._compute_tou_plan(demand_forecast_full)
        print(f"\n  [TOU 計畫] 目標功率:")
        for h in range(24):
            rate = HOURLY_RATE.get(h, 2.5)
            target = self._tou_power_target.get(h, 0)
            label = "離峰" if h in self._offpeak_hours else ("尖峰" if h in self._peak_hours else "半尖")
            print(f"    H{h:02d} [{label}] rate={rate:.2f} target={target:.0f}kW")

        # Step 2: 滾動排程
        pool_level = POOL_LEVEL_INIT
        daily_schedule: List[HourlyAction] = []
        total_energy = 0.0
        total_flow = 0.0
        total_cost = 0.0
        actual_demands: List[float] = []

        for hour in range(24):
            # 剩餘時段需求預測 (只最佳化前 6 步，用 24h 需求做預充判斷)
            remaining_demand = demand_forecast_full[hour:]
            lookahead = min(6, 24 - hour)  # 貪心視窗 6 步

            horizon_actions = self.optimize_horizon(
                start_hour=hour,
                end_hour=min(hour + lookahead - 1, 23),
                current_pool_level=pool_level,
                demand_forecast=remaining_demand[:lookahead],
            )

            # 取第一步動作
            action = horizon_actions[0]

            # 模擬真實需求（加入小擾動）
            rng = np.random.RandomState(hour + 100)
            actual_demand = demand_forecast_full[hour] * (1 + rng.normal(0, 0.02))
            actual_demands.append(actual_demand)

            # 以實際需求重算水位
            net_flow = action.total_flow_m3h - actual_demand
            delta_level = net_flow / POOL_AREA_M2
            actual_pool = pool_level + delta_level

            # 水位擾動
            disturbance = rng.normal(0, 0.001)
            actual_pool += disturbance
            actual_pool = np.clip(actual_pool, POOL_LEVEL_MIN * 0.8, POOL_LEVEL_MAX * 1.1)

            action.pool_level = actual_pool
            action.demand_m3h = actual_demand

            # 記錄預充事件
            if action.is_precharge:
                self.precharge_events += 1

            daily_schedule.append(action)

            # 狀態更新
            pool_level = actual_pool
            total_energy += action.total_power_kw
            total_flow += action.total_flow_m3h
            total_cost += action.electricity_cost_ntd

            # 列印排程
            active_pumps = [
                f"{pid}@{hz:.0f}Hz"
                for pid, hz in action.pump_states.items()
                if hz >= PUMPS[pid]["min_hz"]
            ]
            precharge_tag = " [預充]" if action.is_precharge else ""
            peak_tag = " [尖峰]" if hour in self._peak_hours else ""
            print(f"    H{hour:02d} | "
                  f"flow={action.total_flow_m3h:7.0f} m3/h | "
                  f"power={action.total_power_kw:6.1f} kW | "
                  f"pool={action.pool_level:.3f}m | "
                  f"rate={HOURLY_RATE.get(hour, 2.5):.2f} | "
                  f"{', '.join(active_pumps)}{precharge_tag}{peak_tag}")

        # 匯總
        supply_rate = total_flow / DAILY_TARGET_CMD * 100
        avg_power = total_energy / 24.0

        # 需求預測精度
        forecast_mae = np.mean(np.abs(
            demand_forecast_full - np.array(actual_demands)
        ))
        forecast_mape = np.mean(np.abs(
            (demand_forecast_full - np.array(actual_demands)) /
            (np.array(actual_demands) + 1e-8)
        )) * 100

        result = {
            "date": date_str,
            "combo": "combo_04_PINNs-MPC-DynamicBL_v2",
            "version": "v2_24h_forecast",
            "schedule": [a.to_dict() for a in daily_schedule],
            "demand_forecast": demand_forecast_full.tolist(),
            "actual_demands": actual_demands,
            "summary": {
                "total_energy_kwh": round(total_energy, 1),
                "total_flow_m3": round(total_flow, 1),
                "total_cost_ntd": round(total_cost, 0),
                "avg_power_kw": round(avg_power, 1),
                "supply_compliance_pct": round(supply_rate, 2),
                "final_pool_level_m": round(pool_level, 3),
                "re_optimization_count": self.re_optimization_count,
                "constraint_violations": len(self.constraint_violations),
                "sec_kwh_per_m3": round(total_energy / (total_flow + 1e-8), 4),
                "precharge_events": self.precharge_events,
                "forecast_mae_m3h": round(float(forecast_mae), 1),
                "forecast_mape_pct": round(float(forecast_mape), 2),
            },
            "constraint_violations": self.constraint_violations,
        }

        # TOU 分段電費
        tou_breakdown = {"peak": 0.0, "semi": 0.0, "offpeak": 0.0}
        for action in daily_schedule:
            h = action.hour
            cost = action.electricity_cost_ntd
            if h in TOU["peak"]["hours"]:
                tou_breakdown["peak"] += cost
            elif h in TOU["semi"]["hours"]:
                tou_breakdown["semi"] += cost
            else:
                tou_breakdown["offpeak"] += cost
        result["summary"]["tou_breakdown_ntd"] = {
            k: round(v, 0) for k, v in tou_breakdown.items()
        }

        print(f"\n{'='*60}")
        print(f"[MPC v2] 日排程完成")
        print(f"  總耗電: {total_energy:.0f} kWh")
        print(f"  總供水: {total_flow:.0f} m3")
        print(f"  總電費: {total_cost:.0f} NTD")
        print(f"  平均功率: {avg_power:.1f} kW")
        print(f"  供水達成率: {supply_rate:.1f}%")
        print(f"  SEC: {total_energy / (total_flow + 1e-8):.4f} kWh/m3")
        print(f"  預充事件: {self.precharge_events} 次")
        print(f"  需求預測 MAE: {forecast_mae:.1f} m3/h")
        print(f"  需求預測 MAPE: {forecast_mape:.2f}%")
        print(f"  重新最佳化次數: {self.re_optimization_count}")
        print(f"  約束違反: {len(self.constraint_violations)} 次")
        print(f"  TOU 電費: peak={tou_breakdown['peak']:.0f}, "
              f"semi={tou_breakdown['semi']:.0f}, "
              f"offpeak={tou_breakdown['offpeak']:.0f} NTD")
        print(f"{'='*60}")

        # 儲存排程
        self._save_schedule(result, date_str)

        return result

    # -----------------------------------------
    # 靜態需求因子（回退用）
    # -----------------------------------------
    @staticmethod
    def _static_factor(h: int) -> float:
        if 7 <= h <= 9:
            return 1.20
        elif 17 <= h <= 21:
            return 1.15
        elif 0 <= h <= 6:
            return 0.85
        elif 22 <= h <= 23:
            return 0.90
        else:
            return 1.05

    # -----------------------------------------
    # 擾動模擬
    # -----------------------------------------
    def simulate_disturbance(
        self, hour: int, disturbance_type: str = "pump_trip"
    ) -> dict:
        """模擬即時擾動事件。"""
        print(f"\n[MPC v2] 模擬擾動: {disturbance_type} @ hour={hour}")
        if disturbance_type == "pump_trip":
            print("  事件: P1 跳脫，MPC 自動重新分配負載")
            return {"event": "pump_trip", "affected": "P1", "hour": hour}
        elif disturbance_type == "demand_spike":
            print(f"  事件: 需水量突增 30%")
            return {"event": "demand_spike", "factor": 1.3, "hour": hour}
        elif disturbance_type == "pool_drop":
            print("  事件: 水位意外下降 0.3m")
            return {"event": "pool_drop", "delta": -0.3, "hour": hour}
        return {}

    # -----------------------------------------
    # 儲存
    # -----------------------------------------
    def _save_schedule(self, result: dict, date_str: str) -> None:
        schedule_dir = os.path.join(OUTPUT_DIR, "schedules")
        os.makedirs(schedule_dir, exist_ok=True)

        path = os.path.join(schedule_dir, f"schedule_{date_str}_v2.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[MPC v2] 排程 -> {path}")


# -----------------------------------------
# 直接執行測試
# -----------------------------------------
if __name__ == "__main__":
    print("訓練 PINNs 代理模型...")
    surrogate = PINNsSurrogate()
    surrogate.train()

    print("\n訓練需求預測模型...")
    forecaster = DemandForecaster()
    forecaster.fit(365)

    print("\n執行 MPC v2 滾動排程...")
    mpc = MPCOptimizer(surrogate, forecaster)
    result = mpc.run_day("2024-01-15")
