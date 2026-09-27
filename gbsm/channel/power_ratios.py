from __future__ import annotations
import math
from dataclasses import dataclass

from config.paper_parameters import EVAPORATION_DUCT
from config.simulation_config import SimulationConfig, default_config
from gbsm.geometry.scenarios import ClusterKind, PowerRatios


@dataclass(frozen=True)
class ComponentWeights:
    line_of_sight: float
    reflection: float
    nlos1: float
    nlos2: float
    k_factor: float
    duct_flag: int
    nlos1_ratio: float
    nlos2_ratio: float
    reflection_ratio: float


class PowerRatioResolver:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self._config = config or default_config()
        self._duct_height = EVAPORATION_DUCT.duct_height_m

    def duct_flag(self, tx_height_m: float, rx_height_m: float) -> int:
        return 1 if tx_height_m < self._duct_height and rx_height_m < self._duct_height else 0

    def resolve(
        self,
        power_ratios: PowerRatios,
        nlos1_kind: ClusterKind,
        nlos2_kind: ClusterKind,
        tx_height_m: float,
        rx_height_m: float,
    ) -> ComponentWeights:
        flag = self.duct_flag(tx_height_m, rx_height_m)
        nlos1 = power_ratios.nlos1 * (flag if nlos1_kind is ClusterKind.EVAPORATION_DUCT else 1)
        nlos2 = power_ratios.nlos2 * (flag if nlos2_kind is ClusterKind.EVAPORATION_DUCT else 1)
        reflection = power_ratios.reflection
        total = nlos1 + nlos2 + reflection
        if total <= 0.0:
            nlos1, nlos2, reflection, total = 1.0, 0.0, 0.0, 1.0
        nlos1 /= total
        nlos2 /= total
        reflection /= total
        k_factor = power_ratios.k_factor
        denominator = k_factor + 1.0
        return ComponentWeights(
            line_of_sight=math.sqrt(k_factor / denominator),
            reflection=math.sqrt(reflection / denominator),
            nlos1=math.sqrt(nlos1 / denominator),
            nlos2=math.sqrt(nlos2 / denominator),
            k_factor=k_factor,
            duct_flag=flag,
            nlos1_ratio=nlos1,
            nlos2_ratio=nlos2,
            reflection_ratio=reflection,
        )


__all__ = ["ComponentWeights", "PowerRatioResolver"]