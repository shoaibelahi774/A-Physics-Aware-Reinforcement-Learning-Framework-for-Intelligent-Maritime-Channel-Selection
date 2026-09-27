from __future__ import annotations
from dataclasses import dataclass, field
from math import isfinite
from typing import Any


def _finite_value(value: Any, name: str) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_value(value: Any, name: str) -> float:
    result = _finite_value(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _nonnegative_value(value: Any, name: str) -> float:
    result = _finite_value(value, name)
    if result < 0.0:
        raise ValueError(f"{name} cannot be negative.")
    return result


def _unit_interval_value(value: Any, name: str) -> float:
    result = _finite_value(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return result


@dataclass
class Channel:
    name: str
    latency: float
    bandwidth: float
    cost: float
    stability: float
    available: bool = True
    metrics: dict[str, Any] = field(default_factory=dict)

    _configured_latency_ms: float = field(init=False, repr=False)
    _configured_bandwidth_mbps: float = field(init=False, repr=False)
    _configured_cost_index: float = field(init=False, repr=False)
    _configured_stability: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.latency = _positive_value(self.latency, "latency")
        self.bandwidth = _positive_value(self.bandwidth, "bandwidth")
        self.cost = _nonnegative_value(self.cost, "cost")
        self.stability = _unit_interval_value(self.stability, "stability")

        self._configured_latency_ms = self.latency
        self._configured_bandwidth_mbps = self.bandwidth
        self._configured_cost_index = self.cost
        self._configured_stability = self.stability

    @property
    def configured_latency_ms(self) -> float:
        return self._configured_latency_ms

    @property
    def configured_bandwidth_mbps(self) -> float:
        return self._configured_bandwidth_mbps

    @property
    def configured_cost_index(self) -> float:
        return self._configured_cost_index

    @property
    def configured_stability(self) -> float:
        return self._configured_stability

    def reset_physical_metrics(self) -> None:
        self.available = True
        self.metrics.clear()
        self.latency = self._configured_latency_ms
        self.bandwidth = self._configured_bandwidth_mbps
        self.cost = self._configured_cost_index
        self.stability = self._configured_stability

        for attribute in tuple(vars(self)):
            if attribute.startswith("gbsm_") or attribute.startswith("leo_"):
                delattr(self, attribute)

    def attach_metrics(self, metrics: dict[str, Any]) -> None:
        values = dict(metrics)
        self.metrics = values
        self.available = bool(values.get("physical_available", False))

        self.latency = _positive_value(
            values.get("latency_ms", self._configured_latency_ms),
            "latency_ms",
        )
        self.cost = _nonnegative_value(
            values.get("cost_index", self._configured_cost_index),
            "cost_index",
        )
        self.stability = _unit_interval_value(
            values.get("stability_score", self._configured_stability),
            "stability_score",
        )

        aliases = {
            "capacity_mbps": "gbsm_capacity_mbps",
            "capacity_qos": "gbsm_capacity_qos",
            "snr_db": "gbsm_snr_db",
            "los_score": "gbsm_los_score",
            "rms_delay_spread_s": "gbsm_rms_delay_spread_s",
            "capacity_variation": "gbsm_capacity_variation",
            "path_loss_db": "gbsm_path_loss_db",
            "physical_bandwidth_mhz": "gbsm_physical_bandwidth_mhz",
            "configured_physical_bandwidth_mhz": (
                "gbsm_configured_physical_bandwidth_mhz"
            ),
            "bandwidth_allocation_factor": (
                "gbsm_bandwidth_allocation_factor"
            ),
            "availability_reason": "gbsm_availability_reason",
            "model_note": "gbsm_model_note",
            "doppler_hz": "leo_doppler_hz",
            "doppler_penalty": "leo_doppler_penalty",
        }
        for source_name, attribute_name in aliases.items():
            if source_name in values:
                setattr(self, attribute_name, values[source_name])

    def get_status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "configured_latency_ms": self._configured_latency_ms,
            "configured_bandwidth_mbps": self._configured_bandwidth_mbps,
            "configured_cost_index": self._configured_cost_index,
            "configured_stability": self._configured_stability,
            "operating_latency_ms": self.latency,
            "operating_cost_index": self.cost,
            "operating_stability": self.stability,
            "available": self.available,
            "physical_metrics": dict(self.metrics),
        }
