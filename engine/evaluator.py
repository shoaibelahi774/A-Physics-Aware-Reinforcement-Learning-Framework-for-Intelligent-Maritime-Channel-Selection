from utils.normalizer import normalize


class Evaluator:
    def __init__(self):
        self.weights = {
            "bandwidth": 0.25,
            "latency": 0.20,
            "cost": 0.15,
            "stability": 0.15,
            "capacity": 0.15,   
            "los": 0.10,        
        }

    def _physical(self, channel, key, default=0.0):
        metrics = getattr(channel, "metrics", None) or {}
        return metrics.get(key, default)

    def evaluate(self, channels):
        if not channels:
            return None, []

        if all((getattr(c, "metrics", None) or {}).get("qos_score") is not None for c in channels):
            scored = sorted(((c, c.metrics["qos_score"]) for c in channels),
                            key=lambda x: x[1], reverse=True)
            return scored[0][0], scored

        latencies = [c.latency for c in channels]
        bandwidths = [c.bandwidth for c in channels]
        costs = [c.cost for c in channels]
        stabilities = [c.stability for c in channels]
        capacities = [self._physical(c, "capacity_qos", 0.0) for c in channels]
        los_scores = [self._physical(c, "los_score", 0.0) for c in channels]

        min_lat, max_lat = min(latencies), max(latencies)
        min_bw, max_bw = min(bandwidths), max(bandwidths)
        min_cost, max_cost = min(costs), max(costs)
        min_stab, max_stab = min(stabilities), max(stabilities)
        min_cap, max_cap = min(capacities), max(capacities)
        min_los, max_los = min(los_scores), max(los_scores)

        scored_channels = []
        for c in channels:
            latency_norm = normalize(c.latency, min_lat, max_lat)
            bandwidth_norm = normalize(c.bandwidth, min_bw, max_bw)
            cost_norm = normalize(c.cost, min_cost, max_cost)
            stability_norm = normalize(c.stability, min_stab, max_stab)
            capacity_norm = normalize(self._physical(c, "capacity_qos", 0.0), min_cap, max_cap)
            los_norm = normalize(self._physical(c, "los_score", 0.0), min_los, max_los)
            score = (
                self.weights["bandwidth"] * bandwidth_norm
                - self.weights["latency"] * latency_norm
                - self.weights["cost"] * cost_norm
                + self.weights["stability"] * stability_norm
                + self.weights["capacity"] * capacity_norm
                + self.weights["los"] * los_norm
            )
            scored_channels.append((c, score))

        scored_channels.sort(key=lambda x: x[1], reverse=True)
        best_channel, best_score = scored_channels[0]
        return best_channel, scored_channels