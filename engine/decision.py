class DecisionEngine:
    def __init__(self, base_threshold=0.05, min_threshold=0.01, max_threshold=0.30, threshold=None):
        # 'threshold' kept for backward compatibility with the old fixed API
        self.base_threshold = base_threshold if threshold is None else threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.current_channel = None
        self.current_q_value = 0.0
        self.last_threshold = self.base_threshold

    def _dynamic_threshold(self, context):
        if not context:
            return self.base_threshold
        threshold = self.base_threshold
        threshold += 0.10 * context.get("sea_state_norm", 0.0)      # rough seas -> stay put
        threshold += 0.10 * context.get("doppler_norm", 0.0)        # fast fading -> stay put
        threshold += 0.08 * context.get("recent_switch_rate", 0.0)  # was flapping -> stabilise
        threshold += 0.05 * (1.0 - context.get("stability", 1.0))   # low confidence -> stay put
        threshold += 0.05 * context.get("wind_norm", 0.0)           # high wind -> stay put
        threshold += 0.04 * context.get("distance_norm", 0.0)       # far offshore -> stabilise
        threshold -= 0.12 * context.get("capacity_gap_norm", 0.0)   # much better option -> switch
        return max(self.min_threshold, min(self.max_threshold, threshold))

    def decide(self, chosen_channel, chosen_q_value, context=None, fallback_channel=None):
        threshold = self._dynamic_threshold(context)
        self.last_threshold = threshold
        # A very large capacity gap always permits a switch regardless of hysteresis.
        large_gap = bool(context and context.get("capacity_gap_norm", 0.0) >= 0.9)
        switched = False
        if self.current_channel is None:
            self.current_channel = chosen_channel
            self.current_q_value = chosen_q_value
            switched = True
        elif not self.current_channel.available:
            # forced switch: go to the best available channel, not a stray exploration
            target = fallback_channel if fallback_channel is not None else chosen_channel
            self.current_channel = target
            self.current_q_value = chosen_q_value
            switched = True
        elif chosen_channel is not self.current_channel and (
                chosen_q_value > self.current_q_value + threshold or large_gap):
            self.current_channel = chosen_channel
            self.current_q_value = chosen_q_value
            switched = True
        return self.current_channel, switched

    def reset(self):
        self.current_channel = None
        self.current_q_value = 0.0