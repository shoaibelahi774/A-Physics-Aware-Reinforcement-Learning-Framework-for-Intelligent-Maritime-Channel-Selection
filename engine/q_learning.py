import random
import collections
from config.paper_parameters import CHANNEL_DOPPLER_REFERENCE_HZ


class QLearningAgent:
    def __init__(
        self,
        n_actions,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.2,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        switch_penalty=0.15,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.switch_penalty = switch_penalty
        self.q_table = collections.defaultdict(lambda: [0.0] * self.n_actions)
        self.visit_counts = collections.defaultdict(lambda: [0] * self.n_actions)

    # ---- legacy binning (used only when no GBSM snapshot is supplied) ----
    def _bin_latency(self, latency_ms):
        if latency_ms < 100:
            return 0
        elif latency_ms < 500:
            return 1
        return 2

    def _bin_bandwidth(self, bandwidth_mbps):
        if bandwidth_mbps < 10:
            return 0
        elif bandwidth_mbps < 50:
            return 1
        return 2

    @staticmethod
    def _bin_distance(distance_km):
        if distance_km is None:
            return 0
        if distance_km < 5.0:
            return 0     # near coast
        if distance_km < 50.0:
            return 1     # mid range
        return 2         # far offshore

    def get_state(self, channel, packet, snapshot=None, distance_km=None, current_index=None):
        if snapshot is None:
            latency_bin = self._bin_latency(channel.latency)
            bandwidth_bin = self._bin_bandwidth(channel.bandwidth)
            return (latency_bin, bandwidth_bin, packet.priority)
        capacity_bin = 0 if snapshot.capacity_bit_s_hz < 5.0 else (1 if snapshot.capacity_bit_s_hz < 8.0 else 2)
        los_bin = 1 if snapshot.los_probability >= 0.5 else 0
        doppler_bin = 1 if snapshot.max_doppler_hz >= 150.0 else 0
        satellite_bin = 1 if snapshot.satellite_visible else 0
        distance_bin = self._bin_distance(distance_km)
        # Include the CURRENTLY-selected technology in the state. Without it the
        # switching penalty is not learnable and the action cannot influence the
        # next state, which reduces the problem to a contextual bandit. With it,
        # Q(s,a) captures the sequential switch-cost trade-off (a true MDP).
        current_bin = -1 if current_index is None else int(current_index)
        return (capacity_bin, los_bin, doppler_bin, satellite_bin, distance_bin, packet.priority, current_bin)

    def select_action(self, state, available_indices, preferred_index=None):
        if not available_indices:
            return None
        q_values = self.q_table[state]
        if random.random() < self.epsilon:
            if (preferred_index is not None and preferred_index in available_indices
                    and random.random() < 0.5):
                return preferred_index
            return random.choice(available_indices)
        max_q = max(q_values[i] for i in available_indices)
        best_actions = [i for i in available_indices if q_values[i] == max_q]
        if len(best_actions) > 1 and preferred_index in best_actions:
            return preferred_index
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, available_next_indices):
        current_q = self.q_table[state][action]
        if available_next_indices:
            next_q_values = self.q_table[next_state]
            best_next_q = max(next_q_values[i] for i in available_next_indices)
        else:
            best_next_q = 0.0
        new_q = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)
        self.q_table[state][action] = new_q
        self.visit_counts[state][action] += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def compute_reward(self, channel, packet, snapshot=None, switched=False):
        metrics = getattr(channel, "metrics", None) or {}
        capacity_qos = metrics.get("capacity_qos", min(channel.bandwidth / 200.0, 1.0))
        qos = metrics.get("qos_score", channel.stability)
        stability = channel.stability
        availability = 1.0 if getattr(channel, "available", True) else 0.0
        packet_loss = metrics.get("packet_loss", 0.0)
        doppler_hz = abs(metrics.get("maritime_doppler_hz", metrics.get("doppler_hz", 0.0)))
        doppler_penalty = min(doppler_hz / CHANNEL_DOPPLER_REFERENCE_HZ, 1.0)
        latency_penalty = min(channel.latency / 600.0, 1.0)
        cost_penalty = min(channel.cost / 5.0, 1.0)
        switch_penalty = self.switch_penalty if switched else 0.0
        reward = (0.28 * capacity_qos + 0.20 * qos + 0.12 * availability + 0.14 * stability
                  - 0.10 * packet_loss - 0.06 * latency_penalty - 0.06 * cost_penalty
                  - 0.05 * doppler_penalty - switch_penalty)
        return max(-1.0, min(1.0, reward))

    def print_q_table(self, channel_names):
        print("\n--- Q-Table Summary (visited states only) ---")
        print("  physics state = (capacity, LoS, Doppler, satellite, distance, priority, current_tech)")
        print(f"{'State':<38} {'Channel':<20} {'Q-value':>9}  {'Visits':>6}")
        print("-" * 80)
        for state, q_values in sorted(self.q_table.items(), key=lambda kv: str(kv[0])):
            state_str = str(state)
            for action_idx, q_val in enumerate(q_values):
                visits = self.visit_counts[state][action_idx]
                if visits > 0:
                    ch_name = channel_names[action_idx] if action_idx < len(channel_names) else str(action_idx)
                    print(f"{state_str:<38} {ch_name:<20} {q_val:>9.4f}  {visits:>6}")
        print(f"\nEpsilon (exploration rate): {self.epsilon:.4f}")
        print("-" * 80)