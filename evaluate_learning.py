from __future__ import annotations
import json
import random

import numpy as np

from config.simulation_config import RESULTS_DIR, default_config
from gbsm.channel_generator import ChannelGenerator
from channels.coverage import CoveragePhysicsModel
from simulator.environment import Environment
from simulator.data_packet import DataPacket
from engine.evaluator import Evaluator
from engine.q_learning import QLearningAgent

_START_KM = 0.3
_MAX_KM = 700.0


def _round_trip_distances(n: int) -> np.ndarray:
    """Port -> offshore -> port, log-spaced so near ranges (terrestrial) and far
    ranges (satellite) are both well sampled."""
    half = max(2, n // 2)
    out = np.geomspace(_START_KM, _MAX_KM, half)
    back = out[::-1]
    return np.concatenate([out, back])[:n]


def _availability(environment, coverage, by_name, all_channels, distance_km, snapshot):
    for channel in all_channels:
        channel.reset_physical_metrics()
    coverage.apply(by_name, distance_km, snapshot)
    available = environment.get_available_channels()
    if not available:
        available = [by_name["GEO Satellite"]]
    return available


def _run_episode(agent, evaluator, environment, coverage, by_name, all_channels,
                 snapshots, distances, packets, explore: bool, learn: bool):
    """Run one voyage. Returns (mean_reward, switch_count, selection_counts)."""
    rewards, switches = [], 0
    counts = {c.name: 0 for c in all_channels}
    prev_index = -1
    prev = None  # (state, action, reward)
    for snapshot, distance_km, packet in zip(snapshots, distances, packets):
        available = _availability(environment, coverage, by_name, all_channels,
                                  float(distance_km), snapshot)
        available_indices = [all_channels.index(c) for c in available]
        _, ranked = evaluator.evaluate(available)
        best_channel = ranked[0][0]
        state = agent.get_state(best_channel, packet, snapshot, float(distance_km),
                                current_index=prev_index)
        if explore:
            action = agent.select_action(state, available_indices,
                                         preferred_index=all_channels.index(best_channel))
        else:  # pure exploitation of the learned policy
            q = agent.q_table[state]
            best_q = max(q[i] for i in available_indices)
            action = random.choice([i for i in available_indices if q[i] == best_q])
        chosen = all_channels[action]
        switched = prev_index != -1 and action != prev_index
        reward = agent.compute_reward(chosen, packet, snapshot, switched)
        if learn and prev is not None:
            agent.update(prev[0], prev[1], prev[2], state, available_indices)
        prev = (state, action, reward)
        rewards.append(reward)
        switches += int(switched)
        counts[chosen.name] += 1
        prev_index = action
    if learn and prev is not None:
        agent.update(prev[0], prev[1], prev[2], prev[0], [prev[1]])
    return float(np.mean(rewards)), switches, counts


def _greedy_episode(evaluator, environment, coverage, by_name, all_channels,
                    snapshots, distances, packets, agent, random_policy=False):
    rewards, switches = [], 0
    counts = {c.name: 0 for c in all_channels}
    prev_index = -1
    for snapshot, distance_km, packet in zip(snapshots, distances, packets):
        available = _availability(environment, coverage, by_name, all_channels,
                                  float(distance_km), snapshot)
        available_indices = [all_channels.index(c) for c in available]
        _, ranked = evaluator.evaluate(available)
        if random_policy:
            chosen = all_channels[random.choice(available_indices)]
        else:
            chosen = ranked[0][0]                      # greedy: best instantaneous QoS
        action = all_channels.index(chosen)
        switched = prev_index != -1 and action != prev_index
        rewards.append(agent.compute_reward(chosen, packet, snapshot, switched))
        switches += int(switched)
        counts[chosen.name] += 1
        prev_index = action
    return float(np.mean(rewards)), switches, counts


def _sweep_switch_cost(scenario, snapshots, distances, packets, evaluator, environment,
                       coverage, by_name, all_channels, penalties, episodes, seed):
    """For each switch-cost, train a fresh agent and compare its converged policy
    to the greedy-QoS baseline. Returns rows of (penalty, greedy_R, q_R, q_switches)."""
    rows = []
    for sp in penalties:
        random.seed(seed)
        agent = QLearningAgent(n_actions=len(all_channels), alpha=0.2, gamma=0.9,
                               epsilon=0.9, epsilon_decay=0.94, epsilon_min=0.02, switch_penalty=sp)
        greedy_r, greedy_sw, _ = _greedy_episode(
            evaluator, environment, coverage, by_name, all_channels, snapshots, distances, packets, agent)
        for _ in range(episodes):
            _run_episode(agent, evaluator, environment, coverage, by_name, all_channels,
                         snapshots, distances, packets, explore=True, learn=True)
        q_r, q_sw, _ = _run_episode(agent, evaluator, environment, coverage, by_name, all_channels,
                                    snapshots, distances, packets, explore=False, learn=False)
        rows.append({"switch_penalty": sp, "greedy_reward": round(greedy_r, 4),
                     "q_reward": round(q_r, 4), "greedy_switches": greedy_sw, "q_switches": q_sw,
                     "q_vs_greedy_pct": round(100.0 * (q_r - greedy_r) / abs(greedy_r), 1) if greedy_r else None})
    return rows


def main(scenario: str = "S2L", steps: int = 120, episodes: int = 60, seed: int = 7) -> dict:
    config = default_config()
    random.seed(seed)
    np.random.seed(seed)

    generator = ChannelGenerator(config, operating_snr_db=20.0)
    bundle = generator.build_bundle(write_audit=False)
    record_count = len(list(bundle.sea_state))
    steps = min(steps, record_count)

    distances = _round_trip_distances(steps)
    snapshots = generator.generate(scenario, bundle, max_snapshots=steps,
                                   distances_m=[float(d * 1000.0) for d in distances])
    steps = len(snapshots)
    distances = distances[:steps]

    coverage = CoveragePhysicsModel(config)
    environment = Environment()
    evaluator = Evaluator()
    all_channels = environment.get_all_channels()
    by_name = {c.name: c for c in all_channels}

    # Fixed packet sequence reused everywhere so only the policy varies.
    random.seed(seed)
    packets = [DataPacket.generate_random_packet() for _ in range(steps)]

    agent = QLearningAgent(n_actions=len(all_channels), alpha=0.2, gamma=0.9,
                           epsilon=0.9, epsilon_decay=0.94, epsilon_min=0.02)

    # --- baselines (policy-independent environment) ---
    greedy_reward, greedy_sw, greedy_counts = _greedy_episode(
        evaluator, environment, coverage, by_name, all_channels, snapshots, distances, packets, agent)
    random.seed(seed + 1)
    random_reward, random_sw, _ = _greedy_episode(
        evaluator, environment, coverage, by_name, all_channels, snapshots, distances, packets, agent,
        random_policy=True)

    # --- train the Q-agent, record the learning curve ---
    curve = []
    for ep in range(episodes):
        r, sw, _ = _run_episode(agent, evaluator, environment, coverage, by_name, all_channels,
                                snapshots, distances, packets, explore=True, learn=True)
        curve.append(r)

    # --- evaluate the converged (greedy) policy ---
    conv_reward, conv_sw, conv_counts = _run_episode(
        agent, evaluator, environment, coverage, by_name, all_channels,
        snapshots, distances, packets, explore=False, learn=False)

    # --- switch-cost crossover: when does RL beat greedy? ---
    sweep = _sweep_switch_cost(scenario, snapshots, distances, packets, evaluator, environment,
                               coverage, by_name, all_channels,
                               penalties=[0.15, 0.30, 0.45, 0.60, 0.75], episodes=episodes, seed=seed)

    report = {
        "scenario": scenario, "steps": steps, "episodes": episodes,
        "voyage": "round trip (port -> offshore -> port), log-spaced",
        "learning_curve_mean_reward": [round(x, 4) for x in curve],
        "q_learned_policy": {
            "mean_reward": round(conv_reward, 4), "switches": conv_sw,
            "selection_counts": conv_counts,
        },
        "greedy_qos_baseline": {
            "mean_reward": round(greedy_reward, 4), "switches": greedy_sw,
            "selection_counts": greedy_counts,
        },
        "random_baseline": {"mean_reward": round(random_reward, 4), "switches": random_sw},
        "improvement_over_greedy_pct": round(100.0 * (conv_reward - greedy_reward) / abs(greedy_reward), 1)
        if greedy_reward else None,
        "switch_cost_crossover": sweep,
        "q_states_visited": len(agent.q_table),
        "q_total_updates": int(sum(sum(v) for v in agent.visit_counts.values())),
        "final_epsilon": round(agent.epsilon, 4),
        "first_vs_last_episode_reward": [round(curve[0], 4), round(curve[-1], 4)],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "learning_evaluation.json").write_text(json.dumps(report, indent=2))

    print(f"Q-learning evaluation over {episodes} episodes of a {steps}-step round-trip voyage")
    print(f"  episode 1 mean reward     = {curve[0]:.4f}")
    print(f"  episode {episodes} mean reward    = {curve[-1]:.4f}")
    print(f"  converged Q policy reward = {conv_reward:.4f}   (switches={conv_sw})")
    print(f"  greedy-QoS baseline       = {greedy_reward:.4f}   (switches={greedy_sw})")
    print(f"  random baseline           = {random_reward:.4f}")
    if report["improvement_over_greedy_pct"] is not None:
        print(f"  Q vs greedy               = {report['improvement_over_greedy_pct']:+.1f}%")
    print(f"  states visited            = {report['q_states_visited']}")
    print(f"\nReport written to {RESULTS_DIR / 'learning_evaluation.json'}")
    return report


if __name__ == "__main__":
    main()