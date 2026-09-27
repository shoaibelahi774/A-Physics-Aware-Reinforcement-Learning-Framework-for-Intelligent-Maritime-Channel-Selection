from __future__ import annotations
import argparse
import json
import random
import time
from collections import Counter

import numpy as np

from config.simulation_config import RESULTS_DIR, default_config
from gbsm.channel_generator import ChannelGenerator
from channels.coverage import CoveragePhysicsModel
from simulator.environment import Environment
from simulator.data_packet import DataPacket
from engine.evaluator import Evaluator
from engine.q_learning import QLearningAgent
from engine.decision import DecisionEngine
from visualization import RunRecorder, render_all

_START_KM = 0.3
_DEFAULT_MAX_KM = 150.0


def run_simulation(scenario: str = "S2L", steps: int = 60, pause: float = 0.0,
                   max_offshore_km: float = _DEFAULT_MAX_KM, voyage: str = "transit",
                   visualize: bool = True) -> dict:
    config = default_config()
    random.seed(config.random_seed)

    generator = ChannelGenerator(config, operating_snr_db=20.0)
    bundle = generator.build_bundle()
    mean_speed = _mean_ais_speed(bundle)
    record_count = len(list(bundle.sea_state))
    steps = min(steps, record_count)
    distances_km, distance_source = _real_offshore_distance_km(bundle, steps, max_offshore_km, voyage)
    distances_m = [float(d * 1000.0) for d in distances_km]
    voyage_hours = (float(distances_km[-1]) * 1000.0) / max(mean_speed, 1e-6) / 3600.0

    snapshots = generator.generate(scenario, bundle, max_snapshots=steps, distances_m=distances_m)
    steps = len(snapshots)

    coverage = CoveragePhysicsModel(config)
    environment = Environment()
    evaluator = Evaluator()
    decision_engine = DecisionEngine(base_threshold=0.04)
    all_channels = environment.get_all_channels()
    by_name = {c.name: c for c in all_channels}
    channel_names = [c.name for c in all_channels]

    agent = QLearningAgent(n_actions=len(all_channels), alpha=0.2, gamma=0.9,
                           epsilon=0.35, epsilon_decay=0.97, epsilon_min=0.03)

    print("\n" + "=" * 104)
    print(f"  INTELLIGENT MARITIME MULTI-CHANNEL SELECTOR   |   GBSM scenario {scenario}   |   {steps} steps")
    print(f"  Outbound voyage {float(distances_km[0]):.1f} -> {float(distances_km[-1]):.0f} km "
          f"({distance_source}) at real AIS mean speed {mean_speed:.1f} m/s (~{voyage_hours:.1f} h)")
    print("=" * 104)

    selection_counter: Counter = Counter()
    recorder = RunRecorder()
    rewards: list[float] = []
    handovers: list = []
    switch_history: list[int] = []
    real_sgp4_steps = [0]
    last_final = None
    prev_state = prev_action = prev_reward = None

    for step, (snapshot, distance_km) in enumerate(zip(snapshots, distances_km), start=1):
        distance_km = float(distance_km)
        for channel in all_channels:
            channel.reset_physical_metrics()
        coverage.apply(by_name, distance_km, snapshot)

        available = environment.get_available_channels()
        available_indices = [all_channels.index(c) for c in available]
        packet = DataPacket.generate_random_packet()

        if not available:
            # Guarantee at least the always-on satellites are considered.
            available = [by_name["GEO Satellite"]]
            available_indices = [all_channels.index(available[0])]

        _, ranked = evaluator.evaluate(available)
        best_channel = ranked[0][0]
        current_idx = all_channels.index(decision_engine.current_channel) if decision_engine.current_channel in all_channels else -1
        state = agent.get_state(best_channel, packet, snapshot, distance_km, current_index=current_idx)
        preferred_index = all_channels.index(best_channel)
        chosen_index = agent.select_action(state, available_indices, preferred_index)
        rl_channel = all_channels[chosen_index]
        chosen_q = agent.q_table[state][chosen_index]
        context = _hysteresis_context(snapshot, ranked, rl_channel, switch_history, distance_km)
        final_channel, switched = decision_engine.decide(rl_channel, chosen_q, context, fallback_channel=best_channel)

        if final_channel.name != last_final:
            handovers.append((step, distance_km, final_channel.name))
            last_final = final_channel.name
        switch_history.append(1 if (switched and step > 1) else 0)

        transmit_reward = agent.compute_reward(final_channel, packet, snapshot, switched)
        learning_reward = agent.compute_reward(rl_channel, packet, snapshot, switched)
        rewards.append(transmit_reward)
        if prev_state is not None:
            next_idx = [all_channels.index(c) for c in environment.get_available_channels()] or [preferred_index]
            agent.update(prev_state, prev_action, prev_reward, state, next_idx)
        prev_state, prev_action, prev_reward = state, chosen_index, learning_reward
        selection_counter[final_channel.name] += 1
        recorder.record(distance_km, snapshot, by_name, final_channel,
                        transmit_reward, switched, decision_engine.last_threshold,
                        agent.epsilon)
        leo_meta = by_name["LEO Satellite"].metrics or {}
        if leo_meta.get("leo_source") == "sgp4":
            real_sgp4_steps[0] += 1

        _print_step(step, steps, distance_km, snapshot, packet, all_channels,
                    rl_channel, final_channel, switched, agent, state,
                    transmit_reward, decision_engine.last_threshold)
        if step % 10 == 0:
            _print_learning(agent)
        if pause > 0.0:
            time.sleep(pause)

    if prev_state is not None:
        next_idx = [all_channels.index(c) for c in environment.get_available_channels()] or [prev_action]
        agent.update(prev_state, prev_action, prev_reward, prev_state, next_idx)

    _print_final(agent, channel_names, selection_counter, rewards, handovers)
    report = {
        "scenario": scenario,
        "steps": steps,
        "mean_ais_speed_m_s": round(mean_speed, 3),
        "voyage_offshore_km": [round(_START_KM, 2), round(max_offshore_km, 2)],
        "selection_counts": dict(selection_counter),
        "average_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "handovers": [{"step": s, "distance_km": round(d, 2), "technology": n} for s, d, n in handovers],
        "q_states_visited": len(agent.q_table),
        "q_total_updates": sum(sum(v) for v in agent.visit_counts.values()),
        "final_epsilon": round(agent.epsilon, 4),
        "leo_real_sgp4_steps": real_sgp4_steps[0],
        "leo_satellite_count": len(bundle.satellite_ephemeris),
        "leo_visibility_note": (
            f"LEO uses real SGP4 geometry/Doppler on resolved passes "
            f"({real_sgp4_steps[0]}/{steps} steps) from the historical March-2026 "
            f"ORBCOMM constellation ({len(bundle.satellite_ephemeris)} satellites); "
            f"a labelled constellation model fills any steps without a resolved pass."
        ),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "simulation_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nRun summary written to {RESULTS_DIR / 'simulation_report.json'}")

    if visualize:
        print("\nAll iterations complete - rendering result visualizations...")
        paths = render_all(recorder, show=False)
        print(f"{len(paths)} figures written to {paths[0].parent}")
    return report


def _mean_ais_speed(bundle) -> float:
    speeds = [s.speed_m_s for track in bundle.trajectories.values() for s in track.samples]
    return float(np.mean(speeds)) if speeds else 6.5


def _real_offshore_distance_km(bundle, steps: int, max_offshore_km: float, voyage: str = "transit"):
    best = None
    for track in bundle.trajectories.values():
        pos = np.array([s.position_enu_m for s in track.samples], dtype=float)[:, :2]
        if pos.shape[0] < 2:
            continue
        seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        cum_km = np.concatenate([[0.0], np.cumsum(seg)]) / 1000.0
        if best is None or cum_km[-1] > best[-1]:
            best = cum_km
    if best is None or best[-1] <= 0.0:
        d_max = max_offshore_km
        source = "modelled ramp (no AIS)"
    else:
        d_max = min(float(best[-1]), max_offshore_km)
        source = f"AIS-derived range 0.3->{d_max:.0f} km"

    if voyage == "ais" and best is not None and best[-1] > 0.0:
        # Raw AIS along-track distance (monotonic outbound). This is the true
        # vessel behaviour, but the real ships head offshore quickly, so the
        # terrestrial regimes are only briefly in range (satellite-dominated).
        idx = np.linspace(0, len(best) - 1, steps).astype(int)
        d = np.minimum(np.maximum.accumulate(best[idx]), max_offshore_km)
        d[0] = max(float(d[0]), _START_KM)
        return d, source + ", raw AIS along-track"

    # Default "transit": a smooth monotonic outbound ramp over the AIS-derived
    # range so the voyage actually dwells in each coverage regime
    # (WiFi -> 4G -> Short-Range Radio -> LEO -> GEO). Range and mean speed stay
    # AIS-derived; only the distance-vs-step profile is regularised (Mo1).
    d = np.linspace(_START_KM, d_max, steps)
    return d, source + ", smooth outbound transit"


def _hysteresis_context(snapshot, ranked, rl_channel, switch_history, distance_km):
    gap = (ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else 0.0
    recent = switch_history[-5:]
    return {
        "sea_state_norm": min(snapshot.significant_wave_height_m / 3.0, 1.0),
        "doppler_norm": min(snapshot.max_doppler_hz / 300.0, 1.0),
        "wind_norm": min(snapshot.wind_speed_reference_m_s / 20.0, 1.0),
        "distance_norm": min(distance_km / 500.0, 1.0),
        "capacity_gap_norm": min(max(gap, 0.0) * 3.0, 1.0),
        "recent_switch_rate": (sum(recent) / len(recent)) if recent else 0.0,
        "stability": rl_channel.stability,
    }


def _print_step(step, total, distance_km, snapshot, packet, all_channels,
                rl_channel, final_channel, switched, agent, state, reward, threshold):
    los = "LoS" if snapshot.los_present else "NLoS"
    print(f"\n--------- Time Step {step}/{total}  ({snapshot.timestamp:%Y-%m-%d %H:%M})  "
          f"distance={distance_km:.1f} km ---------")
    print(
        f"Sea state: Hs={snapshot.significant_wave_height_m:.2f} m  "
        f"wind={snapshot.wind_speed_reference_m_s:.1f} m/s  |  "
        f"GBSM: capacity={snapshot.capacity_bit_s_hz:.2f} bit/s/Hz  "
        f"delay-spread={snapshot.rms_delay_spread_s * 1e9:.0f} ns  "
        f"max-Doppler={snapshot.max_doppler_hz:.0f} Hz  "
        f"P_LoS={snapshot.los_probability:.2f} ({los})"
    )
    if snapshot.satellite_visible:
        print(f"LEO link: {snapshot.satellite_name}  elevation="
              f"{snapshot.satellite_elevation_rad * 57.2958:.1f} deg  "
              f"Doppler={snapshot.satellite_doppler_hz / 1000:.1f} kHz")
    print(f"Packet: type={packet.packet_type:<10} size={packet.size:.2f} MB  priority={packet.priority}")

    _print_status_table(all_channels, agent, state, final_channel, rl_channel)

    tag = "   ** HANDOVER **" if (switched and step > 1) else ""
    hyst = "" if final_channel is rl_channel else f"  (hysteresis kept {final_channel.name})"
    print(f"DECISION  RL-selected={rl_channel.name}{hyst}   ->  FINAL={final_channel.name}{tag}")
    print(f"Reward(transmit {final_channel.name})={reward:+.3f}   "
          f"hysteresis_thr={threshold:.3f}   epsilon={agent.epsilon:.3f}")


def _print_status_table(all_channels, agent, state, final_channel, rl_channel):
    print("\nChannel Status Table:")
    print("-" * 125)
    print(
        f"{'Technology':<18}{'Cap(Mbps)':<10}{'BW(MHz)':<9}{'Lat(ms)':<9}{'QoS':<6}{'SNR':<7}"
        f"{'Cost':<6}{'Stab':<6}{'Status':<17}{'Q-Value':<9}"
    )
    print("-" * 125)
    for channel in all_channels:
        m = channel.metrics or {}
        index = all_channels.index(channel)
        q_value = agent.q_table[state][index]
        status = getattr(channel, "status", "Available" if channel.available else "Unavailable")
        marker = ""
        if channel is final_channel and channel is rl_channel:
            marker = " <- selected"
        elif channel is final_channel:
            marker = " <- selected (hysteresis)"
        elif channel is rl_channel:
            marker = " (agent wanted, rejected)"
        print(
            f"{channel.name:<18}{m.get('capacity_mbps', 0):<10.1f}{m.get('bandwidth_mhz', channel.bandwidth):<9.0f}"
            f"{channel.latency:<9.1f}{m.get('qos_score', 0):<6.2f}{m.get('snr_db', 0):<7.1f}"
            f"{channel.cost:<6.2f}{channel.stability:<6.2f}{status:<17}{q_value:<9.4f}{marker}"
        )
    print("-" * 125)


def _print_learning(agent):
    q_all = [q for row in agent.q_table.values() for q in row]
    spread = (max(q_all) - min(q_all)) if q_all else 0.0
    updates = sum(sum(v) for v in agent.visit_counts.values())
    print(f"   [Q-LEARNING]  states={len(agent.q_table)}  updates={updates}  "
          f"Q-spread={spread:.3f}  epsilon={agent.epsilon:.3f}")


def _print_final(agent, channel_names, selection_counter, rewards, handovers):
    print("\n" + "=" * 104)
    print("  VOYAGE COMPLETE")
    print("=" * 104)
    print("  Handover timeline:")
    for step, dist, name in handovers:
        print(f"    step {step:>3}   {dist:7.1f} km   ->  {name}")
    print(f"\n  Selection counts: {dict(selection_counter)}")
    if rewards:
        print(f"  Average reward:   {sum(rewards)/len(rewards):.4f}")
    agent.print_q_table(channel_names)


def _parse_args():
    parser = argparse.ArgumentParser(description="Intelligent Multi-Channel Maritime Communication Selector")
    parser.add_argument("--scenario", default="S2L", choices=["S2S", "S2L", "U2L", "U2S"])
    parser.add_argument("--steps", type=int, default=224)
    parser.add_argument("--pause", type=float, default=0.4, help="seconds to wait between steps")
    parser.add_argument("--max-km", type=float, default=_DEFAULT_MAX_KM, help="max offshore distance")
    parser.add_argument("--no-viz", action="store_true", help="skip figures")
    parser.add_argument("--voyage", default="transit", choices=["transit", "ais"],
                        help="'transit' = smooth ramp through all coverage regimes; "
                             "'ais' = raw AIS along-track (satellite-dominated)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_simulation(scenario=args.scenario, steps=args.steps, pause=args.pause,
                   max_offshore_km=args.max_km, voyage=args.voyage,
                   visualize=not args.no_viz)