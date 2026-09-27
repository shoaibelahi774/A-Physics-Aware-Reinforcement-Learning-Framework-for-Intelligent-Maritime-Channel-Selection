from __future__ import annotations
import json

import numpy as np

from config.simulation_config import RESULTS_DIR, default_config
from gbsm.channel_generator import ChannelGenerator
from gbsm import qmc


def _capacity_series(generator, scenario, bundle, distances_m):
    snaps = generator.generate(scenario, bundle, max_snapshots=len(distances_m),
                               distances_m=distances_m)
    return np.array([s.capacity_bit_s_hz for s in snaps]), \
           np.array([s.rms_delay_spread_s * 1e9 for s in snaps])


def main(scenario: str = "S2L", n_snapshots: int = 30, n_realisations: int = 24, seed: int = 7) -> dict:
    config = default_config()
    generator = ChannelGenerator(config, operating_snr_db=20.0)
    bundle = generator.build_bundle(write_audit=False)
    n = min(n_snapshots, len(list(bundle.sea_state)))
    # A fixed distance sweep so the statistic (mean capacity) is well defined and
    # the only thing that varies between realisations is the stochastic channel.
    distances_m = list(np.geomspace(500.0, 40_000.0, n) * 1.0)

    phases = qmc.phases_for_ensemble(n_realisations, seed=seed)  # phases[0] == 0.0
    cap_means, ds_means, cap_pool = [], [], []
    for phi in phases:
        qmc.set_phase(float(phi))
        cap, ds = _capacity_series(generator, scenario, bundle, distances_m)
        cap_means.append(float(cap.mean()))
        ds_means.append(float(ds.mean()))
        cap_pool.append(cap)
    qmc.reset()

    cap_means = np.array(cap_means)
    deterministic = float(cap_means[0])          # phase 0 = deterministic run
    ensemble = cap_means[1:]                      # independent random realisations
    ens_mean = float(ensemble.mean())
    ens_std = float(ensemble.std(ddof=1))         # spread of individual realisations
    stderr = ens_std / np.sqrt(ensemble.size)
    ci95_mean = (ens_mean - 1.96 * stderr, ens_mean + 1.96 * stderr)   # CI of the TRUE mean
    spread95 = (ens_mean - 1.96 * ens_std, ens_mean + 1.96 * ens_std)  # spread of realisations
    # (a) is the deterministic run a TYPICAL realisation? (within the ensemble spread)
    typical = bool(spread95[0] <= deterministic <= spread95[1])
    # (b) how close is the deterministic run to the true (ensemble) mean?
    rel_bias = 100.0 * (deterministic - ens_mean) / ens_mean if ens_mean else 0.0

    # running ensemble mean (convergence of the RQMC estimator)
    running = np.array([ensemble[:k].mean() for k in range(1, ensemble.size + 1)])

    # pooled capacity distribution (all realisations) vs the deterministic run
    pooled = np.concatenate(cap_pool)
    det_series = cap_pool[0]

    report = {
        "scenario": scenario, "n_snapshots": n, "n_realisations": n_realisations,
        "method": "randomized QMC (Cranley-Patterson rotation of the golden-ratio sequence)",
        "deterministic_mean_capacity_bit_s_hz": round(deterministic, 4),
        "ensemble_mean_capacity_bit_s_hz": round(ens_mean, 4),
        "ensemble_std_bit_s_hz": round(ens_std, 4),
        "ensemble_95pct_ci_of_mean": [round(ci95_mean[0], 4), round(ci95_mean[1], 4)],
        "realisation_95pct_spread": [round(spread95[0], 4), round(spread95[1], 4)],
        "deterministic_is_typical_realisation": typical,
        "deterministic_vs_ensemble_mean_bias_pct": round(rel_bias, 2),
        "running_ensemble_mean": [round(float(x), 4) for x in running],
        "pooled_capacity_mean": round(float(pooled.mean()), 4),
        "pooled_capacity_std": round(float(pooled.std()), 4),
        "deterministic_delay_spread_ns": round(ds_means[0], 2),
        "ensemble_delay_spread_ns_mean": round(float(np.mean(ds_means[1:])), 2),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "ensemble_evaluation.json").write_text(json.dumps(report, indent=2))
    # keep the raw arrays for the figure
    np.savez(RESULTS_DIR / "ensemble_arrays.npz",
             cap_means=cap_means, running=running, pooled=pooled, det_series=det_series)

    print(f"RQMC ensemble over {n_realisations} realisations ({n}-snapshot sweep, {scenario})")
    print(f"  deterministic mean capacity = {deterministic:.4f} bit/s/Hz  (phase 0, reproducible)")
    print(f"  ensemble mean capacity      = {ens_mean:.4f} bit/s/Hz  (true stochastic mean)")
    print(f"  ensemble 95% CI of the mean = [{ci95_mean[0]:.4f}, {ci95_mean[1]:.4f}]")
    print(f"  realisation spread (95%)    = [{spread95[0]:.4f}, {spread95[1]:.4f}]  std={ens_std:.4f}")
    print(f"  deterministic is typical    = {typical}   (bias vs ensemble mean {rel_bias:+.2f}%)")
    print(f"\nReport written to {RESULTS_DIR / 'ensemble_evaluation.json'}")
    return report


if __name__ == "__main__":
    main()