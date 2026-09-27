# A Physics-Aware Reinforcement Learning Framework for Intelligent Maritime Channel Selection

A real-data-driven **3D non-stationary GBSM** (Geometry-Based Stochastic Model) coupled with a
**Q-learning** selector that automatically chooses the best wireless technology for a ship sailing
offshore, across five technologies in coverage order:

> **WiFi → 4G Cellular → Short-Range Marine Radio → LEO Satellite → GEO Satellite**

The system rebuilds the maritime radio channel from **real measured data** (waves, wind, AIS ship
tracks, buildings, satellite orbits), scores each technology with a **physical link budget**, and
selects the best one at every step with a Q-learning agent stabilised by **dynamic hysteresis** to
prevent ping-pong handovers.

**Study area:** Aarhus / Kattegat, Denmark (56.15° N, 10.21° E).

---

## Status

| Check                               | Result |
| GBSM statistical validation         | **8 / 8 passed** |
| Automated test suite                | **30 / 30 passing** |
| Satellite constellation             | **65 real ORBCOMM satellites** (real SGP4 on the majority of steps) |
| Determinism vs stochastic ensemble  | deterministic run within **~1.3 %** of the ensemble mean |
| Handover chain (transit voyage)     | **WiFi → 4G → SRR → LEO → GEO** at physical range boundaries |

---

## Highlights

- **Real-data-forced physics.** Every environmental input is measured data — CMEMS waves, ERA5 wind,
  Danish Maritime Authority AIS, OpenStreetMap buildings, Space-Track TLEs — all co-located.
- **Physical link budgets.** Terrestrial `SNR = EIRP + G_rx − PL(d,f) − N`; satellite
  `C/N = EIRP + G/T − FSPL − k − B`, using the real SGP4 slant range.
- **Radio-horizon reach + two-ray sea propagation.** Coverage is derived from antenna heights and
  Earth curvature, not assumed constants.
- **3D non-stationary GBSM.** Cluster birth–death, sea-surface reflection, polarization/XPR, an 8×8
  MIMO matrix normalised to `‖H‖² = N_t·N_r`, and ergodic capacity `C = log₂ det(I + (ρ/n_T) H Hᴴ)`.
- **Reproducible by default,** with a randomized quasi-Monte-Carlo mode for the stochastic ensemble.
- **Q-learning selector** with a genuine MDP state (includes the current technology) and dynamic
  hysteresis.

---

## Requirements

- **Python 3.12+** (developed and run on Windows with Python 3.14)
- Dependencies (`requirements.txt`): `numpy`, `pandas`, `xarray`, `netCDF4`, `sgp4`, `shapely`, `pytest`

```bash
pip install -r requirements.txt
```

---


### Evaluation harnesses

```bash
python evaluate_learning.py     # learning curve + greedy baseline + switch-cost crossover
python evaluate_ensemble.py     # deterministic run vs randomized-QMC stochastic ensemble
```

### `main.py` options

| Flag         | Default                  | Meaning |
| `--scenario` | `S2L`                    | link geometry: `S2S`, `S2L`, `U2L`, `U2S` |
| `--steps`    | `448`                    | number of voyage steps |
| `--pause`    | `0.4`                    | seconds between steps (use `0` for a fast run) |
| `--max-km`   | —                        | maximum offshore distance |
| `--voyage`   | `transit`                |   `transit` = smooth ramp through all regimes; `ais` = raw AIS along-track (satellite-dominated) |

---

## Project structure

```
Intelligent Multi-Channel Communication Selector/
├── main.py                      # the voyage loop
├── evaluate_learning.py         # Q-learning evaluation (learning curve, baselines)
├── evaluate_ensemble.py         # randomized-QMC stochastic ensemble
├── config/
│   ├── simulation_config.py     # carrier 5.8 GHz, 8×8 MIMO, seed, dataset paths
│   └── paper_parameters.py      # GBSM physical constants + Doppler references
├── gbsm/                        # the physics engine
│   ├── channel_generator.py     # orchestrator → ChannelSnapshot (+ fast-axis series)
│   ├── los_probability.py       # line-of-sight probability
│   ├── validation.py            # 8 statistical checks
│   ├── qmc.py                   # randomized quasi-Monte-Carlo rotation
│   ├── data_ingestion/          # sea, wind, AIS, buildings, TLE loaders + audit
│   ├── environment/             # sea spectrum, sea surface, antenna motion
│   ├── geometry/                # scenarios, clusters, reflection, satellite dynamics
│   ├── channel/                 # propagation, CIR, 8×8 MIMO matrix, polarization
│   └── statistics/              # capacity, delay spread, stationarity, correlation
├── channels/                    # per-technology link budgets
│   ├── coverage.py              # link budget, two-ray, radio horizon, satellite C/N
│   ├── channel.py               # base class
│   └── wifi / cellular / radio / satellite / gbsm_channel
├── engine/
│   ├── evaluator.py             # QoS ranking
│   ├── q_learning.py            # the learning agent
│   └── decision.py              # dynamic hysteresis
├── simulator/
│   ├── environment.py           # channel registry
│   └── data_packet.py           # traffic with priority
├── tests/                       # 30 automated tests
├── utils/                       # helpers (normalizer)
└── data/                        # real datasets (waves, wind, AIS, buildings, TLE)
```

---

## How it works

The system is a one-directional pipeline. Everything communicates through two objects: a
`DataBundle` (ingestion → physics) and a `ChannelSnapshot` (physics → decision).

```
Real datasets → Data ingestion → Environment model → GBSM channel
   → Physical metrics → QoS ranking → Q-learning → Dynamic hysteresis
   → Best technology → Console table + JSON report
```

Each voyage step:

1. Generate a `ChannelSnapshot` from the GBSM.
2. Compute per-technology SNR / capacity / availability (`coverage.py`).
3. Rank the available technologies by QoS (`evaluator.py`).
4. Select a technology with the Q-learning agent (state → action → reward → update).
5. Stabilise with dynamic hysteresis, or force a switch on coverage loss.
6. Log the choice and metrics to the console and the JSON report.

---

## Datasets

All real and co-located around Aarhus / Kattegat.

| Source               | File              | Provider                  | Coverage |
| Waves | `data/Denmark_Wave_56_Days.nc` | CMEMS Baltic (hourly) | 1344 records (56 days) |
| Wind | `data/Denmark_Wind_{March,April}_2026.nc` | ECMWF ERA5 | March–April 2026 |
| AIS | `data/ais/ships_multiday.csv` | Danish Maritime Authority | 5 vessels, 29,994 fixes (21 days) |
| Buildings | `data/buildings/aarhus_coast_buildings.geojson` | OpenStreetMap | 41,385 footprints |
| Satellites | `data/orbcomm_constellation_2026-03.3le.txt` | Space-Track (historical) | 65 ORBCOMM satellites |


---

## Communication technologies

| Technology | Standard / band | Effective reach | Role |
|---|---|---|---|
| WiFi | IEEE 802.11 · 2.4 GHz | ~8–11 km | in-harbour / very near shore |
| 4G Cellular | 3GPP · 3.5 GHz | ~39 km | coastal waters |
| Short-Range Radio | ITU-R M.1371 · 0.9 GHz | ~53 km | near-offshore (radio-horizon limited) |
| LEO Satellite | 3GPP TR 38.811 / DVB-S2X | global | offshore, high-elevation passes |
| GEO Satellite | DVB-S2X | global | far offshore / stable fallback |

Reaches are **derived** from a physical link budget combined with the radio horizon
`d = 4.12·(√h_tx + √h_rx)` km — not assumed.

---



