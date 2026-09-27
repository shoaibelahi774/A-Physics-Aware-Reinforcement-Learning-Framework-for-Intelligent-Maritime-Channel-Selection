from __future__ import annotations

import os
import sys
import webbrowser
from dataclasses import dataclass, field

import numpy as np

from config.simulation_config import RESULTS_DIR

TECH_ORDER = ["WiFi", "4G Cellular", "Short-Range Radio", "LEO Satellite", "GEO Satellite"]
TECH_COLOR = {"WiFi": "#1f77b4", "4G Cellular": "#ff7f0e", "Short-Range Radio": "#2ca02c",
              "LEO Satellite": "#d62728", "GEO Satellite": "#9467bd"}
FIG_DIR = RESULTS_DIR / "figures"


@dataclass
class RunRecorder:
    """Collects one row per iteration; filled inside the main voyage loop."""

    distances_km: list = field(default_factory=list)
    chosen: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    switched: list = field(default_factory=list)
    thresholds: list = field(default_factory=list)
    epsilons: list = field(default_factory=list)
    # channel snapshot scalars
    capacity: list = field(default_factory=list)
    delay_ns: list = field(default_factory=list)
    doppler_hz: list = field(default_factory=list)
    sat_doppler_khz: list = field(default_factory=list)
    los_prob: list = field(default_factory=list)
    wave_m: list = field(default_factory=list)
    wind_ms: list = field(default_factory=list)
    # per-technology metric traces
    tech: dict = field(default_factory=lambda: {
        name: {k: [] for k in ("snr", "cap", "lat", "loss", "cost", "avail")}
        for name in TECH_ORDER})

    def record(self, distance_km, snapshot, by_name, final_channel,
               reward, switched, threshold, epsilon) -> None:
        self.distances_km.append(float(distance_km))
        self.chosen.append(final_channel.name)
        self.rewards.append(float(reward))
        self.switched.append(1 if switched else 0)
        self.thresholds.append(float(threshold))
        self.epsilons.append(float(epsilon))
        self.capacity.append(float(snapshot.capacity_bit_s_hz))
        self.delay_ns.append(float(snapshot.rms_delay_spread_s) * 1e9)
        self.doppler_hz.append(abs(float(snapshot.max_doppler_hz)))
        self.sat_doppler_khz.append(abs(float(snapshot.satellite_doppler_hz)) / 1e3)
        self.los_prob.append(float(snapshot.los_probability))
        self.wave_m.append(float(snapshot.significant_wave_height_m))
        self.wind_ms.append(float(snapshot.wind_speed_reference_m_s))
        for name in TECH_ORDER:
            m = by_name[name].metrics or {}
            t = self.tech[name]
            t["snr"].append(m.get("snr_db", np.nan))
            t["cap"].append(m.get("capacity_mbps", np.nan))
            t["lat"].append(m.get("latency_ms", np.nan))
            t["loss"].append(m.get("packet_loss", np.nan))
            t["cost"].append(m.get("cost_index", np.nan))
            t["avail"].append(1.0 if m.get("physical_available") else 0.0)

    # ------------------------------------------------------------------
    def ping_pong_events(self, window: int = 3) -> int:
        """Handovers that reverse to the previous technology within `window`
        steps (the classic ping-pong definition)."""
        events = 0
        names = self.chosen
        for i in range(1, len(names)):
            if names[i] != names[i - 1]:                       # a handover at i
                for j in range(i + 1, min(i + 1 + window, len(names))):
                    if names[j] != names[j - 1]:               # next handover
                        if names[j] == names[i - 1]:
                            events += 1
                        break
        return events


# ----------------------------------------------------------------------
def render_trajectory(bundle, voyage_distances_km=None, show: bool = False):
    """Render a SINGLE-panel, IEEE-column-ready vessel-trajectory figure from
    the project's own loaded AIS data: all real vessel tracks in latitude /
    longitude, start (green circle) / end (red square) markers, and the
    reference point. Sized for an IEEE single column (~3.5 in wide), serif
    fonts, white background. Saves PNG (300 dpi) + vector PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LAT0, LON0 = 56.15, 10.21
    KM_LAT = 111.32
    KM_LON = 111.32 * np.cos(np.radians(LAT0))

    plt.rcParams.update({"font.size": 8, "font.family": "serif",
                         "axes.grid": True, "grid.alpha": 0.35,
                         "grid.linestyle": ":", "axes.linewidth": 0.7,
                         "figure.facecolor": "white", "axes.facecolor": "white"})

    fig, ax = plt.subplots(figsize=(3.6, 3.5))     # IEEE single column
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, (mmsi, tr) in enumerate(bundle.trajectories.items()):
        pos = np.array([s.position_enu_m for s in tr.samples]) / 1000.0
        lon = LON0 + pos[:, 0] / KM_LON
        lat = LAT0 + pos[:, 1] / KM_LAT
        step = max(1, len(lat) // 2000)            # decimate dense tracks
        lon, lat = lon[::step], lat[::step]
        ax.plot(lon, lat, lw=0.8, color=colors[i], alpha=0.95,
                label=f"{mmsi}")
        ax.scatter(lon[0], lat[0], marker="o", s=26, color="#2ca02c",
                   edgecolor="k", zorder=5, linewidth=0.6)
        ax.scatter(lon[-1], lat[-1], marker="s", s=26, color="#d62728",
                   edgecolor="k", zorder=5, linewidth=0.6)
    ax.scatter([LON0], [LAT0], marker="*", s=120, c="#1C7293",
               edgecolor="k", linewidth=0.6, zorder=6)

    ax.set_xlabel("Longitude (\u00b0E)")
    ax.set_ylabel("Latitude (\u00b0N)")
    ax.set_aspect(1.0 / np.cos(np.radians(LAT0)))
    ax.tick_params(labelsize=7)

    handles, labels = ax.get_legend_handles_labels()
    extra = [Line2D([], [], marker="o", ls="", mfc="#2ca02c", mec="k",
                    ms=5, label="start"),
             Line2D([], [], marker="s", ls="", mfc="#d62728", mec="k",
                    ms=5, label="end"),
             Line2D([], [], marker="*", ls="", mfc="#1C7293", mec="k",
                    ms=8, label="reference")]
    ax.legend(handles + extra, labels + ["start", "end", "reference"],
              fontsize=5.6, loc="lower right", framealpha=0.92,
              title="MMSI", title_fontsize=6, handlelength=1.4,
              borderpad=0.4, labelspacing=0.35)

    fig.tight_layout(pad=0.4)
    png = FIG_DIR / "fig0_vessel_trajectory.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig0_vessel_trajectory.pdf", bbox_inches="tight")
    plt.close(fig)
    if show:
        _open(png)
    return png



def _distance_axis(ax, x):
    """Symlog distance axis whose largest tick is the ACTUAL maximum range,
    so the simulated extent (e.g. 150 km) is visible on the figure."""
    from matplotlib.ticker import FuncFormatter, FixedLocator
    xmax = float(np.nanmax(x))
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(0.0, xmax)
    ticks = [t for t in (0, 10, 50, 100, 200, 500) if t < 0.9 * xmax]
    ticks.append(round(xmax))
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def render_all(rec: RunRecorder, show: bool = True) -> list:
    """Render every figure from the recorded run; return the file paths.
    Opens a gallery with the OS default viewer when show=True."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 125, "savefig.dpi": 150, "font.size": 9.5,
        "axes.grid": True, "grid.alpha": 0.3,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.fontsize": 8,
    })
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    x = np.asarray(rec.distances_km)
    paths = []

    def save(fig, name):
        p = FIG_DIR / name
        fig.tight_layout()
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

    # 1 - per-technology SNR + coverage strip -----------------------------
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.4]})
    for name in TECH_ORDER:
        a1.plot(x, rec.tech[name]["snr"], color=TECH_COLOR[name], label=name, lw=1.4)
    a1.axhline(2.0, ls="--", c="grey", lw=1, label="usable SNR (2 dB)")
    a1.set_ylabel("link SNR (dB)"); _distance_axis(a1, x)
    a1.set_title("(a) Per-technology SNR vs offshore distance (physical link budget)")
    a1.legend(ncol=3); a1.set_ylim(-12, 70)
    for i, name in enumerate(TECH_ORDER):
        av = np.asarray(rec.tech[name]["avail"])
        a2.fill_between(x, i + 0.08, i + 0.92, where=av > 0.5,
                        color=TECH_COLOR[name], alpha=0.75, step="mid")
    a2.set_yticks([i + 0.5 for i in range(5)]); a2.set_yticklabels(TECH_ORDER, fontsize=8)
    a2.set_xlabel("offshore distance (km)")
    a2.set_title("(b) Coverage availability (radio horizon + SNR limit)")
    save(fig, "fig1_snr_coverage.png")

    # 2 - GBSM capacity + measured environment ---------------------------
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.2, 5.2), sharex=True)
    a1.plot(x, rec.capacity, c="#0E7C86", lw=1.4)
    a1.set_ylabel("GBSM capacity (bit/s/Hz)")
    a1.set_title("(a) GBSM ergodic MIMO capacity along the voyage")
    a2.plot(x, rec.wave_m, c="#1f77b4", label="significant wave height (m)")
    a2b = a2.twinx(); a2b.plot(x, rec.wind_ms, c="#ff7f0e", alpha=0.8)
    a2.set_ylabel("$H_s$ (m)", color="#1f77b4"); a2b.set_ylabel("wind (m/s)", color="#ff7f0e")
    a2b.grid(False)
    a2.set_xlabel("offshore distance (km)"); _distance_axis(a2, x)
    a2.set_title("(b) Measured sea state and wind (CMEMS / ERA5)")
    save(fig, "fig2_capacity_environment.png")

    # 3 - handover timeline ----------------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    tidx = {t: i for i, t in enumerate(TECH_ORDER)}
    ys = [tidx[n] for n in rec.chosen]
    ax.step(x, ys, where="post", c="#1C7293", lw=2)
    ho_x = [x[i] for i in range(1, len(ys)) if ys[i] != ys[i - 1]]
    ho_y = [ys[i] for i in range(1, len(ys)) if ys[i] != ys[i - 1]]
    ax.scatter(ho_x, ho_y, c="#d62728", zorder=3, s=28)
    ax.set_yticks(range(5)); ax.set_yticklabels(TECH_ORDER, fontsize=8.5)
    _distance_axis(ax, x); ax.set_xlabel("offshore distance (km)")
    pp = rec.ping_pong_events()
    ax.set_title(f"Selected technology along the voyage "
                 f"({len(ho_x)} handovers, {pp} ping-pong events)")
    save(fig, "fig3_handover_timeline.png")

    # 4 - QoS metrics 2x2 -------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0), sharex=True)
    panels = [("cap", "throughput (Mbps)", "(a) Throughput"),
              ("lat", "latency (ms)", "(b) Latency"),
              ("loss", "packet loss", "(c) Packet loss"),
              ("cost", "cost index", "(d) Communication cost")]
    for ax, (key, ylab, title) in zip(axes.flat, panels):
        for name in TECH_ORDER:
            vals = np.asarray(rec.tech[name][key], dtype=float)
            av = np.asarray(rec.tech[name]["avail"]) > 0.5
            ax.plot(x, np.where(av, vals, np.nan), color=TECH_COLOR[name], lw=1.2, label=name)
        ax.set_ylabel(ylab); ax.set_title(title); _distance_axis(ax, x)
        if key == "lat":
            ax.set_yscale("log")
    axes[1, 0].set_xlabel("offshore distance (km)")
    axes[1, 1].set_xlabel("offshore distance (km)")
    axes[0, 0].legend(ncol=2, fontsize=7)
    fig.suptitle("Per-technology QoS metrics (shown where in coverage)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "fig4_qos_metrics.png"); plt.close(fig)
    paths.append(FIG_DIR / "fig4_qos_metrics.png")

    # 5 - selection distribution + cumulative reward ----------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    names, counts = np.unique(rec.chosen, return_counts=True)
    order = [n for n in TECH_ORDER if n in names]
    cnt = [int(counts[list(names).index(n)]) for n in order]
    a1.bar(order, cnt, color=[TECH_COLOR[n] for n in order])
    a1.set_ylabel("steps selected"); a1.tick_params(axis="x", labelsize=7.5)
    a1.set_title("(a) Technology selection distribution")
    r = np.asarray(rec.rewards)
    a2.plot(np.arange(1, r.size + 1), np.cumsum(r), c="#0E7C86", lw=1.5)
    a2.set_xlabel("iteration"); a2.set_ylabel("cumulative reward")
    a2.set_title(f"(b) Cumulative reward (mean/step = {r.mean():.3f})")
    save(fig, "fig5_selection_reward.png")

    # 6 - channel statistics + controller state ---------------------------
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0))
    axes[0, 0].plot(x, rec.delay_ns, c="#1f77b4", lw=1.2)
    axes[0, 0].set_ylabel("RMS delay spread (ns)"); axes[0, 0].set_title("(a) RMS delay spread")
    axes[0, 1].plot(x, rec.doppler_hz, c="#ff7f0e", lw=1.2)
    axb = axes[0, 1].twinx(); axb.plot(x, rec.sat_doppler_khz, c="#d62728", lw=1.0, alpha=0.7)
    axb.set_ylabel("satellite Doppler (kHz)", color="#d62728"); axb.grid(False)
    axes[0, 1].set_ylabel("channel Doppler (Hz)"); axes[0, 1].set_title("(b) Doppler")
    axes[1, 0].plot(x, rec.los_prob, c="#9467bd", lw=1.2)
    axes[1, 0].set_ylabel("LoS probability"); axes[1, 0].set_title("(c) Line-of-sight probability")
    axes[1, 1].plot(np.arange(1, len(rec.thresholds) + 1), rec.thresholds,
                    c="#2ca02c", lw=1.2, label="hysteresis threshold")
    axes[1, 1].plot(np.arange(1, len(rec.epsilons) + 1), rec.epsilons,
                    c="#7f7f7f", lw=1.2, ls="--", label="exploration epsilon")
    axes[1, 1].set_xlabel("iteration"); axes[1, 1].legend()
    axes[1, 1].set_title("(d) Hysteresis threshold and exploration")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        _distance_axis(ax, x)
    axes[1, 0].set_xlabel("offshore distance (km)")
    fig.suptitle("Channel statistics and controller state over the run",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "fig6_channel_and_controller.png"); plt.close(fig)
    paths.append(FIG_DIR / "fig6_channel_and_controller.png")

    # gallery page so one click shows everything --------------------------
    # include the trajectory figure (fig0_*) if it was rendered before us
    gallery_paths = sorted(FIG_DIR.glob("fig0_*.png")) + paths
    gallery = FIG_DIR / "results_gallery.html"
    items = "\n".join(
        f'<h3>{p.name}</h3><img src="{p.name}" style="max-width:980px;width:100%;'
        f'border:1px solid #ccc;margin-bottom:24px">' for p in gallery_paths)
    gallery.write_text(
        "<html><head><title>Simulation results</title></head>"
        "<body style='font-family:Segoe UI,Arial;margin:24px'>"
        "<h1>Intelligent Multi-Channel Maritime Communication Selector &mdash; run results</h1>"
        f"{items}</body></html>", encoding="utf-8")

    if show:
        _open(gallery)
    return paths


def _open(path) -> None:
    """Open a file with the OS default viewer (Windows/macOS/Linux safe)."""
    try:
        if os.name == "nt":
            os.startfile(str(path))                      # noqa
        elif sys.platform == "darwin":
            os.system(f'open "{path}"')                  # noqa
        else:
            webbrowser.open(f"file://{path}")
    except Exception as exc:                             # never crash the run
        print(f"  (could not auto-open figures: {exc})")