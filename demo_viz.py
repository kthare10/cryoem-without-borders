"""Demo visualizations for cryoem_experiment.ipynb (run locally, in the notebook kernel).

- topology():   world map — FABRIC sites and links, AmLight, and this slice's nodes + XFER link
- dashboard():  batch summary — drift, high-resolution gain, CTF fit resolution, mean spectra
- transfer():   iperf3 baseline over time vs. achieved rsync rate, checksum badge
- timeline():   end-to-end stage timeline (data → GPUs → CTF → network)
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np

BLUE, RED, GREEN, GREY, ORANGE = "#2471a3", "#c0392b", "#1e8449", "#7f8c8d", "#ca6f1e"


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


# --- stage timing ---------------------------------------------------------------
STAGES: dict = {}


@contextmanager
def stage(name: str):
    """``with stage("MotionCor3"): ...`` records wall-clock start/end for timeline()."""
    t0 = time.time()
    try:
        yield
    finally:
        STAGES[name] = (t0, time.time())


# --- 1. slice map -----------------------------------------------------------------
# AmLight points of presence and links, traced from the AmLight network map
# (positions are city coordinates; capacities as labelled on the map).
AMLIGHT_POPS = {
    "Atlanta": (33.75, -84.39), "Jacksonville": (30.33, -81.66), "Boca Raton": (26.37, -80.13),
    "Miami": (25.76, -80.19), "San Juan": (18.47, -66.11), "Panama City": (8.98, -79.52),
    "Fortaleza": (-3.73, -38.53), "Sao Paulo": (-23.55, -46.63), "Porto Alegre": (-30.03, -51.23),
    "Buenos Aires": (-34.60, -58.38), "Santiago": (-33.45, -70.67), "Sines": (37.96, -8.87),
    "Sangano": (-9.59, 13.24), "Cape Town": (-33.92, 18.42), "SLAC": (37.42, -122.20),
}
AMLIGHT_LINKS = [  # (a, b, capacity, kind)
    ("Atlanta", "Jacksonville", "100G", "spectrum"), ("Atlanta", "Boca Raton", "300G", "spectrum"),
    ("Jacksonville", "Miami", "200G", "spectrum"), ("Boca Raton", "Miami", "", "spectrum"),
    ("Boca Raton", "Fortaleza", "1.1 Tbps (AmLight Express)", "spectrum"),
    ("Fortaleza", "Sao Paulo", "1.2 Tbps", "spectrum"), ("Fortaleza", "Sangano", "100G", "spectrum"),
    ("Sao Paulo", "Porto Alegre", "100G", "spectrum"), ("Porto Alegre", "Buenos Aires", "100G", "spectrum"),
    ("Sao Paulo", "Buenos Aires", "400G", "spectrum"), ("Buenos Aires", "Santiago", "300G", "spectrum"),
    ("Miami", "San Juan", "100G", "leased"), ("San Juan", "Fortaleza", "100G", "leased"),
    ("Miami", "Panama City", "100G", "leased"), ("Panama City", "Santiago", "100G (AmLight Protect)", "leased"),
    ("Fortaleza", "Sao Paulo", "100G", "leased"), ("Santiago", "Sao Paulo", "100G", "leased"),
    ("Panama City", "Fortaleza", "200G", "partner"), ("Fortaleza", "Sines", "200G", "partner"),
    ("Sangano", "Cape Town", "100G", "partner"), ("SLAC", "Atlanta", "100G (ESnet)", "partner"),
]
AMLIGHT_COLORS = {"spectrum": "#8fbc1f", "leased": "#5d6d7e", "partner": "#b0479f"}
FABRIC_LINK, FABRIC_SITE = "#9fb4c7", "#34495e"

# FABRIC reports some new sites at placeholder coordinates (CAPE: CIEN's Ottawa address,
# while the site is still in maintenance). Plot them where they really are.
SITE_LOCATION_OVERRIDES = {"CAPE": (-33.92, 18.42)}  # Cape Town, South Africa
# The planned downstream site that will replace the FIU receiver.
FUTURE_RECEIVER = ("CAPE", "Cape Town, South Africa")
PURPLE = "#7d3c98"


def site_locations(fablib, sites=None) -> dict:
    """site -> (lat, lon) for ``sites`` (default: every FABRIC site)."""
    res = fablib.get_resources()
    out = {}
    for s in sites if sites is not None else res.get_site_names():
        try:
            lat, lon = res.get_location_lat_long(s)
            out[s] = (float(lat), float(lon))
        except Exception:
            pass
        if s in SITE_LOCATION_OVERRIDES:
            out[s] = SITE_LOCATION_OVERRIDES[s]
    return out


def fabric_links(fablib) -> list:
    """Inter-site FABRIC links as (site_a, site_b, bandwidth Gbit/s), one per site pair."""
    pairs = {}
    try:
        for link in fablib.list_links(output="list", quiet=True) or []:
            sites = sorted(set(link.get("sites") or []))
            if len(sites) == 2:
                key = tuple(sites)
                pairs[key] = max(pairs.get(key, 0), int(link.get("bandwidth") or 0))
    except Exception:
        pass
    return [(a, b, bw) for (a, b), bw in pairs.items()]


def topology(fablib, slice_obj, transfer: dict | None = None, ceph_cluster: str = "west",
             amlight: bool = True):
    """Map of the whole FABRIC footprint (sites + inter-site links), the AmLight network,
    and this slice on top. Uses plotly (interactive world map) when available, else matplotlib."""
    nodes = [(n.get_name(), n.get_site()) for n in slice_obj.get_nodes()]
    loc = site_locations(fablib)
    links = fabric_links(fablib)
    gpu = next(((n, s) for n, s in nodes if n == "cryoem"), None)
    rx = next(((n, s) for n, s in nodes if n == "receiver"), None)
    xfer = "XFER (this slice, dedicated L2)"
    if transfer:
        xfer += f" — {transfer.get('rsync Gbit/s')} Gbit/s rsync / {transfer.get('iperf3 Gbit/s')} iperf3"
    roles = {"cryoem": f"GPU node: MotionCor3 + CTFFIND4<br>input: CephFS '{ceph_cluster}' volume",
             "receiver": "Receiver: ~/cryoem-inbox"}
    am_links = [(AMLIGHT_POPS[a], AMLIGHT_POPS[b], cap, kind) for a, b, cap, kind in AMLIGHT_LINKS] if amlight else []
    cape_site, cape_city = FUTURE_RECEIVER
    cape = loc.get(cape_site)
    cape_note = (f"{cape_site} ({cape_city}): planned downstream site, replacing the FIU receiver.<br>"
                 "FABRIC site in maintenance; reached via AmLight (Fortaleza–Sangano–Cape Town) and STAR.")

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        for i, (a, b, bw) in enumerate(links):          # FABRIC backbone
            if a in loc and b in loc:
                fig.add_trace(go.Scattergeo(
                    lat=[loc[a][0], loc[b][0]], lon=[loc[a][1], loc[b][1]], mode="lines",
                    line=dict(width=1.5, color=FABRIC_LINK), hoverinfo="text",
                    text=f"FABRIC {a}–{b}: {bw} Gbit/s", name="FABRIC links",
                    legendgroup="fabric", showlegend=i == 0))
        shown = set()
        for (la1, lo1), (la2, lo2), cap, kind in am_links:  # AmLight
            fig.add_trace(go.Scattergeo(
                lat=[la1, la2], lon=[lo1, lo2], mode="lines",
                line=dict(width=2.5, color=AMLIGHT_COLORS[kind]), hoverinfo="text",
                text=f"AmLight ({kind}) {cap}", name=f"AmLight {kind}",
                legendgroup=f"am-{kind}", showlegend=kind not in shown))
            shown.add(kind)
        if amlight:
            fig.add_trace(go.Scattergeo(
                lat=[v[0] for v in AMLIGHT_POPS.values()], lon=[v[1] for v in AMLIGHT_POPS.values()],
                mode="markers", marker=dict(size=7, color="#f4d03f", line=dict(width=1, color="#7d6608")),
                text=list(AMLIGHT_POPS), hoverinfo="text", name="AmLight PoPs"))
        used = {s for _, s in nodes}
        fig.add_trace(go.Scattergeo(                    # every FABRIC site
            lat=[loc[s][0] for s in loc], lon=[loc[s][1] for s in loc], mode="markers+text",
            text=list(loc), textposition="bottom center", textfont=dict(size=9, color=FABRIC_SITE),
            marker=dict(size=[6 if s not in used else 0 for s in loc], color=FABRIC_SITE),
            hoverinfo="text", name="FABRIC sites"))
        if gpu and rx and gpu[1] in loc and rx[1] in loc:
            (la1, lo1), (la2, lo2) = loc[gpu[1]], loc[rx[1]]
            fig.add_trace(go.Scattergeo(lat=[la1, la2], lon=[lo1, lo2], mode="lines",
                                        line=dict(width=5, color=GREEN), name=xfer, hoverinfo="name"))
        if cape and gpu and gpu[1] in loc and (not rx or rx[1] != cape_site):
            (la1, lo1), (la2, lo2) = loc[gpu[1]], loc[cape_site]
            fig.add_trace(go.Scattergeo(lat=[la1, la2], lon=[lo1, lo2], mode="lines",
                                        line=dict(width=3, color=PURPLE, dash="dash"),
                                        name=f"planned: XFER → {cape_site}", hoverinfo="name"))
        if cape:
            fig.add_trace(go.Scattergeo(
                lat=[cape[0]], lon=[cape[1]], mode="markers+text",
                text=[f"<b>{cape_site}</b> — future receiver"], textposition="middle right",
                textfont=dict(size=12, color=PURPLE), hovertext=cape_note, hoverinfo="text",
                marker=dict(size=20, symbol="star", color=PURPLE, line=dict(width=2, color="white")),
                name=f"{cape_site} ({cape_city}): future receiver"))
        for name, site in nodes:
            if site not in loc:
                continue
            la, lo = loc[site]
            fig.add_trace(go.Scattergeo(
                lat=[la], lon=[lo], mode="markers+text", text=[f"<b>{name}</b> @ {site}"],
                textposition="top center", hovertext=roles.get(name, name), hoverinfo="text",
                marker=dict(size=16, color=BLUE if name == "cryoem" else ORANGE,
                            line=dict(width=2, color="white")), name=f"{name} ({site})"))
        fig.update_geos(projection_type="natural earth", showland=True, landcolor="#f2f3f4",
                        showocean=True, oceancolor="#eaf2f8", showcountries=True,
                        countrycolor="#cfd8dc", lataxis_range=[-50, 65], lonaxis_range=[-170, 160])
        fig.update_layout(title="FABRIC footprint and AmLight, with this slice: "
                                "Ceph volume → GPU processing → transfer downstream",
                          height=620, margin=dict(l=0, r=0, t=50, b=0),
                          legend=dict(orientation="h", y=-0.02))
        return fig
    except ImportError:
        fig, ax = plt.subplots(figsize=(14, 7))
        for a, b, _ in links:
            if a in loc and b in loc:
                ax.plot([loc[a][1], loc[b][1]], [loc[a][0], loc[b][0]], color=FABRIC_LINK, lw=1)
        for (la1, lo1), (la2, lo2), _, kind in am_links:
            ax.plot([lo1, lo2], [la1, la2], color=AMLIGHT_COLORS[kind], lw=2)
        for s, (la, lo) in loc.items():
            ax.scatter(lo, la, s=12, color=FABRIC_SITE, zorder=3)
            ax.annotate(s, (lo, la), xytext=(3, -9), textcoords="offset points", fontsize=7)
        if gpu and rx and gpu[1] in loc and rx[1] in loc:
            (la1, lo1), (la2, lo2) = loc[gpu[1]], loc[rx[1]]
            ax.plot([lo1, lo2], [la1, la2], color=GREEN, lw=4, label=xfer)
        if cape and gpu and gpu[1] in loc and (not rx or rx[1] != cape_site):
            (la1, lo1), (la2, lo2) = loc[gpu[1]], cape
            ax.plot([lo1, lo2], [la1, la2], color=PURPLE, lw=2, ls="--", label=f"planned: XFER → {cape_site}")
        if cape:
            ax.scatter(cape[1], cape[0], s=400, marker="*", color=PURPLE, zorder=5)
            ax.annotate(f"{cape_site} ({cape_city})\nfuture receiver", (cape[1], cape[0]), xytext=(10, -4),
                        textcoords="offset points", fontsize=9, weight="bold", color=PURPLE)
        for name, site in nodes:
            if site in loc:
                la, lo = loc[site]
                ax.scatter(lo, la, s=200, color=BLUE if name == "cryoem" else ORANGE, zorder=4)
                ax.annotate(f"{name} @ {site}", (lo, la), xytext=(8, 8), textcoords="offset points",
                            fontsize=9, weight="bold")
        ax.set_xlim(-170, 160), ax.set_ylim(-50, 65)
        ax.set_xlabel("longitude"), ax.set_ylabel("latitude")
        ax.set_title("FABRIC footprint and AmLight (install plotly for a map background)")
        ax.legend(loc="lower left", frameon=False)
        _clean(ax)
        return fig


# --- 2. batch dashboard -------------------------------------------------------------
def dashboard(summary: list, mc_seconds: float, gpu_ids: str, pixel_a: float | None = None):
    def short(stem):  # FoilHole_6683041_Data_6680842_36_2025... -> 6683041·36
        p = stem.split("_")
        return f"{p[1]}·{p[4]}" if stem.startswith("FoilHole_") and len(p) > 4 else stem[:16]
    names = [short(s["micrograph"]) for s in summary]
    drift = [s.get("total_drift_px") or 0 for s in summary]
    gain = [s.get("hf_gain") or np.nan for s in summary]
    fitres = [(s.get("ctf") or {}).get("fit_resolution_A") for s in summary]
    fitres = [f for f in fitres if f]
    n_gpu = len(gpu_ids.split())
    rate = 60 * len(summary) / mc_seconds if mc_seconds else 0

    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(f"{len(summary)} micrographs · MotionCor3 {mc_seconds:.0f} s on {n_gpu} GPU(s) "
                 f"= {rate:.1f} movies/min", fontsize=15, fontweight="bold")

    a = ax[0, 0]
    a.bar(range(len(drift)), drift, color=BLUE)
    a.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    a.set_ylabel("total drift (pixels)")
    a.set_title("Beam-induced motion removed per movie")
    if drift:
        a.axhline(np.mean(drift), color=GREY, ls="--", lw=1)
        a.text(-0.4, np.mean(drift), f"mean {np.mean(drift):.1f} px", va="bottom", ha="left", color=GREY,
               bbox=dict(fc="white", ec="none", pad=1))
    _clean(a)

    a = ax[0, 1]
    a.bar(range(len(gain)), gain, color=[GREEN if g >= 1 else RED for g in np.nan_to_num(gain, nan=1)])
    a.axhline(1, color="k", lw=1)
    a.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    a.set_ylabel("after / before power, 10–5 Å")
    a.set_title("High-resolution signal recovered (×, >1 = better)")
    _clean(a)

    a = ax[1, 0]
    pairs = sorted(((s.get("ctf") or {}).get("fit_resolution_A"), n) for s, n in zip(summary, names)
                   if (s.get("ctf") or {}).get("fit_resolution_A"))
    if pairs:
        vals = [v for v, _ in pairs]
        a.bar(range(len(vals)), vals, color=[GREEN if v <= 5 else ORANGE for v in vals])
        a.set_xticks(range(len(vals)), [n for _, n in pairs], rotation=45, ha="right", fontsize=8)
        a.axhline(np.median(vals), color="k", ls="--", lw=1)
        a.text(-0.4, np.median(vals), f"median {np.median(vals):.1f} Å", va="bottom",
               bbox=dict(fc="white", ec="none", pad=1))
    a.set_ylabel("CTF fit resolution (Å, lower = better)")
    a.set_title("CTF estimation quality per micrograph (green ≤ 5 Å)")
    _clean(a)

    a = ax[1, 1]
    profs = [s["profile"] for s in summary if s.get("profile")]
    if profs:
        f = np.array(profs[0]["freq"])
        b = np.exp(np.mean([np.log(p["before"]) for p in profs], axis=0))
        af = np.exp(np.mean([np.log(p["after"]) for p in profs], axis=0))
        a.semilogy(f, b, color=RED, lw=2, label="before (no alignment)")
        a.semilogy(f, af, color=BLUE, lw=2, label="after (MotionCor3)")
        a.axvspan(1 / 10, 1 / 5, color="0.9", zorder=0)
        ticks = [20, 10, 6, 4, 3]
        nyq = 2 * pixel_a if pixel_a else 0
        a.set_xticks([1 / t for t in ticks if t > nyq], [f"{t} Å" for t in ticks if t > nyq])
        a.set_xlim(1 / 30, min(f[-1], 1 / max(3.0, nyq)))
        from matplotlib.ticker import NullFormatter, ScalarFormatter
        a.yaxis.set_major_formatter(ScalarFormatter())
        a.yaxis.set_minor_formatter(NullFormatter())
        a.legend(frameon=False)
    a.set_xlabel("resolution")
    a.set_ylabel("mean power (same scale before/after)")
    a.set_title("Average signal vs. resolution, all micrographs")
    _clean(a)
    fig.tight_layout()
    return fig


# --- 3. transfer ------------------------------------------------------------------
def transfer(iperf: dict, result: dict):
    ok = result.get("checksums") == "verified"
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})
    fig.suptitle(f"{result['source']}  →  {result['destination']}   ·   {result['GB']} GB, "
                 f"{result['files']} files in {result['seconds']} s", fontsize=14, fontweight="bold")

    a = ax[0]
    iv = iperf.get("intervals", [])
    t = [i["sum"]["end"] for i in iv]
    g = [i["sum"]["bits_per_second"] / 1e9 for i in iv]
    a.plot(t, g, "o-", color=BLUE, lw=2, ms=4, label=f"iperf3 ({len(iv[0]['streams']) if iv else '?'} streams)")
    a.fill_between(t, g, color=BLUE, alpha=0.12)
    a.axhline(result["rsync Gbit/s"], color=GREEN, lw=2.5, ls="--",
              label=f"rsync achieved: {result['rsync Gbit/s']} Gbit/s")
    a.set_ylim(0, max(g + [result["rsync Gbit/s"]]) * 1.25 if g else 1)
    a.set_xlabel("seconds")
    a.set_ylabel("Gbit/s")
    a.set_title(f"Network baseline on the XFER link (MTU {result.get('MTU')})")
    a.legend(frameon=False, loc="lower right")
    _clean(a)

    a = ax[1]
    vals = [result["iperf3 Gbit/s"], result["rsync Gbit/s"]]
    bars = a.bar(["link capacity\n(iperf3)", f"data transfer\n({result.get('method', 'rsync/SSH')})"], vals,
                 color=[BLUE, GREEN])
    for b_, v in zip(bars, vals):
        a.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=12)
    a.set_ylabel("Gbit/s")
    a.set_title(f"Efficiency {result.get('efficiency')}")
    a.text(0.5, 0.93, "✔ checksums verified" if ok else "✘ checksum MISMATCH", transform=a.transAxes,
           ha="center", fontsize=12, fontweight="bold", color="white",
           bbox=dict(boxstyle="round,pad=0.4", fc=GREEN if ok else RED, ec="none"))
    a.set_ylim(0, max(vals) * 1.35)
    _clean(a)
    fig.tight_layout()
    return fig


# --- 5. timeline ------------------------------------------------------------------
def timeline(stages: dict | None = None):
    stages = stages or STAGES
    if not stages:
        raise ValueError("no stages recorded")
    items = sorted(stages.items(), key=lambda kv: kv[1][0])
    t0 = items[0][1][0]
    colors = {"data": GREY, "gpu": BLUE, "cpu": ORANGE, "net": GREEN}

    def kind(name):
        n = name.lower()
        if any(k in n for k in ("motioncor", "gpu")):
            return "gpu"
        if any(k in n for k in ("iperf", "transfer", "rsync", "verify")):
            return "net"
        if any(k in n for k in ("ctf", "preview")):
            return "cpu"
        return "data"

    fig, ax = plt.subplots(figsize=(15, 0.55 * len(items) + 1.6))
    for i, (name, (a, b)) in enumerate(items):
        ax.barh(i, b - a, left=a - t0, color=colors[kind(name)], height=0.6)
        ax.text(b - t0, i, f"  {b - a:.0f} s", va="center", fontsize=10)
    ax.set_yticks(range(len(items)), [n for n, _ in items])
    ax.invert_yaxis()
    total = items[-1][1][1] - t0
    ax.set_xlabel("seconds since start")
    ax.set_title(f"End-to-end pipeline: {total / 60:.1f} min from volume to verified copy at the receiver",
                 fontsize=13, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for l, c in
                       (("storage", GREY), ("GPU", BLUE), ("CPU", ORANGE), ("network", GREEN))],
              frameon=False, loc="upper right")
    ax.set_xlim(0, total * 1.12)
    _clean(ax)
    fig.tight_layout()
    return fig
