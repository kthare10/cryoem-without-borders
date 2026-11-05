#!/usr/bin/env python3
"""Before/after figures for the CryoEM pre-processing demo (runs on the node).

For every motion-corrected micrograph in OUT_DIR this renders one PNG with:
  row 1: unaligned sum (MotionCor3 -Align 0) | aligned micrograph | CTFFIND fit
  row 2: power spectrum of each       | drift trajectory
and writes summary.json with drift and CTFFIND numbers per micrograph.

    ~/cryoem-venv/bin/python cryoem_preview.py MOVIE_DIR OUT_DIR [--raw-dir DIR] [--max N]

The "before" image is taken from --raw-dir (same frames, gain-corrected, summed without
alignment). Without it, MRC/TIFF movies are summed here; EER needs --raw-dir.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mrcfile  # noqa: E402
import numpy as np  # noqa: E402

BLUE_C = "#2471a3"
MOVIE_EXTS = (".eer", ".mrcs", ".mrc", ".tif", ".tiff")


def bin2d(img: np.ndarray, target: int = 1024) -> np.ndarray:
    """Block-average down to roughly ``target`` pixels on the long edge."""
    f = max(1, max(img.shape) // target)
    h, w = (img.shape[0] // f) * f, (img.shape[1] // f) * f
    return img[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def power_spectrum(img: np.ndarray, box: int = 1024) -> np.ndarray:
    """Log power spectrum of the central ``box`` crop (shows Thon rings)."""
    cy, cx = img.shape[0] // 2, img.shape[1] // 2
    b = min(box, img.shape[0], img.shape[1]) // 2
    crop = img[cy - b:cy + b, cx - b:cx + b].astype(np.float32)
    crop = crop - crop.mean()
    ps = np.abs(np.fft.fftshift(np.fft.fft2(crop))) ** 2
    ps = np.log1p(ps)
    c = ps.shape[0] // 2
    ps[c - 2:c + 3, c - 2:c + 3] = np.median(ps)        # hide the DC spike
    return ps


def radial_profile(img: np.ndarray, pixel_a: float, box: int = 1024):
    """Rotationally averaged power spectrum of the central crop.

    Returns (spatial frequency in 1/Å, power). Averaging over 4 quadrant tiles
    reduces noise so before/after curves are easy to compare.
    """
    b = min(box, img.shape[0] // 2, img.shape[1] // 2)
    cy, cx = img.shape[0] // 2, img.shape[1] // 2
    tiles = [img[cy - b:cy, cx - b:cx], img[cy - b:cy, cx:cx + b],
             img[cy:cy + b, cx - b:cx], img[cy:cy + b, cx:cx + b]]
    ps = np.zeros((b, b), dtype=np.float64)
    for t in tiles:
        t = t.astype(np.float64) - t.mean()
        ps += np.abs(np.fft.fftshift(np.fft.fft2(t * np.hanning(b)[:, None] * np.hanning(b)[None, :]))) ** 2
    yy, xx = np.indices(ps.shape)
    r = np.hypot(yy - b // 2, xx - b // 2).astype(int)
    prof = np.bincount(r.ravel(), ps.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    n = b // 2
    freq = np.arange(n) / (b * pixel_a)
    return freq[1:n], prof[1:n]


def hf_gain(freq, before, after, band=(1 / 10.0, 1 / 5.0)) -> float:
    """After/before power ratio in a high-resolution band (10-5 Å).

    Both images are sums of the SAME gain-corrected frames, so total dose and shot
    noise are identical and absolute power is directly comparable: the extra power
    after alignment is signal that motion had blurred away."""
    m = (freq >= band[0]) & (freq <= band[1])
    return float(after[m].mean() / before[m].mean())


def stretch(img: np.ndarray, lo: float = 1, hi: float = 99) -> tuple:
    a, b = np.percentile(img, [lo, hi])
    return a, b


def raw_sum(movie: str) -> np.ndarray | None:
    """Plain sum of all frames, i.e. what the detector saw without alignment."""
    ext = os.path.splitext(movie)[1].lower()
    if ext == ".eer":
        return None                     # EER can't be decoded here; use --raw-dir
    if ext in (".tif", ".tiff"):
        import tifffile

        return tifffile.imread(movie).astype(np.float32).sum(axis=0)
    with mrcfile.mmap(movie, permissive=True) as m:
        data = m.data
        return data.astype(np.float32).sum(axis=0) if data.ndim == 3 else data.astype(np.float32)


def read_mrc(path: str) -> np.ndarray:
    with mrcfile.open(path, permissive=True) as m:
        d = m.data
        return (d[0] if d.ndim == 3 else d).astype(np.float32)


def read_drift(log_dir: str, stem: str):
    """Global frame shifts from MotionCor3's ``<stem>*-Full.log`` (x, y in pixels)."""
    for path in glob.glob(os.path.join(log_dir, f"{stem}*Full.log")):
        xs, ys = [], []
        with open(path) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 3 and not line.lstrip().startswith("#"):
                    try:
                        xs.append(float(parts[1]))
                        ys.append(float(parts[2]))
                    except ValueError:
                        pass
        if xs:
            return np.array(xs), np.array(ys)
    return None


def read_ctffind(txt: str) -> dict | None:
    """Parse CTFFIND4's ``*_ctf.txt`` (last non-comment line)."""
    if not os.path.exists(txt):
        return None
    rows = [ln.split() for ln in open(txt) if ln.strip() and not ln.startswith("#")]
    if not rows:
        return None
    v = [float(x) for x in rows[-1]]
    return {"defocus1_A": v[1], "defocus2_A": v[2], "astig_angle_deg": v[3],
            "phase_shift_rad": v[4], "cross_corr": v[5], "fit_resolution_A": v[6]}


def find_movie(movie_dir: str, stem: str):
    for ext in MOVIE_EXTS:
        p = os.path.join(movie_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("movie_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--raw-dir", default=None, help="unaligned sums (MotionCor3 -Align 0)")
    ap.add_argument("--max", type=int, default=6)
    ap.add_argument("--pixel", type=float, default=None, help="micrograph pixel size (Å) for resolution axes")
    args = ap.parse_args()

    out = os.path.expanduser(args.out_dir)
    movies = os.path.expanduser(args.movie_dir)
    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Only the plain aligned sums: skip dose-weighted sums, stacks, and CTF outputs
    # (MotionCor3's own <stem>_Ctf.mrc and CTFFIND's <stem>_ctf.mrc).
    mics = sorted(p for p in glob.glob(os.path.join(out, "*.mrc"))
                  if not re.search(r"(_DW|_DWS|_Stk|_ctf)\.mrc$", p, re.IGNORECASE))
    summary = []
    for mic in mics[:args.max]:
        stem = os.path.splitext(os.path.basename(mic))[0]
        movie = find_movie(movies, stem)
        corrected = read_mrc(mic)
        raw_mic = os.path.join(os.path.expanduser(args.raw_dir), f"{stem}.mrc") if args.raw_dir else None
        if raw_mic and os.path.exists(raw_mic):
            before = read_mrc(raw_mic)
        else:
            before = raw_sum(movie) if movie else None
        ctf = read_ctffind(os.path.join(out, "ctf", f"{stem}_ctf.txt"))
        ctf_img = os.path.join(out, "ctf", f"{stem}_ctf.mrc")
        drift = read_drift(os.path.join(out, "logs"), stem)

        fig, ax = plt.subplots(2, 3, figsize=(15, 10))
        for a in ax.ravel():
            a.set_xticks([]), a.set_yticks([])
        if before is not None:
            b = bin2d(before)
            ax[0, 0].imshow(b, cmap="gray", vmin=stretch(b)[0], vmax=stretch(b)[1])
            ps = power_spectrum(before)
            ax[1, 0].imshow(ps, cmap="gray", vmin=stretch(ps, 5, 99.5)[0], vmax=stretch(ps, 5, 99.5)[1])
        ax[0, 0].set_title("BEFORE: frame sum, no alignment")
        ax[1, 0].set_title("BEFORE: power spectrum")

        c = bin2d(corrected)
        ax[0, 1].imshow(c, cmap="gray", vmin=stretch(c)[0], vmax=stretch(c)[1])
        ax[0, 1].set_title("AFTER: MotionCor3 micrograph")
        ps = power_spectrum(corrected)
        ax[1, 1].imshow(ps, cmap="gray", vmin=stretch(ps, 5, 99.5)[0], vmax=stretch(ps, 5, 99.5)[1])
        ax[1, 1].set_title("AFTER: power spectrum (Thon rings)")

        if os.path.exists(ctf_img):
            d = read_mrc(ctf_img)
            ax[0, 2].imshow(d, cmap="gray", vmin=stretch(d)[0], vmax=stretch(d)[1])
        title = "CTFFIND4 fit"
        if ctf:
            title += (f"\ndefocus {ctf['defocus1_A'] / 1e4:.2f}/{ctf['defocus2_A'] / 1e4:.2f} µm, "
                      f"fit to {ctf['fit_resolution_A']:.1f} Å")
        ax[0, 2].set_title(title)

        drift_px = None
        if drift is not None:
            x, y = drift
            ax[1, 2].plot(x, y, "o-", ms=3)
            ax[1, 2].plot(x[0], y[0], "gs", label="first frame")
            ax[1, 2].set_aspect("equal", adjustable="datalim")
            ax[1, 2].set_xticks(np.linspace(x.min(), x.max(), 3).round(1))
            ax[1, 2].set_yticks(np.linspace(y.min(), y.max(), 3).round(1))
            ax[1, 2].legend(loc="best")
            drift_px = float(np.hypot(np.diff(x), np.diff(y)).sum())
            ax[1, 2].set_title(f"Beam-induced drift: {drift_px:.1f} px total")
        else:
            ax[1, 2].set_title("Drift log not found")

        fig.suptitle(stem, fontsize=14)
        fig.tight_layout()
        png = os.path.join(fig_dir, f"{stem}.png")
        fig.savefig(png, dpi=90)
        plt.close(fig)

        # --- zoom + resolution figure: what motion correction buys you ---
        zoom_png, gain, profile = None, None, None
        if before is not None:
            fig, ax = plt.subplots(1, 3, figsize=(16, 5.4), gridspec_kw={"width_ratios": [1, 1, 1.3]})
            z = 384                                     # full-resolution crop, no binning
            cy, cx = corrected.shape[0] // 2, corrected.shape[1] // 2
            for a, img, title in ((ax[0], before, "BEFORE (no alignment)"),
                                  (ax[1], corrected, "AFTER (MotionCor3)")):
                crop = img[cy - z:cy + z, cx - z:cx + z]
                lo, hi = stretch(crop, 2, 98)
                a.imshow(crop, cmap="gray", vmin=lo, vmax=hi)
                a.set_title(f"{title}: full-resolution crop", fontsize=12)
                a.set_xticks([]), a.set_yticks([])
            if args.pixel:
                f, pb = radial_profile(before, args.pixel)
                _, pa = radial_profile(corrected, args.pixel)
                # One common scale for both curves (same frames, same dose).
                scale = pb[(f >= 1 / 50) & (f <= 1 / 20)].mean()
                pb_n, pa_n = pb / scale, pa / scale
                k = max(3, len(f) // 150)
                smooth = lambda v: np.convolve(v, np.ones(k) / k, "same")
                ax[2].semilogy(f, smooth(pb_n), color="#c0392b", lw=1.6, label="before")
                ax[2].semilogy(f, smooth(pa_n), color="#2471a3", lw=1.6, label="after")
                gain = hf_gain(f, pb, pa)
                ax[2].axvspan(1 / 10, 1 / 5, color="0.85", zorder=0)
                ax[2].text(1 / 7.07, ax[2].get_ylim()[1], f"10–5 Å\nsignal ×{gain:.2f}", ha="center", va="top",
                           fontsize=10, fontweight="bold", color=BLUE_C)
                if ctf and ctf.get("fit_resolution_A"):
                    ax[2].axvline(1 / ctf["fit_resolution_A"], color="k", ls="--", lw=1)
                    ax[2].text(1 / ctf["fit_resolution_A"], ax[2].get_ylim()[1], f"CTF fit {ctf['fit_resolution_A']:.1f} Å ",
                               rotation=90, va="top", ha="right", fontsize=9)
                ticks = [20, 10, 6, 4, 3]
                nyq = 2 * args.pixel
                ax[2].set_xticks([1 / t for t in ticks if t > nyq])
                ax[2].set_xticklabels([f"{t} Å" for t in ticks if t > nyq])
                ax[2].set_xlim(1 / 30, min(f[-1], 1 / max(3.0, nyq)))
                ax[2].set_xlabel("resolution")
                ax[2].set_ylabel("power (same scale for both)")
                ax[2].set_title("Signal vs. resolution", fontsize=12)
                ax[2].legend(frameon=False)
                ax[2].spines[["top", "right"]].set_visible(False)
                sel = np.linspace(0, len(f) - 1, 200).astype(int)
                profile = {"freq": f[sel].tolist(), "before": pb_n[sel].tolist(), "after": pa_n[sel].tolist()}
            else:
                ax[2].axis("off")
            fig.tight_layout()
            zoom_png = os.path.join(fig_dir, f"{stem}_zoom.png")
            fig.savefig(zoom_png, dpi=90)
            plt.close(fig)

        summary.append({"micrograph": stem, "movie": movie, "figure": png, "zoom_figure": zoom_png,
                        "total_drift_px": drift_px, "hf_gain": gain, "profile": profile, "ctf": ctf})
        print(f"preview: {stem} drift={drift_px} ctf={ctf}")

    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {len(summary)} figure(s) to {fig_dir}")


if __name__ == "__main__":
    main()
