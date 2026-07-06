"""generate_chime_mask.py

Generates a simple CHIME-like detection probability mask:

  - Zero for dec < DEC_MIN  (below CHIME's southern horizon)
  - Cosine-shaped in declination:
        p(dec) = cos(|dec - CHIME_LAT|)
    which peaks at the zenith (dec = CHIME latitude) and smoothly falls to
    zero at the horizon (90° away), matching the projection of the cylinder
    beam onto the sky.
  - Uniform in RA (transit telescope, full rotation of the Earth).

Output
------
  chime_mask_cosine_nside<NSIDE>.fits  – HEALPix probability map
  chime_mask_cosine_nside<NSIDE>.png   – Mollweide projection plot
"""

import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
CHIME_LAT = 49.32    # DRAO latitude [degrees North]
DEC_MIN   = -11.0    # southernmost declination accessible to CHIME [degrees]
NSIDE     = 1024

_HERE       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FITS = os.path.join(_HERE, f"chime_mask_cosine_nside{NSIDE}.fits")
OUTPUT_PLOT = os.path.join(_HERE, f"chime_mask_cosine_nside{NSIDE}.png")


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def build_chime_mask(nside: int, chime_lat: float, dec_min: float) -> np.ndarray:
    """Return a HEALPix map (RING ordering) with the CHIME cosine mask.

    The detection probability at declination δ is
        p(δ) = max(0, cos(|δ - chime_lat|))
    for δ ≥ dec_min, and 0 elsewhere.
    """
    npix  = hp.nside2npix(nside)
    theta, _ = hp.pix2ang(nside, np.arange(npix))
    dec   = 90.0 - np.degrees(theta)          # colatitude → declination

    zenith_angle = np.abs(dec - chime_lat)    # degrees from zenith at transit
    prob = np.cos(np.radians(zenith_angle))   # 1 at zenith, 0 at horizon

    prob[dec < dec_min] = 0.0                 # below southern horizon
    prob = np.clip(prob, 0.0, 1.0)
    return prob.astype(np.float32)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_mask(mask: np.ndarray, output_path: str) -> None:
    fig = plt.figure(figsize=(11, 5.5))
    hp.mollview(
        mask,
        fig=fig.number,
        title=f"CHIME detection probability mask  (cosine beam, NSIDE={NSIDE})",
        unit="detection probability",
        cmap="viridis",
        min=0.0,
        max=1.0,
        hold=True,
    )
    hp.graticule(dpar=30, dmer=60, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {output_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Building CHIME cosine mask  (NSIDE={NSIDE}, lat={CHIME_LAT}°N, dec_min={DEC_MIN}°)")
    mask = build_chime_mask(NSIDE, CHIME_LAT, DEC_MIN)

    n_nonzero = int((mask > 0).sum())
    f_sky     = n_nonzero / hp.nside2npix(NSIDE)
    print(f"  Non-zero pixels : {n_nonzero}  (f_sky = {f_sky:.3f})")
    print(f"  Peak value      : {mask.max():.6f}  (at dec ≈ {CHIME_LAT}°)")

    hp.write_map(
        OUTPUT_FITS, mask,
        overwrite=True,
        dtype=np.float32,
        column_names=["PROB"],
        extra_header=[
            ("CHIME_LAT", CHIME_LAT, "telescope latitude [deg N]"),
            ("DEC_MIN",   DEC_MIN,   "southern horizon cutoff [deg]"),
            ("MASKTYPE",  "cosine",  "beam model"),
        ],
    )
    print(f"  FITS saved      : {OUTPUT_FITS}")

    plot_mask(mask, OUTPUT_PLOT)


if __name__ == "__main__":
    main()
