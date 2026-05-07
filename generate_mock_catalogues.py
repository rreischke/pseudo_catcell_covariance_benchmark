"""generate_mock_catalogues.py

Generate many realisations of a mock FRB catalogue from an input angular power
spectrum and write them incrementally to a FITS file.

Each realisation gets its own FITS binary-table extension named REAL_NNNNN so
the file is fully self-contained and grows by one extension per completed
realisation.  A fixed CATALOG extension stores the input sky positions and a
CELL extension stores the interpolated C_ell array that was passed to synfast.

Configuration via environment variables (all optional, sane defaults given):
    CL_PATH          – path to the .npz file with 'ell' and 'cell' keys
                       [default: data/cell_chime.npz  relative to this script]
    POSITIONS_PATH   – path to FITS table with source positions
                       [default: data/chimefrbcat2.fits]
    OUTPUT_PATH      – path to write the output FITS file
                       [default: output/mock_catalogues.fits]
    NSIDE            – HEALPix NSIDE for synfast  [default: 4096]
    N_REALISATIONS   – number of realisations to generate  [default: 100]
    NOISE_MEAN       – mean of additive noise  [default: 50.0]
    NOISE_VAR        – variance (sigma^2) of additive noise  [default: 2500.0]
    SEED_START       – base random seed; realisation i uses SEED_START + i
                       [default: 0]
    LMAX_FACTOR      – lmax = min(max_input_ell, LMAX_FACTOR * NSIDE - 1)
                       [default: 3]
    RESUME           – if 'true', skip already-completed realisations found in
                       an existing output file  [default: true]
"""

import os
import sys
import numpy as np
import healpy as hp
from astropy.table import Table
from astropy.io import fits
from scipy.interpolate import InterpolatedUnivariateSpline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

CL_PATH        = os.getenv("CL_PATH",        os.path.join(_HERE, "data", "cell_chime.npz"))
POSITIONS_PATH = os.getenv("POSITIONS_PATH", os.path.join(_HERE, "data", "chimefrbcat2.fits"))
OUTPUT_PATH    = os.getenv("OUTPUT_PATH",    os.path.join(_HERE, "output/chime", "mock_catalogues.fits"))
NSIDE          = int(os.getenv("NSIDE",           "128"))
N_REALISATIONS = int(os.getenv("N_REALISATIONS",  "100"))
NOISE_MEAN     = float(os.getenv("NOISE_MEAN",    "50.0"))
NOISE_VAR      = float(os.getenv("NOISE_VAR",     "2500.0"))
SEED_START     = int(os.getenv("SEED_START",      "0"))
LMAX_FACTOR    = int(os.getenv("LMAX_FACTOR",     "3"))
RESUME         = os.getenv("RESUME", "true").strip().lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Helper: load source positions from FITS catalogue
# ---------------------------------------------------------------------------

def _pick_col(colnames_lower_map, candidates):
    for c in candidates:
        if c in colnames_lower_map:
            return colnames_lower_map[c]
    return None


def load_positions(fits_path):
    """Return (ra, dec) arrays in degrees from the first binary table in *fits_path*."""
    tab = Table.read(fits_path)
    lower_map = {c.lower(): c for c in tab.colnames}

    ra_col  = _pick_col(lower_map, ["ra", "ra_deg", "raj2000", "raj"])
    dec_col = _pick_col(lower_map, ["dec", "dec_deg", "dej2000", "decj"])

    if ra_col is None or dec_col is None:
        raise ValueError(
            f"Could not find RA/Dec columns in {fits_path}. "
            f"Available columns: {tab.colnames}"
        )

    # De-duplicate repeaters if the column exists
    rep_col = _pick_col(lower_map, ["repeater_name", "repeater", "src_name", "source_name"])
    if rep_col is not None:
        seen, keep = set(), []
        for i, val in enumerate(tab[rep_col]):
            name = "" if val is np.ma.masked else str(val).strip()
            if name.lower() in {"", "none", "nan", "--"}:
                keep.append(i)
            elif name not in seen:
                seen.add(name)
                keep.append(i)
        tab = tab[keep]

    ra  = np.asarray(tab[ra_col],  dtype=float)
    dec = np.asarray(tab[dec_col], dtype=float)
    return ra, dec


# ---------------------------------------------------------------------------
# Helper: build interpolated C_ell array for hp.synfast
# ---------------------------------------------------------------------------

def build_cl_full(cl_path, nside, lmax_factor):
    cl_npz = np.load(cl_path)
    if not {"ell", "cell"}.issubset(cl_npz.files):
        raise ValueError(f"{cl_path} must contain 'ell' and 'cell' arrays")

    ell   = np.asarray(cl_npz["ell"],  dtype=int)
    cl_in = np.asarray(cl_npz["cell"], dtype=float)

    valid = np.isfinite(ell) & np.isfinite(cl_in) & (ell >= 2) & (cl_in > 0)
    ell, cl_in = ell[valid], cl_in[valid]

    sort_idx = np.argsort(ell)
    unique_ell, unique_idx = np.unique(ell[sort_idx], return_index=True)
    unique_cl = cl_in[sort_idx][unique_idx]

    if len(unique_ell) < 4:
        raise ValueError("Need at least 4 unique C_ell points for a cubic spline.")

    cl_spline = InterpolatedUnivariateSpline(
        np.log(unique_ell.astype(float)), np.log(unique_cl), k=3
    )

    lmax_syn = min(int(unique_ell.max()), lmax_factor * nside - 1)
    ells_full = np.arange(lmax_syn + 1, dtype=float)
    cl_full = np.zeros(lmax_syn + 1, dtype=float)

    mask_eval = (ells_full >= max(2, unique_ell.min())) & (ells_full <= unique_ell.max())
    cl_full[mask_eval] = np.exp(cl_spline(np.log(ells_full[mask_eval])))
    cl_full[0] = 0.0
    if lmax_syn >= 1:
        cl_full[1] = 0.0
    cl_full = np.maximum(cl_full, 0.0)

    return cl_full, lmax_syn


# ---------------------------------------------------------------------------
# Helper: generate one catalogue realisation
# ---------------------------------------------------------------------------

def generate_realisation(cl_full, lmax_syn, ra, dec, noise_mean, noise_var, seed):
    """
    Draw a new synfast map and noise realisation.

    Returns a dict with keys: DM, DM_gaussian, DM_lognormal.
    """
    rng = np.random.default_rng(seed)

    # New HEALPix map from the same C_ell
    alm = hp.synalm(cl_full, lmax=lmax_syn, new=True)
    # Use rng-derived seed for healpy (which uses the global numpy RNG)
    np.random.seed(int(rng.integers(0, 2**31)))
    map_real = hp.alm2map(alm, nside=NSIDE, lmax=lmax_syn)

    # Evaluate map at catalogue positions
    ra_hp  = np.where(ra < 0, ra + 360.0, ra)
    theta  = np.clip(np.radians(90.0 - dec), 1e-6, np.pi - 1e-6)
    phi    = np.radians(ra_hp)

    valid  = np.isfinite(theta) & np.isfinite(phi)
    dm     = np.full(len(theta), np.nan)
    dm[valid] = hp.get_interp_val(map_real, theta[valid], phi[valid])

    # Additive Gaussian noise
    noise_gauss = rng.normal(loc=noise_mean, scale=np.sqrt(noise_var), size=len(dm))

    # Additive log-normal noise (matched mean/variance)
    if noise_mean <= 0:
        raise ValueError("NOISE_MEAN must be > 0 for the log-normal noise model.")
    s_sq = np.log(1.0 + noise_var / noise_mean**2)
    s    = np.sqrt(s_sq)
    m    = np.log(noise_mean) - 0.5 * s_sq
    noise_lognorm = rng.lognormal(mean=m, sigma=s, size=len(dm))

    return {
        "DM":          dm,
        "DM_gaussian": dm + noise_gauss,
        "DM_lognormal": dm + noise_lognorm,
    }


# ---------------------------------------------------------------------------
# FITS helpers
# ---------------------------------------------------------------------------

def _completed_realisations(fits_path):
    """Return the set of realisation indices already stored in *fits_path*."""
    if not os.path.exists(fits_path):
        return set()
    with fits.open(fits_path) as hdul:
        done = set()
        for hdu in hdul:
            name = hdu.name
            if name.startswith("REAL_"):
                try:
                    done.add(int(name.split("_")[1]))
                except ValueError:
                    pass
    return done


def _init_fits(fits_path, ra, dec, cl_full):
    """Create a new FITS file with a primary HDU, CATALOG, and CELL extensions."""
    os.makedirs(os.path.dirname(os.path.abspath(fits_path)), exist_ok=True)

    # Positions table
    cat_tab = Table()
    cat_tab["RA"]  = np.asarray(ra,  dtype=np.float64)
    cat_tab["Dec"] = np.asarray(dec, dtype=np.float64)
    cat_tab["RA"].unit  = "deg"
    cat_tab["Dec"].unit = "deg"
    cat_hdu = fits.BinTableHDU(cat_tab, name="CATALOG")
    cat_hdu.header["N_SRC"]    = len(ra)
    cat_hdu.header["NSIDE"]    = NSIDE

    # Interpolated C_ell
    cl_tab = Table()
    cl_tab["ell"]  = np.arange(len(cl_full), dtype=np.int32)
    cl_tab["cell"] = np.asarray(cl_full, dtype=np.float64)
    cl_hdu = fits.BinTableHDU(cl_tab, name="CELL")
    cl_hdu.header["NSIDE"]   = NSIDE
    cl_hdu.header["LMAX"]    = len(cl_full) - 1

    primary = fits.PrimaryHDU()
    primary.header["NREAL"]    = 0
    primary.header["NSIDE"]    = NSIDE
    primary.header["NOMEAN"]   = NOISE_MEAN
    primary.header["NOVAR"]    = NOISE_VAR
    primary.header["SEEDBASE"] = SEED_START

    hdul = fits.HDUList([primary, cat_hdu, cl_hdu])
    hdul.writeto(fits_path, overwrite=True)


def _append_realisation(fits_path, idx, data):
    """Append one realisation as a new BinTable extension and update NREAL."""
    real_tab = Table()
    real_tab["DM"]          = np.asarray(data["DM"],          dtype=np.float64)
    real_tab["DM_gaussian"] = np.asarray(data["DM_gaussian"], dtype=np.float64)
    real_tab["DM_lognormal"]= np.asarray(data["DM_lognormal"],dtype=np.float64)
    for col in real_tab.colnames:
        real_tab[col].unit = "pc / cm3"

    real_hdu = fits.BinTableHDU(real_tab, name=f"REAL_{idx:05d}")
    real_hdu.header["SEED"] = SEED_START + idx
    real_hdu.header["IDX"]  = idx

    with fits.open(fits_path, mode="append") as hdul:
        hdul.append(real_hdu)
        # Update NREAL in primary header
        hdul[0].header["NREAL"] = hdul[0].header.get("NREAL", 0) + 1
        hdul.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"=== generate_mock_catalogues.py ===")
    print(f"  CL_PATH:        {CL_PATH}")
    print(f"  POSITIONS_PATH: {POSITIONS_PATH}")
    print(f"  OUTPUT_PATH:    {OUTPUT_PATH}")
    print(f"  NSIDE:          {NSIDE}")
    print(f"  N_REALISATIONS: {N_REALISATIONS}")
    print(f"  NOISE_MEAN:     {NOISE_MEAN}")
    print(f"  NOISE_VAR:      {NOISE_VAR}")
    print(f"  SEED_START:     {SEED_START}")
    print(f"  RESUME:         {RESUME}")

    # Build C_ell
    print("\nBuilding interpolated C_ell...")
    cl_full, lmax_syn = build_cl_full(CL_PATH, NSIDE, LMAX_FACTOR)
    print(f"  lmax_syn = {lmax_syn}, cl_full shape = {cl_full.shape}")

    # Load positions
    print("Loading source positions...")
    ra, dec = load_positions(POSITIONS_PATH)
    print(f"  {len(ra)} sources loaded")

    # Determine which realisations still need to be done
    completed = _completed_realisations(OUTPUT_PATH) if RESUME else set()
    todo = [i for i in range(N_REALISATIONS) if i not in completed]

    if not todo:
        print("All realisations already completed.")
        return

    # Initialise FITS file if it does not exist yet (or if not resuming)
    if not os.path.exists(OUTPUT_PATH) or not RESUME:
        print(f"Initialising output FITS: {OUTPUT_PATH}")
        _init_fits(OUTPUT_PATH, ra, dec, cl_full)

    print(f"\nGenerating {len(todo)} realisation(s) "
          f"({len(completed)} already done, {N_REALISATIONS} total)...")

    for count, idx in enumerate(todo):
        seed = SEED_START + idx
        data = generate_realisation(cl_full, lmax_syn, ra, dec, NOISE_MEAN, NOISE_VAR, seed)
        _append_realisation(OUTPUT_PATH, idx, data)
        print(f"  [{count+1}/{len(todo)}] realisation {idx:5d} done  "
              f"(DM mean = {np.nanmean(data['DM']):.4e})", flush=True)

    print(f"\nFinished. Output written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
