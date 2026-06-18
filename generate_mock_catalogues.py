"""generate_mock_catalogues.py

Generate many realisations of a mock FRB catalogue from an input angular power
spectrum and write them incrementally to a FITS file.

Each realisation gets its own FITS binary-table extension named REAL_NNNNN so
the file is fully self-contained and grows by one extension per completed
realisation.  A fixed CATALOG extension stores the input sky positions and a
CELL extension stores the interpolated C_ell array that was passed to synfast.
If enabled, each realisation also gets a PCL_NNNNN extension with a binned
pseudo-C_ell measurement derived from the sampled catalogue values.

Configuration via environment variables (all optional, sane defaults given):
    CL_PATH          - path to the .npz file with 'ell' and 'cell' keys
                       [default: data/cell_chime.npz  relative to this script]
    POSITIONS_PATH   - path to text file with source positions (RA, Dec)
                       [default: data/chimefrbcat2_radec.txt]
    OUTPUT_PATH      - base path to write the output FITS file; the script
                       appends a suffix describing the position mode
                       (catalog, masked_fixed, masked_resampled)
                       [default: output/mock_catalogues.fits]
    NSIDE            - HEALPix NSIDE for synfast  [default: 4096]
    N_REALISATIONS   - number of realisations to generate  [default: 100]
    NOISE_MEAN       - mean of additive noise  [default: 50.0]
    NOISE_VAR        - variance (sigma^2) of additive noise  [default: 2500.0]
    SEED_START       - base random seed; realisation i uses SEED_START + i
                       [default: 0]
    LMAX_FACTOR      - lmax = min(max_input_ell, LMAX_FACTOR * NSIDE - 1)
                       [default: 3]
    MEASURE_PCL      - if 'true', measure a pseudo-C_ell with NaMaster
                       [default: true]
    PCL_LMIN         - minimum multipole used for pseudo-C_ell binning
                       [default: 40]
    PCL_LMAX         - maximum multipole used for pseudo-C_ell binning
                       [default: auto (uses lmax_syn and 3*NSIDE-1)]
    PCL_NBINS        - number of geometric ell bins for the pseudo-C_ell
                       [default: 9]
    USE_HEALPIX_MAP  - if 'true', build a HEALPix map and interpolate from it
                       [default: false]
    POSITION_SOURCE  - 'catalog' or 'mask' position generation
                       [default: catalog]
    POSITION_MASK_PATH - path to non-binary probability mask FITS map
                       [default: output/chime/detection_probability_mask_nside128.fits]
    POSITION_MASK_MODE - when POSITION_SOURCE='mask':
                       'fixed' (sample once, reuse all realisations) or
                       'resample' (new sampled positions per realisation)
                       [default: fixed]
    N_SOURCES        - number of positions to sample when POSITION_SOURCE='mask';
                       if <=0, use the size of POSITIONS_PATH catalogue
                       [default: 0]
    WRITE_EVERY      - flush FITS updates to disk every N realisations
                       [default: 100]
    RESUME           - if 'true', skip already-completed realisations found in
                       an existing output file  [default: true]
"""

import os
import sys
import numpy as np
import healpy as hp
from astropy.table import Table
from astropy.io import fits
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import sph_harm
from scipy.integrate import simpson
from scipy.spatial import cKDTree


def _with_position_suffix(output_path, position_source, position_mask_mode):
    """Append a descriptive position-source suffix to the output FITS filename."""
    if output_path.endswith(".fits.gz"):
        stem = output_path[:-8]
        ext = ".fits.gz"
    else:
        stem, ext = os.path.splitext(output_path)

    if position_source == "catalog":
        suffix = "_catalog"
    elif position_source == "mask":
        if position_mask_mode == "fixed":
            suffix = "_masked_fixed"
        elif position_mask_mode == "resample":
            suffix = "_masked_resampled"
        else:
            suffix = f"_masked_{position_mask_mode}"
    else:
        suffix = f"_{position_source}"

    if stem.endswith(suffix):
        return output_path
    return f"{stem}{suffix}{ext}"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

CL_PATH        = os.getenv("CL_PATH",        os.path.join(_HERE, "data", "cell_chime.npz"))
POSITIONS_PATH = os.getenv("POSITIONS_PATH", os.path.join(_HERE, "data", "chimefrbcat2_radec.txt"))
_OUTPUT_PATH_RAW = os.getenv("OUTPUT_PATH",  os.path.join(_HERE, "output/forecast", "mock_catalogues.fits"))
NSIDE          = int(os.getenv("NSIDE",           "256"))
N_REALISATIONS = int(os.getenv("N_REALISATIONS",  "10000"))
NOISE_MEAN     = float(os.getenv("NOISE_MEAN",    "50.0"))
NOISE_VAR      = float(os.getenv("NOISE_VAR",     "2500.0"))
SEED_START     = int(os.getenv("SEED_START",      "0"))
LMAX_FACTOR    = int(os.getenv("LMAX_FACTOR",     "3"))
MEASURE_PCL    = os.getenv("MEASURE_PCL", "true").strip().lower() in ("1", "true", "yes")
PCL_LMIN       = int(os.getenv("PCL_LMIN",       "2"))
_PCL_LMAX_RAW  = os.getenv("PCL_LMAX", "").strip()
PCL_LMAX       = int(_PCL_LMAX_RAW) if _PCL_LMAX_RAW else None
PCL_NBINS      = int(os.getenv("PCL_NBINS",      "12"))
RESUME         = os.getenv("RESUME", "false").strip().lower() in ("1", "true", "yes")
USE_HEALPIX_MAP = os.getenv("USE_HEALPIX_MAP", "true").strip().lower() in ("1", "true", "yes")
WRITE_EVERY    = int(os.getenv("WRITE_EVERY",    "500"))
POSITION_SOURCE = os.getenv("POSITION_SOURCE", "mask").strip().lower()
POSITION_MASK_PATH = os.getenv(
    "POSITION_MASK_PATH",
    os.path.join(_HERE, "output", "chime", "detection_probability_mask_nside128.fits"),
)
POSITION_MASK_MODE = os.getenv("POSITION_MASK_MODE", "fixed").strip().lower()
N_SOURCES = int(os.getenv("N_SOURCES", "50000"))
OUTPUT_PATH = _with_position_suffix(_OUTPUT_PATH_RAW, POSITION_SOURCE, POSITION_MASK_MODE)


def next_power_of_two(value):
    """Return the smallest power of two greater than or equal to value."""
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def load_probability_mask(mask_path):
    """Load and sanitize a non-binary probability mask map."""
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    mask_map = hp.read_map(mask_path, verbose=False)
    mask_map = np.asarray(mask_map, dtype=float)
    if mask_map.ndim > 1:
        mask_map = mask_map[0]

    mask_map = np.where(np.isfinite(mask_map), mask_map, 0.0)
    mask_map = np.clip(mask_map, 0.0, None)
    max_val = np.max(mask_map)
    if max_val <= 0.0:
        raise ValueError("Probability mask has no positive entries.")
    return mask_map / max_val


def sample_positions_from_probability_mask(prob_mask, n_samples, rng):
    """Sample RA/Dec positions from a probability mask."""
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    npix = len(prob_mask)
    pix = np.arange(npix, dtype=np.int64)
    weights = np.asarray(prob_mask, dtype=float)
    weight_sum = weights.sum()
    if weight_sum <= 0.0:
        raise ValueError("Probability mask sum is not positive.")
    probs = weights / weight_sum

    chosen_pix = rng.choice(pix, size=int(n_samples), p=probs)
    theta, phi = hp.pix2ang(hp.npix2nside(npix), chosen_pix)
    ra = np.degrees(phi)
    dec = 90.0 - np.degrees(theta)
    return ra.astype(float), dec.astype(float)

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
    ra = np.asarray(tab[ra_col], dtype=float)
    dec = np.asarray(tab[dec_col], dtype=float)

    # Drop rows with invalid positions before returning coordinates.
    valid_pos = np.isfinite(ra) & np.isfinite(dec)
    n_bad = int((~valid_pos).sum())
    if n_bad > 0:
        print(f"Removed {n_bad} rows with invalid RA/Dec (NaN or inf) from input catalogue.")
        ra = ra[valid_pos]
        dec = dec[valid_pos]

    return ra, dec


def load_positions_from_output(fits_path):
    """Return (ra, dec) arrays in degrees from the saved CATALOG extension."""
    with fits.open(fits_path) as hdul:
        if "CATALOG" not in hdul:
            raise ValueError(f"Existing output file {fits_path} has no CATALOG extension")
        tab = Table(hdul["CATALOG"].data)

    lower_map = {c.lower(): c for c in tab.colnames}
    ra_col = _pick_col(lower_map, ["ra", "ra_deg"])
    dec_col = _pick_col(lower_map, ["dec", "dec_deg"])
    if ra_col is None or dec_col is None:
        raise ValueError(
            f"Existing output file {fits_path} CATALOG extension is missing RA/Dec columns. "
            f"Available columns: {tab.colnames}"
        )

    ra = np.asarray(tab[ra_col], dtype=float)
    dec = np.asarray(tab[dec_col], dtype=float)

    valid_pos = np.isfinite(ra) & np.isfinite(dec)
    n_bad = int((~valid_pos).sum())
    if n_bad > 0:
        print(f"Removed {n_bad} rows with invalid RA/Dec (NaN or inf) from saved output catalogue.")
        ra = ra[valid_pos]
        dec = dec[valid_pos]

    return ra, dec


def compute_mean_interparticle_distance(ra, dec):
    """Return the mean nearest-neighbor separation on the sphere in radians."""
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)

    valid = np.isfinite(ra) & np.isfinite(dec)
    ra = ra[valid]
    dec = dec[valid]

    n_obj = ra.size
    if n_obj < 2:
        return np.nan

    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    cos_dec = np.cos(dec_rad)
    xyz = np.column_stack([
        cos_dec * np.cos(ra_rad),
        cos_dec * np.sin(ra_rad),
        np.sin(dec_rad),
    ])

    tree = cKDTree(xyz)
    distances, _ = tree.query(xyz, k=2)
    nearest_chord = np.clip(distances[:, 1], 0.0, 2.0)
    nearest_angle = 2.0 * np.arcsin(0.5 * nearest_chord)
    return float(np.mean(nearest_angle))


def derive_resolution_from_mean_separation(mean_sep_rad, nside_min=1):
    """Convert mean separation into a characteristic ell_max and NSIDE.

    Uses the standard rule-of-thumb ell ~ pi / theta and enforces
    lmax <= 3 * nside - 1 via the smallest power-of-two NSIDE satisfying it.
    """
    if not np.isfinite(mean_sep_rad) or mean_sep_rad <= 0.0:
        raise ValueError("mean_sep_rad must be finite and positive")

    ell_max = max(2, int(np.floor(np.pi / mean_sep_rad)))
    nside_required = next_power_of_two(int(np.ceil((ell_max + 1) / 1.0)))
    return ell_max, max(int(nside_min), nside_required)


def get_positions_for_realisation(idx, base_ra, base_dec, position_source, position_mask_mode, prob_mask, n_samples):
    """Return the RA/Dec positions to use for a given realisation index."""
    if position_source != "mask":
        return base_ra, base_dec

    if position_mask_mode == "fixed":
        return base_ra, base_dec

    if position_mask_mode == "resample":
        rng_pos = np.random.default_rng(SEED_START + 1000000 + int(idx))
        return sample_positions_from_probability_mask(prob_mask, n_samples=n_samples, rng=rng_pos)

    raise ValueError(f"Unknown POSITION_MASK_MODE: {position_mask_mode}")


# ---------------------------------------------------------------------------
# Helper: build interpolated C_ell array for hp.synfast
# ---------------------------------------------------------------------------

def build_cl_full(cl_path, nside, lmax_factor, ell_max_cap=None):
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
    if ell_max_cap is not None:
        lmax_syn = min(lmax_syn, int(ell_max_cap))
    ells_full = np.arange(lmax_syn + 1, dtype=float)
    cl_full = np.zeros(lmax_syn + 1, dtype=float)

    mask_eval = (ells_full >= max(2, unique_ell.min())) & (ells_full <= unique_ell.max())
    cl_full[mask_eval] = np.exp(cl_spline(np.log(ells_full[mask_eval])))
    cl_full[0] = 0.0
    if lmax_syn >= 1:
        cl_full[1] = 0.0
    cl_full = np.maximum(cl_full, 0.0)

    return cl_full, lmax_syn, cl_spline


def evaluate_alm_at_positions(alm, lmax, theta, phi, chunk_size=256):
    """Evaluate a scalar field from packed alm coefficients at arbitrary positions.

    The field is computed directly from the spherical-harmonic expansion,
    avoiding an intermediate HEALPix map.
    """
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    if theta.shape != phi.shape:
        raise ValueError("theta and phi must have the same shape")

    values = np.zeros(theta.size, dtype=np.float64)
    if theta.size == 0:
        return values

    for start in range(0, theta.size, int(chunk_size)):
        stop = min(start + int(chunk_size), theta.size)
        theta_chunk = theta[start:stop]
        phi_chunk = phi[start:stop]
        chunk_values = np.zeros(theta_chunk.size, dtype=np.float64)

        for ell in range(lmax + 1):
            idx_0 = hp.Alm.getidx(lmax, ell, 0)
            chunk_values += np.real(alm[idx_0] * sph_harm(0, ell, phi_chunk, theta_chunk))

            for m in range(1, ell + 1):
                idx = hp.Alm.getidx(lmax, ell, m)
                y_lm = sph_harm(m, ell, phi_chunk, theta_chunk)
                chunk_values += 2.0 * np.real(alm[idx] * y_lm)

        values[start:stop] = chunk_values

    return values


def evaluate_map_at_positions(map_real, theta, phi):
    """Evaluate a HEALPix map at arbitrary positions with interpolation."""
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    return hp.get_interp_val(map_real, theta, phi)


def _get_pymaster():
    try:
        import pymaster as nmt
    except ImportError as exc:
        raise ImportError(
            "MEASURE_PCL is enabled but pymaster is not installed. "
            "Install NaMaster/pymaster or set MEASURE_PCL=false."
        ) from exc
    return nmt


def build_pcl_binner(nside, lmax_pcl, lmin, nbins, lmax_user=None):
    """Build the geometric ell binning used for the pseudo-C_ell measurement."""
    nmt = _get_pymaster()

    lmin = max(2, int(lmin))
    if nbins < 1:
        raise ValueError("PCL_NBINS must be >= 1.")

    lmax_cap = min(int(lmax_pcl), 3 * int(nside) - 1)
    if lmax_user is not None:
        if int(lmax_user) < 2:
            raise ValueError("PCL_LMAX must be >= 2 when provided.")
        lmax_cap = min(lmax_cap, int(lmax_user))

    ell_stop = lmax_cap + 1
    if ell_stop <= lmin:
        raise ValueError(
            f"Pseudo-C_ell binning is invalid: lmin={lmin} but ell_stop={ell_stop}."
        )

    raw_edges = np.geomspace(lmin, ell_stop, nbins + 1)
    edges = np.rint(raw_edges).astype(int)
    edges[0] = lmin
    edges[-1] = ell_stop
    edges = np.unique(edges)
    if len(edges) < 2:
        raise ValueError("Pseudo-C_ell binning collapsed to fewer than one bandpower.")

    if POSITION_SOURCE == "mask":
        np.save("./output/forecast/pcl_edges_chime_5000.npy", edges)
    else:
        np.save("./output/chime/pcl_edges_chime.npy", edges)

    binner = nmt.NmtBin.from_edges(edges[:-1], edges[1:])
    leff = np.asarray(binner.get_effective_ells(), dtype=float)
    return binner, leff, edges


def measure_pseudo_cl(ra, dec, data_columns, lmax_pcl, binner):
    """Measure binned pseudo-C_ell values from catalogue samples using NaMaster."""
    nmt = _get_pymaster()

    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    ra_wrapped = np.where(ra < 0.0, ra + 360.0, ra)

    pseudo_cls = {}
    for name, values in data_columns.items():
        values = np.asarray(values, dtype=float)
        valid = np.isfinite(ra_wrapped) & np.isfinite(dec) & np.isfinite(values)
        if not np.any(valid):
            pseudo_cls[name] = np.full(len(binner.get_effective_ells()), np.nan, dtype=float)
            continue

        pos_data = np.vstack([ra_wrapped[valid], dec[valid]])
        weights = np.ones(valid.sum(), dtype=float)
        field_values = values[valid][np.newaxis, :]
        field = nmt.NmtFieldCatalog(pos_data, weights, field_values- np.mean(field_values), lmax=lmax_pcl, lonlat=True)
        workspace = nmt.NmtWorkspace.from_fields(field, field, binner)
        coupled = nmt.compute_coupled_cell(field, field)
        pseudo_cls[name] = np.asarray(workspace.decouple_cell(coupled)[0], dtype=float)

    return pseudo_cls


# ---------------------------------------------------------------------------
# Helper: generate one catalogue realisation
# ---------------------------------------------------------------------------

def generate_realisation(cl_full, lmax_syn, nside_sim, ra, dec, noise_mean, noise_var, seed, use_healpix_map=False, theory_var=None):
    """
    Draw a new alm realisation, evaluate the field directly at the catalogue
    positions, and add noise.

    Returns a dict with keys: DM, DM_gaussian, DM_lognormal.
    """
    rng = np.random.default_rng(seed)

    # New alm realisation from the same C_ell
    np.random.seed(int(rng.integers(0, 2**31)))
    alm = hp.synalm(cl_full, lmax=lmax_syn, new=True)

    # Evaluate the field either directly from alm or via a HEALPix map
    ra_hp  = np.where(ra < 0, ra + 360.0, ra)
    theta  = np.clip(np.radians(90.0 - dec), 1e-6, np.pi - 1e-6)
    phi    = np.radians(ra_hp)

    valid  = np.isfinite(theta) & np.isfinite(phi)
    dm     = np.full(len(theta), np.nan)
    if use_healpix_map:
        map_real = hp.alm2map(alm, nside=nside_sim, lmax=lmax_syn)
        dm[valid] = evaluate_map_at_positions(map_real, theta[valid], phi[valid])
    else:
        dm[valid] = evaluate_alm_at_positions(alm, lmax_syn, theta[valid], phi[valid])
    if theory_var is not None:
        diff_var = theory_var - np.var(dm)

    # Additive Gaussian noise
    dm += rng.normal(loc=0, scale=np.sqrt(diff_var), size=len(dm))
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
    """Return the sets of realisation and pseudo-C_ell indices stored in *fits_path*."""
    if not os.path.exists(fits_path):
        return set(), set()
    with fits.open(fits_path) as hdul:
        real_done = set()
        pcl_done = set()
        for hdu in hdul:
            name = hdu.name
            if name.startswith("REAL_"):
                try:
                    real_done.add(int(name.split("_")[1]))
                except ValueError:
                    pass
            elif name.startswith("PCL_"):
                try:
                    pcl_done.add(int(name.split("_")[1]))
                except ValueError:
                    pass
    return real_done, pcl_done


def _init_fits(
    fits_path,
    ra,
    dec,
    cl_full,
    nside_sim,
    mean_sep_rad,
    ell_max_catalog,
    pcl_edges=None,
    position_source="catalog",
    position_mask_mode="fixed",
    position_mask_path="",
):
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
    cat_hdu.header["NSIDE"]    = nside_sim
    cat_hdu.header["MEANSEPR"] = float(mean_sep_rad)
    cat_hdu.header["ELLCATA"]  = int(ell_max_catalog)

    # Interpolated C_ell
    cl_tab = Table()
    cl_tab["ell"]  = np.arange(len(cl_full), dtype=np.int32)
    cl_tab["cell"] = np.asarray(cl_full, dtype=np.float64)
    cl_hdu = fits.BinTableHDU(cl_tab, name="CELL")
    cl_hdu.header["NSIDE"]   = nside_sim
    cl_hdu.header["LMAX"]    = len(cl_full) - 1
    cl_hdu.header["ELLCATA"] = int(ell_max_catalog)

    primary = fits.PrimaryHDU()
    primary.header["NREAL"]    = 0
    primary.header["NSIDE"]    = nside_sim
    primary.header["NOMEAN"]   = NOISE_MEAN
    primary.header["NOVAR"]    = NOISE_VAR
    primary.header["SEEDBASE"] = SEED_START
    primary.header["MEASPCL"]  = int(MEASURE_PCL)
    primary.header["NPCL"]     = 0
    primary.header["MEANSEPR"] = float(mean_sep_rad)
    primary.header["ELLCATA"]  = int(ell_max_catalog)
    primary.header["POSSRC"]   = str(position_source)[:16]
    primary.header["POSMODE"]  = str(position_mask_mode)[:16]
    if position_mask_path:
        primary.header["POSPATH"] = str(position_mask_path)[:68]
    if pcl_edges is not None:
        primary.header["PCLLMIN"] = int(pcl_edges[0])
        primary.header["PCLNBIN"] = len(pcl_edges) - 1
        primary.header["PCLLMAX"] = int(pcl_edges[-1] - 1)

    hdul = fits.HDUList([primary, cat_hdu, cl_hdu])
    hdul.writeto(fits_path, overwrite=True)


def _append_realisation(hdul, idx, data):
    """Append one realisation to an open FITS file and update NREAL."""
    real_tab = Table()
    real_tab["DM"]          = np.asarray(data["DM"],          dtype=np.float64)
    real_tab["DM_gaussian"] = np.asarray(data["DM_gaussian"], dtype=np.float64)
    real_tab["DM_lognormal"]= np.asarray(data["DM_lognormal"],dtype=np.float64)
    for col in real_tab.colnames:
        real_tab[col].unit = "pc / cm3"

    real_hdu = fits.BinTableHDU(real_tab, name=f"REAL_{idx:05d}")
    real_hdu.header["SEED"] = SEED_START + idx
    real_hdu.header["IDX"]  = idx

    hdul.append(real_hdu)
    hdul[0].header["NREAL"] = hdul[0].header.get("NREAL", 0) + 1


def _append_pseudo_cl(hdul, idx, leff, pseudo_cls, pcl_edges, lmax_pcl):
    """Append the binned pseudo-C_ell measurement to an open FITS file."""
    pcl_tab = Table()
    pcl_tab["ell_eff"] = np.asarray(leff, dtype=np.float64)
    pcl_tab["pcl_dm"] = np.asarray(pseudo_cls["DM"], dtype=np.float64)
    pcl_tab["pcl_dm_gaussian"] = np.asarray(pseudo_cls["DM_gaussian"], dtype=np.float64)
    pcl_tab["pcl_dm_lognormal"] = np.asarray(pseudo_cls["DM_lognormal"], dtype=np.float64)
    pcl_tab["ell_eff"].unit = "1"
    for col in ("pcl_dm", "pcl_dm_gaussian", "pcl_dm_lognormal"):
        pcl_tab[col].unit = "pc2 / cm6"

    pcl_hdu = fits.BinTableHDU(pcl_tab, name=f"PCL_{idx:05d}")
    pcl_hdu.header["SEED"] = SEED_START + idx
    pcl_hdu.header["IDX"] = idx
    pcl_hdu.header["LMIN"] = int(pcl_edges[0])
    pcl_hdu.header["LMAX"] = int(lmax_pcl)
    pcl_hdu.header["NBIN"] = len(leff)

    hdul.append(pcl_hdu)
    hdul[0].header["NPCL"] = hdul[0].header.get("NPCL", 0) + 1


def _load_realisation(fits_path, idx):
    """Load one stored realisation from a REAL_NNNNN extension."""
    with fits.open(fits_path) as hdul:
        ext_name = f"REAL_{idx:05d}"
        if ext_name not in hdul:
            raise ValueError(f"Missing expected extension {ext_name} in {fits_path}")
        tab = Table(hdul[ext_name].data)

    return {
        "DM": np.asarray(tab["DM"], dtype=float),
        "DM_gaussian": np.asarray(tab["DM_gaussian"], dtype=float),
        "DM_lognormal": np.asarray(tab["DM_lognormal"], dtype=float),
    }


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
    print(f"  MEASURE_PCL:    {MEASURE_PCL}")
    if MEASURE_PCL:
        print(f"  PCL_LMIN:       {PCL_LMIN}")
        print(f"  PCL_LMAX:       {PCL_LMAX if PCL_LMAX is not None else 'auto'}")
        print(f"  PCL_NBINS:      {PCL_NBINS}")
    print(f"  RESUME:         {RESUME}")
    print(f"  USE_HEALPIX_MAP: {USE_HEALPIX_MAP}")
    print(f"  WRITE_EVERY:    {WRITE_EVERY}")
    print(f"  POSITION_SOURCE: {POSITION_SOURCE}")
    if POSITION_SOURCE == "mask":
        print(f"  POSITION_MASK_PATH: {POSITION_MASK_PATH}")
        print(f"  POSITION_MASK_MODE: {POSITION_MASK_MODE}")
        print(f"  N_SOURCES: {N_SOURCES if N_SOURCES > 0 else 'auto(from catalog)'}")

    # Load positions
    print("Loading source positions...")
    if POSITION_SOURCE not in {"catalog", "mask"}:
        raise ValueError("POSITION_SOURCE must be either 'catalog' or 'mask'")
    if POSITION_MASK_MODE not in {"fixed", "resample"}:
        raise ValueError("POSITION_MASK_MODE must be either 'fixed' or 'resample'")

    prob_mask = None
    if POSITION_SOURCE == "catalog":
        if os.path.exists(OUTPUT_PATH) and RESUME:
            ra, dec = load_positions_from_output(OUTPUT_PATH)
            print(f"  {len(ra)} sources loaded from existing output catalogue")
        else:
            ra, dec = np.loadtxt(POSITIONS_PATH, usecols=(0, 1), unpack=True)
            print(f"  {len(ra)} sources loaded from input catalogue")
    else:
        prob_mask = load_probability_mask(POSITION_MASK_PATH)
        if N_SOURCES > 0:
            n_sources = int(N_SOURCES)
        else:
            # Use input catalogue size as default target number of sampled sources.
            ra_ref, dec_ref = np.loadtxt(POSITIONS_PATH, usecols=(0, 1), unpack=True)
            n_sources = len(ra_ref)
        if n_sources < 1:
            raise ValueError("Number of mask-sampled sources must be >= 1")

        if os.path.exists(OUTPUT_PATH) and RESUME and POSITION_MASK_MODE == "fixed":
            ra, dec = load_positions_from_output(OUTPUT_PATH)
            print(f"  {len(ra)} fixed mask-sampled sources loaded from existing output catalogue")
        else:
            rng_initial = np.random.default_rng(SEED_START + 999999)
            ra, dec = sample_positions_from_probability_mask(prob_mask, n_sources, rng_initial)
            print(f"  {len(ra)} sources sampled from probability mask")

    mean_sep_rad = compute_mean_interparticle_distance(ra, dec)
    if np.isfinite(mean_sep_rad):
        mean_sep_deg = np.degrees(mean_sep_rad)
        print(f"  mean inter-particle separation = {mean_sep_deg:.6f} deg")
        print(f"                                 = {mean_sep_deg * 60.0:.3f} arcmin")
    else:
        print("  mean inter-particle separation = undefined (need at least 2 valid positions)")

    ell_max_catalog, nside_sim = derive_resolution_from_mean_separation(mean_sep_rad, nside_min=NSIDE)
    print(f"  derived catalog ell_max       = {ell_max_catalog}")
    print(f"  simulation NSIDE in use       = {nside_sim}")

    # Build C_ell using the catalog-derived angular resolution.
    print("\nBuilding interpolated C_ell...")
    cl_full, lmax_syn, cl_spline = build_cl_full(CL_PATH, nside_sim, LMAX_FACTOR, ell_max_cap=ell_max_catalog)
    ell_int = np.geomspace(2, int(1e5), num=1000)
    cl_int = np.exp(cl_spline(np.log(ell_int)))
    theory_var = simpson(cl_int * ell_int / (2 * np.pi), x=ell_int)

    print(f"  lmax_syn = {lmax_syn}, cl_full shape = {cl_full.shape}")

    pcl_binner = None
    pcl_leff = None
    pcl_edges = None
    if MEASURE_PCL:
        print("Building pseudo-C_ell binning...")
        pcl_binner, pcl_leff, pcl_edges = build_pcl_binner(
            nside_sim, lmax_syn, PCL_LMIN, PCL_NBINS, lmax_user=PCL_LMAX
        )
        print(f"  pseudo-C_ell bandpowers = {len(pcl_leff)}")

    
    # Determine which realisations still need to be done
    real_done, pcl_done = _completed_realisations(OUTPUT_PATH) if RESUME else (set(), set())
    todo_real = [i for i in range(N_REALISATIONS) if i not in real_done]
    todo_pcl = [i for i in range(N_REALISATIONS) if MEASURE_PCL and i in real_done and i not in pcl_done]

    if not todo_real and not todo_pcl:
        print("All realisations already completed.")
        return

    # Initialise FITS file if it does not exist yet (or if not resuming)
    if not os.path.exists(OUTPUT_PATH) or not RESUME:
        print(f"Initialising output FITS: {OUTPUT_PATH}")
        _init_fits(
            OUTPUT_PATH,
            ra,
            dec,
            cl_full,
            nside_sim,
            mean_sep_rad,
            ell_max_catalog,
            pcl_edges=pcl_edges,
            position_source=POSITION_SOURCE,
            position_mask_mode=POSITION_MASK_MODE,
            position_mask_path=POSITION_MASK_PATH if POSITION_SOURCE == "mask" else "",
        )

    if todo_pcl:
        print(f"\nBackfilling pseudo-C_ell for {len(todo_pcl)} completed realisation(s)...")
        with fits.open(OUTPUT_PATH, mode="append") as hdul:
            for count, idx in enumerate(todo_pcl):
                data = _load_realisation(OUTPUT_PATH, idx)
                ra_i, dec_i = get_positions_for_realisation(
                    idx,
                    ra,
                    dec,
                    POSITION_SOURCE,
                    POSITION_MASK_MODE,
                    prob_mask,
                    len(ra),
                )
                pseudo_cls = measure_pseudo_cl(ra_i, dec_i, data, lmax_syn, pcl_binner)
                _append_pseudo_cl(hdul, idx, pcl_leff, pseudo_cls, pcl_edges, lmax_syn)
                should_flush = ((count + 1) % WRITE_EVERY == 0) or (count + 1 == len(todo_pcl))
                if should_flush:
                    hdul.flush()
                print(f"  [{count+1}/{len(todo_pcl)}] pseudo-C_ell {idx:5d} done", flush=True)

    print(f"\nGenerating {len(todo_real)} realisation(s) "
          f"({len(real_done)} already done, {N_REALISATIONS} total)...")

    with fits.open(OUTPUT_PATH, mode="append") as hdul:
        for count, idx in enumerate(todo_real):
            seed = SEED_START + idx
            ra_i, dec_i = get_positions_for_realisation(
                idx,
                ra,
                dec,
                POSITION_SOURCE,
                POSITION_MASK_MODE,
                prob_mask,
                len(ra),
            )
            data = generate_realisation(
                cl_full,
                lmax_syn,
                nside_sim,
                ra_i,
                dec_i,
                NOISE_MEAN,
                NOISE_VAR,
                seed,
                use_healpix_map=USE_HEALPIX_MAP,
                theory_var=theory_var,
            )
            _append_realisation(hdul, idx, data)
            if MEASURE_PCL:
                pseudo_cls = measure_pseudo_cl(ra_i, dec_i, data, lmax_syn, pcl_binner)
                _append_pseudo_cl(hdul, idx, pcl_leff, pseudo_cls, pcl_edges, lmax_syn)

            should_flush = ((count + 1) % WRITE_EVERY == 0) or (count + 1 == len(todo_real))
            if should_flush:
                hdul.flush()

            print(f"  [{count+1}/{len(todo_real)}] realisation {idx:5d} done  "
                  f"(DM mean = {np.nanmean(data['DM']):.4e})", flush=True)

    print(f"\nFinished. Output written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
