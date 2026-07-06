"""generate_mock_catalogues.py

Configuration via environment variables (all optional, sane defaults given):
    CL_PATH          - path to the .npz file with 'ell' and 'cell' keys
                       [default: data/cell_chime_multipoles.npz  relative to this script]
    POSITIONS_PATH   - path to text file with source positions (RA, Dec)
                       [default: data/chimefrbcat2_radec.txt]
    OUTPUT_PATH      - base path to write the output NPZ file; the script
                       appends a suffix describing the position mode
                       (catalog, masked_fixed, masked_resampled)
                       [default: output/mock.npz]
    NSIDE            - HEALPix NSIDE for synfast  [default: 256]
    N_REALISATIONS   - number of realisations to generate  [default: 10000]
    NOISE_MEAN       - mean of additive noise  [default: 50.0]
    NOISE_VAR        - variance (sigma^2) of additive noise  [default: 2500.0]
    SEED_START       - base random seed; realisation i uses SEED_START + i
                       [default: 0]
    LMAX_FACTOR      - lmax = min(max_input_ell, LMAX_FACTOR * NSIDE - 1)
                       [default: 3]
    MEASURE_PCL      - if 'true', measure a pseudo-C_ell with NaMaster
                       [default: true]
    PCL_LMIN         - minimum multipole used for pseudo-C_ell binning
                       [default: 2]
    PCL_LMAX         - maximum multipole used for pseudo-C_ell binning
                       [default: auto (uses lmax_syn and 3*NSIDE-1)]
    PCL_NBINS        - number of geometric ell bins for the pseudo-C_ell
                       [default: 12]
    PCL_COUPLED      - if 'true', store the coupled (per-multipole) pseudo-C_ell
                       instead of the decoupled bandpower estimates
                       [default: false]
    POSITION_SOURCE  - 'catalog' or 'mask' position generation
                       [default: mask]
    POSITION_MASK_PATH - path to non-binary probability mask FITS map
                       [default: output/chime/detection_probability_mask_nside128.fits]
    POSITION_MASK_MODE - when POSITION_SOURCE='mask':
                       'fixed' (sample once, reuse all realisations) or
                       'resample' (new sampled positions per realisation)
                       [default: fixed]
    N_SOURCES        - number of positions to sample when POSITION_SOURCE='mask';
                       if <=0, use the size of POSITIONS_PATH catalogue
                       [default: 50000]
    WRITE_EVERY      - flush NPZ updates to disk every N realisations
                       [default: 500]
    RESUME           - if 'true', skip already-completed realisations found in
                       an existing output file  [default: false]
"""

# CHANGELOG:
# * Take CHIME C_ells from per-ell file
# * Don't interpolate theory C_ells
# * Sample random source uniformly across the sky
# * Do not subtract mean from field values.
# * Load CHIME positions from original file, and ignore POSITION_PATH
#   (This is the change that fixes the bug)

import os
import numpy as np
import healpy as hp
from scipy.interpolate import InterpolatedUnivariateSpline
import pymaster as nmt


def _with_position_suffix(output_path, position_source, position_mask_mode):
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


_HERE = os.path.dirname(os.path.abspath(__file__))

CL_PATH        = os.getenv("CL_PATH",        os.path.join(_HERE, "data", "cell_chime_multipoles.npz"))
POSITIONS_PATH = os.getenv("POSITIONS_PATH", os.path.join(_HERE, "data", "chimefrbcat2_radec.txt"))
_OUTPUT_PATH_RAW = os.getenv("OUTPUT_PATH",  os.path.join(_HERE, "output", "mock_mod.npz"))
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
PCL_COUPLED    = os.getenv("PCL_COUPLED", "false").strip().lower() in ("1", "true", "yes")
RESUME         = os.getenv("RESUME", "false").strip().lower() in ("1", "true", "yes")
WRITE_EVERY    = int(os.getenv("WRITE_EVERY",    "500"))
POSITION_SOURCE = os.getenv("POSITION_SOURCE", "mask").strip().lower()
POSITION_MASK_PATH = os.getenv(
    "POSITION_MASK_PATH",
    os.path.join(_HERE, "output", "chime", "detection_probability_mask_nside128.fits"),
)
POSITION_MASK_MODE = os.getenv("POSITION_MASK_MODE", "fixed").strip().lower()
N_SOURCES = int(os.getenv("N_SOURCES", "50000"))
OUTPUT_PATH = _with_position_suffix(_OUTPUT_PATH_RAW, POSITION_SOURCE, POSITION_MASK_MODE)


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


def get_catalog(npoints, m, seed, verbose=False):
    """
    Generates point catalog given a modulating mask m.

    Parameters:
        npoints: upper limit to number of points generated
        m: spin-0 map. modulation mask
        seed: random seed
    Returns:
        pos: Catalog positions
    """
#    np.random.seed(seed)
    m = m / np.amax(m)  # new: normalize to [0,1]
    npix = len(m)
    npoints_get = int(npoints/np.mean(m))
    if npoints_get > 1e7 and verbose:
        print(f"WARNING: Npoints = {npoints_get}. "
              "Consider decreasing nhope or steepening power spectrum slope.")
    phi = 2*np.pi*np.random.rand(npoints_get)
    th = np.arccos(-1+2*np.random.rand(npoints_get))
    ipix = hp.ang2pix(hp.npix2nside(npix), th, phi)
    mv = m[ipix]
    u = np.random.rand(npoints_get)
    keep = u <= mv
    th = th[keep]
    phi = phi[keep]
    return np.array([th, phi], dtype=np.float64)


def sample_positions_from_probability_mask(prob_mask, n_samples, seed):
    """Sample RA/Dec positions from a probability mask."""
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    theta, phi = get_catalog(n_samples, prob_mask, seed)
    ra = np.deg2rad(phi)
    dec = 90. - np.rad2deg(theta)
    return ra.astype(float), dec.astype(float)

    npix = len(prob_mask)
    probs = np.asarray(prob_mask, dtype=float)
    probs = probs / probs.sum()

    # TODO: Here, we may instead distribute sources continuously across RA/DEC
    # and up/downsample them according to the mask
    chosen_pix = rng.choice(npix, size=int(n_samples), p=probs)
    theta, phi = hp.pix2ang(hp.npix2nside(npix), chosen_pix)
    ra = np.degrees(phi)
    dec = 90.0 - np.degrees(theta)
    return ra.astype(float), dec.astype(float)


def load_CHIME_catalog(lmax):
    """
    Returns CHIME catalog positions, theory Cls (up to lmax),
    and dispersion measure catalogs noiseless, with Gaussian noise,
    and with lognormal noise, respectively.
    """
    from astropy.io import fits
    # NOTE: THIS PATH NEEDS TO BE MODIFIED
    chime_fn = "/Users/robert/Documents/Git/pseudo_catcell_covariance_benchmark/data/chimefrbcat2.fits"#"/global/homes/k/kwolz/CatalogCovariancesSandbox/data/chime_catalogue_nside4096_noise.fits"  # noqa: E501
    with fits.open(chime_fn) as hdul:
        print(hdul[1].columns.names)
        data = hdul[1].data
        ra, dec, dm, dm_gauss, dm_lognorm = [
            np.asarray(data[k], dtype=np.float64)
            for k in ["RA", "DEC", "DM", "DM_with_gaussian_noise",
                      "DM_with_lognormal_noise"]]
        cl_th = np.asarray(hdul[2].data["cell"], dtype=np.float64)[:lmax+1]
    pos = np.array([np.deg2rad(90.-dec), np.deg2rad(ra)], dtype=np.float64)
    msk = np.logical_and(~np.isnan(dec), ~np.isnan(ra), ~np.isnan(dm))
    pos, dm, dm_gauss, dm_lognorm = pos[:, msk], dm[msk], dm_gauss[msk], dm_lognorm[msk]
    pos, idx = np.unique(pos, axis=1, return_index=True)

    return cl_th, pos, dm[idx], dm_gauss[idx], dm_lognorm[idx]


def load_positions_from_output(npz_path):
    """Return (ra, dec) arrays in degrees from a saved output NPZ file."""
    theta, phi = load_CHIME_catalog(1000)[1]
    ra = np.rad2deg(phi)
    dec = 90. - np.rad2deg(theta)

    return ra, dec

    data = np.load(npz_path, allow_pickle=False)
    if "ra" not in data or "dec" not in data:
        raise ValueError(
            f"Existing output file {npz_path} is missing 'ra' or 'dec' arrays. "
            f"Available keys: {list(data.keys())}"
        )
    ra  = np.asarray(data["ra"],  dtype=float)
    dec = np.asarray(data["dec"], dtype=float)

    valid_pos = np.isfinite(ra) & np.isfinite(dec)
    n_bad = int((~valid_pos).sum())
    if n_bad > 0:
        print(f"Removed {n_bad} rows with invalid RA/Dec (NaN or inf) from saved output.")
        ra  = ra[valid_pos]
        dec = dec[valid_pos]

    return ra, dec


def get_positions_for_realisation(idx, base_ra, base_dec, prob_mask, n_samples):
    """Return the RA/Dec positions to use for a given realisation index."""
    if POSITION_SOURCE != "mask" or POSITION_MASK_MODE == "fixed":
        return base_ra, base_dec

    rng_pos = np.random.default_rng(SEED_START + 1000000 + int(idx))
    return sample_positions_from_probability_mask(prob_mask, n_samples=n_samples, rng=rng_pos)


def build_cl_full(cl_path, nside, lmax_factor, ell_max_cap=None):
    # NOTE: Here we simply read the per-multipole input data
    cl_npz = np.load(cl_path)
    if not {"ell", "cell"}.issubset(cl_npz.files):
        raise ValueError(f"{cl_path} must contain 'ell' and 'cell' arrays")

    ell   = np.asarray(cl_npz["ell"],  dtype=int)
    cl_in = np.asarray(cl_npz["cell"], dtype=float)

    return cl_in, ell[-1]

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

    return cl_full, lmax_factor * nside - 1


def evaluate_alm_at_positions(alm, lmax, theta, phi, epsilon=1e-7):
    """Evaluate a scalar field from packed alm coefficients at arbitrary positions.

    Uses ducc0.sht.synthesis_general for fast NUFFT-style evaluation
    (healpy-packed alm, colatitude theta, longitude phi in radians).
    """
    import ducc0
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    if theta.shape != phi.shape:
        raise ValueError("theta and phi must have the same shape")

    if theta.size == 0:
        return np.zeros(0, dtype=np.float64)

    loc = np.stack([theta.ravel(), phi.ravel()], axis=1)
    result = ducc0.sht.synthesis_general(
        alm=np.asarray(alm, dtype=np.complex128)[np.newaxis],
        spin=0, lmax=lmax, loc=loc, epsilon=epsilon,
    )
    return result[0].real.reshape(theta.shape)


def build_pcl_binner(lmax_pcl, lmin, nbins, lmax_user=None):
    """Build the geometric ell binning used for the pseudo-C_ell measurement."""
    lmin = max(2, int(lmin))
    if nbins < 1:
        raise ValueError("PCL_NBINS must be >= 1.")

    lmax_cap = min(int(lmax_pcl), 3 * int(NSIDE) - 1)
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

    binner = nmt.NmtBin.from_edges(edges[:-1], edges[1:])
    leff = np.asarray(binner.get_effective_ells(), dtype=float)
    return binner, leff, edges


def measure_pseudo_cl(ra, dec, data_columns, binner, workspace=None,
                      return_coupled=False, lmax_syn=None):
    """Measure pseudo-C_ell values from catalogue samples using NaMaster.

    Pass a pre-built NmtWorkspace via *workspace* to skip recomputing the
    coupling matrix (safe when positions/weights are fixed across realisations).
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    ra_wrapped = np.where(ra < 0.0, ra + 360.0, ra)

    n_out = binner.lmax + 1 if return_coupled else len(binner.get_effective_ells())
    pseudo_cls = {}
    for name, values in data_columns.items():
        values = np.asarray(values, dtype=float)
        valid = np.isfinite(ra_wrapped) & np.isfinite(dec) & np.isfinite(values)
        if not np.any(valid):
            pseudo_cls[name] = np.full(n_out, np.nan, dtype=float)
            continue
        n_valid = int(valid.sum())
        pos_data = np.vstack([ra_wrapped[valid], dec[valid]])
        weights = np.ones(n_valid, dtype=float)
        # NOTE: The catalog-based estimator is agnostic to white noise,
        # so subtracting the mean shouldn't be necessary.
        # mean_sub = values[valid][np.newaxis, :] - np.mean(values[valid])
        field = nmt.NmtFieldCatalog(pos_data, weights, values[valid],
                                    lmax=lmax_syn, lonlat=True)
        ws = (workspace if workspace is not None
              else nmt.NmtWorkspace.from_fields(field, field, binner))
        coupled = nmt.compute_coupled_cell(field, field)
        if return_coupled:
            pseudo_cls[name] = np.asarray(coupled[0], dtype=float)
        else:
            pseudo_cls[name] = np.asarray(ws.decouple_cell(coupled)[0], dtype=float)

    return pseudo_cls


def generate_realisation(cl_full, lmax_syn, ra, dec, noise_mean, noise_var, seed):
    """Draw a new alm realisation, evaluate at catalogue positions, and add noise.

    Returns a dict with keys: DM, DM_gaussian, DM_lognormal.
    """
    rng = np.random.default_rng(seed)

    ra_hp  = np.where(ra < 0, ra + 360.0, ra)
    theta  = np.clip(np.radians(90.0 - dec), 1e-6, np.pi - 1e-6)
    phi    = np.radians(ra_hp)
    valid  = np.isfinite(theta) & np.isfinite(phi)
    dm     = np.full(len(theta), np.nan)

    np.random.seed(int(rng.integers(0, 2**31)))
    alm = hp.synalm(cl_full, lmax=lmax_syn, new=True)
    dm[valid] = evaluate_alm_at_positions(alm, lmax_syn, theta[valid], phi[valid])

    noise_gauss = rng.normal(loc=noise_mean, scale=np.sqrt(noise_var), size=len(dm))

    if noise_mean <= 0:
        raise ValueError("NOISE_MEAN must be > 0 for the log-normal noise model.")
    s_sq = np.log(1.0 + noise_var / noise_mean**2)
    m    = np.log(noise_mean) - 0.5 * s_sq
    noise_lognorm = rng.lognormal(mean=m, sigma=np.sqrt(s_sq), size=len(dm))

    return {
        "DM":           dm,
        "DM_gaussian":  dm + noise_gauss,
        "DM_lognormal": dm + noise_lognorm,
    }


def _load_npz_for_resume(npz_path):
    data = np.load(npz_path, allow_pickle=False)
    n_completed = int(data["n_completed"]) if "n_completed" in data else 0
    return dict(data), n_completed


def _save_npz(npz_path, **arrays):
    os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
    stem = npz_path[:-4] if npz_path.endswith(".npz") else npz_path
    tmp = stem + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, npz_path)


def main():
    print(f"=== generate_mock_catalogues.py ===")
    print(f"  CL_PATH:         {CL_PATH}")
    print(f"  POSITIONS_PATH:  {POSITIONS_PATH}")
    print(f"  OUTPUT_PATH:     {OUTPUT_PATH}")
    print(f"  NSIDE:           {NSIDE}")
    print(f"  N_REALISATIONS:  {N_REALISATIONS}")
    print(f"  NOISE_MEAN:      {NOISE_MEAN}")
    print(f"  NOISE_VAR:       {NOISE_VAR}")
    print(f"  SEED_START:      {SEED_START}")
    print(f"  MEASURE_PCL:     {MEASURE_PCL}")
    if MEASURE_PCL:
        print(f"  PCL_LMIN:        {PCL_LMIN}")
        print(f"  PCL_LMAX:        {PCL_LMAX if PCL_LMAX is not None else 'auto'}")
        print(f"  PCL_NBINS:       {PCL_NBINS}")
        print(f"  PCL_COUPLED:     {PCL_COUPLED}")
    print(f"  RESUME:          {RESUME}")
    print(f"  WRITE_EVERY:     {WRITE_EVERY}")
    print(f"  POSITION_SOURCE: {POSITION_SOURCE}")
    if POSITION_SOURCE == "mask":
        print(f"  POSITION_MASK_PATH: {POSITION_MASK_PATH}")
        print(f"  POSITION_MASK_MODE: {POSITION_MASK_MODE}")
        print(f"  N_SOURCES:          {N_SOURCES if N_SOURCES > 0 else 'auto (from catalog)'}")

    if POSITION_SOURCE not in {"catalog", "mask"}:
        raise ValueError("POSITION_SOURCE must be either 'catalog' or 'mask'")
    if POSITION_MASK_MODE not in {"fixed", "resample"}:
        raise ValueError("POSITION_MASK_MODE must be either 'fixed' or 'resample'")

    # Load positions
    print("Loading source positions...")
    prob_mask = None
    if POSITION_SOURCE == "catalog":
        #if os.path.exists(OUTPUT_PATH):  # and RESUME:
        ra, dec = load_positions_from_output(OUTPUT_PATH)
        # else:
        #     ra, dec = np.loadtxt(POSITIONS_PATH, usecols=(0, 1), unpack=True)
        print(f"  {len(ra)} sources")
    else:
        prob_mask = load_probability_mask(POSITION_MASK_PATH)
        n_sources = N_SOURCES if N_SOURCES > 0 else len(np.loadtxt(POSITIONS_PATH, usecols=(0,)))
        # if n_sources < 1:
        #     raise ValueError("Number of mask-sampled sources must be >= 1")

        if os.path.exists(OUTPUT_PATH) and POSITION_MASK_MODE == "fixed":  #and RESUME
            ra, dec = load_positions_from_output(OUTPUT_PATH)
        else:
            rng_initial = np.random.default_rng(SEED_START + 999999)
            ra, dec = sample_positions_from_probability_mask(prob_mask, n_sources, rng_initial)
        print(f"  {len(ra)} sources")

    print("\nBuilding interpolated C_ell...")
    cl_full, lmax_syn = build_cl_full(CL_PATH, NSIDE, LMAX_FACTOR, ell_max_cap=3 * NSIDE - 1)
    print(f"  lmax_syn = {lmax_syn}", NSIDE)

    pcl_binner = None
    pcl_leff = None
    pcl_edges = None
    pcl_workspace = None
    pcl_theory_bandpowers = None
    if MEASURE_PCL:
        print("Building pseudo-C_ell binning...")
        pcl_binner, pcl_leff, pcl_edges = build_pcl_binner(
            lmax_syn, PCL_LMIN, PCL_NBINS, lmax_user=PCL_LMAX,
        )
        if PCL_COUPLED:
            pcl_leff = np.arange(pcl_binner.lmax + 1, dtype=float)
        print(f"  {'multipoles' if PCL_COUPLED else 'bandpowers'}: {len(pcl_leff)}")

        if POSITION_SOURCE == "catalog" or POSITION_MASK_MODE == "fixed":
            print("  Building NaMaster workspace from fixed source positions...")
            _pos_ws = np.vstack([np.where(ra < 0, ra + 360.0, ra).astype(float), dec.astype(float)])
            _w_ws   = np.ones(len(ra), dtype=float)
            _fv_ws  = np.ones(len(ra), dtype=float)
            _f_ws   = nmt.NmtFieldCatalog(_pos_ws, _w_ws, _fv_ws, lmax=pcl_binner.lmax, lonlat=True)
            pcl_workspace = nmt.NmtWorkspace.from_fields(_f_ws, _f_ws, pcl_binner)

            _theory_cl = cl_full[:pcl_binner.lmax + 1]
            if PCL_COUPLED:
                pcl_theory_bandpowers = np.asarray(
                    pcl_workspace.couple_cell([_theory_cl])[0], dtype=np.float64
                )
            else:
                pcl_theory_bandpowers = np.asarray(
                    pcl_workspace.decouple_cell(
                        pcl_workspace.couple_cell([_theory_cl])
                    )[0], dtype=np.float64
                )
            print(f"  theory: {np.array2string(pcl_theory_bandpowers, precision=3)}")

    n_src  = len(ra)
    n_bins = (pcl_binner.lmax + 1 if PCL_COUPLED else len(pcl_leff)) if MEASURE_PCL else 0

    dm_all         = np.full((N_REALISATIONS, n_src),   np.nan, dtype=np.float64)
    dm_gauss_all   = np.full((N_REALISATIONS, n_src),   np.nan, dtype=np.float64)
    dm_lognorm_all = np.full((N_REALISATIONS, n_src),   np.nan, dtype=np.float64)
    if MEASURE_PCL:
        pcl_dm_all       = np.full((N_REALISATIONS, n_bins), np.nan, dtype=np.float64)
        pcl_dm_gauss_all = np.full((N_REALISATIONS, n_bins), np.nan, dtype=np.float64)
        pcl_dm_ln_all    = np.full((N_REALISATIONS, n_bins), np.nan, dtype=np.float64)

    n_done = 0
    if RESUME and os.path.exists(OUTPUT_PATH):
        print(f"Resuming from existing output: {OUTPUT_PATH}")
        saved, n_done = _load_npz_for_resume(OUTPUT_PATH)
        if n_done > 0 and "DM" in saved:
            nc = min(n_done, N_REALISATIONS)
            dm_all[:nc]         = saved["DM"][:nc]
            dm_gauss_all[:nc]   = saved["DM_gaussian"][:nc]
            dm_lognorm_all[:nc] = saved["DM_lognormal"][:nc]
        if MEASURE_PCL and n_done > 0 and "pcl_dm" in saved:
            nc = min(n_done, N_REALISATIONS)
            pcl_dm_all[:nc]       = saved["pcl_dm"][:nc]
            pcl_dm_gauss_all[:nc] = saved["pcl_dm_gaussian"][:nc]
            pcl_dm_ln_all[:nc]    = saved["pcl_dm_lognormal"][:nc]
        del saved

    todo = list(range(n_done, N_REALISATIONS))
    if not todo:
        print("All realisations already completed.")
        return

    print(f"\nGenerating {len(todo)} realisation(s) "
          f"({n_done} already done, {N_REALISATIONS} total)...")

    def _build_save_dict(n_completed):
        nc = int(n_completed)
        d = dict(
            ra=ra, dec=dec,
            ell=np.arange(len(cl_full), dtype=np.int32),
            cell=cl_full,
            nside_sim=np.int32(NSIDE),
            noise_mean=np.float64(NOISE_MEAN),
            noise_var=np.float64(NOISE_VAR),
            position_source=np.bytes_(POSITION_SOURCE),
            position_mask_mode=np.bytes_(POSITION_MASK_MODE),
            DM=dm_all[:nc],
            DM_gaussian=dm_gauss_all[:nc],
            DM_lognormal=dm_lognorm_all[:nc],
            n_completed=np.int32(nc),
        )
        if MEASURE_PCL:
            d.update(
                pcl_ell_eff=pcl_leff,
                pcl_edges=pcl_edges,
                pcl_dm=pcl_dm_all[:nc],
                pcl_dm_gaussian=pcl_dm_gauss_all[:nc],
                pcl_dm_lognormal=pcl_dm_ln_all[:nc],
            )
            if pcl_theory_bandpowers is not None:
                d["pcl_theory"] = pcl_theory_bandpowers
        return d

    for count, idx in enumerate(todo):
        ra_i, dec_i = get_positions_for_realisation(idx, ra, dec, prob_mask, n_src)
        data = generate_realisation(
            cl_full, lmax_syn, ra_i, dec_i,
            NOISE_MEAN, NOISE_VAR, seed=SEED_START + idx,
        )
        dm_all[idx]         = data["DM"]
        dm_gauss_all[idx]   = data["DM_gaussian"]
        dm_lognorm_all[idx] = data["DM_lognormal"]

        if MEASURE_PCL:
            pseudo_cls = measure_pseudo_cl(ra_i, dec_i, data, pcl_binner,
                                           workspace=pcl_workspace,
                                           return_coupled=PCL_COUPLED,
                                           lmax_syn=lmax_syn)
            pcl_dm_all[idx]       = pseudo_cls["DM"]
            pcl_dm_gauss_all[idx] = pseudo_cls["DM_gaussian"]
            pcl_dm_ln_all[idx]    = pseudo_cls["DM_lognormal"]

        n_done = idx + 1
        if ((count + 1) % WRITE_EVERY == 0) or (count + 1 == len(todo)):
            _save_npz(OUTPUT_PATH, **_build_save_dict(n_done))

        print(f"  [{count+1}/{len(todo)}] realisation {idx:5d} done  "
              f"(DM mean = {np.nanmean(data['DM']):.4e})", flush=True)

    print(f"\nFinished. Output written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
