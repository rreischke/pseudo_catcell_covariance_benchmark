#!/usr/bin/env bash
# Usage: ./run_example.sh catalog
#        ./run_example.sh mask
set -e

case "${1:-}" in
  catalog)
    POSITION_SOURCE=catalog \
    POSITIONS_PATH=data/chimefrbcat2_radec.txt \
    CL_PATH=data/cell_chime.npz \
    N_REALISATIONS=1000 \
    NSIDE=128 \
    WRITE_EVERY=100 \
    PCL_COUPLED=false\
    NOISE_MEAN=5000 \
    PCL_LMIN=1 \
    PCL_BINNING=linear \
    python generate_mock_catalogues.py
    ;;
  mask)
    POSITION_SOURCE=mask \
    POSITION_MASK_PATH=output/chime/chime_mask_cosine_nside1024.fits \
    POSITION_MASK_MODE=fixed \
    N_SOURCES=50000 \
    CL_PATH=data/cell_chime.npz \
    N_REALISATIONS=1000 \
    NSIDE=1024 \
    PCL_LMIN=2 \
    PCL_LMAX=300 \
    WRITE_EVERY=100 \
    PCL_BINNING=linear \
    python generate_mock_catalogues.py
    ;;
  *)
    echo "Usage: $0 catalog|mask" >&2
    exit 1
    ;;
esac
