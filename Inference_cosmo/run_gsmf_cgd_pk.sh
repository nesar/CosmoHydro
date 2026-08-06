#!/usr/bin/env bash
# Joint GSMF+CGD+KiDS-Pk MCMC launcher.  usage: ./run_gsmf_cgd_pk.sh <config.yaml> [nworkers]
set -u -o pipefail
cd "$(dirname "$0")" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MCMC_NWORKERS=${2:-8}
cfg="$1"; trial=$(basename "$cfg" .yaml)
echo "=== $trial started $(date) (NWORKERS=$MCMC_NWORKERS) ==="
if [ -f "results/samples_${trial}.npy" ]; then echo "already done — skipping"; exit 0; fi
python run_mcmc_cosmo.py "$cfg" 2>&1 | tee "results/run_${trial}.log"
echo "=== $trial done $(date) ==="
