#!/usr/bin/env bash
set -u -o pipefail
cd "$(dirname "$0")" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MCMC_NWORKERS=4
echo "=== GSMF_CGD_7p_wide (flat cosmo prior) started $(date) ==="
if [ -f results/samples_GSMF_CGD_7p_wide.npy ]; then
  echo "already done — skipping"; exit 0
fi
python run_mcmc.py configs/GSMF_CGD_7p_wide.yaml 2>&1 | tee results/run_GSMF_CGD_7p_wide.log
echo "=== done $(date) ==="
