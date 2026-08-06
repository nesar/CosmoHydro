#!/usr/bin/env bash
#
# Fixed-hydro scan, WIDE-PRIOR variant: same scan points A-D plus the fiducial
# fix, but with a FLAT cosmology prior over the FULL design box (see the
# *_wide.yaml configs). Separate outputs; does NOT touch the original scan.
#
# The plot also includes two chains that already exist and are NOT re-run here:
#   GSMF_CGD_7p_planck      hydro marginalized       (red)
#   GSMF_CGD_2cosmo_planck  Frontier-E fiducial fix  (black, ~22 sigma from the 7p peak)
#
# Usage (inside screen):
#     screen -S hydscan_wide
#     ./run_fixed_hydro_scan_wide.sh
#     # Ctrl-A then D to detach;  screen -r hydscan_wide  to come back
#
# Resumable: a point whose samples_*.npy already exists is skipped, so if the
# session dies you can just re-run this script.

set -u -o pipefail

cd "$(dirname "$0")" || exit 1

# --- CRITICAL: one BLAS thread per worker ------------------------------------
# run_mcmc.py runs emcee with a multiprocessing Pool (~24 workers). Without these,
# every worker's numpy/BLAS spawns a full thread pool (all cores), so 24 workers x
# ~24 threads oversubscribe the 24-core box ~25x and each run crawls for DAYS.
# Pinning to 1 thread/worker makes 24 workers == 24 cores and cuts runtime from
# days to minutes. Must be exported BEFORE python imports numpy.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Worker count: for this cheap 2-param GSMF+CGD likelihood, emcee's Pool overhead
# dominates. A benchmark (200 steps x 400 walkers) gave:
#   serial 551s | pool-4 317s | pool-8 520s | pool-24 far worse (never finished)
# so a SMALL pool wins; 24 workers left a single point unfinished after 4+ hours.
export MCMC_NWORKERS=4

SUITE=GSMF_CGD
TRIALS=(${SUITE}_2cosmo_wide ${SUITE}_2cosmo_hydA_wide ${SUITE}_2cosmo_hydB_wide ${SUITE}_2cosmo_hydC_wide ${SUITE}_2cosmo_hydD_wide)

echo "=============================================================="
echo " Fixed-hydro scan (WIDE prior) — started $(date)"
echo " workdir: $PWD"
echo "=============================================================="

# ---- sanity: configs present, and a dry-run on the first trial --------------
for trial in "${TRIALS[@]}"; do
    if [[ ! -f "configs/${trial}.yaml" ]]; then
        echo "ERROR: missing configs/${trial}.yaml"
        exit 1
    fi
done

echo
echo "--- dry-run check on ${TRIALS[0]} ---"
if ! python run_mcmc.py "configs/${TRIALS[0]}.yaml" --dry-run; then
    echo "ERROR: dry-run failed — aborting before the long runs."
    exit 1
fi

# ---- the four runs, sequential ---------------------------------------------
failed=()
for trial in "${TRIALS[@]}"; do
    out="results/samples_${trial}.npy"

    echo
    echo "=============================================================="
    if [[ -f "$out" ]]; then
        echo " SKIP  $trial — $out already exists"
        continue
    fi
    echo " RUN   $trial   ($(date +%H:%M:%S))"
    echo "=============================================================="

    if python run_mcmc.py "configs/${trial}.yaml" 2>&1 | tee "results/run_${trial}.log"; then
        echo " DONE  $trial   ($(date +%H:%M:%S))"
    else
        echo " FAIL  $trial — see results/run_${trial}.log"
        failed+=("$trial")
    fi
done

# ---- plot -------------------------------------------------------------------
echo
echo "=============================================================="
echo " Plotting the scan   ($(date +%H:%M:%S))"
echo "=============================================================="
python diagnostics/check_fixed_hydro_scan_wide.py

echo
echo "=============================================================="
if (( ${#failed[@]} )); then
    echo " Finished WITH FAILURES: ${failed[*]}"
else
    echo " All done — $(date)"
fi
echo "   plot:    diagnostics/fixed_hydro_scan_wide.png"
echo "   summary: diagnostics/fixed_hydro_scan_wide_summary.txt"
echo "=============================================================="
