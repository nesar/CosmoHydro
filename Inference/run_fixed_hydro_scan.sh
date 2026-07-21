#!/usr/bin/env bash
#
# Fixed-hydro scan: run the four 2p-cosmology MCMCs (hydro pinned at scan points
# A-D drawn from the GSMF_CGD_7p_pk posterior), then make the overlay plot.
#
# The plot also includes two chains that already exist and are NOT re-run here:
#   GSMF_CGD_7p_pk      hydro marginalized       (red)
#   GSMF_CGD_2cosmo_pk  Frontier-E fiducial fix  (black, ~22 sigma from the 7p peak)
#
# Usage (inside screen):
#     screen -S hydscan
#     ./run_fixed_hydro_scan.sh
#     # Ctrl-A then D to detach;  screen -r hydscan  to come back
#
# Resumable: a point whose samples_*.npy already exists is skipped, so if the
# session dies you can just re-run this script.

set -u -o pipefail

cd "$(dirname "$0")" || exit 1
POINTS=(A B C D)
SUITE=GSMF_CGD

echo "=============================================================="
echo " Fixed-hydro scan — started $(date)"
echo " workdir: $PWD"
echo "=============================================================="

# ---- sanity: configs present, and a dry-run on the first point --------------
for L in "${POINTS[@]}"; do
    cfg="configs/${SUITE}_2cosmo_hyd${L}.yaml"
    if [[ ! -f "$cfg" ]]; then
        echo "ERROR: missing $cfg"
        echo "       regenerate with: python diagnostics/select_fixed_hydro_points.py"
        exit 1
    fi
done

echo
echo "--- dry-run check on point ${POINTS[0]} ---"
if ! python run_mcmc.py "configs/${SUITE}_2cosmo_hyd${POINTS[0]}.yaml" --dry-run; then
    echo "ERROR: dry-run failed — aborting before the long runs."
    exit 1
fi

# ---- the four runs, sequential ---------------------------------------------
failed=()
for L in "${POINTS[@]}"; do
    trial="${SUITE}_2cosmo_hyd${L}"
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
python diagnostics/check_fixed_hydro_scan.py

echo
echo "=============================================================="
if (( ${#failed[@]} )); then
    echo " Finished WITH FAILURES: ${failed[*]}"
else
    echo " All done — $(date)"
fi
echo "   plot:    diagnostics/fixed_hydro_scan.png"
echo "   summary: diagnostics/fixed_hydro_scan_summary.txt"
echo "=============================================================="
