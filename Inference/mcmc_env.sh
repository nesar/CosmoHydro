# mcmc_env.sh — source before launching any run_mcmc*.py job so it plays nicely
# on a shared box. Two things:
#   1) pins BLAS/OpenMP to 1 thread per worker  -> no thread-explosion / thrashing
#   2) optionally caps the multiprocessing Pool -> divide cores across parallel jobs
#
# Usage:
#   source mcmc_env.sh          # solo job: all cores, 1 thread each (fast)
#   source mcmc_env.sh 6        # share: this job uses ~6 workers (for ~4 parallel jobs)
#
# Rule of thumb on this 24-core box: set the arg to  24 / (number of jobs you run
# at once). Running one job? omit it. The MCMC code is unchanged unless
# MCMC_NWORKERS is set (run_mcmc.py reads it; default = all cores).

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

if [ -n "${1:-}" ]; then
    export MCMC_NWORKERS="$1"
    echo "[mcmc_env] 1 thread/worker; Pool capped to $MCMC_NWORKERS workers"
else
    unset MCMC_NWORKERS
    echo "[mcmc_env] 1 thread/worker; Pool = all cores (solo-job mode)"
fi
