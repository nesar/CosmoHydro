"""Summary figure for the joint GSMF + CGD + KiDS-Pk (2-redshift) runs:
triangle (left) + posterior-predictive panels for GSMF, CGD and KiDS P_m (right),
one best-fit curve per chain.

Chains (7-parameter union triangle):
  GSMF_CGD_Pk_7p            all 7 free, marginalized     RED outline
  GSMF_CGD_Pk_5p_fid_cosmo  5 subgrid, cosmo fixed       GREEN fill
  GSMF_CGD_Pk_2cosmo_hydA   2 cosmo, hydro fixed at A    BLUE fill

Output: results/plot_summary_gsmf_cgd_pk_7p_vs_2cosmoA_5subgrid.png
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import plot_summary_cosmo as M                     # panels + triangle machinery
from cosmo_hydro_emu.load_hacc import PARAM_NAME


def main():
    # optional suffix, e.g. '_emuvar', to plot the emulator-variance run set
    suffix = sys.argv[1] if len(sys.argv) > 1 else ''
    tag = {'': '', '_emuvar': ' [+emu variance]',
           '_cal': ' [calibrated errors: GSMF Jacobian, CGD Ghirardini+19, '
                   'KiDS k$\\in$[0.05,4]]',
           '_cal2': ' [GSMF Jacobian, CGD inter-survey errors, KiDS '
                    'k$\\in$[0.05,4], +emu var]'}.get(suffix, f' [{suffix.lstrip("_")}]')
    # (trial, label, color, filled, linewidth)
    chain_specs = [
        (f'GSMF_CGD_Pk_7p{suffix}',           '7p marginalized',        '#d62728', False, 2.6),
        (f'GSMF_CGD_Pk_5p_fid_cosmo{suffix}', '5 subgrid (cosmo fixed)', '#2ca02c', True, 1.6),
        (f'GSMF_CGD_Pk_2cosmo_hydA{suffix}',  '2 cosmo @ A (hydro fixed)', '#1f77b4', True, 1.6),
    ]
    # For the _cal set, show the CORRECTED CGD errors in the panel (plot_mcmc's
    # loader hard-codes the legacy flat 5%). NOTE: no GSMF entry on purpose —
    # the GSMF panel plots log10(y)=phi against sigma_phi, which is already the
    # consistent pair; the err_jacobian fix belongs to the LIKELIHOOD's 10^phi
    # space only, so applying it here would double-count. See CALIBRATION_FIXES.md.
    gc_fixes = {'_cal':  {'CGD': {'err_model': 'ghirardini19'}},
                '_cal2': {'CGD': {'err_model': 'intersurvey'}}}.get(suffix)
    ctx = M.build_ctx(need_hmf=False, need_kids=True, need_gc=True,
                      gc_err_fixes=gc_fixes)
    M.make_figure_multi(
        chain_specs,
        all_labels=list(PARAM_NAME),
        title='GSMF + CGD + KiDS $P_m$ (2 redshifts): 7p marginalized (red) vs '
              '2-cosmo @ A (blue) vs 5-subgrid (green)' + tag +
              '\n— with posterior-predictive summary (GSMF, CGD, $P_m$)',
        out_name=f'plot_summary_gsmf_cgd_pk_7p_vs_2cosmoA_5subgrid{suffix}.png',
        panels=['gsmf', 'cgd', 'kids'],
        ctx=ctx,
    )


if __name__ == '__main__':
    main()
