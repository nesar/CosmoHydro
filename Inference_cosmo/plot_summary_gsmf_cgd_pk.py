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
    # (trial, label, color, filled, linewidth)
    chain_specs = [
        ('GSMF_CGD_Pk_7p',           '7p marginalized',        '#d62728', False, 2.6),
        ('GSMF_CGD_Pk_5p_fid_cosmo', '5 subgrid (cosmo fixed)', '#2ca02c', True, 1.6),
        ('GSMF_CGD_Pk_2cosmo_hydA',  '2 cosmo @ A (hydro fixed)', '#1f77b4', True, 1.6),
    ]
    ctx = M.build_ctx(need_hmf=False, need_kids=True, need_gc=True)
    M.make_figure_multi(
        chain_specs,
        all_labels=list(PARAM_NAME),
        title='GSMF + CGD + KiDS $P_m$ (2 redshifts): 7p marginalized (red) vs '
              '2-cosmo @ A (blue) vs 5-subgrid (green)\n— with posterior-predictive '
              'summary (GSMF, CGD, $P_m$)',
        out_name='plot_summary_gsmf_cgd_pk_7p_vs_2cosmoA_5subgrid.png',
        panels=['gsmf', 'cgd', 'kids'],
        ctx=ctx,
    )


if __name__ == '__main__':
    main()
