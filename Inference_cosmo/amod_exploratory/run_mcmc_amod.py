#!/usr/bin/env python
"""EXPLORATORY A_mod MCMC runner — see README.md for why this is quarantined.

Registers the 'amod' target kind into the standard driver's plug-in registry
and delegates everything else to run_mcmc_cosmo (same YAML schema, same
sampler, same results format). Configs live in amod_exploratory/configs/ and
write into amod_exploratory/results/ (set via output_dir in each config).

    python run_mcmc_amod.py configs/Pk_amod_5subgrid_fidcosmo.yaml
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..')))
sys.path.insert(0, _HERE)

import run_mcmc_cosmo as R                          # noqa: E402
from amod_likelihood import load_amod, AmodLikelihood   # noqa: E402
from linear_theory import LinearPk                  # noqa: E402


def _build_amod(spec, get_emu):
    target = load_amod(spec.get('constraint', 'DES_Y3_Planck'))
    like = AmodLikelihood(target, get_emu(), linear_pk=LinearPk(),
                          k_fit_min=spec.get('k_fit_min', 0.1),
                          k_fit_max=spec.get('k_fit_max', 8.0))
    return f'amod_{spec.get("constraint", "DES_Y3_Planck")}', like


R.EXTRA_TARGET_KINDS['amod'] = _build_amod

if __name__ == '__main__':
    print('*** EXPLORATORY A_mod run — see amod_exploratory/README.md ***')
    R.main()
