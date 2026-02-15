__all__ = ['scale_factor_from_snapshot', 'redshift_from_scale_factor',
           'SNAPSHOT_IDS', 'get_snapshot_redshifts']

import numpy as np


# SNAPSHOT_IDS = [59, 66, 74, 86, 101, 122, 153, 175, 205, 224,
#                 247, 275, 310, 355, 415, 479, 498, 567, 624]


# SNAPSHOT_IDS = [205, 224, 247, 275, 310, 355, 415, 479, 498, 567, 624]
SNAPSHOT_IDS = [205, 224, 247, 275, 310, 355, 415, 479, 498, 567, 624]


def scale_factor_from_snapshot(snapshot_number, z_initial, n_snaps=625):
    a_min = 1.0 / (1.0 + z_initial)
    if not (0 <= snapshot_number <= n_snaps - 1):
        raise ValueError(f"snapshot_number must be in [0, {n_snaps-1}]")
    return a_min + (1.0 - a_min) * ((snapshot_number + 1.0) / n_snaps)


def redshift_from_scale_factor(a):
    a = float(a)
    if a <= 0:
        raise ValueError("scale factor a must be > 0")
    return 1.0 / a - 1.0


def get_snapshot_redshifts(snapshot_ids=None, z_initial=200, n_snaps=625):
    """Compute redshifts for a list of snapshot IDs."""
    if snapshot_ids is None:
        snapshot_ids = SNAPSHOT_IDS
    scale_factors = np.array([scale_factor_from_snapshot(s, z_initial, n_snaps)
                              for s in snapshot_ids])
    redshifts = np.array([redshift_from_scale_factor(a) for a in scale_factors])
    return redshifts, scale_factors
