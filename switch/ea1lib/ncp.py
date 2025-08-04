from typing import Callable, Iterator

import numpy as np
import numpy.typing as npt

from switch.ea1lib import skeleton
from switch.mjplib.skeleton import sample_ppp, sample_batch_ppp
from switch.misc.exceptions import BudgetConstraintError


FloatArr = npt.NDArray[np.floating]
NormOp = Callable[[FloatArr, FloatArr], FloatArr]


def sample_bridge(
    fin_t: float,
    denorm_rsde: NormOp,
    eval_disc: Callable[[FloatArr], FloatArr],
    eval_bounds_disc: Callable[[float, float], tuple[float, float]],
    ome: np.random.Generator,
    max_props: int = int(1e4),
) -> Iterator[skeleton.Skeleton]:

    assert 0 <= max_props

    while True:

        for _ in range(max_props):
            proposal = skeleton.init_skeleton(fin_t, 0, 0, ome)
            coin = gen_poisson_coin(proposal, denorm_rsde, eval_disc, eval_bounds_disc, ome)
            success, proposal = next(coin)
            log_weight = integrate_disc_bound(proposal, denorm_rsde, eval_bounds_disc)
            if success and np.log(ome.uniform()) < log_weight:
                break
        else:
            raise BudgetConstraintError('None of the proposals were accepted.')

        yield proposal


def gen_poisson_coin(
    skel: skeleton.Skeleton,
    denorm_rsde: NormOp,
    eval_disc: Callable[[FloatArr], FloatArr],
    eval_bounds_disc: Callable[[float, float], tuple[float, float]],
    ome: np.random.Generator,
    batch_size: int = 2,
) -> Iterator[tuple[bool, skeleton.Skeleton]]:

    inf_phi, sup_phi = eval_bounds_disc(-np.inf, np.inf)

    while True:
        success, skel = flip_poisson_coin(skel, (inf_phi, sup_phi), denorm_rsde, eval_disc, ome, batch_size)
        yield success, skel


def flip_poisson_coin(
    skel: skeleton.Skeleton,
    bounds_disc: tuple[float, float],
    denorm_rsde: NormOp,
    eval_disc: Callable[[FloatArr], FloatArr],
    ome: np.random.Generator,
    batch_size: int,
) -> tuple[bool, skeleton.Skeleton]:

    inf_phi, sup_phi = bounds_disc
    intensity = np.max(sup_phi - inf_phi)
    if np.isinf(intensity):
        return False, skel

    new_skel = skel
    for ppp in sample_batch_ppp(1, (intensity, skel.t[-1]), batch_size, ome):
        crit_phi, new_t = ppp.T[:, np.argsort(ppp.T[1])]
        new_skel, new_x = interpolate_skel(new_skel, new_t, denorm_rsde, ome)
        new_phi = eval_disc(new_x)
        if np.any(new_phi - inf_phi > crit_phi):
            return False, new_skel
    return True, new_skel


def interpolate_skel(
    skel: skeleton.Skeleton, 
    new_t: FloatArr,
    denorm_rsde: NormOp,
    ome: np.random.Generator,
) -> tuple[skeleton.Skeleton, FloatArr]:

    new_skel, new_z = skeleton.interpolate_skeleton(skel, new_t, ome)
    new_x = denorm_rsde(new_t, new_z)
    return new_skel, new_x


def integrate_disc_bound(
    skel: skeleton.Skeleton,
    denorm_rsde: NormOp,
    eval_bounds_disc: Callable[[float, float], tuple[float, float]],
) -> float:

    inf_phi, _ = eval_bounds_disc(-np.inf, np.inf)
    return -skel.t[-1] * inf_phi
