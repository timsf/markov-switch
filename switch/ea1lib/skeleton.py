from typing import Callable, Iterator, NamedTuple

import numpy as np
import numpy.typing as npt

from switch.mjplib import skeleton
from switch.sdelib import interpolators
from switch.misc.exceptions import BudgetConstraintError


FloatArr = npt.NDArray[np.floating]


class Skeleton(NamedTuple):
    t: FloatArr
    zt: FloatArr


def sample_skeleton(
    fin_t: float, 
    init_x: float,
    endpoint_sampler: Iterator[float], 
    eval_disc: Callable[[FloatArr], FloatArr],
    eval_bounds_disc: Callable[[float, float], tuple[float, float]],
    ome: np.random.Generator, 
    max_props: int = int(1e4),
) -> Iterator[Skeleton]:

    while True:

        for _ in range(max_props):
            proposal = init_skeleton(fin_t, init_x, next(endpoint_sampler), ome)
            success, proposal = flip_poisson_coin(proposal, eval_disc, eval_bounds_disc, ome)
            if success:
                break
        else:
            raise BudgetConstraintError('None of the proposals were accepted.')

        yield proposal


def init_skeleton(fin_t: float, init_x: float, fin_x: float, ome: np.random.Generator) -> Skeleton:

    skel = Skeleton(np.array([0, fin_t]), np.array([init_x, fin_x]))
    return skel


def flip_poisson_coin(
    skel: Skeleton, 
    eval_disc: Callable[[FloatArr], FloatArr],
    eval_bounds_disc: Callable[[float, float], tuple[float, float]],
    ome: np.random.Generator, 
    batch_size: int = 2,
) -> tuple[bool, Skeleton]:

    inf_phi, sup_phi = eval_bounds_disc(-np.inf, np.inf)

    new_skel = skel
    for ppp in skeleton.sample_batch_ppp(1, (sup_phi - inf_phi, skel.t[-1]), batch_size, ome):
        crit_phi, new_t = ppp.T[:, np.argsort(ppp.T[1])]
        new_skel, new_x = interpolate_skeleton(new_skel, new_t, ome)
        if np.any(eval_disc(new_x) - inf_phi > crit_phi):
            return False, new_skel
    return True, new_skel


def interpolate_skeleton(
    skel: Skeleton, 
    new_t: FloatArr, 
    ome: np.random.Generator,
) -> tuple[Skeleton, FloatArr]:

    if len(new_t) == 0:
        return skel, np.array([])

    stack_t, stack_xt = interpolators.fill_brownbr_path(skel.t, skel.zt[np.newaxis], new_t, ome)
    new_skel = Skeleton(stack_t, stack_xt[0])
    return new_skel, new_skel.zt[np.isin(new_skel.t, new_t)]
