import sys
from typing import Callable, Iterator, NamedTuple

import numpy as np
import numpy.typing as npt

import switch.mjplib.skeleton
from switch.sdelib import bessel, interpolators, transforms
from switch.misc.exceptions import BudgetConstraintError


BoolArr = npt.NDArray[np.bool_]
FloatArr = npt.NDArray[np.floating]


class Skeleton(NamedTuple):
    bl: tuple[FloatArr, FloatArr]
    br: tuple[FloatArr, FloatArr]
    init: tuple[float, float]
    fin: tuple[float, float]
    min: tuple[float, float]
    ll: tuple[float, float]
    ul: tuple[float, float]
    hit: tuple[bool, bool]
    flip: bool


def init_skeleton(
    fin_t: float,
    init_x: float,
    fin_x: float,
    ome: np.random.Generator,
) -> Skeleton:
    
    init = (0, init_x)
    fin = (fin_t, fin_x)
    min_, ll, ul, hit, flip = bessel.sample_minimal_skel(fin[0] - init[0], init[1], fin[1], -np.inf, np.inf, ome)
    min_ = (min_[0] + init[0], min_[1])
    bl = (np.array([0, min_[0] - init[0]]), np.zeros((3, 2)))
    br = (np.array([0, fin[0] - min_[0]]), np.zeros((3, 2)))
    return Skeleton(bl, br, init, fin, min_, ll, ul, hit, flip)


def sample_skeleton(
    fin_t: float,
    init_x: float,
    endpoint_sampler: Iterator[float],
    eval_disc: Callable[[FloatArr, FloatArr], FloatArr],
    eval_bounds_disc: Callable[[float, float, float, float], tuple[float, float]],
    global_lb_disc: float,
    ome: np.random.Generator, 
    max_props: int = int(1e4),
) -> Iterator[Skeleton]:

    while True:
        for _ in range(max_props):
            proposal = init_skeleton(fin_t, init_x, next(endpoint_sampler), ome)
            inf_phi, sup_phi = bound_disc_path(proposal, eval_bounds_disc)
            success, proposal = flip_coin(proposal, eval_disc, eval_bounds_disc, ome, (inf_phi, sup_phi))
            if success and np.log(ome.uniform()) < fin_t * (global_lb_disc - inf_phi):
                break
        else:
            raise BudgetConstraintError('None of the proposals were accepted.')

        yield proposal


def flip_coin(
    skel: Skeleton, 
    eval_disc: Callable[[FloatArr, FloatArr], FloatArr],
    eval_bounds_disc: Callable[[float, float, float, float], tuple[float, float]],
    ome: np.random.Generator, 
    bounds_phi: tuple[float, float] | None,
    batch_size: int = 2,
) -> tuple[bool, Skeleton]:

    if bounds_phi is None:
        inf_phi, sup_phi = bound_disc_path(skel, eval_bounds_disc)
    else:
        inf_phi, sup_phi = bounds_phi
    new_skel = skel
    for ppp in switch.mjplib.skeleton.sample_batch_ppp(1, (sup_phi - inf_phi, skel.fin[0] - skel.init[0]), batch_size, ome):
        crit_phi, new_t = ppp.T[:, np.argsort(ppp.T[1])]
        new_skel, new_x = interpolate_skel(new_skel, new_t + skel.init[0], ome)
        if np.any(eval_disc(new_t + skel.init[0], new_x) > crit_phi + inf_phi):
            return False, new_skel
    return True, new_skel


def est_poisson_cond(
    skel: Skeleton,
    eval_disc: Callable[[FloatArr, FloatArr], FloatArr],
    eval_bounds_disc: Callable[[float, float, float, float], tuple[float, float]],
    ome: np.random.Generator,
) -> tuple[float, Skeleton]:

    ppp = (np.empty(shape=(2,0)), 0.0)
    return est_poisson_dcond(skel, ppp, eval_disc, eval_bounds_disc, ome)[:-1]


def est_poisson_dcond(
    skel: Skeleton,
    ppp: tuple[FloatArr, float],
    eval_disc: Callable[[FloatArr, FloatArr], FloatArr],
    eval_bounds_disc: Callable[[float, float, float, float], tuple[float, float]],
    ome: np.random.Generator,
    max_intensity: float | None = None,
) -> tuple[float, Skeleton, tuple[FloatArr, float]]:

    inf_phi, sup_phi = bound_disc_path(skel, eval_bounds_disc)
    intensity = sup_phi - inf_phi
    if np.isinf(intensity):
        return -np.inf, skel, ppp
    if max_intensity is not None and intensity > max_intensity:
        print('Exceeded maximum Poisson process intensity. The current proposal will be rejected.', file=sys.stderr)
        return -np.inf, skel, ppp

    inc_intensity = max(0, intensity - ppp[1])
    new_points = np.hstack([ppp[0], (switch.mjplib.skeleton.sample_ppp(1, (inc_intensity, skel.fin[0] - skel.init[0]), ome) + np.array([ppp[1], 0])).T])
    new_ppp = (new_points[:, np.argsort(new_points[1])], ppp[1] + inc_intensity)

    new_t = new_ppp[0][1, new_ppp[0][0] < intensity]
    if len(new_t) > 0:
        new_skel, new_x = interpolate_skel(skel, new_t + skel.init[0], ome)
        new_phi = eval_disc(new_t, new_x)
        return np.sum(np.log(sup_phi - new_phi)) - len(new_phi) * np.log(intensity), new_skel, new_ppp
    return 0, skel, new_ppp


def est_poisson_coin(
    skel: Skeleton,
    eval_disc: Callable[[FloatArr, FloatArr], FloatArr],
    eval_bounds_disc: Callable[[float, float, float, float], tuple[float, float]],
    ome: np.random.Generator,
) -> tuple[float, Skeleton]:

    inf_phi, sup_phi = bound_disc_path(skel, eval_bounds_disc)
    intensity = sup_phi - inf_phi

    new_t, = switch.mjplib.skeleton.sample_ppp(intensity, (skel.fin[0] - skel.init[0],), ome).T
    if len(new_t) > 0:
        new_skel, new_x = interpolate_skel(skel, new_t + skel.init[0], ome)
        new_phi = eval_disc(new_t + skel.init[0], new_x)
        return np.sum(np.log(sup_phi - new_phi)) - len(new_t) * np.log(intensity), new_skel 
    return 0, skel


def est_poisson_log(
    skel: Skeleton,
    eval_disc: Callable[[FloatArr, FloatArr], FloatArr],
    new_t: FloatArr,
    ome: np.random.Generator,
) -> tuple[float, Skeleton]:

    if len(new_t) == 0:
        return 0, skel
    new_skel, new_x = interpolate_skel(skel, new_t, ome)
    new_phi = eval_disc(new_t, new_x)
    return float(-(skel.fin[0] - skel.init[0]) * np.mean(new_phi)), new_skel


def interpolate_skel(skel: Skeleton, new_t: FloatArr, ome: np.random.Generator) -> tuple[Skeleton, FloatArr]:

    if len(new_t) == 0:
        return skel, np.array([])

    if not skel.flip:
        new_t_std = new_t
    else:
        new_t_std, _ = transforms.reverse(new_t, np.array([]), skel.fin[0])

    new_bl, new_br, new_x_std = interpolate_mincase(skel, new_t_std, ome)
    new_skel = Skeleton(new_bl, new_br, *skel[2:])

    if not skel.flip:
        new_x = new_x_std
    else:
        new_x = transforms.reflect(new_x_std[::-1], (skel.init[1] + skel.fin[1]) / 2)

    return new_skel, new_x


def interpolate_mincase(
    skel: Skeleton,
    new_t: FloatArr,
    ome: np.random.Generator,
) -> tuple[tuple[FloatArr, FloatArr], tuple[FloatArr, FloatArr], FloatArr]:
    
    if skel.hit[0]:
        ubl, hitl = skel.ul[1] - skel.min[1], skel.ul[0] - skel.min[1]
    else:
        ubl, hitl = skel.ul[0] - skel.min[1], None
    if skel.hit[1]:
        ubr, hitr = skel.ul[1] - skel.min[1], skel.ul[0] - skel.min[1]
    else:
        ubr, hitr = skel.ul[0] - skel.min[1], None
    new_sl = skel.min[0] - new_t[new_t < skel.min[0]][::-1]
    new_sr = new_t[new_t > skel.min[0]] - skel.min[0]
    new_bl = interpolators.fill_besselbr_path(*skel.bl, skel.init[1] - skel.min[1], ubl, hitl, new_sl, ome)
    new_br = interpolators.fill_besselbr_path(*skel.br, skel.fin[1] - skel.min[1], ubr, hitr, new_sr, ome)
    _, xt = restore_ea2path(new_bl, new_br, skel.init[1], skel.fin[1], skel.min)
    is_new_t = np.hstack([np.isin(new_bl[0], new_sl)[::-1][:-1], np.isin(new_br[0], new_sr)[1:]])
    return new_bl, new_br, xt[is_new_t]


def restore_ea2path(
    bl: tuple[FloatArr, FloatArr],
    br: tuple[FloatArr, FloatArr],
    init_x: float,
    fin_x: float,
    min_: tuple[float, float],
) -> tuple[FloatArr, FloatArr]:
    
    rl = interpolators.restore_besselbr(*bl, bl[0][-1], init_x - min_[1])
    rr = interpolators.restore_besselbr(*br, br[0][-1], fin_x - min_[1])
    t = np.hstack([-bl[0][1:][::-1], br[0][1:]]) + min_[0]
    x = np.hstack([rl[1:][::-1], rr[1:]]) + min_[1]
    return t, x


def bound_disc_path(
    skel: Skeleton,
    eval_bounds_disc: Callable[[float, float, float, float], tuple[float, float]],
) -> tuple[float, float]:

    return eval_bounds_disc(skel.init[0], skel.fin[0], *bound_skeleton(skel))


def bound_skeleton(skel: Skeleton) -> tuple[float, float]:

    if not skel.flip:
        return skel.ll[0], skel.ul[1]
    return tuple(transforms.reflect(np.array([skel.ll[0], skel.ul[1]]), (skel.init[1] + skel.fin[1]) / 2))[::-1]
