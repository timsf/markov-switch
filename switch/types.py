from typing import NamedTuple

import numpy as np
import numpy.typing as npt

import switch.ea3lib.skeleton
import switch.mjplib.skeleton


IntArr = npt.NDArray[np.integer]
FloatArr = npt.NDArray[np.floating]


class Scaffold(NamedTuple):
    t: FloatArr
    yt: IntArr
    vt: FloatArr


def prune_anchorage(h: Scaffold) -> switch.mjplib.skeleton.Skeleton:

    is_self_trans = np.hstack([False, np.diff(h.yt) == 0])
    return switch.mjplib.skeleton.Skeleton(h.t[~is_self_trans], h.yt[~is_self_trans], h.t[-1])


def split_anchorage(h: Scaffold, new_t: FloatArr) -> list[Scaffold]:

    break_r = [0] + list(np.where(np.isin(h.t[1:-1], new_t))[0] + 1) + [len(h.t) - 1]
    h_part = [Scaffold(h.t[r0:r1+1], h.yt[r0:r1+1], h.vt[r0:r1+1])
              for r0, r1 in zip(break_r, break_r[1:])]
    return h_part


def paste_anchorage(h_part: list[Scaffold]) -> Scaffold:

    h = h_part[0]
    for h_ in h_part[1:]:
        h = Scaffold(np.append(h.t, h_.t[1:]), np.append(h.yt, h_.yt[1:]), np.append(h.vt, h_.vt[1:]))
    return h


def split_hidden(
    h: Scaffold, z: list[switch.ea3lib.skeleton.Skeleton], new_t: FloatArr,
) -> tuple[list[Scaffold], list[list[switch.ea3lib.skeleton.Skeleton]]]:

    break_r = [0] + list(np.where(np.isin(h.t[1:-1], new_t))[0] + 1) + [len(h.t) - 1]
    z_part = [z[r0:r1] for r0, r1 in zip(break_r, break_r[1:])]
    return split_anchorage(h, new_t), z_part


def paste_hidden(
    h_part: list[Scaffold], z_part: list[list[switch.ea3lib.skeleton.Skeleton]],
) -> tuple[Scaffold, list[switch.ea3lib.skeleton.Skeleton]]:

    return paste_anchorage(h_part), sum(z_part, [])


def construct_gen_hyperprior(
    n_regimes: int, alp: float, bet: float,
) -> tuple[FloatArr, FloatArr]:

    ones = np.ones((n_regimes, n_regimes)) + np.diag(np.repeat(np.nan, n_regimes))
    return (alp * ones, bet * ones)