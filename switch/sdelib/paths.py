import numpy as np
import numpy.typing as npt

from switch.sdelib import paths, bessel
from switch.misc.exceptions import BudgetConstraintError


BoolArr = npt.NDArray[np.bool_]
FloatArr = npt.NDArray[np.floating]


""""""
def sample_wiener(t: FloatArr, init_x: float, mu: float, ome: np.random.Generator) -> FloatArr:

    if len(t) == 0:
        return np.empty(0)
    dt = np.ediff1d(t, to_begin=(t[0],))
    dxt = np.sqrt(dt) * ome.standard_normal(size=len(t))
    if mu is not None:
        dxt += dt * mu
    xt = np.cumsum(dxt) + init_x
    return xt


""""""
def sample_brownbr(
    t: FloatArr, 
    fin_t: float, 
    init_x: float, 
    fin_x: float,
    ome: np.random.Generator,
) -> FloatArr:

    std_t = t / fin_t
    std_init_x = init_x / np.sqrt(fin_t)
    std_fin_x = fin_x / np.sqrt(fin_t)
    std_xt = (1 - std_t) * sample_wiener(std_t / (1 - std_t), std_init_x, std_fin_x, ome)
    xt = np.sqrt(fin_t) * std_xt
    return xt


""""""
def sample_wienervec(t: FloatArr, init_x: FloatArr, mu: FloatArr, ome: np.random.Generator) -> FloatArr:

    dt = np.ediff1d(t, to_begin=(t[0],))
    dxt = np.sqrt(dt) * ome.standard_normal((len(init_x), len(t)))
    if mu is not None:
        dxt += dt * np.expand_dims(mu, 1)
    xt = np.empty_like(dxt)
    for i in range(len(init_x)):
        xt[i] = np.cumsum(dxt[i]) + init_x[i]
    return xt


""""""
def sample_brownbrvec(
    t: FloatArr, 
    fin_t: float, 
    init_x: FloatArr, 
    fin_x: FloatArr,
    ome: np.random.Generator,
) -> FloatArr:

    std_t = t / fin_t
    std_init_x = init_x / np.sqrt(fin_t)
    std_fin_x = fin_x / np.sqrt(fin_t)
    std_xt = (1 - std_t) * sample_wienervec(std_t / (1 - std_t), std_init_x, std_fin_x, ome)
    xt = np.sqrt(fin_t) * std_xt
    return xt


""""""
def sample_layerbr(
    t: FloatArr, 
    fin_t: float, 
    init_x: float, 
    fin_x: float, 
    inf_x: float,
    ub_sup_x: float | None, 
    lb_sup_x: float | None,
    ome: np.random.Generator,
    max_props: int = int(1e9),
) -> tuple[FloatArr, BoolArr | None]:

    for _ in range(max_props):
        xt, hit_i = propose_layerbr(t, fin_t, init_x, fin_x, inf_x, ub_sup_x, lb_sup_x, ome)
        if lb_sup_x is None:
            return xt, None
        if np.any(hit_i):
            return xt, hit_i
    else:
        raise BudgetConstraintError('None of the proposals were accepted.')


""""""
def sample_besselbr(
    t: FloatArr, 
    fin_t: float, 
    fin_x: float, 
    dim: int, 
    ome: np.random.Generator,
) -> FloatArr:

    xt = np.square(paths.sample_brownbr(t, fin_t, 0, fin_x, ome))
    for _ in range(dim - 1):
        xt += np.square(paths.sample_brownbr(t, fin_t, 0, 0, ome))
    return np.sqrt(xt)


""""""
def propose_layerbr_edge(
    t: FloatArr, 
    fin_t: float, 
    fin_x: float, 
    ub_sup_x: float | None, 
    lb_sup_x: float | None,
    ome: np.random.Generator,
    max_props: int = int(1e9),
) -> tuple[FloatArr, BoolArr]:

    for _ in range(max_props):
        if len(t) == 0:
            xt = np.array((), dtype=np.float64)
        else:
            xt = sample_besselbr(t, fin_t, fin_x, 3, ome)
        t_, xt_ = np.hstack((np.zeros(1), t, np.array([fin_t]))), np.hstack((np.zeros(1), xt, np.array([fin_x])))
        if ub_sup_x is None or not np.any(np.bool_([bessel.sample_bessel3br_esc(dt, x0, x1, ub_sup_x, None, None, ome)
                                                    for dt, x0, x1 in zip(np.diff(t_), xt_, xt_[1:])])):
            if lb_sup_x is None:
                return xt, np.array((), dtype=np.bool_)
            else:
                return xt, np.array([bessel.sample_bessel3br_esc(dt, x0, x1, lb_sup_x, ub_sup_x, None, ome)
                                     for dt, x0, x1 in zip(np.diff(t_), xt_, xt_[1:])], dtype=np.bool_)
    else:
        raise BudgetConstraintError('None of the proposals were accepted.')


""""""
def propose_layerbr(
    t: FloatArr, 
    fin_t: float, 
    init_x: float, 
    fin_x: float,
    inf_x: float, 
    ub_sup_x: float | None, 
    lb_sup_x: float | None,
    ome: np.random.Generator,
) -> tuple[FloatArr, BoolArr]:

    # handle edge minimum case
    if inf_x == init_x or inf_x == fin_x:
        if lb_sup_x is not None:
            lb_sup_x_ = lb_sup_x - inf_x
        else:
            lb_sup_x_ = None
        if ub_sup_x is not None:
            ub_sup_x_ = ub_sup_x - inf_x
        else:
            ub_sup_x_ = ub_sup_x_ = None
        if inf_x == fin_x:
            xt, hit_i = propose_layerbr_edge(fin_t - t[::-1], fin_t, init_x - inf_x, ub_sup_x_, lb_sup_x_, ome)
            return xt[::-1] + inf_x, hit_i[::-1]
        else:
            xt, hit_i = propose_layerbr_edge(t, fin_t, fin_x - inf_x, ub_sup_x_, lb_sup_x_, ome)
            return xt + inf_x, hit_i

    # handle exterior minimum case
    else:
        min_t, min_x = bessel.sample_brownbr_ub_min(fin_t, init_x, fin_x, ub_sup_x, inf_x, None, ome)
        t0, t1 = np.split(t, np.searchsorted(t, (min_t,)))
        x0, i0 = propose_layerbr(t0, min_t, init_x, min_x, min_x, ub_sup_x, lb_sup_x, ome)
        x1, i1 = propose_layerbr(t1 - min_t, fin_t - min_t, min_x, fin_x, min_x, ub_sup_x, lb_sup_x, ome)
        xt = np.hstack((x0, x1))
        if lb_sup_x is None:
            return xt, np.array((), dtype=np.bool_)
        return xt, np.hstack((i0[:-1], np.array([i0[-1] or i1[0]]), i1[1:]))
