from itertools import repeat

import numpy as np
import numpy.typing as npt

import switch.ea3lib.skeleton
import switch.mjplib.skeleton
from switch import factory


FloatArr = npt.NDArray[np.floating]


def sample_forward_euler(
    thi: FloatArr,
    fin_t: float,
    init_y: int,
    init_v: float,
    mod: factory.IntractableModel,
    ome: np.random.Generator,
    n_steps: int,
) -> float:

    x = list(mod.eval_eta(thi[[init_y]].T, np.array([init_v])))
    for dt in np.diff(np.linspace(0, fin_t, n_steps + 1)):
        x.append(ome.normal(x[-1] + dt * mod.eval_dil(thi[init_y].T, x[-1]), np.sqrt(dt) * mod.eval_rho(thi[init_y].T)))
    return mod.eval_ieta(thi[[init_y]].T, np.array([x[-1]]))[0]


def sample_forward(
    thi: FloatArr,
    fin_t: float,
    init_y: int,
    init_v: float,
    mod: factory.IntractableModel,
    ome: np.random.Generator,
    n_steps: int = 1,
) -> float:
    
    init_x, = mod.eval_eta(thi[init_y], np.array([init_v]))
    fin_v = 0
    for _ in range(n_steps):
        fin_x = 0
        for fin_x in mod.sample_conv_endpt(thi, fin_t / n_steps, init_y, init_x, ome):
            fin_v, = mod.eval_ieta(thi[init_y], np.array([fin_x]))
            eval_phi, eval_bounds_phi = mod.bind_params_ext(thi, np.array([0, fin_t / n_steps]), np.array([init_y, init_y]), np.array([init_v, fin_v]))[0]
            gmin_phi = mod.eval_global_min_phi(thi[init_y])
            try:
                next(switch.ea3lib.skeleton.sample_skeleton(fin_t, 0, repeat(0), eval_phi, eval_bounds_phi, gmin_phi, ome, 1))
                break
            except switch.ea3lib.skeleton.BudgetConstraintError:
                continue
        init_x = fin_x
    return fin_v


def sample_bridge(
    thi: FloatArr,
    t: FloatArr,
    fin_t: float,
    init_y: int,
    init_v: float,
    fin_v: float,
    mod: factory.IntractableModel,
    ome: np.random.Generator,
) -> FloatArr:
    
    init_x, fin_x = mod.eval_eta(thi[init_y], np.array([init_v, fin_v]))    
    eval_phi, eval_bounds_phi = mod.bind_params_ext(thi, np.array([0, fin_t]), np.array([init_y, init_y]), np.array([init_v, fin_v]))[0]
    gmin_phi = mod.eval_global_min_phi(thi[init_y])
    z = next(switch.ea3lib.skeleton.sample_skeleton(fin_t, 0, repeat(0), eval_phi, eval_bounds_phi, gmin_phi, ome))
    return mod.eval_ieta(thi[init_y], mod.denormalize(thi[init_y], fin_t, init_x, fin_x, t, switch.ea3lib.skeleton.interpolate_skel(z, t, ome)[1]))


def sample_mssde_forward(
    thi: FloatArr,
    lam: FloatArr,
    fin_t: float,
    init_y: int,
    init_v: float,
    mod: factory.IntractableModel,
    ome: np.random.Generator,
    n_steps: int = 1
) -> float:

    y = switch.mjplib.skeleton.sample_forwards(0.0, fin_t, init_y, lam, ome)
    if n_steps != 1:
        steps = np.linspace(0, fin_t, n_steps + 1)[1:-1]
        i_steps = np.searchsorted(y.t, steps)
        y = switch.mjplib.skeleton.Skeleton(np.insert(y.t, i_steps, steps), np.insert(y.xt, i_steps, y(steps)), y.fin_t)

    vt = [init_v]
    for dt, y0 in zip(np.diff(np.append(y.t, y.fin_t)), y.xt):
        vt.append(sample_forward(thi, dt, y0, vt[-1], mod, ome))
    return vt[-1]
