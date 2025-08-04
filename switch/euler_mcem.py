from typing import Callable, Iterator, NamedTuple

import numpy as np
import numpy.typing as npt
import scipy.optimize

import switch.ea1lib.skeleton
import switch.misc.adapt
import switch.mjplib.inference
import switch.mjplib.skeleton
from switch import euler_mcmc, factory, mcem, types


FloatArr = npt.NDArray[np.floating]


class Controls(NamedTuple):
    bound_buffer: float = 1e-9
    init_scale_h: float = 0.0
    imputation_rate: float = 1.0
    learning_rate: float = 1.5
    n_init: int = 10000
    n_thin_mcmc: int = 1
    opt_acc_prob: float = 0.234
    use_barker: bool = False


def maximize_posterior(
    t: FloatArr, 
    vt: FloatArr, 
    n_regimes: int, 
    a0: float, 
    b0: float, 
    mod: factory.SwitchingModel, 
    ome: np.random.Generator, 
    **kwargs,
) -> Iterator[tuple[float, FloatArr, FloatArr, 
                    list[types.Scaffold], 
                    list[list[switch.ea1lib.skeleton.Skeleton]], 
                    list[list[tuple[FloatArr, float]]]]]:

    ctrl = Controls(**kwargs)
    hyper_lam = types.construct_gen_hyperprior(n_regimes, a0, b0)
    lam = switch.mjplib.inference.get_ev(*hyper_lam)
    thi = np.repeat(mod.def_thi[np.newaxis], lam.shape[0], axis=0)
    y = switch.mjplib.skeleton.sample_forwards(t[0], t[-1], None, lam, ome)
    hh, zz = zip(mod.sample_aug1(thi, y, t, vt, np.array([]), ome))
    pp = [euler_mcmc.update_imputation_times(h_, ctrl.imputation_rate, ome) for h_ in hh]
    node_sampler = switch.misc.adapt.MyopicSkelSampler(t[1:-1], np.repeat(ctrl.init_scale_h, len(t) - 2), ctrl.opt_acc_prob)
    return resume_maxim(thi, lam, hh, zz, pp, t, vt, hyper_lam, node_sampler, mod, ctrl, ome)


def resume_maxim(
    thi: FloatArr, 
    lam: FloatArr, 
    hh: list[types.Scaffold], 
    zz: list[list[switch.ea1lib.skeleton.Skeleton]], 
    pp: list[list[tuple[FloatArr, float]]],
    t: FloatArr, 
    vt: FloatArr, 
    hyper_lam: tuple[FloatArr, FloatArr], 
    node_sampler: switch.misc.adapt.MyopicSkelSampler, 
    mod: factory.SwitchingModel, 
    ctrl: Controls, 
    ome: np.random.Generator,
) -> Iterator[tuple[float, FloatArr, FloatArr, 
                    list[types.Scaffold], 
                    list[list[switch.ea1lib.skeleton.Skeleton]], 
                    list[list[tuple[FloatArr, float]]]]]:

    while True:
        n_particles = int((len(hh) ** (1 / ctrl.learning_rate) + 1) ** ctrl.learning_rate) if ctrl.learning_rate > 0 else len(hh)
        obj, thi, lam, hh, zz, pp = update_joint(thi, lam, hh[-1], zz[-1], pp[-1], n_particles, t, vt, hyper_lam, node_sampler, 
                                                 mod, ctrl, ome)
        yield obj, thi, lam, hh, zz, pp


def update_joint(
    thi: FloatArr, 
    lam: FloatArr, 
    h: types.Scaffold, 
    z: list[switch.ea1lib.skeleton.Skeleton], 
    psi: list[tuple[FloatArr, float]],
    n_particles: int, 
    t: FloatArr, 
    vt: FloatArr, 
    hyper_lam: tuple[FloatArr, FloatArr], 
    node_sampler: switch.misc.adapt.MyopicSkelSampler, 
    mod: factory.SwitchingModel, 
    ctrl: Controls, 
    ome: np.random.Generator,
) -> tuple[float, FloatArr, FloatArr, 
           list[types.Scaffold], 
           list[list[switch.ea1lib.skeleton.Skeleton]], 
           list[list[tuple[FloatArr, float]]]]:

    hh, zz, pp = update_particles(thi, lam, h, z, psi, n_particles, t, vt, node_sampler, mod, ctrl, ome)
    thi, obj_thi = update_param(thi, hh, zz, pp, mod, ctrl, ome)
    lam, obj_lam = mcem.update_generator(hh, hyper_lam)
    stat = np.mean([switch.mjplib.skeleton.est_stat(types.prune_anchorage(h_), np.arange(thi.shape[0])) for h_ in hh], 0)
    ymax = np.argmax(stat)
    for i in range(len(stat)):
        if stat[i] == 0:
            thi[i] = thi[ymax]
            lam[i] = lam[ymax]
            lam[i, i], lam[i, ymax] = lam[i, ymax], lam[i, i]
    for i in range(len(stat)):
        if np.any(lam[i] == 0):
            lam[i, lam[i] == 0] = np.max(lam[i])
            lam[i, i] -= np.max(lam[i])
    return obj_thi + obj_lam, thi, lam, hh, zz, pp


def update_param(
    thi_nil: FloatArr, 
    hh: list[types.Scaffold], 
    zz: list[list[switch.ea1lib.skeleton.Skeleton]],
    pp: list[list[tuple[FloatArr, float]]],
    mod: factory.SwitchingModel, 
    ctrl: Controls, 
    ome: np.random.Generator,
) -> tuple[FloatArr, float]:

    def f_wrap_obj(thi_: FloatArr, i: int) -> float:
        return f_obj(np.vstack([thi_nil[:i], thi_, thi_nil[i+1:]]), i)

    opt_bounds = mcem.buffer_bounds(mod.bounds_thi[0], mod.bounds_thi[1], len(thi_nil), ctrl.bound_buffer)
    f_obj = update_objective(thi_nil, hh, zz, pp, mod, ctrl, ome)
    
    opt = [scipy.optimize.minimize(f_wrap_obj, thi_nil[i], (i,), jac=False, bounds=list(zip(*opt_bounds)))
           for i in range(thi_nil.shape[0])]
           
    thi_prime = np.array([opt_.x for opt_ in opt])
    obj = -sum([opt_.fun for opt_ in opt])
    return thi_prime, obj


def update_particles(
    thi: FloatArr, 
    lam: FloatArr, 
    init_h: types.Scaffold, 
    init_z: list[switch.ea1lib.skeleton.Skeleton], 
    init_psi: list[tuple[FloatArr, float]],
    n_particles: int, 
    t: FloatArr,
    vt: FloatArr, 
    node_sampler: switch.misc.adapt.MyopicSkelSampler, 
    mod: factory.SwitchingModel, 
    ctrl: Controls, 
    ome: np.random.Generator,
) -> tuple[list[types.Scaffold], 
           list[list[switch.ea1lib.skeleton.Skeleton]], 
           list[list[tuple[FloatArr, float]]]]:

    h, z, psi, hh, zz, pp = init_h, init_z, init_psi, [], [], []
    mcmc_ctrl = euler_mcmc.Controls(opt_acc_prob=ctrl.opt_acc_prob, use_barker=True)
    for i in range(n_particles * ctrl.n_thin_mcmc):
        h, z, psi = euler_mcmc.update_hidden(thi, lam, h, z, psi, t, vt, node_sampler, mod, mcmc_ctrl, ome)
        if not i % ctrl.n_thin_mcmc:
            hh.append(h)
            zz.append(z)
            pp.append(psi)
    return hh, zz, pp


def update_objective(
    thi: FloatArr,
    hh: list[types.Scaffold],
    zz: list[list[switch.ea1lib.skeleton.Skeleton]],
    pp: list[list[tuple[FloatArr, float]]],
    mod: factory.SwitchingModel,
    ctrl: Controls,
    ome: np.random.Generator,
) -> Callable[[FloatArr, int], float]:
                     
    def f_obj(thi_prime: FloatArr, restrict: int) -> float:
        loglik, _, _ = zip(*[euler_mcmc.eval_loglik(thi_prime, h, z, psi, mod, ome, restrict) 
                                       for h, z, psi in zip(hh, new_zz, pp)])
        return -float(np.mean(loglik))

    new_zz = zz
    for i in range(thi.shape[0]):
        _, new_zz, _ = zip(*[euler_mcmc.eval_loglik(thi, h, z, psi, mod, ome, i) for h, z, psi in zip(hh, new_zz, pp)])

    return f_obj
