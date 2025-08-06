from typing import Callable, Iterator, NamedTuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

import switch.ea3lib.skeleton
import switch.mjplib.inference
import switch.mjplib.skeleton
import switch.misc.adapt
from switch import euler_mcem, factory, mcmc, types


FloatArr = npt.NDArray[np.floating]


class Controls(NamedTuple):
    bound_buffer: float = 1e-9
    init_scale_h: float = 0.0
    imputation_rate: float = 1.0
    learning_rate: float = 1.5
    n_init: int = 10
    n_path_segments: int = 1
    n_thin_mcmc: int = 1
    opt_acc_prob: float = 0.2
    pr_portkey: float = 0.01


def maximize_posterior(
    t: FloatArr, 
    vt: FloatArr, 
    n_regimes: int, 
    a0: float, 
    b0: float, 
    mod: factory.IntractableModel, 
    ome: np.random.Generator, 
    **kwargs,
) -> Iterator[tuple[float, FloatArr, FloatArr, 
                    list[types.Scaffold], list[list[switch.ea3lib.skeleton.Skeleton]]]]:

    ctrl = Controls(**kwargs)
    hyper_lam = types.construct_gen_hyperprior(n_regimes, a0, b0)
    lam = switch.mjplib.inference.get_ev(*hyper_lam)
    thi = np.repeat(mod.def_thi[np.newaxis], lam.shape[0], axis=0)
    y = switch.mjplib.skeleton.sample_forwards(t[0], t[-1], None, lam, ome)
    hh = [mod.sample_scaffold(thi, y, t, vt, np.array([]), ome)]
    node_sampler = switch.misc.adapt.MyopicSkelSampler(t[1:-1], np.repeat(ctrl.init_scale_h, len(t) - 2), ctrl.opt_acc_prob)
    thi, lam, hh, node_sampler = pre_adapt(thi, lam, hh, t, vt, hyper_lam, node_sampler, mod, ctrl, ome)
    zz = [mod.sample_bridges3(thi, *h, ctrl.n_path_segments, ome) for h in hh]
    return resume_maxim(thi, lam, hh, zz, t, vt, hyper_lam, node_sampler, mod, ctrl, ome)


def pre_adapt(
    thi: FloatArr, 
    lam: FloatArr, 
    hh: list[types.Scaffold], 
    t: FloatArr, 
    vt: FloatArr,
    hyper_lam: tuple[FloatArr, FloatArr], 
    node_sampler: switch.misc.adapt.MyopicSkelSampler, 
    mod: factory.IntractableModel, 
    ctrl: Controls, 
    ome: np.random.Generator,
) -> tuple[FloatArr, FloatArr, list[types.Scaffold], switch.misc.adapt.MyopicSkelSampler]:

    adapt_ctrl = euler_mcem.Controls(bound_buffer=ctrl.bound_buffer, init_scale_h=ctrl.init_scale_h, imputation_rate=ctrl.imputation_rate,
                                     learning_rate=0, n_thin_mcmc=ctrl.n_thin_mcmc, opt_acc_prob=ctrl.opt_acc_prob,
                                     use_barker=True)
    zz = [mod.sample_bridges1(thi, *h_, ome) for h_ in hh]
    pp = [euler_mcem.euler_mcmc.update_imputation_times(h_, adapt_ctrl.imputation_rate, ome) for h_ in hh]
    estimator = euler_mcem.resume_maxim(thi, lam, hh, zz, pp, t, vt, hyper_lam, node_sampler, 
                                        mod, adapt_ctrl, ome)
    for _ in range(ctrl.n_init):
        _, thi, lam, hh, _, _ = next(estimator)
    return thi, lam, hh[-1:], node_sampler.reset()


def resume_maxim(
    thi: FloatArr, 
    lam: FloatArr, 
    hh: list[types.Scaffold], 
    zz: list[list[switch.ea3lib.skeleton.Skeleton]],
    t: FloatArr, 
    vt: FloatArr, 
    hyper_lam: tuple[FloatArr, FloatArr], 
    node_sampler: switch.misc.adapt.MyopicSkelSampler, 
    mod: factory.IntractableModel, 
    ctrl: Controls, 
    ome: np.random.Generator,
) -> Iterator[tuple[float, FloatArr, FloatArr, 
                    list[types.Scaffold], list[list[switch.ea3lib.skeleton.Skeleton]]]]:

    while True:
        n_particles = int((len(hh) ** (1 / ctrl.learning_rate) + 1) ** ctrl.learning_rate) if ctrl.learning_rate > 0 else len(hh)
        obj, thi, lam, hh, zz = update_joint(thi, lam, hh[-1], zz[-1], n_particles, t, vt, hyper_lam, node_sampler, 
                                             mod, ctrl, ome)
        yield obj, thi, lam, hh, zz


def update_joint(
    thi: FloatArr, 
    lam: FloatArr, 
    h: types.Scaffold, 
    z: list[switch.ea3lib.skeleton.Skeleton], 
    n_particles: int, 
    t: FloatArr, 
    vt: FloatArr, 
    hyper_lam: tuple[FloatArr, FloatArr], 
    node_sampler: switch.misc.adapt.MyopicSkelSampler, 
    mod: factory.IntractableModel, 
    ctrl: Controls, 
    ome: np.random.Generator,
) -> tuple[float, FloatArr, FloatArr, list[types.Scaffold], list[list[switch.ea3lib.skeleton.Skeleton]]]:

    hh, zz = update_particles(thi, lam, h, z, n_particles, t, vt, node_sampler, mod, ctrl, ome)
    thi, obj_thi = update_param(thi, hh, zz, mod, ctrl, ome)
    lam, obj_lam = update_generator(hh, hyper_lam)
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
    return obj_thi + obj_lam, thi, lam, hh, zz


def update_param(
    thi_nil: FloatArr, 
    hh: list[types.Scaffold], 
    zz: list[list[switch.ea3lib.skeleton.Skeleton]],
    mod: factory.IntractableModel, 
    ctrl: Controls, 
    ome: np.random.Generator,
) -> tuple[FloatArr, float]:

    def f_wrap_obj(thi_: FloatArr, i: int) -> float: #tuple[float, FloatArr]:
        return f_obj(np.vstack([thi_nil[:i], thi_, thi_nil[i+1:]]), i)

    opt_bounds = buffer_bounds(mod.bounds_thi[0], mod.bounds_thi[1], len(thi_nil), ctrl.bound_buffer)
    f_obj = update_objective(thi_nil, hh, zz, mod, ctrl, ome)
    
    opt = [minimize(f_wrap_obj, thi_nil[i], (i,), jac=False, bounds=list(zip(*opt_bounds)))
           for i in range(thi_nil.shape[0])]

    thi_prime = np.array([opt_.x for opt_ in opt])
    obj = -sum([opt_.fun for opt_ in opt])
    return thi_prime, obj


def update_particles(
    thi: FloatArr, 
    lam: FloatArr, 
    init_h: types.Scaffold, 
    init_z: list[switch.ea3lib.skeleton.Skeleton], 
    n_particles: int, 
    t: FloatArr, 
    vt: FloatArr,
    node_sampler: switch.misc.adapt.MyopicSkelSampler, 
    mod: factory.IntractableModel, 
    ctrl: Controls, 
    ome: np.random.Generator,
) -> tuple[list[types.Scaffold], list[list[switch.ea3lib.skeleton.Skeleton]]]:

    h, z, hh, zz = init_h, init_z, [], []
    mcmc_ctrl = mcmc.Controls(n_path_segments=ctrl.n_path_segments, opt_acc_prob=ctrl.opt_acc_prob)
    for i in range(n_particles * ctrl.n_thin_mcmc):
        h, z = mcmc.update_hidden(thi, lam, h, z, t, vt, node_sampler, mod, mcmc_ctrl, ome)
        if not i % ctrl.n_thin_mcmc:
            hh.append(h)
            zz.append(z)
    return hh, zz


def update_objective(
    thi: FloatArr, 
    hh: list[types.Scaffold], 
    zz: list[list[switch.ea3lib.skeleton.Skeleton]],
    mod: factory.IntractableModel, 
    ctrl: Controls, 
    ome: np.random.Generator,
) -> Callable[[FloatArr, int], float]:
    
    def f_obj(thi_prime: FloatArr, state: int) -> float:
        loglik, _ = zip(*[est_loglik(thi_prime, h, z, psi, mod, ome, state) 
                          for h, z, psi in zip(hh, new_zz, pp)])
        return -float(np.nanmean(loglik))

    pp = [euler_mcem.euler_mcmc.update_imputation_times(h_, ctrl.imputation_rate, ome) for h_ in hh]
    new_zz = zz
    for i in range(thi.shape[0]):
        _, new_zz = zip(*[est_loglik(thi, h, z, psi, mod, ome, i) for h, z, psi in zip(hh, new_zz, pp)])

    return f_obj


def update_generator(
    hh: list[types.Scaffold], 
    hyper_lam: tuple[FloatArr, FloatArr],
) -> tuple[FloatArr, float]:

    yy = [types.prune_anchorage(h) for h in hh]
    alp, bet = zip(*[switch.mjplib.inference.update(y, *hyper_lam) for y in yy])
    lam_prime = np.mean([(a - 1) for a in alp], 0) / np.mean([b for b in bet], 0)
    lam_prime[np.isnan(lam_prime)] = -np.nansum(lam_prime, 1)
    log_post = np.mean([switch.mjplib.inference.eval_loglik(y, lam_prime) for y in yy]) \
        + switch.mjplib.inference.eval_logprior(lam_prime, *hyper_lam)
    return lam_prime, float(log_post)


def buffer_bounds(
    lb: FloatArr, 
    ub: FloatArr, 
    n_dim: int, 
    buffer: float = 1e-2,
) -> tuple[FloatArr, FloatArr]:

    lbinf = np.repeat(-np.inf, n_dim) if lb is None else lb
    ubinf = np.repeat(np.inf, n_dim) if ub is None else ub
    return lbinf + buffer, ubinf - buffer


def est_loglik(
    thi: FloatArr, 
    h: types.Scaffold, 
    z: list[switch.ea3lib.skeleton.Skeleton],
    psi: list[tuple[FloatArr, float]],
    mod: factory.IntractableModel,
    ome: np.random.Generator, 
    state: int,
) -> tuple[float, list[switch.ea3lib.skeleton.Skeleton]]:

    include = h.yt == state
    st_disc, new_z = zip(*[switch.ea3lib.skeleton.est_poisson_log(z_, ops_[0], psi_[0], ome) 
                           if i else (0, z_) 
                           for i, z_, psi_, ops_ in zip(include, z, psi, mod.bind_params_ext(thi, *h))])
    return sum(mod.eval_biased_log_lik(thi, *h)) + sum(st_disc) + mod.eval_log_prior(thi), new_z
