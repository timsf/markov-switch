from functools import partial
from typing import Iterator, NamedTuple

import numpy as np
import numpy.typing as npt

import switch.ea1lib.ncp
import switch.ea1lib.skeleton
import switch.misc.adapt
import switch.mjplib.inference
import switch.mjplib.skeleton
from switch import factory, mcmc, types


FloatArr = npt.NDArray[np.floating]


class Controls(NamedTuple):
    imputation_rate: float = 0.0
    init_scale_thi: float = 0.0
    init_scale_h: float = 0.0
    n_init: int = 10000
    opt_acc_prob: float = 0.234
    use_barker: bool = False


def sample_posterior(
    t: FloatArr,
    vt: FloatArr,
    n_regimes: int,
    a0: float,
    b0: float,
    mod: factory.SwitchingModel,
    ome: np.random.Generator,
    **kwargs,
) -> Iterator[tuple[FloatArr, FloatArr, types.Scaffold, list[switch.ea1lib.skeleton.Skeleton], list[tuple[FloatArr, float]]]]:

    ctrl = Controls(**kwargs)
    hyper_lam = types.construct_gen_hyperprior(n_regimes, a0, b0)
    lam = switch.mjplib.inference.get_ev(*hyper_lam)
    thi = np.repeat(mod.def_thi[np.newaxis], lam.shape[0], axis=0)
    y = switch.mjplib.skeleton.sample_forwards(t[0], t[-1], None, lam, ome)
    h, z = mod.sample_aug1(thi, y, t, vt, np.array([]), ome)
    psi = update_imputation_times(h, ctrl.imputation_rate, ome)
    thi_samplers = [switch.misc.adapt.MyopicRwSampler(np.zeros(len(thi_)), np.identity(len(thi_)), ctrl.init_scale_thi, mod.bounds_thi, ctrl.opt_acc_prob)
                    for thi_ in thi]
    node_sampler = switch.misc.adapt.MyopicSkelSampler(t[1:-1], np.repeat(ctrl.init_scale_h, len(t) - 2), ctrl.opt_acc_prob)
    thi, lam, h, z, psi, thi_samplers, node_sampler = pre_adapt(thi, lam, h, z, psi, t, vt, hyper_lam, thi_samplers,
                                                                node_sampler, mod, ctrl, ome)
    return resume_sampling(thi, lam, h, z, psi, t, vt, hyper_lam, thi_samplers, node_sampler, mod, ctrl, ome)


def pre_adapt(
    thi: FloatArr,
    lam: FloatArr,
    h: types.Scaffold,
    z: list[switch.ea1lib.skeleton.Skeleton],
    psi: list[tuple[FloatArr, float]],
    t: FloatArr,
    vt: FloatArr,
    hyper_lam: tuple[FloatArr, FloatArr],
    thi_samplers: list[switch.misc.adapt.MyopicRwSampler],
    node_sampler: switch.misc.adapt.MyopicSkelSampler,
    mod: factory.SwitchingModel,
    ctrl: Controls,
    ome: np.random.Generator,
) -> tuple[FloatArr, FloatArr, types.Scaffold, list[switch.ea1lib.skeleton.Skeleton], list[tuple[FloatArr, float]],
           list[switch.misc.adapt.MyopicRwSampler], switch.misc.adapt.MyopicSkelSampler]:

    adapt_ctrl = ctrl._asdict()
    adapt_ctrl['imputation_rate'] = 0.0
    sampler = resume_sampling(thi, lam, h, z, psi, t, vt, hyper_lam, thi_samplers, node_sampler,
                              mod, Controls(**adapt_ctrl), ome)
    for _ in range(ctrl.n_init):
        thi, lam, h, z, psi = next(sampler)
    return thi, lam, h, z, psi, [param_sampler.reset() for param_sampler in thi_samplers], node_sampler.reset()


def resume_sampling(
    thi: FloatArr,
    lam: FloatArr,
    h: types.Scaffold,
    z: list[switch.ea1lib.skeleton.Skeleton],
    psi: list[tuple[FloatArr, float]],
    t: FloatArr,
    vt: FloatArr,
    hyper_lam: tuple[FloatArr, FloatArr],
    thi_samplers: list[switch.misc.adapt.MyopicRwSampler],
    node_sampler: switch.misc.adapt.MyopicSkelSampler,
    mod: factory.SwitchingModel,
    ctrl: Controls,
    ome: np.random.Generator,
) -> Iterator[tuple[FloatArr, FloatArr, types.Scaffold, list[switch.ea1lib.skeleton.Skeleton], list[tuple[FloatArr, float]]]]:

    while True:
        thi, lam, h, z, psi = update_joint(thi, lam, h, z, psi, t, vt, hyper_lam,
                                           thi_samplers, node_sampler, mod, ctrl, ome)
        yield thi, lam, h, z, psi


def update_joint(
    thi: FloatArr,
    lam: FloatArr,
    h: types.Scaffold,
    z: list[switch.ea1lib.skeleton.Skeleton],
    psi: list[tuple[FloatArr, float]],
    t: FloatArr,
    vt: FloatArr,
    hyper_lam: tuple[FloatArr, FloatArr],
    thi_samplers: list[switch.misc.adapt.MyopicRwSampler],
    node_sampler: switch.misc.adapt.MyopicSkelSampler,
    mod: factory.SwitchingModel,
    ctrl: Controls,
    ome: np.random.Generator,
) -> tuple[FloatArr, FloatArr, types.Scaffold, list[switch.ea1lib.skeleton.Skeleton], list[tuple[FloatArr, float]]]:

    stat = switch.mjplib.skeleton.est_stat(types.prune_anchorage(h), np.arange(thi.shape[0]))
    ymax = np.argmax(stat)
    for i in range(len(stat)):
        if stat[i] == 0:
            thi[i] = thi[ymax]
    lam = mcmc.update_generator(h, hyper_lam, ome)
    thi, z, psi = update_params(thi, h, z, psi, thi_samplers, mod, ctrl, ome)
    h, z, psi = update_hidden(thi, lam, h, z, psi, t, vt, node_sampler, mod, ctrl, ome)
    return thi, lam, h, z, psi


def update_params(
    thi_nil: FloatArr,
    h: types.Scaffold,
    z: list[switch.ea1lib.skeleton.Skeleton],
    psi: list[tuple[FloatArr, float]],
    thi_samplers: list[switch.misc.adapt.MyopicRwSampler],
    mod: factory.SwitchingModel,
    ctrl: Controls,
    ome: np.random.Generator,
) -> tuple[FloatArr, list[switch.ea1lib.skeleton.Skeleton], list[tuple[FloatArr, float]]]:

    acc_part, thi_acc_part, z_part, psi_part = zip(*[
        update_params_section(thi_nil, h, z, psi, thi_samplers, mod, ctrl, i, ome)
        for i in range(thi_nil.shape[0])])

    thi_acc = np.vstack(thi_acc_part)
    new_z = [z_part[y0][i] for i, y0 in enumerate(h.yt[:-1])]
    new_psi = [psi_part[y0][i] for i, y0 in enumerate(h.yt[:-1])]
    for i in range(thi_nil.shape[0]):
        thi_samplers[i].adapt(thi_acc[i], float(acc_part[i]))
    return thi_acc, new_z, new_psi


def update_params_section(
    thi_nil: FloatArr,
    h: types.Scaffold,
    z: list[switch.ea1lib.skeleton.Skeleton],
    psi: list[tuple[FloatArr, float]],
    thi_samplers: list[switch.misc.adapt.MyopicRwSampler],
    mod: factory.SwitchingModel,
    ctrl: Controls,
    state: int,
    ome: np.random.Generator,
) -> tuple[float, FloatArr, list[switch.ea1lib.skeleton.Skeleton], list[tuple[FloatArr, float]]]:

    prop, log_q_forw, log_q_back = thi_samplers[state].propose(thi_nil[state], ome)
    thi_prime = thi_nil.copy()
    thi_prime[state] = prop
    if not np.all((mod.bounds_thi[0] < thi_prime) & (thi_prime < mod.bounds_thi[1])):
        return 0.0, thi_nil[state], z, psi
    new_z, new_psi = z, psi
    log_p_nil, new_z, new_psi = eval_loglik(thi_nil, h, new_z, new_psi, mod, ome, state)
    log_p_prime, new_z, new_psi = eval_loglik(thi_prime, h, new_z, new_psi, mod, ome, state)
    log_odds = (log_p_prime - log_p_nil) - (log_q_forw - log_q_back)
    if ctrl.use_barker:
        log_acc_prob = log_odds - np.logaddexp(0, log_odds)
    else:
        log_acc_prob = min(0, log_odds)
    if np.log(ome.uniform()) < log_acc_prob:
        return np.exp(log_acc_prob), thi_prime[state], new_z, new_psi
    return np.exp(log_acc_prob), thi_nil[state], new_z, new_psi


def update_hidden(
    thi: FloatArr,
    lam: FloatArr,
    h_nil: types.Scaffold,
    z_nil: list[switch.ea1lib.skeleton.Skeleton],
    psi_nil: list[tuple[FloatArr, float]],
    t: FloatArr,
    vt: FloatArr,
    node_sampler: switch.misc.adapt.MyopicSkelSampler,
    mod: factory.SwitchingModel,
    ctrl: Controls,
    ome: np.random.Generator,
) -> tuple[types.Scaffold, list[switch.ea1lib.skeleton.Skeleton], list[tuple[FloatArr, float]]]:

    y_nil = types.prune_anchorage(h_nil)
    t_node_nil = np.setdiff1d(h_nil.t, y_nil.t)
    t_node_prime = node_sampler.propose(ome)
    t_node = np.intersect1d(t_node_nil, t_node_prime)
    y_prime = switch.mjplib.skeleton.sample_leapfrog(y_nil, t_node, lam, ome)
    t_cond = np.union1d(t, t_node)
    vt_cond = h_nil.vt[np.isin(h_nil.t, t_cond)]
    h_prime, z_prime = mod.sample_aug1(
        thi, y_prime, t_cond, vt_cond, np.setdiff1d(t_node_prime, t_cond), ome)
    psi_prime = update_imputation_times(h_prime, ctrl.imputation_rate, ome)
    t_split = np.union1d(t_node, t[1:-1][y_nil(t[1:-1]) == y_prime(t[1:-1])])
    h_nil_part, z_nil_part, psi_nil_part = split_hidden(
        h_nil, z_nil, psi_nil, t_split)
    h_prime_part, z_prime_part, psi_prime_part = split_hidden(
        h_prime, z_prime, psi_prime, t_split)

    acc_part, h_acc_part, z_acc_part, psi_acc_part = zip(*[
        update_hidden_section(thi, h_nil_, z_nil_, psi_nil_,
                              h_prime_, z_prime_, psi_prime_, mod, ctrl, ome)
        for h_nil_, h_prime_, z_nil_, z_prime_, psi_nil_, psi_prime_
        in zip(h_nil_part, h_prime_part, z_nil_part, z_prime_part, psi_nil_part, psi_prime_part)])

    h_acc, z_acc, psi_acc = paste_hidden(h_acc_part, z_acc_part, psi_acc_part)
    node_sampler.adapt(np.hstack([t[0], t_split, t[-1]]), np.array(acc_part))
    return h_acc, z_acc, psi_acc


def update_hidden_section(
    thi: FloatArr,
    h_nil: types.Scaffold,
    z_nil: list[switch.ea1lib.skeleton.Skeleton],
    psi_nil: list[tuple[FloatArr, float]],
    h_prime: types.Scaffold,
    z_prime: list[switch.ea1lib.skeleton.Skeleton],
    psi_prime: list[tuple[FloatArr, float]],
    mod: factory.SwitchingModel,
    ctrl: Controls,
    ome: np.random.Generator,
) -> tuple[float, types.Scaffold, list[switch.ea1lib.skeleton.Skeleton], list[tuple[FloatArr, float]]]:

    if not np.all((mod.bounds_v[0] < h_prime.vt) & (h_prime.vt < mod.bounds_v[1])):
        return 0.0, h_nil, z_nil, psi_nil
    is_data = np.intersect1d(h_nil.t, h_prime.t)
    log_q_forw = mod.eval_log_prop(thi, *h_prime, np.isin(h_prime.t, is_data))
    log_q_back = mod.eval_log_prop(thi, *h_nil, np.isin(h_nil.t, is_data))
    log_p_nil, new_z_nil, new_psi_nil = eval_loglik(thi, h_nil, z_nil, psi_nil, mod, ome)
    log_p_prime, new_z_prime, new_psi_prime = eval_loglik(thi, h_prime, z_prime, psi_prime, mod, ome)
    log_odds = (log_p_prime - log_p_nil) - (log_q_forw - log_q_back)
    if ctrl.use_barker:
        log_acc_prob = log_odds - np.logaddexp(0, log_odds)
    else:
        log_acc_prob = min(0, log_odds)
    if np.log(ome.uniform()) < log_acc_prob:
        return np.exp(log_acc_prob), h_prime, new_z_prime, new_psi_prime
    return np.exp(log_acc_prob), h_nil, new_z_nil, new_psi_nil


def update_imputation_times(
    h: types.Scaffold,
    rate: float,
    ome: np.random.Generator,
) -> list[tuple[FloatArr, float]]:

    psi = np.linspace(h.t[0], h.t[-1], int(rate * (h.t[-1] - h.t[0])))
    psi = np.split(psi[~np.isin(psi, h.t)], np.searchsorted(psi[~np.isin(psi, h.t)], h.t[1:-1]))
    psi = [(t_ - t0, 0.0) for t0, t_ in zip(h.t, psi)]
    return psi


def eval_loglik(
    thi: FloatArr,
    h: types.Scaffold,
    z: list[switch.ea1lib.skeleton.Skeleton],
    psi: list[tuple[FloatArr, float]],
    mod: factory.SwitchingModel,
    ome: np.random.Generator,
    state: int | None = None,
) -> tuple[float, list[switch.ea1lib.skeleton.Skeleton], list[tuple[FloatArr, float]]]:

    include = np.repeat(True, len(h.t)) if state is None else h.yt == state
    xt = mod.eval_eta(thi[h.yt].T, h.vt)
    denorm = [partial(mod.denormalize, thi[y0], dt, x0, x1) for dt, y0, x0, x1 in zip(np.diff(h.t), h.yt, xt, xt[1:])]
    new_z, new_x = zip(*[switch.ea1lib.ncp.interpolate_skel(z_, t_, f_, ome) if (i and len(t_) != 0) else (z_, np.array([]))
                         for i, z_, (t_, _), f_
                         in zip(include, z, psi, denorm)])
    t = np.hstack([np.append(t0, t_ + t0) if i else t0
                   for i, t0, (t_, _) in zip(include, h.t, psi)] + [h.t[-1]])
    yt = np.hstack([np.repeat(y0, len(t_) + 1) if i else y0
                    for i, y0, (t_, _) in zip(include, h.yt, psi)] + [h.yt[-1]])
    vt = np.hstack([np.append(v0, mod.eval_ieta(thi[np.repeat(y0, len(x_))].T, x_)) if i else v0
                    for i, y0, v0, x_ in zip(include, h.yt, h.vt, new_x)] + [h.vt[-1]])
    log_prop = np.sum([eval_brownbr(*z_) if i else 0 for i, z_, (t_, lt_) in zip(include, new_z, psi)])
    log_jac = np.sum([np.sum(np.log(mod.eval_rho(thi[y0]) / np.abs(mod.eval_dv_eta(thi[y0], mod.eval_ieta(thi[y0], x_)))))
                      if (i and len(x_) != 0) else 0 for i, y0, x_ in zip(include, h.yt, new_x)])
    return log_jac + mod.eval_approx_log_lik(thi, t, yt, vt) + mod.eval_log_prior(thi) - log_prop, list(new_z), psi


def split_hidden(
    h: types.Scaffold,
    z: list[switch.ea1lib.skeleton.Skeleton],
    psi: list[tuple[FloatArr, float]],
    new_t: FloatArr,
) -> tuple[list[types.Scaffold], list[list[switch.ea1lib.skeleton.Skeleton]], list[list[tuple[FloatArr, float]]]]:

    break_r = [0] + list(np.where(np.isin(h.t[1:-1], new_t))[0] + 1) + [len(h.t) - 1]
    z_part = [z[r0:r1] for r0, r1 in zip(break_r, break_r[1:])]
    psi_part = [psi[r0:r1] for r0, r1 in zip(break_r, break_r[1:])]
    return types.split_anchorage(h, new_t), z_part, psi_part


def paste_hidden(
    h_part: list[types.Scaffold],
    z_part: list[list[switch.ea1lib.skeleton.Skeleton]],
    psi_part: list[list[tuple[FloatArr, float]]],
) -> tuple[types.Scaffold, list[switch.ea1lib.skeleton.Skeleton], list[tuple[FloatArr, float]]]:

    return types.paste_anchorage(h_part), sum(z_part, []), sum(psi_part, [])


def eval_brownbr(t: FloatArr, xt: FloatArr) -> float:

    dt = np.diff(t)
    dxt = np.diff(xt)
    return (np.log(2 * np.pi * np.sum(dt)) + np.square(np.sum(dxt)) / np.sum(dt)) / 2 \
            - np.sum(np.log(2 * np.pi * dt) + np.square(dxt) / dt) / 2


def eval_poisson(k: int, lam: float) -> float:

    return k * np.log(lam) - lam - np.sum(np.log(np.arange(1, k + 1)))