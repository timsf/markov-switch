from typing import Callable, Iterator, NamedTuple, TypeVar

import numpy as np
import numpy.typing as npt

import switch.ea3lib.skeleton
import switch.misc.adapt
import switch.misc.bfactories
import switch.mjplib.inference
import switch.mjplib.skeleton
from switch import euler_mcmc, factory, types


FloatArr = npt.NDArray[np.floating]
T = TypeVar('T')


class Controls(NamedTuple):
    init_scale_thi: float = 0.0
    init_scale_h: float = 0.0
    n_dc_splits: int = 2
    n_init: int = 10000
    n_path_segments: int = 1
    opt_acc_prob: float = 0.2
    pr_portkey_thi: float = 0.01
    pr_portkey_h: float = 0.01


def sample_posterior(
    t: FloatArr,
    vt: FloatArr,
    n_regimes: int,
    a0: float,
    b0: float,
    mod: factory.IntractableModel,
    ome: np.random.Generator,
    **kwargs,
) -> Iterator[tuple[FloatArr, FloatArr, types.Scaffold, list[switch.ea3lib.skeleton.Skeleton]]]:

    ctrl = Controls(**kwargs)
    hyper_lam = types.construct_gen_hyperprior(n_regimes, a0, b0)
    lam = switch.mjplib.inference.get_ev(*hyper_lam)
    thi = np.repeat(mod.def_thi[np.newaxis], lam.shape[0], axis=0)
    y = switch.mjplib.skeleton.sample_forwards(t[0], t[-1], None, lam, ome)
    h = mod.sample_scaffold(thi, y, t, vt, np.array([]), ome)
    thi_samplers = [switch.misc.adapt.MyopicRwSampler(np.zeros(len(thi_)), np.identity(len(thi_)), ctrl.init_scale_thi, mod.bounds_thi, ctrl.opt_acc_prob)
                    for thi_ in thi]
    node_sampler = switch.misc.adapt.MyopicSkelSampler(t[1:-1], np.repeat(ctrl.init_scale_h, len(t) - 2), ctrl.opt_acc_prob)
    thi, lam, h, thi_samplers, node_sampler = pre_adapt(thi, lam, h, t, vt, hyper_lam, thi_samplers,
                                                        node_sampler, mod, ctrl, ome)
    z = mod.sample_bridges3(thi, *h, ctrl.n_path_segments, ome)
    return resume_sampling(thi, lam, h, z, t, vt, hyper_lam, thi_samplers, node_sampler, mod, ctrl, ome)


def pre_adapt(
    thi: FloatArr,
    lam: FloatArr,
    h: types.Scaffold,
    t: FloatArr,
    vt: FloatArr,
    hyper_lam: tuple[FloatArr, FloatArr],
    thi_samplers: list[switch.misc.adapt.MyopicRwSampler],
    node_sampler: switch.misc.adapt.MyopicSkelSampler,
    mod: factory.IntractableModel,
    ctrl: Controls,
    ome: np.random.Generator,
) -> tuple[FloatArr, FloatArr, types.Scaffold,
           list[switch.misc.adapt.MyopicRwSampler], switch.misc.adapt.MyopicSkelSampler]:

    adapt_ctrl = euler_mcmc.Controls(use_barker=True, opt_acc_prob=ctrl.opt_acc_prob)
    z = mod.sample_bridges1(thi, *h, ome)
    psi = euler_mcmc.update_imputation_times(h, adapt_ctrl.imputation_rate, ome)
    sampler = euler_mcmc.resume_sampling(thi, lam, h, z, psi, t, vt, hyper_lam, thi_samplers, node_sampler,
                                         mod, adapt_ctrl, ome)
    for _ in range(ctrl.n_init):
        thi, lam, h, z, psi = next(sampler)
    return thi, lam, h, thi_samplers, node_sampler


def resume_sampling(
    thi: FloatArr,
    lam: FloatArr,
    h: types.Scaffold,
    z: list[switch.ea3lib.skeleton.Skeleton],
    t: FloatArr,
    vt: FloatArr,
    hyper_lam: tuple[FloatArr, FloatArr],
    thi_samplers: list[switch.misc.adapt.MyopicRwSampler],
    node_sampler: switch.misc.adapt.MyopicSkelSampler,
    mod: factory.IntractableModel,
    ctrl: Controls,
    ome: np.random.Generator,
) -> Iterator[tuple[FloatArr, FloatArr, types.Scaffold, list[switch.ea3lib.skeleton.Skeleton]]]:

    while True:
        thi, lam, h, z = update_joint(thi, lam, h, z, t, vt, hyper_lam,
                                      thi_samplers, node_sampler, mod, ctrl, ome)
        yield thi, lam, h, z


def update_joint(
    thi: FloatArr,
    lam: FloatArr,
    h: types.Scaffold,
    z: list[switch.ea3lib.skeleton.Skeleton],
    t: FloatArr,
    vt: FloatArr,
    hyper_lam: tuple[FloatArr, FloatArr],
    thi_samplers: list[switch.misc.adapt.MyopicRwSampler],
    node_sampler: switch.misc.adapt.MyopicSkelSampler,
    mod: factory.IntractableModel,
    ctrl: Controls,
    ome: np.random.Generator,
) -> tuple[FloatArr, FloatArr, types.Scaffold, list[switch.ea3lib.skeleton.Skeleton]]:

    stat = switch.mjplib.skeleton.est_stat(types.prune_anchorage(h), np.arange(thi.shape[0]))
    ymax = np.argmax(stat)
    for i in range(len(stat)):
        if stat[i] == 0:
            thi[i] = thi[ymax]
    lam = update_generator(h, hyper_lam, ome)
    thi, z = update_params(thi, h, z, thi_samplers, mod, ctrl, ome)
    h, z = update_hidden(thi, lam, h, z, t, vt, node_sampler, mod, ctrl, ome)
    return thi, lam, h, z


def update_params(
    thi_nil: FloatArr,
    h: types.Scaffold,
    z: list[switch.ea3lib.skeleton.Skeleton],
    thi_samplers: list[switch.misc.adapt.MyopicRwSampler],
    mod: factory.IntractableModel,
    ctrl: Controls,
    ome: np.random.Generator,
) -> tuple[FloatArr, list[switch.ea3lib.skeleton.Skeleton]]:

    acc_part, thi_acc_part, z_part = zip(*[
        update_params_section(thi_nil, h, z, thi_samplers, mod, ctrl, i, ome)
        for i in range(thi_nil.shape[0])])

    thi_acc = np.vstack(thi_acc_part)
    new_z = [z_part[y0][i] for i, y0 in enumerate(h.yt[:-1])]
    for i in range(thi_nil.shape[0]):
        thi_samplers[i].adapt(thi_acc[i], float(acc_part[i]))
    return thi_acc, new_z


def update_params_section(
    thi_nil: FloatArr,
    h: types.Scaffold,
    z: list[switch.ea3lib.skeleton.Skeleton],
    thi_samplers: list[switch.misc.adapt.MyopicRwSampler],
    mod: factory.IntractableModel,
    ctrl: Controls,
    state: int,
    ome: np.random.Generator,
) -> tuple[bool, FloatArr, list[switch.ea3lib.skeleton.Skeleton]]:

    prop, log_q_forw, log_q_back = thi_samplers[state].propose(thi_nil[state], ome)
    thi_prime = thi_nil.copy()
    thi_prime[state] = prop
    if not np.all((mod.bounds_thi[0] < thi_prime) & (thi_prime < mod.bounds_thi[1])):
        return False, thi_nil[state], z

    ops_nil = mod.bind_params_diff(thi_nil, thi_prime, *h)
    ops_prime = mod.bind_params_diff(thi_prime, thi_nil, *h)
    lb_nil, _, coins_nil = zip(*construct_coins(z, ops_nil, ome))
    lb_prime, _, coins_prime = zip(*construct_coins(z, ops_prime, ome))
    weight_nil = eval_weight(thi_nil, h, mod) + (log_q_forw + mod.eval_log_prior(thi_nil)) / len(z) - np.array(lb_nil) * np.diff(h.t)
    weight_prime = eval_weight(thi_prime, h, mod) + (log_q_back + mod.eval_log_prior(thi_prime)) / len(z) - np.array(lb_prime) * np.diff(h.t)
    odds = weight_prime - weight_nil

    try:
        acc, z_acc = next(switch.misc.bfactories.sample_odds_single(z, odds, list(coins_prime), list(coins_nil), ctrl.n_dc_splits, ctrl.pr_portkey_thi, ome))
    except switch.misc.bfactories.BudgetConstraintError:
        acc, z_acc = False, z
    
    if acc:
        return acc, thi_prime[state], z_acc
    return acc, thi_nil[state], z_acc


def update_hidden(
    thi: FloatArr,
    lam: FloatArr,
    h_nil: types.Scaffold,
    z_nil: list[switch.ea3lib.skeleton.Skeleton],
    t: FloatArr,
    vt: FloatArr,
    node_sampler: switch.misc.adapt.MyopicSkelSampler,
    mod: factory.IntractableModel,
    ctrl: Controls,
    ome: np.random.Generator,
) -> tuple[types.Scaffold, list[switch.ea3lib.skeleton.Skeleton]]:

    y_nil = types.prune_anchorage(h_nil)
    t_node_nil = np.setdiff1d(h_nil.t, y_nil.t)
    t_node_prime = node_sampler.propose(ome)
    t_node = np.intersect1d(t_node_nil, t_node_prime)
    y_prime = switch.mjplib.skeleton.sample_leapfrog(y_nil, t_node, lam, ome)
    t_cond = np.union1d(t, t_node)
    vt_cond = h_nil.vt[np.isin(h_nil.t, t_cond)]
    h_prime, z_prime = mod.sample_aug3(thi, y_prime, t_cond, vt_cond, np.setdiff1d(t_node_prime, t_cond), ctrl.n_path_segments, ome)
    t_split = np.union1d(t_node, t[1:-1][y_nil(t[1:-1]) == y_prime(t[1:-1])])
    h_nil_part, z_nil_part = types.split_hidden(h_nil, z_nil, t_split)
    h_prime_part, z_prime_part = types.split_hidden(h_prime, z_prime, t_split)

    acc_part, h_acc_part, z_acc_part = zip(*[
        update_hidden_section(thi, h_nil_, z_nil_, h_prime_, z_prime_, mod, ctrl, ome)
        for h_nil_, h_prime_, z_nil_, z_prime_
        in zip(h_nil_part, h_prime_part, z_nil_part, z_prime_part)])

    h_acc, z_acc = types.paste_hidden(h_acc_part, z_acc_part)
    node_sampler.adapt(np.hstack([t[0], t_split, t[-1]]), np.array(acc_part))
    if len(np.unique(h_acc.yt)) < thi.shape[0]:
        return h_nil, z_nil
    return h_acc, z_acc


def update_hidden_section(
    thi: FloatArr,
    h_nil: types.Scaffold,
    z_nil: list[switch.ea3lib.skeleton.Skeleton],
    h_prime: types.Scaffold,
    z_prime: list[switch.ea3lib.skeleton.Skeleton],
    mod: factory.IntractableModel,
    ctrl: Controls,
    ome: np.random.Generator,
) -> tuple[bool, types.Scaffold, list[switch.ea3lib.skeleton.Skeleton]]:

    if not np.all((mod.bounds_v[0] < h_prime.vt) & (h_prime.vt < mod.bounds_v[1])):
        return False, h_nil, z_nil
    is_data = np.intersect1d(h_nil.t, h_prime.t)
    log_q_forw = mod.eval_log_prop(thi, *h_prime, np.isin(h_prime.t, is_data))
    log_q_back = mod.eval_log_prop(thi, *h_nil, np.isin(h_nil.t, is_data))
    
    ops_nil = mod.bind_params(thi, *h_nil)
    ops_prime = mod.bind_params(thi, *h_prime)
    lb_nil, _, coins_nil = zip(*construct_coins(z_nil, ops_nil, ome))
    lb_prime, _, coins_prime = zip(*construct_coins(z_prime, ops_prime, ome))
    weight_nil = eval_weight(thi, h_nil, mod) + log_q_forw / len(ops_nil) - np.array(lb_nil) * np.diff(h_nil.t) 
    weight_prime = eval_weight(thi, h_prime, mod) + log_q_back / len(ops_prime) - np.array(lb_prime) * np.diff(h_prime.t)

    nodes = np.intersect1d(h_nil.t, h_prime.t)
    splits_nil = np.in1d(h_nil.t, nodes).nonzero()[0]
    splits_prime = np.in1d(h_prime.t, nodes).nonzero()[0]
    weight_nil_grp = [sum(a) for a in np.split(weight_nil, splits_nil[1:-1])]
    weight_prime_grp = [sum(a) for a in np.split(weight_prime, splits_prime[1:-1])]
    coins_nil_grp = [list(coins_nil[i0:i1]) for i0, i1 in zip(splits_nil, splits_nil[1:])]
    coins_prime_grp = [list(coins_prime[i0:i1]) for i0, i1 in zip(splits_prime, splits_prime[1:])]
    z_nil_grp = [list(z_nil[i0:i1]) for i0, i1 in zip(splits_nil, splits_nil[1:])]
    z_prime_grp = [list(z_prime[i0:i1]) for i0, i1 in zip(splits_prime, splits_prime[1:])]
    odds_grp = np.array(weight_prime_grp) - np.array(weight_nil_grp)

    try:
        accept, new_z_prime, new_z_nil = next(switch.misc.bfactories.sample_odds_dual(
            z_prime_grp, z_nil_grp, odds_grp, coins_prime_grp, coins_nil_grp, int(np.log2(np.sqrt(len(nodes)))), ctrl.pr_portkey_h, ome))
    except switch.misc.bfactories.BudgetConstraintError:
        accept, new_z_prime, new_z_nil = False, z_prime_grp, z_nil_grp
    
    if accept:
        return accept, h_prime, sum(new_z_prime, [])
    return accept, h_nil, sum(new_z_nil, [])


def update_generator(h: types.Scaffold, lam0: tuple[FloatArr, FloatArr], ome: np.random.Generator) -> FloatArr:

    y = types.prune_anchorage(h)
    return switch.mjplib.inference.sample_param(*switch.mjplib.inference.update(y, *lam0), ome)


def eval_weight(
    thi: FloatArr,
    h: types.Scaffold,
    mod: factory.IntractableModel,
) -> FloatArr:
    
    return mod.eval_biased_log_lik(thi, *h)


def construct_coins(
    z: list[switch.ea3lib.skeleton.Skeleton],
    ops: list[tuple[Callable[[FloatArr, FloatArr], FloatArr], Callable[[float, float, float, float], tuple[float, float]]]],
    ome: np.random.Generator,
) -> list[tuple[float, float, Callable[[switch.ea3lib.skeleton.Skeleton], tuple[bool, switch.ea3lib.skeleton.Skeleton]]]]:
    
    bounds = [switch.ea3lib.skeleton.bound_disc_path(z_, eval_bounds_) for z_, (_, eval_bounds_) in zip(z, ops)]
    outer_bounds = [(np.min(lb_phi_), np.max(ub_phi_)) for lb_phi_, ub_phi_ in bounds]
    coins = [(*bounds_, lambda z, ops_=ops_, bounds_=bounds_: switch.ea3lib.skeleton.flip_coin(z, ops_[0], ops_[1], ome, bounds_) 
             if bounds_[0] != bounds_[1] else (True, z)) 
             for ops_, bounds_ in zip(ops, outer_bounds)]
    return coins


def take_while(predicate: Callable, iterable: Iterator) -> Iterator:

    for x in iterable:
        if predicate(x):
            yield x
        else:
            yield x
            break
