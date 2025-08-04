from functools import partial
from typing import Callable, Iterator

import numpy as np
import numpy.typing as npt
import sympy as sp

import switch.ea1lib.skeleton
import switch.ea3lib.skeleton
import switch.misc.ars
import switch.mjplib.skeleton
import switch.sdelib.paths
from switch import types


BoolArr = npt.NDArray[np.bool_]
FloatArr = npt.NDArray[np.floating]
IntArr = npt.NDArray[np.integer]
NormOp = Callable[[FloatArr, FloatArr], FloatArr]


def bound_expr_rec(
    f: sp.Expr, 
    s: list[sp.Symbol],
) -> tuple[list[sp.Array], list[sp.Expr]]:

    try:
        return bound_expr(f, s)
    except:
        return expand_and_bound_expr(f, s)


def expand_and_bound_abs_expr(
    f: sp.Expr, 
    s: list[sp.Symbol],
) -> sp.Expr:

    inf_f, sup_f = expand_and_bound_expr(f, s)
    return sp.Max(sp.Abs(inf_f), sp.Abs(sup_f))


def expand_and_bound_expr(
    f: sp.Expr, 
    s: list[sp.Symbol],
) -> tuple[list[sp.Array], list[sp.Expr]]:

    f_exp = f.expand(power_base=False, power_exp=False)
    if not isinstance(f_exp, sp.core.add.Add):
        return bound_expr(f.simplify(), s)
    inf_f, sup_f = zip(*[bound_expr(f_.simplify(), s) for f_ in f_exp.args])
    return sum(inf_f), sum(sup_f)


def bound_expr(
    f: sp.Expr, 
    s: list[sp.Symbol],
) -> tuple[list[sp.Array], list[sp.Expr]]:
    
    roots = sp.solve([f.diff(s_) for s_ in s], s)
    if isinstance(roots, dict):
        roots = [list(roots.values())]
    return [sp.Array(rt) for rt in roots], [f.subs(zip(s, rt)).simplify() for rt in roots]


def symmetrize(
    f: Callable[[FloatArr, FloatArr], FloatArr], 
    lb_domain: float,
    ub_domain: float, 
    lb_codomain: float, 
    ub_codomain: float,
) -> Callable[[FloatArr, FloatArr], FloatArr]:
    
    def f_symm_lb(thi, x):
        y = np.empty_like(x)
        mask = x <= lb_domain
        y[mask] = lb_codomain - f(thi[:, mask] if len(thi.shape) > 1 else thi, lb_domain - x[mask])
        y[~mask] = f(thi[:, ~mask] if len(thi.shape) > 1 else thi, x[~mask])
        return y

    def f_symm_ub(thi, x):
        y = np.empty_like(x)
        mask = ub_domain <= x
        y[mask] = ub_codomain - f(thi[:, mask] if len(thi.shape) > 1 else thi, ub_domain - x[mask])
        y[~mask] = f(thi[:, ~mask] if len(thi.shape) > 1 else thi, x[~mask])
        return y
    
    if not np.isinf(lb_domain) and np.isinf(ub_domain):
        return f_symm_lb
    if np.isinf(lb_domain) and not np.isinf(ub_domain):
        return f_symm_ub
    return f


def bundle(f: list[Callable[..., FloatArr]]) -> Callable[..., FloatArr]:

    def f_bundled(thi, *x):
        return np.array([f_(thi, *x) for f_ in f])
    return f_bundled


class TractableModel(object):

    def __init__(
        self,
        def_thi: FloatArr,
        bounds_v: tuple[float, float],
        bounds_x: tuple[float, float],
        bounds_thi: tuple[FloatArr, FloatArr],
        eval_eta: Callable[[FloatArr, FloatArr], FloatArr],
        eval_dthi_eta: Callable[[FloatArr, FloatArr], FloatArr],
        eval_ieta: Callable[[FloatArr, FloatArr], FloatArr],
        eval_dv_eta: Callable[[FloatArr, FloatArr], FloatArr],
        eval_rho: Callable[[FloatArr], FloatArr],
        eval_dthi_rho: Callable[[FloatArr], FloatArr],
        eval_log_lik: Callable[[FloatArr, FloatArr, IntArr, FloatArr], float],
        eval_dthi_log_lik: Callable[[FloatArr, FloatArr, IntArr, FloatArr], FloatArr],
        eval_log_prior: Callable[[FloatArr], float],
        eval_dthi_log_prior: Callable[[FloatArr], FloatArr],
    ):

        self.def_thi = def_thi
        self.bounds_v = bounds_v
        self.bounds_x = bounds_x
        self.bounds_thi = bounds_thi
        self.eval_eta = eval_eta
        self.eval_dthi_eta = eval_dthi_eta
        self.eval_ieta = eval_ieta
        self.eval_dv_eta = eval_dv_eta
        self.eval_rho = eval_rho
        self.eval_dthi_rho = eval_dthi_rho
        self.eval_approx_log_lik = eval_log_lik
        self.eval_dthi_approx_log_lik = eval_dthi_log_lik
        self.eval_log_prior = eval_log_prior
        self.eval_dthi_log_prior = eval_dthi_log_prior

    def bind_params(
        self,
        thi: FloatArr,
        t: FloatArr,
        yt: IntArr,
        vt: FloatArr,
    ) -> list[tuple[NormOp]]:

        xt = self.eval_eta(thi[yt].T, vt)
        return [(partial(self.denormalize, thi[y0], dt, x0, x1),)
                for dt, y0, x0, x1 in zip(np.diff(t), yt, xt, xt[1:])]

    def bind_params_diff(
        self,
        thi: FloatArr,
        t: FloatArr,
        yt: IntArr,
        vt: FloatArr,
    ) -> list[tuple[NormOp]]:

        dthi_xt = self.eval_dthi_eta(thi[yt].T, vt)
        return [(partial(self.dthi_denormalize, thi[y0], dt, x0, x1),)
                for dt, y0, x0, x1 in zip(np.diff(t), yt, dthi_xt, dthi_xt[1:])]

    def sample_aug1(
        self,
        thi: FloatArr,
        y: switch.mjplib.skeleton.Skeleton,
        t: FloatArr,
        vt: FloatArr,
        new_t: FloatArr,
        ome: np.random.Generator,
    ) -> tuple[types.Scaffold, list[switch.ea1lib.skeleton.Skeleton]]:

        h = self.sample_scaffold(thi, y, t, vt, new_t, ome)
        z = self.sample_bridges1(thi, *h, ome)
        return h, z

    def sample_aug3(
        self,
        thi: FloatArr,
        y: switch.mjplib.skeleton.Skeleton,
        t: FloatArr,
        vt: FloatArr,
        new_t: FloatArr,
        n_segments: int,
        ome: np.random.Generator,
    ) -> tuple[types.Scaffold, list[switch.ea3lib.skeleton.Skeleton]]:

        h = self.sample_scaffold(thi, y, t, vt, new_t, ome)
        z = self.sample_bridges3(thi, *h, n_segments, ome)
        return h, z

    def sample_scaffold(
        self,
        thi: FloatArr,
        y: switch.mjplib.skeleton.Skeleton,
        t: FloatArr,
        vt: FloatArr,
        new_t: FloatArr,
        ome: np.random.Generator,
    ) -> types.Scaffold:

        xt = self.eval_eta(thi[np.append(y(t[:-1]), y.xt[-1])].T, vt)
        i_new_t = np.searchsorted(y.t, new_t)
        y_ext = switch.mjplib.skeleton.Skeleton(np.insert(y.t, i_new_t, new_t), np.insert(y.xt, i_new_t, y(new_t)), t[-1])
        y_part = switch.mjplib.skeleton.partition_skeleton(y_ext, t[1:-1])
        vtt = np.hstack([self.eval_ieta(thi[yt_].T, self.sample_anchors(thi, t_, yt_, fin_t_, x0, x1, ome))
                         for (t_, yt_, fin_t_), x0, x1 in zip(y_part, xt, xt[1:])] + [vt[-1]])
        tt = np.append(np.hstack([y_.t for y_ in y_part]), t[-1])
        ytt = np.append(np.hstack([y_.xt for y_ in y_part]), y.xt[-1])
        return types.Scaffold(tt, ytt, vtt)

    def sample_bridges1(
        self,
        thi: FloatArr,
        t: FloatArr,
        yt: IntArr,
        vt: FloatArr,
        ome: np.random.Generator,
    ) -> list[switch.ea1lib.skeleton.Skeleton]:

        return [switch.ea1lib.skeleton.init_skeleton(dt, 0, 0, ome) for dt in np.diff(t)]

    def sample_bridges3(
        self, 
        thi: FloatArr, 
        t: FloatArr, 
        yt: IntArr,
        vt: FloatArr, 
        n_segments: int,
        ome: np.random.Generator,
    ) -> list[switch.ea3lib.skeleton.Skeleton]:

        return [switch.ea3lib.skeleton.init_skeleton(dt, 0, 0, ome) for dt in np.diff(t)]

    def sample_anchors(
        self,
        thi: FloatArr,
        t: FloatArr,
        yt: IntArr,
        fin_t: float,
        init_x: float,
        fin_x: float,
        ome: np.random.Generator,
    ) -> FloatArr:

        if len(t) == 1:
            return np.array([init_x])
        sig_t = np.sqrt(np.diff(np.append(t, fin_t))) * self.eval_rho(thi[yt].T)
        s = np.cumsum(np.square(sig_t))
        xt = switch.sdelib.paths.sample_brownbr(s[:-1], s[-1], init_x, fin_x, ome)
        return np.hstack([init_x, xt])

    def eval_log_prop(
        self,
        thi: FloatArr,
        t: FloatArr,
        yt: IntArr,
        vt: FloatArr,
        is_obs: BoolArr,
    ) -> float:

        xt = self.eval_eta(thi[yt].T, vt)
        sig_t = np.sqrt(np.diff(t)) * self.eval_rho(thi[yt[:-1]].T)
        sig_t_obs = np.sqrt([np.sum(np.square(sig_t_)) for sig_t_ in np.split(sig_t, np.where(is_obs)[0])[1:-1]])
        dv_lam = np.log(np.abs(self.eval_dv_eta(thi[yt[1:]].T, vt[1:])))
        p_vt = dv_lam - (np.log(2 * np.pi * np.square(sig_t)) + np.square(np.diff(xt) / sig_t)) / 2
        p_vt_obs = dv_lam[is_obs[1:]] \
            - (np.log(2 * np.pi * np.square(sig_t_obs)) + np.square(np.diff(xt[is_obs]) / sig_t_obs)) / 2
        return np.sum(p_vt) - np.sum(p_vt_obs)

    def normalize(
        self,
        thi: FloatArr,
        fin_t: float,
        init_x: float,
        fin_x: float,
        t: FloatArr,
        xt: FloatArr,
    ) -> FloatArr:

        return (xt - init_x - (t / fin_t) * (fin_x - init_x)) / self.eval_rho(thi)

    def denormalize(
        self,
        thi: FloatArr,
        fin_t: float,
        init_x: float,
        fin_x: float,
        t: FloatArr,
        zt: FloatArr,
    ) -> FloatArr:

        return zt * self.eval_rho(thi) + init_x + (t / fin_t) * (fin_x - init_x)

    def dthi_denormalize(
        self,
        thi: FloatArr,
        fin_t: float,
        init_x: FloatArr,
        fin_x: FloatArr,
        t: FloatArr,
        zt: FloatArr,
    ) -> FloatArr:

        return zt * self.eval_dthi_rho(thi)[:, np.newaxis] + init_x[:, np.newaxis] + (t / fin_t) * (fin_x - init_x)[:, np.newaxis]


class IntractableModel(TractableModel):

    def __init__(
        self,
        v: sp.Symbol,
        x: sp.Symbol,
        thi: sp.Array,
        mu_v: sp.Expr,
        sig_v: sp.Expr,
        rho: sp.Expr,
        def_thi: FloatArr,
        bounds_v: tuple[float, float],
        bounds_x: tuple[float, float],
        bounds_thi: tuple[FloatArr, FloatArr],
        eval_log_prior: Callable[[FloatArr], float],
        eval_dthi_log_prior: Callable[[FloatArr], FloatArr],
    ):

        self.def_thi = def_thi
        self.bounds_v = bounds_v
        self.bounds_x = bounds_x
        self.bounds_thi = bounds_thi
        self.eval_log_prior = eval_log_prior
        self.eval_dthi_log_prior = eval_dthi_log_prior

        eta_v = sp.integrate(1/sig_v.expand(), v, conds='separate').simplify()
        dv_sig_v = sp.diff(sig_v, v)
        ieta_x = sp.solve(sp.Eq(eta_v, x), v)[0]
        dv_eta_v = sp.diff(eta_v, v)
        dx_ieta_x = sp.diff(ieta_x, x)
        dil_x = (mu_v / sig_v - dv_sig_v * rho ** 2 / 2).subs(v, ieta_x)
        dx_dil_x = sp.diff(dil_x, x).simplify()
        sx_dil_x = sp.integrate(dil_x.expand(), x, conds='separate').simplify()
        phi_x = ((dil_x / rho) ** 2 + dx_dil_x) / 2
        dx_phi_x = sp.diff(phi_x, x)

        dthi_eta_v = sp.Array([sp.diff(eta_v, thi_) for thi_ in thi])
        dthi_dv_eta_v = sp.Array([sp.diff(dv_eta_v, thi_) for thi_ in thi])
        dthi_rho = sp.diff(rho, thi)
        dthi_dil_x = sp.diff(dil_x, thi)
        dthi_sx_dil_x = sp.diff(sx_dil_x, thi)
        dthi_dx_dil_x = sp.diff(dx_dil_x, thi)
        dthi_phi_x = sp.diff(phi_x, thi)
        dthi_dx_phi_x = sp.diff(dx_phi_x, thi)

        # bound phi in x
        rt_phi_x, ext_phi_x = bound_expr(phi_x, [x])
        # eta_v = sp.integrate(1/sig_v.expand(), v, conds='separate').simplify()
        # dv_sig_v = sp.diff(sig_v, v)
        # ieta_x = sp.solve(sp.Eq(eta_v, x), v)[0]
        # dv_eta_v = sp.diff(eta_v, v)
        # dx_ieta_x = sp.diff(ieta_x, x)
        # dil_x = (mu_v / sig_v - dv_sig_v * rho ** 2 / 2).subs(v, ieta_x)
        # dx_dil_x = sp.diff(dil_x, x).simplify()
        # sx_dil_x = sp.integrate(dil_x.expand(), x, conds='separate').simplify()
        # phi_x = ((dil_x / rho) ** 2 + dx_dil_x) / 2
        # dx_phi_x = sp.diff(phi_x, x)

        self.eval_eta = symmetrize(sp.lambdify((thi, v), eta_v), *bounds_v, *bounds_x)
        self.eval_dv_eta = symmetrize(sp.lambdify((thi, v), sp.Add(dv_eta_v, v, -v, evaluate=False)), *bounds_v, *bounds_x)
        self.eval_ieta = symmetrize(sp.lambdify((thi, x), ieta_x), *bounds_x, *bounds_v)
        self.eval_dx_ieta = symmetrize(sp.lambdify((thi, x), sp.Add(dx_ieta_x, x, -x, evaluate=False)), *bounds_x, *bounds_v)
        self.eval_rho = sp.lambdify((thi,), rho)
        self.eval_dil = sp.lambdify((thi, x), dil_x)
        self.eval_sx_dil = sp.lambdify((thi, x), sx_dil_x)
        self.eval_dx_dil = sp.lambdify((thi, x), dx_dil_x)
        self.eval_phi = sp.lambdify((thi, x), phi_x)
        self.eval_dx_phi = sp.lambdify((thi, x), dx_phi_x)

        self.eval_dthi_rho = bundle([sp.lambdify((thi,), sp.Add(dthi_rho[i], thi[i], -thi[i], evaluate=False)) for i in range(len(thi))])
        self.eval_dthi_eta = bundle([sp.lambdify((thi, v), sp.Add(dthi_eta_v[i], v, -v, evaluate=False)) for i in range(len(thi))])
        self.eval_dthi_dv_eta = bundle([sp.lambdify((thi, v), sp.Add(dthi_dv_eta_v[i], v, -v, evaluate=False)) for i in range(len(thi))])
        self.eval_dthi_dil = bundle([sp.lambdify((thi, x), dthi_dil_x[i]) for i in range(len(thi))])
        self.eval_dthi_sx_dil = bundle([sp.lambdify((thi, x), dthi_sx_dil_x[i]) for i in range(len(thi))])
        self.eval_dthi_dx_dil = bundle([sp.lambdify((thi, x), dthi_dx_dil_x[i]) for i in range(len(thi))])
        self.eval_dthi_phi = bundle([sp.lambdify((thi, x), dthi_phi_x[i]) for i in range(len(thi))])
        self.eval_dthi_dx_phi = bundle([sp.lambdify((thi, x), dthi_dx_phi_x[i]) for i in range(len(thi))])

        self.eval_rt_phi = [sp.lambdify((thi,), rt_[0]) for rt_ in rt_phi_x]
        self.eval_ext_phi = [sp.lambdify((thi,), ext_) for ext_ in ext_phi_x]

    def bind_params(
        self,
        thi: FloatArr,
        t: FloatArr,
        yt: IntArr,
        vt: FloatArr,
    ) -> list[tuple[Callable[[FloatArr, FloatArr], FloatArr], 
                    Callable[[float, float, float, float], tuple[float, float]]]]:

        xt = self.eval_eta(thi[yt].T, vt)
        return [(partial(self.eval_ncp_phi, thi[y0], dt, x0, x1),
                 partial(self.eval_ncp_bounds_phi, thi[y0], dt, x0, x1))
                for dt, y0, x0, x1 in zip(np.diff(t), yt, xt, xt[1:])]

    def bind_params_diff(
        self, 
        thi_num: FloatArr,
        thi_den: FloatArr,
        t: FloatArr,
        yt: IntArr,
        vt: FloatArr,
    ) -> list[tuple[Callable[[FloatArr, FloatArr], FloatArr], 
                    Callable[[float, float, float, float], tuple[float, float]]]]:

        xt_num = self.eval_eta(thi_num[yt].T, vt)
        xt_den = self.eval_eta(thi_den[yt].T, vt)
        return [(partial(self.eval_ncp_dphi, thi_num[y0], thi_den[y0], dt, x0n, x1n, x0d, x1d),
                 partial(self.eval_ncp_bounds_dphi, thi_num[y0], thi_den[y0], dt, x0n, x1n, x0d, x1d))
                for dt, y0, x0n, x1n, x0d, x1d in zip(np.diff(t), yt, xt_num, xt_num[1:], xt_den, xt_den[1:])]

    def eval_approx_log_lik(
        self,
        thi: FloatArr,
        t: FloatArr,
        yt: IntArr,
        vt: FloatArr,
    ) -> float:

        dt = np.diff(t)
        xt = self.eval_eta(thi[yt].T, vt)
        mu_t = np.diff(xt) - dt * self.eval_dil(thi[yt[:-1]].T, xt[:-1])
        sig_t = np.sqrt(dt) * self.eval_rho(thi[yt[:-1]].T)
        p_vt = np.log(np.abs(self.eval_dv_eta(thi[yt[1:]].T, vt[1:]))) \
               - (np.log(2 * np.pi * np.square(sig_t)) + np.square(mu_t / sig_t)) / 2
        return np.sum(p_vt)

    def eval_biased_log_lik(
        self,
        thi: FloatArr,
        t: FloatArr,
        yt: IntArr,
        vt: FloatArr,
    ) -> FloatArr:

        xt = self.eval_eta(thi[yt].T, vt)
        sig_t = np.sqrt(np.diff(t)) * self.eval_rho(thi[yt[:-1]].T)
        dsx_dil = self.eval_sx_dil(thi[yt[:-1]].T, xt[1:]) - self.eval_sx_dil(thi[yt[:-1]].T, xt[:-1])
        p_vt = np.log(np.abs(self.eval_dv_eta(thi[yt[1:]].T, vt[1:]))) \
            + dsx_dil / np.square(self.eval_rho(thi[yt[:-1]].T)) \
            - (np.log(2 * np.pi * np.square(sig_t)) + np.square(np.diff(xt) / sig_t)) / 2
        return p_vt

    def eval_dthi_biased_log_lik(
        self,
        thi: FloatArr,
        t: FloatArr,
        yt: IntArr,
        vt: FloatArr,
    ) -> FloatArr:

        dt = np.diff(t)
        xt = self.eval_eta(thi[yt].T, vt)
        dxt = np.diff(xt)
        sig_t = np.sqrt(dt) * self.eval_rho(thi[yt[:-1]].T)
        dthi_sig_t = np.sqrt(dt) * self.eval_dthi_rho(thi[yt[:-1]].T)
        dthi_xt = self.eval_dthi_eta(thi[yt].T, vt)
        dthi_dxt = np.diff(dthi_xt, 1)
        dthi_log_sig_t = dthi_sig_t / sig_t
        dthi_jac = self.eval_dthi_dv_eta(thi[yt[1:]].T, vt[1:]) / self.eval_dv_eta(thi[yt[1:]].T, vt[1:])
        dthi_n_vt = -(dthi_log_sig_t + (dxt * dthi_dxt - np.square(dxt) * dthi_log_sig_t) / np.square(sig_t))
        dsx_dil = self.eval_sx_dil(thi[yt[:-1]].T, xt[1:]) - self.eval_sx_dil(thi[yt[:-1]].T, xt[:-1])
        dthi_dsx_dil = self.eval_dthi_sx_dil(thi[yt[:-1]].T, xt[1:]) \
                       - self.eval_dthi_sx_dil(thi[yt[:-1]].T, xt[:-1]) \
                       + self.eval_dil(thi[yt[:-1]].T, xt[1:]) * dthi_xt[:, 1:] \
                       - self.eval_dil(thi[yt[:-1]].T, xt[:-1]) * dthi_xt[:, :-1]
        dthi_p_vt = dthi_jac + dthi_n_vt + (dthi_dsx_dil - 2 * dthi_log_sig_t * dsx_dil) / np.square(self.eval_rho(thi[yt[:-1]].T))
        return np.array([np.sum(dthi_p_vt[:, yt[:-1] == y_], 1) for y_ in range(thi.shape[0])])

    def eval_dthi_approx_log_lik(
        self,
        thi: FloatArr,
        t: FloatArr,
        yt: IntArr,
        vt: FloatArr,
    ) -> FloatArr:

        dt = np.diff(t)
        xt = self.eval_eta(thi[yt].T, vt)
        mu_t = np.diff(xt) - dt * self.eval_dil(thi[yt[:-1]].T, xt[:-1])
        sig_t = np.sqrt(dt) * self.eval_rho(thi[yt[:-1]].T)
        dthi_xt = self.eval_dthi_eta(thi[yt].T, vt)
        dthi_mu_t = np.diff(dthi_xt, 1) - dt * (self.eval_dthi_dil(thi[yt[:-1]].T, xt[:-1]) \
                    + self.eval_dx_dil(thi[yt[:-1]].T, xt[:-1]) * dthi_xt[:, :-1])
        dthi_sig_t = np.sqrt(dt) * self.eval_dthi_rho(thi[yt[:-1]].T)
        dthi_p_vt = self.eval_dthi_dv_eta(thi[yt[1:]].T, vt[1:]) / self.eval_dv_eta(thi[yt[1:]].T, vt[1:]) - dthi_sig_t / sig_t \
            - mu_t / sig_t * (dthi_mu_t / sig_t - mu_t * dthi_sig_t / np.square(sig_t))
        return np.array([np.sum(dthi_p_vt[:, yt[:-1] == y_], 1) for y_ in range(thi.shape[0])])
    
    def eval_global_min_phi(
        self, 
        thi: FloatArr,
    ) -> float:

        interior = [g(thi) for g in self.eval_ext_phi]
        if len(interior) == 0:
            raise NotImplementedError
        return min(interior)
    
    def eval_bounds_phi(
        self, 
        thi: FloatArr,
        inf_x: float,
        sup_x: float,
    ) -> tuple[float, float]:

        if inf_x < self.bounds_x[0] or self.bounds_x[1] < sup_x:
            return -np.inf, np.inf
        exterior = list(self.eval_phi(thi, np.array([inf_x, sup_x])))
        interior = [g(thi) for f, g in zip(self.eval_rt_phi, self.eval_ext_phi) if inf_x < f(thi) < sup_x]
        return min(exterior + interior), max(exterior + interior)
    
    def eval_ncp_phi(
        self,
        thi: FloatArr,
        fin_t: float,
        init_x: float,
        fin_x: float,
        t: FloatArr,
        zt: FloatArr,
    ) -> FloatArr:

        return self.eval_phi(thi, self.denormalize(thi, fin_t, init_x, fin_x, t, zt))
    
    def eval_ncp_bounds_phi(
        self,
        thi: FloatArr,
        fin_t: float,
        init_x: float,
        fin_x: float,
        inf_t: float,
        sup_t: float,
        inf_z: float,
        sup_z: float,
    ) -> tuple[float, float]:
        
        inf_x = np.min(self.denormalize(thi, fin_t, init_x, fin_x, np.array([inf_t, sup_t]), np.repeat(inf_z, 2)))
        sup_x = np.max(self.denormalize(thi, fin_t, init_x, fin_x, np.array([inf_t, sup_t]), np.repeat(sup_z, 2)))
        return self.eval_bounds_phi(thi, inf_x, sup_x)
    
    def eval_ncp_dphi(
        self,
        thi_num: FloatArr,
        thi_den: FloatArr,
        fin_t: float,
        init_x_num: float,
        fin_x_num: float,
        init_x_den: float,
        fin_x_den: float,
        t: FloatArr,
        zt: FloatArr,
    ) -> FloatArr:

        phi_num = self.eval_ncp_phi(thi_num, fin_t, init_x_num, fin_x_num, t, zt)
        phi_den = self.eval_ncp_phi(thi_den, fin_t, init_x_den, fin_x_den, t, zt)
        dphi = phi_num - phi_den
        return np.where(dphi > 0, dphi, 0)

    def eval_ncp_bounds_dphi(
        self,
        thi_num: FloatArr,
        thi_den: FloatArr,
        fin_t: float,
        init_x_num: float,
        fin_x_num: float,
        init_x_den: float,
        fin_x_den: float,
        inf_t: float,
        sup_t: float,
        inf_z: float,
        sup_z: float,
    ) -> tuple[float, float]:

        inf_phi_num, sup_phi_num = self.eval_ncp_bounds_phi(thi_num, fin_t, init_x_num, fin_x_num, inf_t, sup_t, inf_z, sup_z)
        inf_phi_den, sup_phi_den = self.eval_ncp_bounds_phi(thi_den, fin_t, init_x_den, fin_x_den, inf_t, sup_t, inf_z, sup_z)
        sup_dphi = sup_phi_num - inf_phi_den
        inf_dphi = inf_phi_num - sup_phi_den
        return inf_dphi if inf_dphi > 0 else 0, sup_dphi if sup_dphi > 0 else 0

    def sample_conv_endpt(
        self,
        thi: FloatArr,
        fin_t: float,
        init_y: int,
        init_x: float,
        ome: np.random.Generator,
    ) -> Iterator[float]:

        def f(x):
            return (self.eval_sx_dil(thi[init_y], x) - np.square(x - init_x) / (2 * fin_t)) / self.eval_rho(thi[[init_y]].T) ** 2
        def df(x):
            return (self.eval_dil(thi[init_y], x) - (x - init_x) / fin_t) / self.eval_rho(thi[[init_y]].T) ** 2
        return switch.misc.ars.sample(f, df, *self.bounds_x, ome=ome)


SwitchingModel = TractableModel | IntractableModel
