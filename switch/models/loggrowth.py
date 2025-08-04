import numpy as np
import sympy as sp

from switch import factory


def eval_log_prior(thi, scale=np.array([1, 1, 1])):
    if np.any(thi <= 0) or np.any(2 * thi[:, 0] * thi[:, 1] < np.square(thi[:, 2])):
        return -np.inf
    return -np.sum(np.log(thi) + np.square(np.log(thi) / scale) / 2)


def eval_dthi_log_prior(thi, scale=np.array([1, 1, 1])):
    if np.any(thi <= 0) or np.any(2 * thi[:, 0] * thi[:, 1] < np.square(thi[:, 2])):
        return np.tile(np.nan, (2, 3))
    return -(1 + np.log(thi) / np.square(scale)) / thi


init_thi = np.array([1.0, 1.0, 1.0])
bounds_thi = (np.array([0.0, 0.0, 0.0]), np.array([np.inf, np.inf, np.inf]))
bounds_v = (0, np.inf)
bounds_x = (-np.inf, np.inf)


v = sp.symbols('v', positive=True)
x = sp.symbols('x', real=True)
b, k, r = sp.symbols('b k r', positive=True)
thi = sp.Array([b, k, r])
mu = b * r * v * (1 - k * v)
rho = r
sig = v


mod = factory.IntractableModel(v, x, thi, mu, sig, rho, init_thi,
                               bounds_v, bounds_x, bounds_thi,
                               eval_log_prior, eval_dthi_log_prior)
