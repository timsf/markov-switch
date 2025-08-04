import numpy as np
import sympy as sp

from switch import factory


def eval_log_prior(thi, scale = np.array([1, 1, 1])):
    return -np.sum(np.square(thi[:, 0] / scale[0]) / 2 + np.square(np.log(thi[:, 1]) / scale[1]) / 2 + np.square(np.log(thi[:, 2]) / scale[2]) / 2)


def eval_dthi_log_prior(thi, scale = np.array([1, 1, 1])):
    return -np.vstack([thi[:, 0] / np.square(scale[0]), (1 + np.log(thi[:, 1]) / np.square(scale[1])) / thi[:, 1], (1 + np.log(thi[:, 2]) / np.square(scale[2])) / thi[:, 2]]).T


init_thi = np.array([0.0, 1.0, 1.0])
bounds_thi = (np.array([-np.inf, 0.0, 0.0]), np.array([np.inf, np.inf, np.inf]))
bounds_v = (-np.inf, np.inf)
bounds_x = (-np.inf, np.inf)


v, x = sp.symbols('v x', real=True)
b, r = sp.symbols('b r', positive=True)
m = sp.symbols('m', real=True)
thi = sp.Array([m, b, r])
mu = r * b * sp.tanh(m - v)
sig = sp.Integer(1)
rho = r


mod = factory.IntractableModel(v, x, thi, mu, sig, rho,
                               init_thi, bounds_v, bounds_x, bounds_thi,
                               eval_log_prior, eval_dthi_log_prior)
