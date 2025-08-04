from typing import Callable, Iterator, TypeVar

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize


Numeric = TypeVar('Numeric', int, float, npt.NDArray[np.number])


def sample(
    f: Callable[[Numeric], Numeric],
    df: Callable[[Numeric], Numeric],
    lb: float = -np.inf,
    ub: float = np.inf,
    init: npt.NDArray[np.number] | None = None,
    max_points: int = 100,
    ome: np.random.Generator = np.random.default_rng(0),
    max_draws: int = int(1e9),
) -> Iterator[float]:
    """Sample from exp(f), where exp(f) is log-concave.

    :param f: log target density, must be concave
    :param df: first derivative of f
    :param lb: lower bound of f
    :param ub: upper bound of f
    :param init: set of starting points. must brace the mode of f
    :param max_points: maximum number of points from which to construct the hull
    :param ome:
    :param max_draws: maximum number of draws to generate
    :return: generator yielding samples from exp(f)

    >>> # sample chi-2
    >>> dof = np.random.poisson() + 3
    >>> f = lambda x_: (dof / 2 - 1) * np.log(x_) - x_ / 2
    >>> df = lambda x_: (dof / 2 - 1) / x_ - 1 / 2
    >>> y = np.array(list(sample(f, df, 0, max_draws=int(1e3))))

    >>> # test
    >>> from scipy.stats import kstest, chi2
    >>> alpha = 1e-2
    >>> alpha < kstest(y, chi2(dof).cdf)[1]
    True
    """

    if init is None:
        init = np.array(brace(f, lb, ub))

    x = np.sort(init)
    fx, dfx = f(x), df(x)

    while max_draws > 0:

        # sample envelope
        y = np.array([lb, *gen_upper(x, fx, dfx), ub])
        x_new = sample_envelope(x, fx, dfx, y, ome)
        lo_new, up_new = eval_lower(x, fx, x_new), eval_upper(x, fx, dfx, y, x_new)
        z = np.log(ome.uniform())

        # squeeze test
        if z < lo_new - up_new:
            max_draws -= 1
            yield x_new

        # rejection test and envelope update
        else:
            fx_new, dfx_new = f(x_new), df(x_new)
            if z < fx_new - up_new:
                max_draws -= 1
                yield x_new
            elif len(x) < max_points:
                i = np.searchsorted(x, x_new)
                x, fx, dfx = [np.insert(l, i, v) for l, v in zip((x, fx, dfx), (x_new, fx_new, dfx_new))]


def brace(
    f: Callable[[Numeric], Numeric], lb: float = -np.inf, ub: float = np.inf,
) -> npt.NDArray[np.number]:
    """Find two points bracing the mode of a function

    :param f: log target density, must be concave
    :param lb: lower bound of f
    :param ub: upper bound of f
    :return: mode bracing points

    >>> dof = 3
    >>> f = lambda y: (dof / 2 - 1) * np.log(y) - y / 2
    >>> df = lambda y: (dof / 2 - 1) / y - 1 / 2
    >>> np.allclose(brace(f, 0, np.inf), (0.5, 2))
    True
    """

    x0 = np.mean([np.sign(x) * np.log(max(abs(np.nan_to_num(x)), 1)) for x in (lb, ub)])
    bounds = [x if not np.isinf(x) else None for x in (lb, ub)]
    res = minimize(lambda x: -f(x[0]), np.array([x0]), bounds=(bounds,))
    mode = res.x[0]

    if np.isinf(lb):
        linit = mode - 1
    else:
        linit = (mode + lb) / 2

    if np.isinf(ub):
        uinit = mode + 1
    else:
        uinit = (mode + ub) / 2

    return np.array([linit, uinit])


def gen_upper(
    x: npt.NDArray[np.number], fx: npt.NDArray[np.number], dfx: npt.NDArray[np.number],
 ) -> npt.NDArray[np.number]:
    """Compute the intersection points of the upper hull line segments

    :param x: hull supports
    :param fx: upper hull value at y
    :param dfx: upper hull derivative value at y
    :return: intersection points of upper hull segments

    >>> y = np.array([-1, 0, 2])
    >>> gen_upper(y, -y**2 / 2, -y)
    array([-0.5,  1. ])
    """

    return np.diff(fx - x * dfx) / (dfx[:-1] - dfx[1:])


def eval_upper(
    x: npt.NDArray[np.number], 
    fx: npt.NDArray[np.number], 
    dfx: npt.NDArray[np.number], 
    y: npt.NDArray[np.number], 
    x_new: float,
) -> float:
    """Evaluate the upper hull at a given point.

    :param x: hull supports
    :param fx: upper hull value at y
    :param dfx: upper hull derivative value at y
    :param y: intersection points of upper hull segments
    :param x_new: point at which to evaluate the upper hull
    :return: upper hull value at x_new

    >>> y = np.array([-1, 0, 2])
    >>> y = [-np.inf] + gen_upper(y, -y**2 / 2, -y) + [np.inf]
    >>> eval_upper(y, -y**2 / 2, -y, y, 1.5)
    -1.0
    """

    i = np.searchsorted(y, x_new) - 1

    return fx[i] + (x_new - x[i]) * dfx[i]


def eval_lower(
        x: npt.NDArray[np.number], fx: npt.NDArray[np.number], x_new: float,
) -> float:
    """Evaluate the lower hull at a given point.

    :param x: hull supports
    :param fx: upper hull value at y
    :param x_new: point at which to evaluate the lower hull
    :return: lower hull value at x_new

    >>> y = np.array([-1, 0, 2])
    >>> eval_lower(y, -y**2 / 2, 1.5)
    -1.5
    """

    i = np.searchsorted(x, x_new) - 1

    if i == -1 or i == len(x) - 1:
        return -np.inf
    return ((x[i + 1] - x_new) * fx[i] + (x_new - x[i]) * fx[i + 1]) / (x[i + 1] - x[i])


def sample_envelope(
    x: npt.NDArray[np.number], 
    fx: npt.NDArray[np.number], 
    dfx: npt.NDArray[np.number], 
    y: npt.NDArray[np.number], 
    ome: np.random.Generator,
) -> float:
    """Generate a sample from the upper hull.

    :param x: hull supports
    :param fx: upper hull value at y
    :param dfx: upper hull derivative value at y
    :param y: intersection points of upper hull segments
    :param ome:
    :return: sample from the upper hull

    >>> y = np.array([-2, -1, 1, 2])
    >>> y = np.array([-np.inf, *gen_upper(y, -y**2 / 2, -y), np.inf])
    >>> sample = [sample_envelope(y, -y**2 / 2, -y, y) for _ in range(int(1e3))]

    >>> # test mean CLT
    >>> from scipy.stats import ttest_1samp
    >>> alpha = 1e-2
    >>> alpha < ttest_1samp(sample, 0)[1]
    True
    """

    # prevent over/underflows
    offset = np.max(dfx * (y[1:] - x) + fx)

    vol_upper = np.exp(dfx * (y[1:] - x) + fx - offset) / dfx
    vol_lower = np.exp(dfx * (y[:-1] - x) + fx - offset) / dfx

    cdf = np.array([0, *np.cumsum(vol_upper - vol_lower)])
    norm = cdf[-1]

    # pick sector to sample from
    z = ome.uniform()
    i = np.searchsorted(cdf / norm, z) - 1

    return x[i] + (np.log((norm * z - cdf[i] + vol_lower[i]) * dfx[i]) - fx[i] + offset) / dfx[i]
