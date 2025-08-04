from typing import Callable

import numpy as np
import numpy.typing as npt


FloatArr = npt.NDArray[np.floating]


def euler_simulate(
    ndraws: int,
    fin_t: float = 1,
    init_x: float = 0,
    dfunc: Callable[[float], float] = (lambda x: 0),
    vfunc: Callable[[float], float] = (lambda x: 1),
    res: float = 1e-3,
    ome: np.random.Generator = np.random.default_rng(),
) -> tuple[FloatArr, FloatArr]:
    """Simulate from a diffusion with given mu and volatility function by way of discrete approximation.

    :param ndraws:
    :param fin_t:
    :param init_x:
    :param dfunc:
    :param vfunc:
    :param res:
    :param ome:
    :return:

    >>> # simulate Bessel-3 on [0, 1] from x0=1
    >>> nsamples = int(1e4)
    >>> t, y = euler_simulate(nsamples, 1, 1, lambda x: 1 / x, lambda x: 1)

    >>> # test
    >>> from scipy.stats import kstest, ncx2
    >>> alpha = 1e-2
    >>> alpha < kstest(y[:, -1] ** 2 / t[-1], ncx2(3, 1 / t[-1]).cdf)[1]
    True
    """

    noise = np.sqrt(res) * ome.standard_normal((int(fin_t / res), ndraws))
    trajectory = [np.array(ndraws * [init_x])]

    for e in noise:
        trajectory.append(trajectory[-1] + dfunc(trajectory[-1]) * res + vfunc(trajectory[-1]) * e)

    return np.linspace(0, fin_t, int(fin_t / res) + 1), np.array(trajectory).T
