from typing import Callable

import numpy as np
import numpy.typing as npt


FloatArr = npt.NDArray[np.floating]


""""""
def reverse(t: np.ndarray, x: np.ndarray, fin_t: float = 1) -> tuple[FloatArr, FloatArr]:
    """
    :param t:
    :param x:
    :param fin_t:
    :return:

    >>> fin_t = np.random.uniform()
    >>> t = np.sort(np.random.uniform(high=fin_t, size=10))
    >>> y = np.random.standard_normal(10)
    >>> t_, x_ = reverse(*reverse(t, y, fin_t), fin_t)
    >>> np.allclose(t, t_), np.allclose(y, x_)
    (True, True)
    """

    if len(t):
        if not 0 <= np.min(t) <= np.max(t) <= fin_t:
            raise Exception
        assert 0 <= np.min(t) <= np.max(t) <= fin_t
    assert 0 <= fin_t

    return fin_t - t[::-1], x[::-1]


""""""
def reflect(x: np.ndarray, reflect_x: float = 0) -> FloatArr:
    """
    :param x:
    :param reflect_x:
    :return:

    >>> reflect_x = np.random.normal()
    >>> y = np.random.standard_normal(10)
    >>> x_ = reflect(reflect(y, reflect_x), reflect_x)
    >>> np.allclose(y, x_)
    True
    """

    return 2 * reflect_x - x


""""""
def double_flip(
    t: np.ndarray, 
    x: np.ndarray, 
    fin_t: float, 
    reflect_x: float = 0,
) -> tuple[FloatArr, FloatArr]:
    """
    :param t:
    :param x:
    :param fin_t:
    :param reflect_x:
    :return:
    """

    it, ix = reverse(t, x, fin_t)
    return it, reflect(ix, reflect_x)


""""""
def pivot_brownbr(
    s: FloatArr,
    z: FloatArr,
    fin_t: float,
    init_x: float,
    fin_x: float,
    norm_time: Callable[[float], float] = lambda x: x,
    denorm_time: Callable[[float], float] = lambda x: x,
) -> tuple[FloatArr, FloatArr]:
    """
    :param s:
    :param z:
    :param fin_t:
    :param init_x:
    :param fin_x:
    :param norm_time:
    :param denorm_time
    :return:
    """

    amp = norm_time(fin_t)
    t = np.array([denorm_time(s_ * amp) for s_ in s])
    x = z * np.sqrt(amp) + init_x + s * (fin_x - init_x)

    return t, x
