from typing import Iterator

import numpy as np
import scipy.special

from switch.misc.exceptions import BudgetConstraintError


""""""
def sample_minimal_skel(
    fin_t: float, 
    init_x: float, 
    fin_x: float, 
    lb_x: float, 
    ub_x: float,
    ome: np.random.Generator,
) -> tuple[tuple[float, float], 
           tuple[float, float], 
           tuple[float, float], 
           tuple[bool, bool], 
           bool]:

    ll, ul = sample_layer(fin_t, init_x, fin_x, lb_x, ub_x, ome)
    return sample_oneside_min(fin_t, init_x, fin_x, ll, ul, ome)


""""""
def sample_layer(
    fin_t: float, 
    init_x: float, 
    fin_x: float, 
    lb_x: float, 
    ub_x: float,
    ome: np.random.Generator,
    max_props: int = int(1e6),
) -> tuple[tuple[float, float], tuple[float, float]]:

    l = 1

    if not np.isinf(lb_x) and not np.isinf(ub_x):
        for _ in range(max_props):
            u = ome.uniform()
            offset, width = (lb_x + ub_x) / 2, ub_x - lb_x
            if not sample_brownbr_esc(fin_t, init_x - offset, fin_x - offset, width / 2, None, u, ome):
                break
        else:
            raise BudgetConstraintError('None of the proposals were accepted.')
    elif not np.isinf(lb_x) and np.isinf(ub_x):
        u = ome.uniform(np.exp(logcdf_brownbr_min(lb_x, fin_t, init_x, fin_x)), 1)
    elif np.isinf(lb_x) and not np.isinf(ub_x):
        u = ome.uniform(np.exp(logcdf_brownbr_min(-ub_x, fin_t, -fin_x, -init_x)), 1)
    else:
        u = ome.uniform()

    while True:
        ol = compute_edges(l, fin_t, init_x, fin_x, lb_x, ub_x)
        offset, width = (ol[0] + ol[1]) / 2, ol[1] - ol[0]
        if not sample_brownbr_esc(fin_t, init_x - offset, fin_x - offset, width / 2, None, u, ome):
            il = compute_edges(l - 1, fin_t, init_x, fin_x, lb_x, ub_x)
            return (ol[0], il[0]), (il[1], ol[1])
        l += 1


""""""
def sample_oneside_min(
    fin_t: float, 
    init_x: float, 
    fin_x: float, 
    ll: tuple[float, float],
    ul: tuple[float, float],
    ome: np.random.Generator,
    max_props: int = int(1e6),
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[bool, bool], bool]:
    
    def sample_mincase(
        ll_: tuple[float, float], 
        ul_: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[bool, bool]]: 
        min_t, min_x = sample_brownbr_ub_min(fin_t, init_x, fin_x, ul_[1], *ll_, ome)
        esc1 = sample_bessel3br_esc(min_t, 0, init_x - min_x, ul_[0] - min_x, ul_[1] - min_x, None, ome)
        esc2 = sample_bessel3br_esc(fin_t - min_t, 0, fin_x - min_x, ul_[0] - min_x, ul_[1] - min_x, None, ome)
        return (min_t, min_x), (esc1, esc2)

    log_p_lo = logcdf_brownbr_min(ll[1], fin_t, init_x, fin_x, ll[0])
    log_p_hi = logcdf_brownbr_min(-ul[0], fin_t, -fin_x, -init_x, -ul[1])
    for _ in range(max_props):
        if np.log(ome.uniform()) < log_p_lo - np.logaddexp(log_p_lo, log_p_hi):
            min_, hit = sample_mincase(ll, ul)
            if (not hit[0] and not hit[1]) or (ome.uniform() < .5):
                return min_, ll, ul, hit, False
        else:
            # use point symmetry of brownian bridge law for maximum
            ll_reflx = (init_x + fin_x - ul[1], init_x + fin_x - ul[0])
            ul_reflx = (init_x + fin_x - ll[1], init_x + fin_x - ll[0])
            min_, hit = sample_mincase(ll_reflx, ul_reflx)
            if (not hit[0] and not hit[1]) or (ome.uniform() < .5):
                return min_, ll_reflx, ul_reflx, hit, True
    else:
        raise BudgetConstraintError('None of the proposals were accepted.')


""""""
def sample_twoside_min(
    fin_t: float, 
    init_x: float, 
    fin_x: float, 
    ll: tuple[float, float],
    ul: tuple[float, float],
    ome: np.random.Generator,
    max_props: int = int(1e6),
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[bool, bool], bool]:
    
    def sample_mincase(
        ll_: tuple[float, float], 
        ul_: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[bool, bool]]: 
        min_t, min_x = sample_brownbr_ub_min(fin_t, init_x, fin_x, ul_[1], *ll_, ome)
        esc1 = sample_bessel3br_esc(min_t, 0, init_x - min_x, ul_[0] - min_x, ul_[1] - min_x, None, ome)
        esc2 = sample_bessel3br_esc(fin_t - min_t, 0, fin_x - min_x, ul_[0] - min_x, ul_[1] - min_x, None, ome)
        return (min_t, min_x), (esc1, esc2)
            
    ll_reflx = (init_x + fin_x - ul[1], init_x + fin_x - ul[0])
    ul_reflx = (init_x + fin_x - ll[1], init_x + fin_x - ll[0])
    for _ in range(max_props):
        if ome.uniform() < .5:
            min_, hit = sample_mincase(ll, ul)
            if hit[0] or hit[1]:
                return min_, ll, ul, hit, False
        else:
            # use point symmetry of brownian bridge law for maximum
            min_, hit = sample_mincase(ll_reflx, ul_reflx)
            if hit[0] or hit[1]:
                return min_, ll_reflx, ul_reflx, hit, True
    else:
        raise BudgetConstraintError('None of the proposals were accepted.')


""""""
def compute_edges(
    i: int, 
    fin_t: float, 
    init_x: float, 
    fin_x: float, 
    lb_x: float = -np.inf, 
    ub_x: float = np.inf,
    scale: float = 1e-1,
) -> tuple[float, float]:

    lo_bar_x = min(init_x, fin_x) - i * scale * np.sqrt(fin_t)
    hi_bar_x = max(init_x, fin_x) + i * scale * np.sqrt(fin_t)

    return lo_bar_x, hi_bar_x


""""""
def logcdf_brownbr_min(
    min_x: float, 
    fin_t: float, 
    init_x: float, 
    fin_x: float,
    lb_min_x: float | None = None, 
    ub_min_x: float | None = None,
) -> float:

    if lb_min_x is None:
        lb_min_x = -np.inf
    if ub_min_x is None:
        ub_min_x = min(init_x, fin_x)

    if min_x < min(init_x, fin_x):
        log_pm = -2 * (init_x - min_x) * (fin_x - min_x) / fin_t
    else:
        log_pm = 0
    if np.isinf(lb_min_x) and ub_min_x == min(init_x, fin_x):
        return log_pm

    log_cdf_lb = logcdf_brownbr_min(lb_min_x, fin_t, init_x, fin_x)
    log_cdf_ub = logcdf_brownbr_min(ub_min_x, fin_t, init_x, fin_x)
    log_p = logsubexp(log_pm, log_cdf_lb) - logsubexp(log_cdf_ub, log_cdf_lb)

    return log_p


""""""
def ppf_brownbr_min(
    p: float, 
    fin_t: float, 
    init_x: float, 
    fin_x: float, 
    lb_min_x: float | None = None, 
    ub_min_x: float | None = None,
) -> float:
                    
    if lb_min_x is None:
        lb_min_x = -np.inf
    if ub_min_x is None:
        ub_min_x = min(init_x, fin_x)

    if np.isinf(lb_min_x) and ub_min_x == min(init_x, fin_x):
        log_pm = np.log(p)
    else:
        log_cdf_lb = logcdf_brownbr_min(lb_min_x, fin_t, init_x, fin_x)
        log_cdf_ub = logcdf_brownbr_min(ub_min_x, fin_t, init_x, fin_x)
        log_pm = np.logaddexp(np.log(p) + logsubexp(log_cdf_ub, log_cdf_lb), log_cdf_lb)

    min_x = (fin_x + init_x - np.sqrt((fin_x - init_x) ** 2 - 2 * fin_t * log_pm)) / 2
    return min_x


""""""
def sample_brownbr_min(
    fin_t: float, 
    init_x: float, 
    fin_x: float, 
    lb_min_x: float | None, 
    ub_min_x: float | None,
    ome: np.random.Generator,
) -> tuple[float, float]:

    if lb_min_x is None:
        lb_min_x = -np.inf
    if ub_min_x is None:
        ub_min_x = min(init_x, fin_x)

    # simulate minimum value
    min_x = ppf_brownbr_min(ome.uniform(), fin_t, init_x, fin_x, lb_min_x, ub_min_x)

    # simulate hitting time of minimum value
    par1 = fin_x - min_x
    par2 = init_x - min_x
    par3 = par1 / par2
    if ome.uniform() < 1 / (1 + par3):
        min_t = fin_t / (1 + ome.wald(par3, par1 ** 2 / fin_t))
    else:
        min_t = fin_t / (1 + 1 / ome.wald(1 / par3, par2 ** 2 / fin_t))

    # hack around numerical instability of wald simulator
    min_t = max(0, min(min_t, fin_t))
    return min_t, min_x


""""""
def sample_brownbr_ub_min(
    fin_t: float, 
    init_x: float, 
    fin_x: float,
    ub_x: float | None, 
    lb_min_x: float, 
    ub_min_x: float,
    ome: np.random.Generator,
    max_props: int = int(1e6),
) -> tuple[float, float]:

    for _ in range(max_props):
        min_t, min_x = sample_brownbr_min(fin_t, init_x, fin_x, lb_min_x, ub_min_x, ome)
        if min_t == 0 or min_t == fin_t:
            continue
        if ub_x is None:
            return min_t, min_x
        esc1 = sample_bessel3br_esc(min_t, 0, init_x - min_x, ub_x - min_x, None, None, ome)
        esc2 = sample_bessel3br_esc(fin_t - min_t, 0, fin_x - min_x, ub_x - min_x, None, None, ome)
        if not esc1 and not esc2:
            return min_t, min_x
    else:
        raise BudgetConstraintError('None of the proposals were accepted.')


""""""
def sample_brownbr_esc(
    fin_t: float, 
    init_x: float, 
    fin_x: float, 
    bar_x: float, 
    ub_x: float | None, 
    seed: float | None,
    ome: np.random.Generator,
) -> bool:

    if seed is None:
        seed = ome.uniform()

    if bar_x <= max(abs(init_x), abs(fin_x)):
        return True
    if ub_x is not None and ub_x < bar_x:
        return False

    for i, log_s in enumerate(series_brownbr_esc(fin_t, init_x, fin_x, bar_x, ub_x), 1):
        if (not i % 2 and np.exp(log_s) > seed) or (i % 2 and np.exp(log_s) < seed):
            return not bool(i % 2)
    else:
        return False


""""""
def sample_bessel3br_esc(
    fin_t: float, 
    init_x: float, 
    fin_x: float, 
    bar_x: float, 
    ub_x: float | None, 
    seed: float | None,
    ome: np.random.Generator,
) -> bool:

    if seed is None:
        seed = ome.uniform()

    if bar_x <= max(init_x, fin_x):
        return True
    if ub_x is not None and ub_x < bar_x:
        return False

    gen = series_bessel3br_esc(fin_t, init_x, fin_x, bar_x, ub_x)

    # fast forward to reach cauchy sequence
    i0 = 1
    if not 3 * bar_x ** 2 > fin_t:
        seq = [0]
        for i, (log_s, sgn_s) in enumerate(gen, i0):
            seq.append(sgn_s * np.exp(log_s))
            if (i > 2 and i % 2 and seq[-1] < 1 and seq[i - 2] >= seq[i] >= seq[i - 1]):
                i0 = i + 1
                break

    for i, (log_s, sgn_s) in enumerate(gen, i0):
        if (not i % 2 and sgn_s * np.exp(log_s) > seed) or (i % 2 and sgn_s * np.exp(log_s) < seed):
            return not bool(i % 2)
    else:
        return False


""""""
def series_brownbr_esc(
    fin_t: float,
    init_x: float,
    fin_x: float,
    bar_x: float,
    ub_x: float | None = None,
    max_terms: int = int(1e9),
) -> Iterator[float]:

    def up(j: int, a: float, b: float) -> float:
        return -2 * (bar_x * (2 * j - 1) - a) * (bar_x * (2 * j - 1) - b) / fin_t

    def down(j: int, a: float, b: float) -> float:
        return -2 * j * (4 * bar_x ** 2 * j + 2 * bar_x * (a - b)) / fin_t

    # # handle trivial case
    # if bar_x == np.abs(init_x) or bar_x == np.abs(fin_x):
    #     while True:
    #         yield 0.0

    # handle bounded case recursively
    if ub_x is not None:
        num_seq = series_brownbr_esc(fin_t, init_x, fin_x, bar_x)
        den_seq = series_brownbr_esc(fin_t, init_x, fin_x, ub_x)
        next(den_seq)
        for log_r, log_s in zip(num_seq, den_seq):
            yield logsubexp(log_r, log_s) - logsubexp(0, log_s) # 1 - (1 - r) / (1 - s)

    # handle unbounded case
    log_s = -np.inf
    for i in range(1, max_terms):
        if i % 2:
            inc = logaddexp(up((i + 1) // 2, init_x, fin_x), up((i + 1) // 2, -init_x, -fin_x))
            log_s = logaddexp(log_s, inc)
        else:
            inc = logaddexp(down(i // 2, init_x, fin_x), down(i // 2, -init_x, -fin_x))
            log_s = logsubexp(log_s, inc)
        yield log_s
    else:
        raise BudgetConstraintError('The maximum sequence length was exceeded.')


""""""
def series_bessel3br_esc(
    fin_t: float,
    init_x: float,
    fin_x: float,
    bar_x: float,
    ub_x: float = None,
    max_terms: int = int(1e9),
) -> Iterator[tuple[float, int]]:

    def inc(i: int, b: float) -> float:
        return np.log(2 * bar_x * i + b) - 2 * bar_x * i * (bar_x * i + b) / fin_t

    # handle unbounded case recursively
    if ub_x is not None:
        num_seq = series_bessel3br_esc(fin_t, init_x, fin_x, bar_x)
        den_seq = series_brownbr_esc(fin_t, init_x, fin_x, ub_x)
        next(den_seq)
        for (log_r, sgn_r), log_s in zip(num_seq, den_seq):
            log_num, sgn_num = logaddexp_safe(log_r, log_s, sgn_r, -1)
            log_den, sgn_den = logaddexp_safe(0, log_s, 1, -1)
            yield log_num - log_den, min(sgn_num, sgn_den) # yield 1 - (1 - r) / (1 - s)

    # handle unbounded case starting from positive value
    if init_x != 0:
        log_s = -2 * init_x * fin_x / fin_t
        for log_r in series_brownbr_esc(fin_t, init_x - bar_x / 2, fin_x - bar_x / 2, bar_x / 2):
            log_num, sgn_num = logaddexp_safe(log_r, log_s, 1, -1)
            log_den, sgn_den = logaddexp_safe(0, log_s, 1, -1)
            yield log_num - log_den, min(sgn_num, sgn_den)  # yield 1 - (1 - s) / (1 - np.exp(-2 * init_x * fin_x / fin_t))

    sgn = 1
    log_s = -np.inf
    for j in range(1, max_terms):
        if j % 2:
            log_s, sgn = logaddexp_safe(log_s, inc((j + 1) // 2, -fin_x) - np.log(fin_x), sgn, 1)
        else:
            log_s, sgn = logaddexp_safe(log_s, inc(j // 2, fin_x) - np.log(fin_x), sgn, -1)
        # if sgn < 1:
        #     yield np.nan
        yield log_s, sgn
    else:
        raise BudgetConstraintError('The maximum sequence length was exceeded.')


def logsubexp(a: float, b: float) -> float:

    return scipy.special.logsumexp((a, b), b=(1, -1))


def logaddexp(a: float, b: float) -> float:

    return scipy.special.logsumexp((a, b), b=(1, 1))


def logaddexp_safe(a: float, b: float, sa: int, sb: int) -> tuple[float, int]:

    return scipy.special.logsumexp((a, b), b=(sa, sb), return_sign=True)


# """"""
# def sample_oneside_min(
#     fin_t: float, 
#     init_x: float, 
#     fin_x: float, 
#     ll: tuple[float, float],
#     ul: tuple[float, float],
#     ome: np.random.Generator,
#     max_props: int = int(1e6),
# ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[bool, bool], bool]:
    
#     def sample_mincase(
#         ll_: tuple[float, float], 
#         ul_: tuple[float, float],
#     ) -> tuple[tuple[float, float], tuple[bool, bool]]: 
#         for _ in range(max_props):
#             min_t, min_x = sample_brownbr_ub_min(fin_t, init_x, fin_x, ul_[1], *ll_, ome)
#             esc1 = sample_bessel3br_esc(min_t, 0, init_x - min_x, ul_[0] - min_x, ul_[1] - min_x, None, ome)
#             esc2 = sample_bessel3br_esc(fin_t - min_t, 0, fin_x - min_x, ul_[0] - min_x, ul_[1] - min_x, None, ome)
#             if (not esc1 and not esc2) or (ome.uniform() < .5):
#                 return (min_t, min_x), (esc1, esc2)
#         else:
#             raise BudgetConstraintError('None of the proposals were accepted.')

#     log_p_lo = logcdf_brownbr_min(ll[1], fin_t, init_x, fin_x, ll[0])
#     log_p_hi = logcdf_brownbr_min(-ul[0], fin_t, -fin_x, -init_x, -ul[1])
#     if np.log(ome.uniform(0, 1)) < log_p_lo - np.logaddexp(log_p_lo, log_p_hi):
#         min_x, hit = sample_mincase(ll, ul)
#         return min_x, ll, ul, hit, False
#     # use point symmetry of brownian bridge law for maximum
#     ll_reflx = (init_x + fin_x - ul[1], min(init_x + fin_x - ul[0], init_x, fin_x))
#     ul_reflx = (max(init_x + fin_x - ll[1], init_x, fin_x), init_x + fin_x - ll[0])
#     min_x, hit = sample_mincase(ll_reflx, ul_reflx)
#     return min_x, ll_reflx, ul_reflx, hit, True


# """"""
# def sample_twoside_min(
#     fin_t: float, 
#     init_x: float, 
#     fin_x: float, 
#     ll: tuple[float, float],
#     ul: tuple[float, float],
#     ome: np.random.Generator,
#     max_props: int = int(1e6),
# ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[bool, bool], bool]:
    
#     def sample_mincase(
#         ll_: tuple[float, float], 
#         ul_: tuple[float, float],
#     ) -> tuple[tuple[float, float], tuple[bool, bool]]: 
#         for _ in range(max_props):
#             min_t, min_x = sample_brownbr_ub_min(fin_t, init_x, fin_x, ul_[1], *ll_, ome)
#             esc1 = sample_bessel3br_esc(min_t, 0, init_x - min_x, ul_[0] - min_x, ul_[1] - min_x, None, ome)
#             esc2 = sample_bessel3br_esc(fin_t - min_t, 0, fin_x - min_x, ul_[0] - min_x, ul_[1] - min_x, None, ome)
#             if (esc1 or esc2):
#                 return (min_t, min_x), (esc1, esc2)
#         else:
#             raise BudgetConstraintError('None of the proposals were accepted.')
    
#     if ome.uniform() < .5:
#         min_x, hit = sample_mincase(ll, ul)
#         return min_x, ll, ul, hit, False
#     # use point symmetry of brownian bridge law for maximum
#     ll_reflx = (init_x + fin_x - ul[1], min(init_x + fin_x - ul[0], init_x, fin_x))
#     ul_reflx = (max(init_x + fin_x - ll[1], init_x, fin_x), init_x + fin_x - ll[0])
#     min_x, hit = sample_mincase(ll_reflx, ul_reflx)
#     return min_x, ll_reflx, ul_reflx, hit, True
