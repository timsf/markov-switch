import itertools as it
from typing import Iterator, NamedTuple

import numpy as np
import numpy.typing as npt
from scipy.special import loggamma


IntArr = npt.NDArray[np.integer]
FloatArr = npt.NDArray[np.floating] 


class Skeleton(NamedTuple):
    t: FloatArr
    xt: IntArr
    fin_t: float

    def __call__(self, t: FloatArr) -> IntArr:
        if np.any(t > self.fin_t):
            raise ValueError
        return self.xt[np.searchsorted(self.t, t, side='right') - 1]

    def __eq__(self, skel) -> bool:
        return all([len(self.t) == len(skel.t) and np.all(self.t == skel.t),
                    len(self.xt) == len(skel.xt) and np.all(self.xt == skel.xt),
                    self.fin_t == skel.fin_t])


Partition = list[Skeleton]


def sample_batch_ppp(
    intensity: float, 
    bounds: tuple[float, ...], 
    batch_size: int,
    ome: np.random.Generator,
) -> Iterator[FloatArr]:
    """Sample from a Poisson point process in batches.

    >>> # generate fixture
    >>> gen = int(1e4)
    >>> bound = np.random.lognormal(size=2)

    >>> # draw iid samples
    >>> y = np.vstack([ppp for ppp in sample_batch_ppp(gen, tuple(bound))])

    >>> # test sampling distribution
    >>> from scipy.stats import kstest, uniform
    >>> alpha = 1e-2
    >>> sample = (y / bound).flatten()
    >>> dist = uniform()
    >>> alpha < kstest(sample, dist.cdf)[1]
    True
    """

    assert 0 < intensity
    assert 0 < min(bounds)
    assert 0 < batch_size

    n_points = ome.poisson(intensity * np.prod(bounds))
    batch_sizes = it.repeat(batch_size, int(n_points // batch_size))
    if n_points % batch_size != 0:
        batch_sizes = it.chain(batch_sizes, [n_points % batch_size])

    ub = 0
    for i in batch_sizes:
        lb, ub = ub, ub + bounds[0] * ome.beta(i, n_points + 1 - i)
        if i == 1:
            u0 = np.array([ub])
        else:
            u0 = np.hstack([np.sort(ome.uniform(lb, ub, i - 1)), ub])
        u = ome.uniform(np.zeros(len(bounds[1:])), bounds[1:], (i, len(bounds[1:])))
        yield np.vstack([u0, u.T]).T


def sample_ppp(
    intensity: float, bound: tuple[float, ...], ome: np.random.Generator,
) -> FloatArr:
    """Sample from a Poisson point process on the plane.

    :param intensity:
    :param bound:
    :param ome:
    :return:

    >>> # generate fixture
    >>> gen = int(1e4)
    >>> bound = np.random.lognormal(size=2)

    >>> # draw iid samples
    >>> y = sample_ppp(gen, tuple(bound))

    >>> # test sampling distribution
    >>> from scipy.stats import kstest, uniform
    >>> alpha = 1e-2
    >>> sample = (y / bound).flatten()
    >>> dist = uniform()
    >>> alpha < kstest(sample, dist.cdf)[1]
    True
    """

    assert 0 <= intensity
    assert 0 <= min(bound)

    n_points = ome.poisson(intensity * np.prod(bound))
    locations = ome.uniform(np.zeros(len(bound)), bound, (n_points, len(bound)))

    return locations[np.argsort(locations[:, 0])]


def sample_forwards(
    init_t: float,
    fin_t: float,
    init_x: int | None,
    gen: FloatArr,
    ome: np.random.Generator,
) -> Skeleton:
    """
    :param init_t:
    :param fin_t:
    :param init_x:
    :param gen:
    :param ome:
    :return:

    >>> # generate fixture
    >>> dim = 3
    >>> fin_t = 1e5
    >>> gen = sample_trial_generator(dim)
    >>> stat = get_stat(gen)

    >>> # test stationary distribution
    >>> skel = sample_forwards(0, fin_t, None, gen)
    >>> probs = [np.sum(np.ediff1d(skel.t, to_end=(fin_t - skel.t[-1],))[skel.xt == i]) / fin_t for i in range(dim)]
    >>> np.allclose(probs, stat, 1e-2)
    True
    """

    dim = gen.shape[0]
    if init_x is None:
        init_x = ome.choice(dim, p=get_stat(gen))

    t, x = [init_t], [init_x]
    while t[-1] < fin_t:
        rate = np.delete(gen[x[-1]], x[-1])
        hold = ome.exponential(1 / np.where(rate == 0, np.nan, rate))
        move = int(np.nanargmin(hold))
        t.append(t[-1] + hold[move])
        x.append(move + int(move >= x[-1]))
    return Skeleton(np.array(t[:-1]), np.array(x[:-1]), fin_t)


def sample_backwards(
    init_t: float,
    fin_t: float,
    fin_x: int,
    gen: FloatArr,
    ome: np.random.Generator,
) -> Skeleton:
    """
    :param init_t:
    :param fin_t:
    :param fin_x:
    :param gen:
    :param ome:
    :return:

    >>> # generate fixture
    >>> dim = 3
    >>> fin_t = 1e5
    >>> gen = sample_trial_generator(dim)
    >>> stat = get_stat(gen)
    >>> fin_x = np.random.choice(dim, 1, p=get_stat(gen)).item()

    >>> # test stationary distribution
    >>> skel = sample_backwards(0, fin_t, fin_x, gen)
    >>> probs = [np.sum(np.ediff1d(skel.t, to_end=(fin_t - skel.t[-1],))[skel.xt == i]) / fin_t for i in range(dim)]
    >>> np.allclose(probs, stat, 1e-2)
    True
    """

    return sample_backwards_eig(init_t, fin_t, fin_x, gen, eig_gen(gen), ome)


def sample_backwards_eig(
    init_t: float,
    fin_t: float,
    fin_x: int,
    gen: FloatArr,
    gen_eig: tuple[FloatArr, FloatArr, FloatArr],
    ome: np.random.Generator,
) -> Skeleton:

    ftrans = get_t_trans(fin_t - init_t, gen_eig, np.arange(gen.shape[1]), np.array([fin_x]))[:, 0]
    btrans = ftrans / np.sum(ftrans)
    init_x = ome.choice(len(btrans), p=btrans)
    return sample_bridge_eig(init_t, fin_t, init_x, fin_x, gen, gen_eig, ome)


def sample_bridge(
    init_t: float,
    fin_t: float,
    init_x: int,
    fin_x: int,
    gen: FloatArr,
    ome: np.random.Generator,
) -> Skeleton:
    """
    :param init_t:
    :param fin_t:
    :param init_x:
    :param fin_x:
    :param gen:
    :param ome:
    :return:

    >>> # generate fixture
    >>> dim = 3
    >>> fin_t = 1e5
    >>> gen = sample_trial_generator(dim)
    >>> stat = get_stat(gen)
    >>> init_x, fin_x = sample_forwards(0, fin_t, None, gen)[1][[0, -1]]

    >>> # test stationary distribution
    >>> skel = sample_bridge(0, fin_t, init_x, fin_x, gen)
    >>> probs = [np.sum(np.ediff1d(skel.t, to_end=(fin_t - skel.t[-1],))[skel.xt == i]) / fin_t for i in range(dim)]
    >>> np.allclose(probs, stat, 1e-2)
    True
    """

    return sample_bridge_eig(init_t, fin_t, init_x, fin_x, gen, eig_gen(gen), ome)


def sample_bridge_eig(
    init_t: float,
    fin_t: float,
    init_x: int,
    fin_x: int,
    gen: FloatArr,
    gen_eig: tuple[FloatArr, FloatArr, FloatArr],
    ome: np.random.Generator,
    min_rej_prob: float = .1,
) -> Skeleton:

    def sample_n_trans() -> int:
        u = np.log(ome.uniform()) + np.log(ctrans) + rate_bound * (fin_t - init_t)
        p = [0 if init_x == fin_x else -np.inf]
        n = 0
        while p[-1] < u:
            n += 1
            pow_dtrans = get_n_trans(n, dtrans_eig, np.array([init_x]), np.array([fin_x]))[0, 0]
            if pow_dtrans == 0:
                p.append(p[-1])
            else:
                p.append(np.logaddexp(p[-1], n * np.log(rate_bound * (fin_t - init_t)) - loggamma(n + 1) + np.log(pow_dtrans)))
        return n

    ctrans = get_t_trans(fin_t - init_t, gen_eig, np.array([init_x]), np.array([fin_x]))[0, 0]
    if ctrans > min_rej_prob:
        return sample_bridge_rej(init_t, fin_t, init_x, fin_x, gen, ome)

    dim = gen.shape[0]
    rate_bound = -np.min(np.diag(gen))
    dtrans = np.identity(dim) + gen / rate_bound
    dtrans_eig = (gen_eig[0] / rate_bound + 1, gen_eig[1], gen_eig[2])
    n_trans = sample_n_trans()

    if n_trans == 0 or (n_trans == 1 and init_x == fin_x):
        return Skeleton(np.array([init_t]), np.array([init_x]), fin_t)

    x = [init_x]
    for i in range(1, n_trans):
        q = dtrans[x[-1]] * get_n_trans(n_trans - i, dtrans_eig, np.arange(dim), np.array([fin_x]))[:, 0]
        x.append(ome.choice(dim, p=q / np.sum(q)))
    x.append(fin_x)

    t = np.hstack([init_t, np.sort(ome.uniform(init_t, fin_t, n_trans))])
    x = np.array(x)
    is_virtual = np.append(False, x[1:] == x[:-1])

    return Skeleton(t[~is_virtual], x[~is_virtual], fin_t)


def sample_bridge_rej(
    init_t: float,
    fin_t: float,
    init_x: int,
    fin_x: int,
    gen: FloatArr,
    ome: np.random.Generator,
) -> Skeleton:

    while True:
        if init_x == fin_x:
            skel = sample_forwards(init_t, fin_t, init_x, gen, ome)
            if skel.xt[-1] == fin_x:
                return skel
        else:
            t1 = np.log(1 - ome.uniform() * (1 - np.exp((fin_t - init_t) * gen[init_x, init_x]))) / gen[init_x, init_x] + init_t
            p1 = np.where(gen[init_x] > 0, gen[init_x], 0) / np.sum(np.delete(gen[init_x], init_x))
            x1 = ome.choice(gen.shape[0], p=p1)
            partial_skel = sample_forwards(t1, fin_t, x1, gen, ome)
            if partial_skel.xt[-1] == fin_x:
                return Skeleton(np.append(0, partial_skel.t), np.append(init_x, partial_skel.xt), partial_skel.fin_t)


def sample_leapfrog(
    skel: Skeleton,
    t_cond: FloatArr,
    gen: FloatArr,
    ome: np.random.Generator,
) -> Skeleton:

    return paste_partition(mutate_partition(partition_skeleton(skel, t_cond), gen, ome))


def sample_partition(
    skel: Skeleton, intensity: float, ome: np.random.Generator,
) -> tuple[Partition, FloatArr]:
    """
    :param skel:
    :param intensity:
    :param ome:
    :return:

    >>> # generate fixture
    >>> dim = 3
    >>> fin_t = 1e2
    >>> gen = sample_trial_generator(dim)
    >>> stat = get_stat(gen)
    >>> skel = sample_forwards(0, fin_t, None, gen)

    >>> # test
    >>> skel2 = paste_partition(sample_partition(skel)[0])
    >>> np.all(skel.t == skel2.t), np.all(skel.xt == skel2.xt), skel.fin_t == skel2.fin_t
    (True, True, True)
    """

    n_breaks = ome.poisson(intensity * skel.fin_t)
    new_t = np.linspace(0, skel.fin_t, n_breaks + 2)

    return partition_skeleton(skel, new_t[1:-1]), new_t[1:-1]


def paste_partition(partition: list[Skeleton]) -> Skeleton:
    """
    :param partition:
    :return:

    >>> # generate fixture
    >>> dim = 3
    >>> fin_t = 1e2
    >>> gen = sample_trial_generator(dim)
    >>> stat = get_stat(gen)
    >>> skel = sample_forwards(0, fin_t, None, gen)

    >>> # test
    >>> skel2 = paste_partition(sample_partition(skel)[0])
    >>> np.all(skel.t == skel2.t), np.all(skel.xt == skel2.xt), skel.fin_t == skel2.fin_t
    (True, True, True)
    """

    partition = [skel for skel in partition if skel.fin_t != 0]
    breaks = np.append(0, [skel.fin_t for skel in partition])
    t, x = [np.hstack(a) for a in zip(*[(skel.t, skel.xt) for _, skel in zip(breaks[:-1], partition)])]
    is_virtual = np.hstack([False, x[1:] == x[:-1]])
    return Skeleton(t[~is_virtual], x[~is_virtual], breaks[-1])


def mutate_partition(
    partition: list[Skeleton], gen: FloatArr, ome: np.random.Generator,
) -> Partition:
    """
    :param partition:
    :param gen:
    :param ome:
    :return:

    >>> # generate fixture
    >>> dim = 3
    >>> fin_t = 1e5
    >>> gen = sample_trial_generator(dim)
    >>> stat = get_stat(gen)
    >>> skel = sample_forwards(0, fin_t, None, gen)

    >>> # test stationary distribution
    >>> skel = paste_partition(mutate_partition(sample_partition(skel)[0], gen))
    >>> probs = [np.sum(np.ediff1d(skel.t, to_end=(fin_t - skel.t[-1],))[skel.xt == i]) / fin_t for i in range(dim)]
    >>> np.allclose(probs, stat, 1e-2)
    True
    """

    if len(partition) > 1:
        gen_eig = eig_gen(gen)
        init_skel = sample_backwards_eig(partition[0].t[0], partition[0].fin_t, partition[0].xt[-1], gen, gen_eig, ome)
        fin_skel = sample_forwards(partition[-1].t[0], partition[-1].fin_t, partition[-1].xt[0], gen, ome)
        new_partition = [init_skel] \
            + [sample_bridge_eig(skel.t[0], skel.fin_t, skel.xt[0], skel.xt[-1], gen, gen_eig, ome) for skel in partition[1:-1]] \
            + [fin_skel]
    else:
        new_partition = [sample_forwards(partition[0].t[0], partition[0].fin_t, None, gen, ome)]

    return new_partition


def partition_skeleton(skel: Skeleton, new_t: FloatArr) -> Partition:
    """
    :param skel:
    :param new_t:

    >>> # generate fixture
    >>> dim = 3
    >>> fin_t = 1e2
    >>> gen = sample_trial_generator(dim)
    >>> stat = get_stat(gen)
    >>> new_t = np.sort(np.random.uniform(0, fin_t, 100))
    >>> skel = sample_forwards(0, fin_t, None, gen)

    >>> # test
    >>> skel2 = paste_partition(partition_skeleton(skel, new_t))
    >>> np.all(skel.t == skel2.t), np.all(skel.xt == skel2.xt), skel.fin_t == skel2.fin_t
    (True, True, True)
    """

    new_t = np.hstack([0, new_t, skel.fin_t])
    slice_right = np.searchsorted(skel.t, new_t, side='right')
    slice_left = np.searchsorted(skel.t, new_t, side='left')

    t = [np.append(init_t, skel.t[i0:i1])
         for init_t, i0, i1
         in zip(new_t, slice_right, slice_left[1:])]
    x = [skel.xt[i0 - 1:i0 - 1 + len(t_)] for i0, t_ in zip(slice_right, t)]

    return [Skeleton(t_, x_, fin_t) for t_, x_, fin_t in zip(t, x, new_t[1:])]


def repartition_skeleton(partition: Partition, new_t: FloatArr) -> Partition:
    """
    :param partition:
    :param new_t:
    :return:

    >>> # generate fixture
    >>> dim = 3
    >>> fin_t = 1e2
    >>> gen = sample_trial_generator(dim)
    >>> stat = get_stat(gen)
    >>> new_t = np.sort(np.random.uniform(0, fin_t, 100))
    >>> skel = sample_forwards(0, fin_t, None, gen)

    >>> # test
    >>> skel2 = paste_partition(repartition_skeleton(partition_skeleton(skel, new_t), new_t))
    >>> np.all(skel.t == skel2.t), np.all(skel.vt == skel2.vt), skel.fin_t == skel2.fin_t
    (True, True, True)
    """

    return partition_skeleton(paste_partition(partition), new_t)


def sample_trial_generator(dim: int, ome: np.random.Generator) -> FloatArr:
    """
    :param dim:
    :parma ome:
    :return:
    """

    gen = ome.lognormal(size=(dim, dim))
    np.fill_diagonal(gen, 0)
    np.fill_diagonal(gen, -np.sum(gen, 1))
    return gen


def est_stat(skel: Skeleton, states: IntArr) -> FloatArr:
    """
    :param skel:
    :param states:
    :return:
    """

    return (skel.xt == states[:, np.newaxis]) \
        @ np.ediff1d(np.float64(skel.t), to_end=(skel.fin_t - skel.t[-1],)) / skel.fin_t


def get_stat(gen: FloatArr) -> FloatArr:
    """
    :param gen:
    :return:
    """

    return np.linalg.solve(1 - gen.T, np.ones(gen.shape[0]))


def get_t_trans(
    fin_t: float,
    gen_eig: tuple[FloatArr, FloatArr, FloatArr],
    rows: IntArr,
    cols: IntArr,
) -> FloatArr:

    t_trans = restore_matrix(np.exp(fin_t * gen_eig[0]), gen_eig[1], gen_eig[2], rows, cols).real
    return np.where(t_trans < 0, 0, t_trans)


def get_n_trans(
    n: int,
    gen_eig: tuple[FloatArr, FloatArr, FloatArr],
    rows: IntArr,
    cols: IntArr,
) -> FloatArr:

    n_trans = restore_matrix(gen_eig[0] ** n, gen_eig[1], gen_eig[2], rows, cols).real
    return np.where(n_trans < 0, 0, n_trans)


def restore_matrix(
    eig_val: FloatArr,
    eig_vec: FloatArr,
    inv_eig_vec: FloatArr,
    rows: IntArr,
    cols: IntArr,
) -> FloatArr:

    return (eig_vec[rows] * eig_val) @ inv_eig_vec[:, cols]


def eig_gen(gen: FloatArr) -> tuple[FloatArr, FloatArr, FloatArr]:

    eig_val, eig_vec = np.linalg.eig(gen)
    return eig_val, eig_vec, np.linalg.inv(eig_vec)
