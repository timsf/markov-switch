from typing import Callable, Iterator, TypeVar

import numpy as np
import numpy.typing as npt

from switch.misc.exceptions import BudgetConstraintError


FloatArr = npt.NDArray[np.floating]


S = TypeVar('S')
T = TypeVar('T')


def sample_odds_dual(
    z1: list[list[S]],
    z2: list[list[S]],
    log_odds: FloatArr,
    c1: list[list[Callable[[S], tuple[bool, S]]]],
    c2: list[list[Callable[[S], tuple[bool, S]]]],
    n_splits: int,
    pr_portkey: float,
    ome: np.random.Generator,
) -> Iterator[tuple[bool, list[S], list[S]]]:
    
    if n_splits == 0 or len(z1) == 1:
        while True:
            log_prob = np.sum(log_odds) - np.logaddexp(0, np.sum(log_odds))
            yield sample_twocoin_dual(z1, z2, log_prob, bundle_coins([bundle_coins(a) for a in c1]), bundle_coins([bundle_coins(a) for a in c2]), pr_portkey, ome)
    split_ix = len(log_odds) // 2
    perm = ome.permutation(np.arange(len(log_odds)))
    ix1, ix2 = perm[:split_ix], perm[split_ix:]
    s1 = sample_odds_dual([z1[i] for i in ix1], [z2[i] for i in ix1], log_odds[ix1], [c1[i] for i in ix1], [c2[i] for i in ix1], n_splits - 1, pr_portkey, ome)
    s2 = sample_odds_dual([z1[i] for i in ix2], [z2[i] for i in ix2], log_odds[ix2], [c1[i] for i in ix2], [c2[i] for i in ix2], n_splits - 1, pr_portkey, ome)
    while True:
        success1, new_z11, new_z21 = next(s1)
        success2, new_z12, new_z22 = next(s2)
        new_z1 = new_z11 + new_z12
        new_z2 = new_z21 + new_z22
        if success1 and success2:
            yield True, [new_z1[i] for i in np.argsort(perm)], [new_z2[i] for i in np.argsort(perm)]
        if not success1 and not success2:
            yield False, [new_z1[i] for i in np.argsort(perm)], [new_z2[i] for i in np.argsort(perm)]
        

def sample_odds_single(
    z: list[S],
    log_odds: FloatArr,
    c1: list[Callable[[S], tuple[bool, S]]],
    c2: list[Callable[[S], tuple[bool, S]]],
    n_splits: int,
    pr_portkey: float,
    ome: np.random.Generator,
) -> Iterator[tuple[bool, list[S]]]:

    if n_splits == 0 or len(z) == 1:
        while True:
            log_prob = np.sum(log_odds) - np.logaddexp(0, np.sum(log_odds))
            yield sample_twocoin_single(z, log_prob, bundle_coins(c1), bundle_coins(c2), pr_portkey, ome)
    split_ix = len(z) // 2
    perm = ome.permutation(np.arange(len(z)))
    ix1, ix2 = perm[:split_ix], perm[split_ix:]
    s1 = sample_odds_single([z[i] for i in ix1], log_odds[ix1], [c1[i] for i in ix1], [c2[i] for i in ix1], n_splits - 1, pr_portkey, ome)
    s2 = sample_odds_single([z[i] for i in ix2], log_odds[ix2], [c1[i] for i in ix2], [c2[i] for i in ix2], n_splits - 1, pr_portkey, ome)
    while True:
        success1, new_z1 = next(s1)
        success2, new_z2 = next(s2)
        new_z = new_z1 + new_z2
        if success1 and success2:
            yield True, [new_z[i] for i in np.argsort(perm)]
        if not success1 and not success2:
            yield False, [new_z[i] for i in np.argsort(perm)]


def sample_twocoin_dual(
    z1: T,
    z2: T,
    log_pr_c1: float,
    coin1: Callable[[T], tuple[bool, T]],
    coin2: Callable[[T], tuple[bool, T]],
    pr_portkey: float,
    ome: np.random.Generator,
    max_props: int = int(1e5),
) -> tuple[bool, T, T]:

    new_z1 = z1
    new_z2 = z2
    for _ in range(max_props):

        # attempt escape or abort pathological proposal
        if ome.uniform() < pr_portkey:
            raise BudgetConstraintError

        # attempt proposal from coin 1
        if np.log(ome.uniform()) < log_pr_c1:
            success, new_z1 = coin1(new_z1)
            if success:
                return True, new_z1, new_z2
        # attempt proposal from coin 2
        else:
            success, new_z2 = coin2(new_z2)
            if success:
                return False, new_z1, new_z2

    # prevent infinite loop
    else:
        raise BudgetConstraintError


def sample_twocoin_single(
    z: T,
    log_pr_c1: float,
    coin1: Callable[[T], tuple[bool, T]],
    coin2: Callable[[T], tuple[bool, T]],
    pr_portkey: float,
    ome: np.random.Generator,
    max_props: int = int(1e5),
) -> tuple[bool, T]:
    
    new_z = z
    for _ in range(max_props):

        # attempt escape or abort pathological proposal
        if ome.uniform() < pr_portkey:
            raise BudgetConstraintError

        # attempt proposal from coin 1
        if np.log(ome.uniform()) < log_pr_c1:
            success, new_z = coin1(new_z)
            if success:
                return True, new_z
        # attempt proposal from coin 2
        else:
            success, new_z = coin2(new_z)
            if success:
                return False, new_z

    # prevent infinite loop
    else:
        raise BudgetConstraintError


def bundle_coins(coins: list[Callable[[S], tuple[bool, S]]]) -> Callable[[list[S]], tuple[bool, list[S]]]:
    
    def flip_coin(z: list[S]) -> tuple[bool, list[S]]:
        gen = (coins_(z_) for coins_, z_ in zip(coins, z))
        success, partial_z = list(zip(*take_while(lambda ret: ret[0], gen)))
        new_z = list(partial_z) + z[len(success):]
        return success[-1], new_z

    return flip_coin


def take_while(predicate: Callable, iterable: Iterator) -> Iterator:

    for x in iterable:
        if predicate(x):
            yield x
        else:
            yield x
            break
