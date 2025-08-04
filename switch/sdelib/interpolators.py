import numpy as np
import numpy.typing as npt

from switch.sdelib import bessel, paths


BoolArr = npt.NDArray[np.bool_]
IntArr = npt.NDArray[np.integer]
FloatArr = npt.NDArray[np.floating]


def restore_besselbr(
    u: FloatArr,
    z: FloatArr,
    fin_u: float,
    fin_r: float,
) -> FloatArr:
    
    return np.sqrt(z[0] ** 2 + z[1] ** 2 + (z[2] + fin_r * u / fin_u) ** 2)


def fill_besselbr_section(
    new_u: FloatArr,
    init_u: float,
    fin_u: float,
    init_z: FloatArr,
    fin_z: FloatArr,
    hor_u: float,
    hor_r: float,
    sup_r: float | None,
    hit_r: float | None,
    ome: np.random.Generator,
) -> tuple[FloatArr, bool | None]:
    
    if len(new_u) == 0:
        if hit_r is not None:
            return np.ndarray(shape=(3, 0)), True
        else:
            return np.ndarray(shape=(3, 0)), None
    
    stack_u = np.hstack([init_u, new_u, fin_u])
    hit = None
    while True:
        new_z = paths.sample_brownbrvec(new_u - init_u, fin_u - init_u, init_z, fin_z, ome)
        stack_z = np.hstack([init_z[:, np.newaxis], new_z, fin_z[:, np.newaxis]])
        stack_r = restore_besselbr(stack_u, stack_z, hor_u, hor_r)
        if sup_r is not None:
            esc = any([bessel.sample_bessel3br_esc(dt, r0, r1, sup_r, None, None, ome) for dt, r0, r1 in zip(np.diff(stack_u), stack_r, stack_r[1:])])
            if esc:
                continue
            if hit_r is not None:
                hit = any([bessel.sample_bessel3br_esc(dt, r0, r1, hit_r, sup_r, None, ome) for dt, r0, r1 in zip(np.diff(stack_u), stack_r, stack_r[1:])])
            else:
                hit = None
        return new_z, hit


def fill_besselbr_path(
    u: FloatArr,
    z: FloatArr,
    fin_r: float,
    sup_r: float | None,
    hit_r: float | None,
    new_u: FloatArr,
    ome: np.random.Generator,
) -> tuple[FloatArr, FloatArr]:

    present_u = np.isin(new_u, u)
    new_uu = new_u[~present_u]
    if len(new_uu) == 0:
        return u, z
    insert_to = np.searchsorted(u, new_uu, side='left')
    stack_u = np.insert(np.float64(u), insert_to, new_uu)
    while True:
        new_z, hit = zip(*[fill_besselbr_section(new_uu[insert_to == i], u[i-1], u[i], z[:, i-1], z[:, i], u[-1], fin_r, sup_r, hit_r, ome) for i in range(1, len(u))])
        stack_z = np.insert(np.float64(z), insert_to, np.hstack(new_z), axis=1)
        if hit_r is not None:
            if not any(hit):
                continue
        return stack_u, np.vstack(list(stack_z))
    

def fill_brownbr_path(
    u: FloatArr,
    z: FloatArr,
    new_u: FloatArr,
    ome: np.random.Generator,
) -> tuple[FloatArr, FloatArr]:
    
    present_u = np.isin(new_u, u)
    new_uu = new_u[~present_u]
    if len(new_uu) == 0:
        return u, z
    insert_to = np.searchsorted(u, new_uu, side='left')
    stack_u = np.insert(np.float64(u), insert_to, new_uu)
    while True:
        new_z = [paths.sample_brownbrvec(new_uu[insert_to == i], u[i] - u[i-1], z[:, i-1], z[:, i], ome) for i in range(1, len(u))]
        stack_z = np.insert(np.float64(z), insert_to, np.hstack(new_z), axis=1)
        return stack_u, np.vstack(list(stack_z))
