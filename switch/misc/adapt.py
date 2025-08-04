import numpy as np
import numpy.typing as npt

from scipy.special import expit


FloatArr = npt.NDArray[np.floating]


def seq_update_normal(
    obs: FloatArr, 
    n: float, 
    mean: FloatArr, 
    cov: FloatArr,
) -> tuple[FloatArr, FloatArr]:
    """
    :param obs:
    :param n:
    :param mean:
    :param cov:
    :return:

    >>> y = np.random.standard_normal((10, 2))
    >>> mean, cov = np.mean(y[:2], 0), np.cov(y[:2].T)
    >>> for i in range(2, 10): mean, cov = seq_update_normal(y[i], i, mean, cov)
    >>> np.allclose(mean, np.mean(y, 0))
    True
    >>> np.allclose(cov, np.cov(y.T))
    True
    """

    dev = obs - mean
    mean = mean + dev / n
    cov = cov + (np.outer(dev, dev) - cov) / n

    return mean, cov


def transform_mv_to_reals(x: FloatArr, lb: FloatArr, ub: FloatArr) -> tuple[FloatArr, float]:

    y, log_jac_y = zip(*[transform_uv_to_reals(x_, lb_, ub_) for x_, lb_, ub_ in zip(x, lb, ub)])
    return np.array(y), sum(log_jac_y)


def transform_uv_to_reals(x: float, lb: float, ub: float) -> tuple[float, float]:

    if np.isinf(lb) and np.isinf(ub):
        return x, 0
    elif not np.isinf(lb) and not np.isinf(ub):
        return np.log((x - lb) / (ub - lb)) - np.log(1 - (x - lb) / (ub - lb)), (1 + 1 / (ub - lb)) / (x - lb)
    elif np.isinf(lb):
        return np.log(ub - x), -np.log(ub - x)
    else:
        return np.log(x - lb), -np.log(x - lb)


def transform_mv_to_const(y: FloatArr, lb: FloatArr, ub: FloatArr) -> tuple[FloatArr, float]:

    x, log_jac_x = zip(*[transform_uv_to_const(y_, lb_, ub_) for y_, lb_, ub_ in zip(y, lb, ub)])
    return np.array(x), sum(log_jac_x)


def transform_uv_to_const(y: float, lb: float, ub: float) -> tuple[float, float]:

    if np.isinf(lb) and np.isinf(ub):
        return y, 0
    elif not np.isinf(lb) and not np.isinf(ub):
        return lb + (ub - lb) / (1 + np.exp(-y)), (2 + np.exp(y) + np.exp(-y)) / (ub - lb)
    elif np.isinf(lb):
        return ub - np.exp(y), -y
    else:
        return lb + np.exp(y), -y


class MyopicRwSampler(object):

    def __init__(
        self, 
        mean: FloatArr, 
        cov: FloatArr, 
        log_scale: float, 
        bounds: tuple[FloatArr, FloatArr],
        opt_prob: float = .234, 
        adapt_decay: float = .25, 
        air: int = 1,
    ):

        self.prop_mean = self.running_mean = mean
        self.prop_cov = self.running_cov = cov
        self.bounds = bounds
        self.opt_prob = opt_prob
        self.adapt_decay = adapt_decay
        self.air = air
        self.log_prop_scale = [log_scale]
        self.emp_prob = [0.0]
        self.adapt_periods = [0, 1]
        self.iter = 0

    def propose(self, state: FloatArr, ome: np.random.Generator) -> tuple[FloatArr, float, float]:

        state_real, log_p_backw = transform_mv_to_reals(state, self.bounds[0], self.bounds[1])
        # prop_real = state_real + np.linalg.cholesky(np.exp(self.log_prop_scale[-1] * 2) * self.prop_cov) @ ome.uniform(-1, 1, size=len(state_real))
        prop_real = ome.multivariate_normal(state_real, np.exp(self.log_prop_scale[-1] * 2) * self.prop_cov)
        prop, log_p_forw = transform_mv_to_const(prop_real, self.bounds[0], self.bounds[1])
        return prop, log_p_forw, log_p_backw

    def adapt(self, state: FloatArr, acc_prob: float):

        self.iter += 1
        self.emp_prob[-1] = ((self.iter - self.adapt_periods[-2] - 1) * self.emp_prob[-1] + acc_prob) / (self.iter - self.adapt_periods[-2])
        state_real, _ = transform_mv_to_reals(state, self.bounds[0], self.bounds[1])
        self.running_mean, self.running_cov = seq_update_normal(state_real, 1 + len(self.emp_prob), self.running_mean, self.running_cov)
        if self.iter == self.adapt_periods[-1]:
            learning_rate = 1 / (len(self.adapt_periods) ** self.adapt_decay)
            self.log_prop_scale.append(self.log_prop_scale[-1] + learning_rate * (self.emp_prob[-1] - self.opt_prob))
            self.adapt_periods.append(self.adapt_periods[-1] + len(self.adapt_periods) ** self.air)
            self.emp_prob.append(0.0)
            self.prop_mean, self.prop_cov = self.running_mean, self.running_cov

    def reset(self):

        return MyopicRwSampler(self.prop_mean, self.prop_cov, self.log_prop_scale[-1], self.bounds,
                               self.opt_prob, self.adapt_decay, self.air)
 
 
class MyopicCompactSampler(object):

    def __init__(
        self, 
        mean: FloatArr, 
        cov: FloatArr, 
        log_scale: float, 
        bounds: tuple[FloatArr, FloatArr],
        opt_prob: float = .234, 
        adapt_decay: float = .25, 
        air: int = 1,
    ):

        self.prop_mean = self.running_mean = mean
        self.prop_cov = self.running_cov = cov
        self.bounds = bounds
        self.opt_prob = opt_prob
        self.adapt_decay = adapt_decay
        self.air = air
        self.log_prop_scale = [log_scale]
        self.emp_prob = [0.0]
        self.adapt_periods = [0, 1]
        self.iter = 0

    def propose(self, state: FloatArr, ome: np.random.Generator) -> tuple[FloatArr, float, float]:

        state_real, log_p_backw = transform_mv_to_reals(state, self.bounds[0], self.bounds[1])
        prop_real = state_real + np.linalg.cholesky(np.exp(self.log_prop_scale[-1] * 2) * self.prop_cov) @ ome.uniform(-1, 1, size=len(state))
        prop, log_p_forw = transform_mv_to_const(prop_real, self.bounds[0], self.bounds[1])
        return prop, log_p_forw, log_p_backw

    def adapt(self, state: FloatArr, acc_prob: float):

        self.iter += 1
        self.emp_prob[-1] = ((self.iter - self.adapt_periods[-2] - 1) * self.emp_prob[-1] + acc_prob) / (self.iter - self.adapt_periods[-2])
        state_real, _ = transform_mv_to_reals(state, self.bounds[0], self.bounds[1])
        self.running_mean, self.running_cov = seq_update_normal(state_real, 1 + len(self.emp_prob), self.running_mean, self.running_cov)
        if self.iter == self.adapt_periods[-1]:
            learning_rate = 1 / (len(self.adapt_periods) ** self.adapt_decay)
            self.log_prop_scale.append(self.log_prop_scale[-1] + learning_rate * (self.emp_prob[-1] - self.opt_prob))
            self.adapt_periods.append(self.adapt_periods[-1] + len(self.adapt_periods) ** self.air)
            self.emp_prob.append(0.0)
            self.prop_mean, self.prop_cov = self.running_mean, self.running_cov

    def reset(self):

        return MyopicCompactSampler(self.prop_mean, self.prop_cov, self.log_prop_scale[-1], self.bounds,
                                    self.opt_prob, self.adapt_decay, self.air)


class MyopicSkelSampler(object):

    def __init__(
        self, 
        breaks: FloatArr,
        init_scale: FloatArr,
        opt_prob: float = .234, 
        adapt_decay: float = .25, 
        air: int = 1,
    ):

        self.breaks = breaks
        self.opt_prob = opt_prob
        self.adapt_decay = adapt_decay
        self.air = air
        self.prop_scale = [[init_scale_] for init_scale_ in init_scale]
        self.emp_prob = [[0.0] for _ in range(len(breaks))]
        self.adapt_periods = [0, 1]
        self.iter = 0

    def propose(self, ome: np.random.Generator) -> FloatArr:

        cond_ix = ome.uniform(size=len(self.breaks)) < expit([-prop_scale_[-1] for prop_scale_ in self.prop_scale])
        return self.breaks[cond_ix]

    def adapt(self, t_split: FloatArr, acc_prob: FloatArr):

        self.iter += 1
        learning_rate = 1 / (len(self.adapt_periods) ** self.adapt_decay)
        for i in range(len(self.breaks)):
            if self.breaks[i] in t_split:
                j = list(t_split).index(self.breaks[i])
                i0 = j-1
                i1 = j+1
            else:
                j = np.searchsorted(t_split, self.breaks[i])
                i0 = j-1
                i1 = j
            prob = np.diff(t_split[i0:i1+1]) @ acc_prob[i0:i1] / (t_split[i1] - t_split[i0])
            self.emp_prob[i][-1] = ((self.iter - self.adapt_periods[-2] - 1) * self.emp_prob[i][-1] + prob) / (self.iter - self.adapt_periods[-2])
        if self.iter == self.adapt_periods[-1]:
            self.adapt_periods.append(self.adapt_periods[-1] + len(self.adapt_periods) ** self.air)
            for i in range(len(self.breaks)):
                self.prop_scale[i].append(self.prop_scale[i][-1] + learning_rate * (self.emp_prob[i][-1] - self.opt_prob))
                self.emp_prob[i].append(0.0)

    def reset(self):

        return MyopicSkelSampler(self.breaks, np.array([a[-1] for a in self.prop_scale]),
                                 self.opt_prob, self.adapt_decay, self.air)
