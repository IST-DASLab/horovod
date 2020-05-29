import numpy as np
import torch
import math
from .dist import TruncNorm, CondNormalTruncHist

EPS = 1e-7


def get_quantile_levels(bits, grad_dist):
    """quantile levels """
    num_levels = 2 << bits - 1
    cdf_points = np.linspace(0, 1, num=num_levels)
    levels = [grad_dist.ppf(level) for level in cdf_points]

    levels[0] = grad_dist.begin
    levels[-1] = grad_dist.end
    return levels

def get_uniform_levels(bits):
    """uniform (QSGD)"""
    num_levels = 2 << bits - 1
    levels_uni = np.linspace(-1, 1, num=num_levels)
    return levels_uni

def get_ternary_levels():
    return np.array([-1, 0, 1])


def get_exp_levels(bits, multiplier=0.5):
    """ exponential (NUQSGD)

    multiplier: is used to modify levels_exp based on the number of bits
    """
    num_levels = 2 << bits - 1

    levels = sum([[-multiplier**j for j in range(num_levels >> 1)],
                  [multiplier**j for j in reversed(range(num_levels >> 1))]],
                 [])
    return np.asarray(levels)


def finite_diff_gradient_descent(f, begin, end, x0=None, niters=10, lr=1):
    """Find the local minima using gradient descent

    Parameters:
        f (function): Function to find the local minima
        begin (int): beginning of the interval
        end (int): end of interval
        x0 (int): initial point
        niters (int): number of iterations
        lr: learning rate
    """
    eps = (end-begin)/1000
    if x0 is None:
        x0 = (begin + end) / 2
    x = x0
    for i in range(niters):
        df = (f(x+eps)-f(x-eps))/(2*eps)
        x -= lr*df
    return x


def bisection(begin, end, f):
    """Find the root using the bisection
    method.

    Parameters:
        begin (int): beginning of the interval
        end (int): end of the interval
        f (function): funciton to find the root for
    """
    x = (begin + end) / 2
    if (np.abs(f(x) - 0) < 1e-7):
        return x
    both_negative = f(begin) < 0 and f(end) < 0
    both_positive = f(begin) > 0 and f(end) > 0
    if both_negative or both_positive:
        print('Bisection failed')

    x_neg_end_pos = f(x) < 0 and f(end) > 0
    x_pos_end_neg = f(x) > 0 and f(end) < 0
    if x_neg_end_pos or x_pos_end_neg:
        return bisection(x, end, f)
    return bisection(begin, x, f)


def amq_norm_based(initial_point, grad_dist, bits, lr=0.1, epochs=50):
    """AMQ Norm-based implementation

    Parameters:
        initial_point (int): the initial multiplier
        grad_dist (dist.Distribution): is the distribution
        bits (int): number of bits
        lr (float): learning rate
        epochs (int): number of epochs
    """

    mul = initial_point
    s = 2 ** (bits - 1) - 1
    all_mul = []
    iter = 0
    for epoch in range(epochs):
        sum = 0.0
        for norm, mean, sigma, coeff in zip(
                grad_dist.norms,
                grad_dist.means,
                grad_dist.sigmas,
                grad_dist.coeff):

            dist_comp = TruncNorm(
                mean, sigma, grad_dist.begin, grad_dist.end, grad_dist.nbins)

            # from eq G.3 in Appendix
            def arg1_1(j):
                return mean * (j * mul ** (j - 1) + (j + 1) * mul ** j) \
                       - (2 * j + 1) * mul ** (2 * j)
            arg1 = np.sum(np.asarray(
                [arg1_1(j)*(dist_comp.cdf(mul**j) - dist_comp.cdf(mul**(j+1)))
                 for j in range(0, s)]))

            def arg2_1(j):
                return j * mul ** (j - 1) + (j + 1) * mul ** j
            arg2 = np.sum(np.asarray(
                [arg2_1(j) * (dist_comp.pdf(mul ** (j + 1))
                              - dist_comp.pdf(mul ** (j)))
                 for j in range(0, s)]))
            sum += coeff * (arg1 + sigma ** 2 * arg2)

        gradient = 2 * s * (mul ** (2 * s - 1)) * \
                   (grad_dist.cdf(mul ** s) - grad_dist.cdf(0)) + sum
        mul = mul - lr * gradient
        iter += 1
        all_mul.append(mul)
    return mul, all_mul


def amq_norm_less(initial_point, grad_dist, bits, lr=0.1, epochs=200):
    """AMQ Norm-less implementation

    Parameters:
        initial_point (int): the initial multiplier
        grad_dist (dist.Distribution): is the distribution
        bits (int): number of bits
        lr (float): learning rate
        epochs (int): number of epochs
    """

    mul = initial_point
    s = 2 ** (bits - 1) - 1
    mean = grad_dist.mean
    sigma = grad_dist.sigma
    all_mul = []
    for epoch in range(epochs):

        def arg1_1(j):
            return mean * (j * mul ** (j - 1) + (j + 1) * mul ** j) \
                   - (2 * j + 1) * mul ** (2 * j)
        arg1 = np.sum(np.asarray([arg1_1(j) * (
                grad_dist.cdf(mul ** j) -
                grad_dist.cdf(mul ** (j+1))) for j in range(0, s)]))

        def arg2_1(j):
            return j * mul ** (j - 1) + (j + 1) * mul ** j
        arg2 = np.sum(np.asarray([
            arg2_1(j) * (grad_dist.pdf(mul ** (j + 1)) -
                         grad_dist.pdf(mul ** (j))) for j in range(0, s)]))

        gradient = 2 * s * (mul ** (2 * s - 1)) * \
                   (grad_dist.cdf(mul ** s) - grad_dist.cdf(0)) \
                   + arg1 + sigma ** 2 * arg2

        mul = mul - lr * gradient
        all_mul.append(mul)

    return mul, all_mul


def alq(initial_levels, grad_dist, epochs, inv=False, sym=True, lr=0.1):
    """ALQ algorithm implementation.
    Parameters:
        grad_dist (dist.Distribution): is the distribution
        epochs (int): number of epochs
        inv (bool): if inv is enabled it uses inverse method
                    instead of gradient descent
        sym (bool): use symmetric levels
        lr(float): learning rate in finite_diff_grad_descent
    """

    losses = []
    # Assuming last level is 1, setting first dummy level to 0
    if sym:
        positive_levels = initial_levels[len(initial_levels) // 2:]
        new_levels = [0] + list(positive_levels).copy()
    else:
        new_levels = list(initial_levels).copy()
    all_levels = [new_levels.copy()]
    for epoch in range(epochs):

        def objective(x, left_level, right_level):
            # from equation below corollary 1
            left_var = grad_dist.est_var_adjacent_levels(left_level, x)
            right_var = grad_dist.est_var_adjacent_levels(x, right_level)
            return left_var+right_var

        for index in range(1, len(new_levels)-1):
            left_level = new_levels[index - 1]
            right_level = new_levels[index + 1]
            if inv:
                new_levels[index] = grad_dist.estimate_variance_adj_inv(
                    left_level, right_level)
            else:
                new_levels[index] = finite_diff_gradient_descent(
                    lambda x: objective(x, left_level, right_level),
                    left_level, right_level, x0=new_levels[index], lr=lr)
            if not (new_levels[index] < right_level and new_levels[index] > left_level):
                print("Means ", grad_dist.means)
                print("Sigmas ", grad_dist.sigmas)
                print("Norms ", grad_dist.norms)
            assert new_levels[index] < right_level and \
                   new_levels[index] > left_level, \
                "New level is not in the interval"
        if sym:
            negative_levels = [-level for level in new_levels]
            negative_levels.reverse()
            losses.append(grad_dist.estimate_variance(
                negative_levels[:-1] + new_levels[1:]))
            all_levels.append(new_levels.copy())
        else:
            losses.append(grad_dist.estimate_variance(new_levels))
            all_levels.append(new_levels.copy())

    if sym:
        # dropping dummy level at 0
        new_levels = new_levels[1:]
        negative_levels = [-level for level in new_levels]
        negative_levels.reverse()
        new_levels = negative_levels + new_levels
    return new_levels, all_levels, losses


class QuantizeMultiBucket(object):
    def __init__(self, method, bits, bucket_size, multiplier, **kwargs):
        """QSGD: qdqL2 + levels_uni
        NUQSGD: qdqL2 + levels_exp
        QSGD-inf: qdqLinf + levels_uni
        """
        self.method = method
        self.multiplier = multiplier
        if kwargs['interval'] is not None:
            self.interval = kwargs['interval']
        if method == 'q':
            self.levels = get_uniform_levels(bits)
            self.norm_type = 'fro'
        elif method == 'nuq':
            self.levels = get_exp_levels(bits, multiplier)
            self.norm_type = 'fro'
        elif method == 'qinf':
            self.levels = get_uniform_levels(bits)
            self.norm_type = float('inf')
        elif method == 'amq':
            self.levels = get_exp_levels(bits, multiplier)
            self.norm_type = 'fro'
        elif method == 'amq_nb':
            self.levels = get_exp_levels(bits, multiplier)
            self.norm_type = 'fro'
        elif method == 'alq':
            self.levels = get_exp_levels(bits, multiplier)
            self.norm_type = 'fro'
        elif method == 'alq_nb':
            self.levels = get_exp_levels(bits, multiplier)
            self.norm_type = 'fro'
        elif method == 'trn':
            self.levels = get_ternary_levels()
            self.norm_type = float('inf')

        elif method == 'none':
            return

        # store the previous best multiplier
        self.previous_best = None

        self.bucket_size = bucket_size
        self.bits = bits
        self.epochs = kwargs['cd_epochs']
        self.path = kwargs['path']
        self.amq_lr = kwargs['amq_lr']
        self.alq_lr = kwargs['learning_rate']
        self.amq_epochs = kwargs['amq_epochs']
        self.symmetric = kwargs['symmetric']
        self.inv = kwargs['inv']
        self.levels = torch.as_tensor(self.levels, dtype=torch.float32).cuda()
        self.mean_weights = 0
        self.variance_weights = 0.1
        self.error = None

    def set_mean_variance(self, stats):
        self.mean = mean = stats['nl']['mean']
        self.variance = stats['nl']['sigma'] ** 2
        self.norms = norms = stats['nb']

        interval = self.interval
        sigma = torch.sqrt(torch.tensor(self.variance)).cpu().item()
        self.grad_dist_nb = CondNormalTruncHist(
            norms['means'], norms['sigmas'], norms['norms'], -interval,
            interval, nbins=100000, bin_type='linear')
        self.grad_dist_nl = TruncNorm(
            mean, sigma, -interval, interval, nbins=100000, bin_type='linear')

        self.error = self.grad_dist_nb.estimate_variance(self.levels.cpu())

    def update_levels(self):
        """Main function to update the levels
        """

        bits = self.bits
        grad_dist_nl = self.grad_dist_nl
        grad_dist_nb = self.grad_dist_nb
        half_point = int(len(self.levels) / 2)
        quantile_levels = get_quantile_levels(bits, grad_dist_nb)
        uniform_levels = get_uniform_levels(
            self.bits)
        exp_levels = get_exp_levels(
            self.bits, 0.5)

        bits = self.bits
        if self.method == 'alq':
            inv = self.inv
            sym = self.symmetric
            epochs = self.epochs

            # Try various initializations and pick the best one
            levels_qua, _, losses_qua = alq(
                quantile_levels, grad_dist_nl, epochs, inv, sym, self.alq_lr)
            levels_uniform, _, losses_uni = alq(
                uniform_levels, grad_dist_nl, epochs, inv, sym, self.alq_lr)
            levels_exp, _, losses_exp = alq(
                exp_levels, grad_dist_nl, epochs, inv, sym, self.alq_lr)
            candidate_levels = np.asarray(
                [levels_qua, levels_uniform, levels_exp])

            candidate_losses = np.asarray(
                [losses_qua[-1], losses_uni[-1], losses_exp[-1]])

            # Select the minimum loss
            self.levels = candidate_levels[np.argsort(candidate_losses)][0]

        elif self.method == 'alq_nb':
            epochs = self.epochs
            inv = self.inv
            sym = self.symmetric
            quantile_levels = get_quantile_levels(bits, grad_dist_nb)
            levels_qua, _, losses_qua = alq(
                quantile_levels, grad_dist_nb, epochs, inv, sym, self.alq_lr)
            levels_uniform, _, losses_uni = alq(
                uniform_levels, grad_dist_nb, epochs, inv, sym, self.alq_lr)
            levels_exp, _, losses_exp = alq(
                exp_levels, grad_dist_nb, epochs, inv, sym, self.alq_lr)
            candidate_levels = np.asarray(
                [levels_qua, levels_uniform, levels_exp])
            candidate_losses = np.asarray(
                [losses_qua[-1], losses_uni[-1], losses_exp[-1]])
            self.levels = candidate_levels[np.argsort(candidate_losses)][0]

        elif self.method == 'amq':
            initial_points = []

            if self.previous_best is None:
                initial_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9]
            else:
                initial_points = [0.1, 0.2, 0.3, 0.4,
                                  self.previous_best, 0.5,  0.8, 0.9]
            optimal_points = []
            for point in initial_points:
                optimal_p, _ = amq_norm_less(
                    point, grad_dist_nl, bits, self.amq_lr, self.amq_epochs)
                optimal_points.append(optimal_p)
            optimal_points_costs = [
                grad_dist_nl.estimate_variance(get_exp_levels(bits, p)[
                                               half_point:]) for p in optimal_points]
            index = np.argmin(optimal_points_costs)
            self.multiplier = optimal_points[index]
            self.previous_best = self.multiplier
            self.levels = get_exp_levels(bits, self.multiplier)

        elif self.method == 'amq_nb':
            initial_points = []

            if self.previous_best is None:
                initial_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9]
            else:
                initial_points = [0.1, 0.2, 0.3, 0.4,
                                  self.previous_best, 0.5, 0.8, 0.9]
            optimal_points = []
            for point in initial_points:
                optimal_p, _ = amq_norm_based(
                    point, grad_dist_nb, bits, self.amq_lr, self.amq_epochs)
                optimal_points.append(optimal_p)
            optimal_points_costs = [
                grad_dist_nb.estimate_variance(get_exp_levels(bits, p)[
                                               half_point:]) for p in optimal_points]
            index = np.argmin(optimal_points_costs)
            self.multiplier = optimal_points[index]
            self.previous_best = self.multiplier
            self.levels = get_exp_levels(self.bits, self.multiplier)

        self.levels = torch.as_tensor(self.levels, dtype=torch.float32, device='cpu')

    def get_levels(self):
        num_levels = len(self.levels) // 2
        levels = self.levels[num_levels:]
        levels[0] = 0.0
        levels_np = levels.numpy()
        levels_np = list(levels_np[::-1])
        return torch.as_tensor(levels_np, dtype=torch.float32, device='cpu')

    def state_dict(self):
        if self.method == 'none':
            return {}
        return {
            'levels': self.levels,
            'means': self.grad_dist_nb.means,
            'sigmas': self.grad_dist_nb.sigmas,
            'norms': self.grad_dist_nb.norms,
            'sigma': self.grad_dist_nl.sigma,
            'mean': self.grad_dist_nl.mean,
            'error': self.error
        }

    def load_state_dict(self, state):
        if self.method == 'none':
            return
        self.levels = state['levels']
        self.grad_dist_nb = CondNormalTruncHist(
            state['means'], state['sigmas'], state['norms'], -1,
            1, nbins=100000, bin_type='linear')

        self.grad_dist_nl = TruncNorm(
            state['mean'], state['sigma'], -1,
            1, nbins=100000, bin_type='linear')
        self.qdq = QDQ(self.levels)

        self.error = state['error']