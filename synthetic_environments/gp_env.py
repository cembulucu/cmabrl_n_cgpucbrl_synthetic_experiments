import itertools

import numpy as np
import matplotlib.pylab as ply
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import multivariate_normal
import sklearn.metrics as skmetrics

from sklearn.neighbors import NearestNeighbors

def rbf_kernel(x1, x2, coeffs):
    data_1n = x1 * coeffs
    data_2n = x2 * coeffs
    dist_matrix = skmetrics.pairwise_distances(data_1n, data_2n)
    kernel_matrix = np.exp(-0.5 * np.square(dist_matrix))
    return kernel_matrix

class InfiniteArmedGPEnvironmentWithIrrelevantDimensions:
    """ Bandit Environment, first dx_bar and da_bar dimensions are relevant(without loss of generality) """
    def __init__(self, granularity=0.2, lims=(0, 1), noise_sigma=1):
        self.dx_bar, self.da_bar = 1, 1
        self.noise_sigma = noise_sigma

        # generate points over context-arm space
        self.points = np.array(list(itertools.product(np.arange(lims[0], lims[1]+granularity/2, granularity), repeat=self.dx_bar + self.da_bar)))
        self.num_samples = self.points.shape[0]
        self.coeffs = np.array([1, 1])
        # calculate the kernel matrix and the exp rew associated with each context-arm pair
        self.gram = rbf_kernel(self.points, self.points, self.coeffs)
        self.exp_rews = np.random.multivariate_normal(np.zeros(shape=(self.num_samples, )), self.gram)
        print(max(self.exp_rews), min(self.exp_rews), np.mean(self.exp_rews))
        self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.points)
        self.lims = lims
        self.granularity = granularity

    def get_reward_at(self, context, arm, return_true_exp_rew=False):
        """ get reward at a given context and arm location"""
        true_exp_rew = self.get_expected_reward_at(context, arm)
        reward = true_exp_rew + np.random.normal(loc=0, scale=self.noise_sigma)
        if return_true_exp_rew:
            return reward, true_exp_rew
        return reward

    def get_expected_reward_at(self, context, arm):
        """ get expected reward at a given context and arm location"""
        p = np.concatenate((context[:self.dx_bar], arm[:self.da_bar]))
        nn_ind = self.nbrs.kneighbors(X=np.atleast_2d(p), return_distance=False)
        exp_rew = np.squeeze(self.exp_rews[nn_ind])
        return exp_rew

    def find_best_exp_rew_for(self, context, return_opt_arm=False):
        """ find the best arm for a given context and the opt exp rew associated with it"""
        arms_1d = np.expand_dims(np.arange(self.lims[0], self.lims[1]+self.granularity/2, self.granularity), axis=-1)
        repeat_c = np.expand_dims(np.repeat(context[:self.dx_bar], repeats=arms_1d.shape[0]), axis=-1)
        conarms = np.concatenate((repeat_c, arms_1d), axis=-1)
        nn_ind = self.nbrs.kneighbors(X=conarms, return_distance=False)
        exp_rews_c = np.squeeze(self.exp_rews[nn_ind])
        max_exp_rew = np.max(exp_rews_c)
        if return_opt_arm:
            max_exp_rew_ind = np.argmax(exp_rews_c)
            return max_exp_rew, arms_1d[max_exp_rew_ind]
        return max_exp_rew


if __name__ == '__main__':
    """Used to generate an environment with 4 dims where first two is relevant, uncomment to do so"""
    dim = 4
    lims = (0, 10)
    xs = np.array(list(itertools.product(np.linspace(lims[0], lims[1], 7), repeat=dim)))
    mean = np.zeros(shape=(xs.shape[0]))
    coeffs = np.array([1, 1, 0, 0])
    gram = rbf_kernel(xs, xs, coeffs=coeffs)
    ys = np.random.multivariate_normal(mean, gram)
    plt.figure()
    subplot_counter = 1
    for i in range(dim):
        for j in range(dim):
            if i >=j :
                continue
            plt.subplot(3, 2, subplot_counter)
            subplot_counter += 1
            grid_2d = np.zeros(shape=(xs.shape[0], 2))
            grid_2d[:, 0] = xs[:, i]
            grid_2d[:, 1] = xs[:, j]
            grid_2d_uniq, uniq_inds, uniq_inv = np.unique(grid_2d, axis=0, return_inverse=True, return_index=True)
            ys_uniq = ys[uniq_inds]
            grid_xs = np.array(list(itertools.product(np.linspace(lims[0], lims[1], 100), repeat=2)))
            grid_vals = griddata(points=grid_2d_uniq, values=ys_uniq, xi=grid_xs, method='linear')
            plt.scatter(grid_xs[:, 0], grid_xs[:, 1], c=grid_vals, s=100, vmin=-3, vmax=3)
            plt.xlabel('Dimension ' + str(i + 1))
            plt.ylabel('Dimension ' + str(j + 1))
            plt.xlim(lims)
            plt.ylim(lims)
            frame1 = plt.gca()
            frame1.axes.get_xaxis().set_ticks([])
            frame1.axes.get_yaxis().set_ticks([])
            # plt.colorbar()

    plt.show()
