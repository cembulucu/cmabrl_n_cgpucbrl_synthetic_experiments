import numpy as np
import sklearn.metrics as spm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import matplotlib.pylab as pyl

bandit_env_dict = {'m': np.array([[0.25, 0.75], [0.5, 0.5]]),
                    'c': 0.0005*np.array([[[100, 60],
                                           [60, 50]],

                                          [[50, -60],
                                           [-60, 100]]]),
                    'w': np.array([0.5, 0.5]),
                    's': 0.25,
                    'dx_bar': 1, 'da_bar': 1}


class InfiniteArmedGMMEnvironmentWithIrrelevantDimensions:
    """ Bandit Environment, first dx_bar and da_bar dimensions are relevant(without loss of generality) """
    def __init__(self, dx, da):
        param_dict = bandit_env_dict
        self.means = param_dict['m']
        self.covs = param_dict['c']
        self.weights = param_dict['w']
        self.scale = param_dict['s']
        self.n_components = self.means.shape[0]
        self.dx, self.dx_bar = dx, param_dict['dx_bar']
        self.da, self.da_bar = da, param_dict['da_bar']

        num_components_check = self.means.shape[0] == self.covs.shape[0] == self.weights.shape[0]
        dimension_check = self.means.shape[1] == self.covs.shape[1] == self.covs.shape[2] == (self.dx_bar + self.da_bar)
        if not (num_components_check and dimension_check):
            raise ValueError('inconsistent dimensions or components')

    def get_approx_opt_1D_arm_n_exp_rew_for(self, context, precision=0.01):
        """ only for 1D arms """
        arm_candidates = np.arange(0, 1.00000000000001, precision)
        exp_rews = np.zeros_like(arm_candidates)
        for i, a in enumerate(arm_candidates):
            exp_rews[i] = self.get_expected_reward_at(context, (a,))
        return arm_candidates[np.argmax(exp_rews)], exp_rews[np.argmax(exp_rews)]

    def get_approx_opt_1D_arms_for(self, contexts, precision=0.01, verbose_period=2000):
        opt_arms, opt_exp_rews = np.zeros(shape=(contexts.shape[0],)), np.zeros(shape=(contexts.shape[0],))
        for i, c in enumerate(contexts):
            if i % verbose_period == 0:
                print('Calculating best arm for ', i, 'th context...')
            a, e = self.get_approx_opt_1D_arm_n_exp_rew_for(c, precision=precision)
            opt_arms[i] = a
            opt_exp_rews[i] = e
        return opt_arms, opt_exp_rews

    def get_reward_at(self, context, arm, return_true_exp_rew=False):
        true_exp_rew = self.get_expected_reward_at(context, arm)
        if return_true_exp_rew:
            return int(np.random.rand() <= true_exp_rew), true_exp_rew
        else:
            return int(np.random.rand() <= true_exp_rew)

    def get_expected_reward_at(self, context, arm):
        p = np.concatenate((context[:self.dx_bar], arm[:self.da_bar]))
        gmm_val_at_p = 0
        for c in range(self.n_components):
            gmm_val_at_p += self.weights[c] * multivariate_normal.pdf(p, self.means[c], self.covs[c])
        return np.clip(gmm_val_at_p*self.scale, 0, 1)

    def generate_uniform_random_contexts(self, num_samples=1, min_val=0.0, max_val=1.0):
        return np.random.uniform(low=min_val, high=max_val, size=(num_samples, self.dx))

    def generate_uniform_random_samples_given_contexts(self, contexts, min_val=0.0, max_val=1.0, get_true_expectation=False):
        num_samples = contexts.shape[0]
        arms = np.random.uniform(low=min_val, high=max_val, size=(num_samples, self.da))
        rewards = np.zeros(shape=(num_samples,))
        for i in range(num_samples):
            if get_true_expectation:
                rewards[i] = self.get_expected_reward_at(contexts[i], arms[i])
            else:
                rewards[i] = self.get_reward_at(contexts[i], arms[i])
        return contexts, arms, rewards

    def visualize_env(self, num_samples=10000, show_opt_arms=False):
        contexts = self.generate_uniform_random_contexts(num_samples=num_samples)
        contexts, arms, exp_rews = self.generate_uniform_random_samples_given_contexts(contexts, get_true_expectation=True)
        contexts_rel, arms_rel = contexts[:, :self.dx_bar], arms[:, :self.da_bar]
        points_rel = np.concatenate((contexts_rel, arms_rel), axis=-1)

        subplot_index, sub_size = 0, points_rel.shape[1]-1
        for i in range(points_rel.shape[1]):
            dim_i = points_rel[:, i]
            for j in range(points_rel.shape[1]):
                if i >= j:
                    continue
                dim_j = points_rel[:, j]
                subplot_index += 1
                sort_inds = np.argsort(exp_rews)
                plt.subplot(sub_size, sub_size, subplot_index)
                plt.scatter(dim_i[sort_inds], dim_j[sort_inds], c=exp_rews[sort_inds], cmap='rainbow')
                plt.colorbar()
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.xlabel('Relevant Context Dimension')
                plt.ylabel('Relevant Arm Dimension')

        if show_opt_arms:
            undersampled_contexts = contexts_rel[np.random.randint(low=0, high=num_samples, size=500)]
            undersampled_contexts = np.expand_dims(np.linspace(0, 1, 200), axis=1)
            opt_arms, opt_exp_rews = self.get_approx_opt_1D_arms_for(undersampled_contexts, precision=0.005, verbose_period=100)
            sort_inds = np.argsort(np.squeeze(undersampled_contexts))
            plt.plot(undersampled_contexts[sort_inds], opt_arms[sort_inds], c='k', label='Optimal Arms')
        plt.legend(loc=3)
        plt.show()

    def print_info(self):
        print('mean: ', self.means)
        print('covs: ', self.covs)
        print('weights: ', self.weights)
        print('scale: ', self.scale)


if __name__ == '__main__':
    print('TEST GMM ENVIRONMENT')
    reps, horizon = 20, 100000
    gmm_env_ = InfiniteArmedGMMEnvironmentWithIrrelevantDimensions(dx=5, da=5)
    contexts = gmm_env_.generate_uniform_random_contexts(num_samples=reps*horizon)
    gmm_env_.visualize_env(num_samples=30000, show_opt_arms=True)
    # opt_arms, opt_exp_rews = gmm_env_.get_approx_opt_1D_arms_for(contexts, precision=0.01, verbose_period=1000)
    # contexts = np.reshape(contexts, newshape=(reps, horizon, -1))
    # opt_arms = np.reshape(opt_arms, newshape=(reps, horizon, -1))
    # opt_exp_rews = np.reshape(opt_exp_rews, newshape=(reps, horizon, -1))

    # path = 'D:/python/projects/bandits/contextual_bandit_with_relevance_learning/data_files/'
    # np.savez((path + 'context_optarms_optexprews_reps' + str(reps) + '_horizon' + str(horizon)), contexts=contexts,
    #          opt_exp_rews=opt_exp_rews, opt_arms=opt_arms)

