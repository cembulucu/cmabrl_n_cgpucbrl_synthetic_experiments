from contextual_bandit_with_relevance_learning.bandit_algorithms.cbrl import ContextualBanditWithRelevanceLearning
import numpy as np
import time

from contextual_bandit_with_relevance_learning.bandit_algorithms.uniform_partition_based_methods import InstanceBasedUniformPartitioning

from contextual_bandit_with_relevance_learning.bandit_algorithms.contextual_x_armed_efficient import ContextualXArmedBanditEfficient
from contextual_bandit_with_relevance_learning.environments.gmm_env import InfiniteArmedGMMEnvironmentWithIrrelevantDimensions

if __name__ == '__main__':
    bandit_name = 'XARM'
    dx, da = 5, 5

    bandit_env = InfiniteArmedGMMEnvironmentWithIrrelevantDimensions(dx=dx, da=da)
    T, dx, da, dx_bar, da_bar = 100000, bandit_env.dx, bandit_env.da, bandit_env.dx_bar, bandit_env.da_bar
    contexts = np.load('D:/python/projects/bandits/contextual_bandit_with_relevance_learning/data_files/uniform[0,1]_contexts_T100000_dx5.npy')

    confidence_scales = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1]
    cs_size = len(confidence_scales)
    rew_hist, arms_hist, ucb_hist = np.zeros(shape=(T, cs_size)), np.zeros(shape=(T, cs_size), dtype=np.int), np.zeros(shape=(T, cs_size))
    total_rews = []
    for i, cs in enumerate(confidence_scales):
        print('Confidence Scale: ', cs)
        rews = 0
        bandit = None
        if bandit_name == 'CBRL':
            bandit = ContextualBanditWithRelevanceLearning(horizon=T, dx=dx, da=da, dx_bar=dx_bar, da_bar=da_bar, lip_c=1.0, conf_scale=cs)
        elif bandit_name == 'UCB1':
            bandit = InstanceBasedUniformPartitioning(horizon=T, dx=dx, da=da, conf_scale=cs)
        elif bandit_name == 'XARM':
            bandit = ContextualXArmedBanditEfficient(horizon=T, dx=dx, da=da, conf_scale=cs)
        else:
            raise ValueError('Bandit name not recognized')

        start_clock = time.time()
        for j, c in enumerate(contexts[:T]):
            y, a, u = bandit.determine_arm_one_round(c, return_winner_arm_ind=True, return_ucb=True)
            r = bandit_env.get_reward_at(c, y)
            bandit.update_statistics(r)
            rews += r
            rew_hist[j, i] = r
            arms_hist[j, i] = a
            ucb_hist[j, i] = u
            if j % 5000 == 0:
                end_lap = time.time()
                print('Round: ', (j + 1), ', total time elapsed: %.2f' % (end_lap - start_clock))
                print(bandit_name, ' stats, avg_rew: %.5f' % (rews / (j + 1)), ', sum rew:', rews, ', Last UCB: %.4f' % ucb_hist[j, i])
                print('')
        print('------------------------------------')
        print('------------------------------------')
        print('-------------XXXXXXXXXX-------------')
        print('------------------------------------')
        print('------------------------------------')
        total_rews.append(rews)

    print('bandit_name: ', bandit_name)
    print('confidence_scales: ', confidence_scales)
    print('total_rews: ', total_rews)