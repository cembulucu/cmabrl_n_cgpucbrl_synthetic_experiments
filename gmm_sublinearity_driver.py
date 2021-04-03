import time

from contextual_bandit_with_relevance_learning.bandit_algorithms.cbrl import ContextualBanditWithRelevanceLearning
from contextual_bandit_with_relevance_learning.bandit_algorithms.contextual_x_armed_efficient import ContextualXArmedBanditEfficient
from contextual_bandit_with_relevance_learning.bandit_algorithms.uniform_partition_based_methods import InstanceBasedUniformPartitioning
from contextual_bandit_with_relevance_learning.environments.gmm_env import InfiniteArmedGMMEnvironmentWithIrrelevantDimensions
import numpy as np

if __name__ == '__main__':
    contexts_path = 'D:/python/projects/bandits/contextual_bandit_with_relevance_learning/data_files/context_optarms_optexprews_reps20_horizon100000.npz'
    contexts_npz_file = np.load(contexts_path)
    contexts_all, opt_arms_all, opt_exp_rews_all = contexts_npz_file['contexts'], contexts_npz_file['opt_arms'], contexts_npz_file['opt_exp_rews']

    reps = 2
    npz_str = 'sublinearity_CBRL_vs_UCB1_vs_Xarmed_reps' + str(reps)
    # npz_str = ''

    # Ts = np.array([5000, 10000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 75000, 80000, 90000, 100000])
    Ts = np.array([50, 100, 200, 250, 300, 400])

    rews_cbrl, regret_hist_cbrl = 0, np.zeros(shape=(reps, np.max(Ts), Ts.size))
    rews_ucb1, regret_hist_ucb1 = 0, np.zeros(shape=(reps, np.max(Ts), Ts.size))
    rews_xarm, regret_hist_xarm = 0, np.zeros(shape=(reps, np.max(Ts), Ts.size))
    rews_random, regret_hist_random = 0, np.zeros(shape=(reps, np.max(Ts), Ts.size))

    dx, da = 5, 5

    bandit_env = InfiniteArmedGMMEnvironmentWithIrrelevantDimensions(dx=dx, da=da)  # see utility for details, like relevant dimension counts
    dx, da, dx_bar, da_bar = bandit_env.dx, bandit_env.da, bandit_env.dx_bar, bandit_env.da_bar

    for j, T in enumerate(Ts):

        for r in range(reps):
            print('Repeat: ', r)
            rews_cbrl, rews_ucb1, rews_xarm, rews_random = 0, 0, 0, 0
            contexts, opt_exp_rews = contexts_all[r], np.squeeze(opt_exp_rews_all[r])

            bandit_cbrl = ContextualBanditWithRelevanceLearning(horizon=T, dx=dx, da=da, dx_bar=dx_bar, da_bar=da_bar, lip_c=1.0, conf_scale=0.001)
            bandit_ucb1 = InstanceBasedUniformPartitioning(horizon=T, dx=dx, da=da, conf_scale=0.01)
            bandit_xarm = ContextualXArmedBanditEfficient(horizon=T, dx=dx, da=da, conf_scale=0.05)

            print('Num CBRL partitions: %d' % (bandit_cbrl.arm_size * bandit_cbrl.W_centers.shape[0]))
            print('Num CBRL arm size: %d' % bandit_cbrl.arm_size)
            print('Num CBRL pw per W size: %d' % (bandit_cbrl.W_centers.shape[0] / bandit_cbrl.V_2dxb_x.shape[0]))
            print('Num CBRL m: %d' % bandit_cbrl.m)
            print('Num CBRL W size: %d' % bandit_cbrl.W_centers.shape[0])
            print('Num CBRL pw size: %d' % bandit_cbrl.V_2dxb_x.shape[0])
            print('Num CBRL one sample needed size: %d' % (bandit_cbrl.arm_size * bandit_cbrl.W_centers.shape[0] // bandit_cbrl.V_2dxb_x.shape[0]))
            print('Num UCB1 partitions: %d' % bandit_ucb1.partition_size)
            print('Num UCB1 m: %d' % bandit_ucb1.m)
            print('Num Xarmed max depth: %d' % bandit_xarm.max_depth)
            print('Num Xarmed max num partitions: %d' % (2 ** bandit_xarm.max_depth))
            print('Num Xarmed v1: %.4f' % bandit_xarm.v1)
            print('Num Xarmed rho: %.4f' % bandit_xarm.rho)

            start_clock = time.time()
            for i, c in enumerate(contexts[:T]):
                y_cbrl, u_cbrl = bandit_cbrl.determine_arm_one_round(c, return_winner_arm_ind=False, return_ucb=True)
                y_ucb1, u_ucb1 = bandit_ucb1.determine_arm_one_round(c, return_winner_arm_ind=False, return_ucb=True)
                y_xarm, u_xarm = bandit_xarm.determine_arm_one_round(c, return_winner_arm_ind=False, return_ucb=True)
                y_random = np.random.rand(da)

                r_cbrl, true_exp_rew_cbrl = bandit_env.get_reward_at(c, y_cbrl, return_true_exp_rew=True)
                r_ucb1, true_exp_rew_ucb1 = bandit_env.get_reward_at(c, y_ucb1, return_true_exp_rew=True)
                r_xarm, true_exp_rew_xarm = bandit_env.get_reward_at(c, y_xarm, return_true_exp_rew=True)
                r_random, true_exp_rew_random = bandit_env.get_reward_at(c, y_random, return_true_exp_rew=True)

                reg_cbrl = opt_exp_rews[i] - true_exp_rew_cbrl
                reg_xarm = opt_exp_rews[i] - true_exp_rew_xarm
                reg_ucb1 = opt_exp_rews[i] - true_exp_rew_ucb1
                reg_random = opt_exp_rews[i] - true_exp_rew_random

                bandit_cbrl.update_statistics(r_cbrl)
                bandit_ucb1.update_statistics(r_ucb1)
                bandit_xarm.update_statistics(r_xarm)

                rews_cbrl += r_cbrl
                rews_ucb1 += r_ucb1
                rews_xarm += r_xarm
                rews_random += r_random

                regret_hist_cbrl[r, i, j] = reg_cbrl
                regret_hist_ucb1[r, i, j] = reg_ucb1
                regret_hist_xarm[r, i, j] = reg_xarm
                regret_hist_random[r, i, j] = reg_random
                if i % (T // 5) == 0:
                    end_lap = time.time()
                    print('Horizon: ', T, ', repeat: ', r, ', round: ', (i + 1), ', total time elapsed: %.2f' % (end_lap - start_clock))
                    print('CBRL stats, avg_rew: %.5f' % (rews_cbrl / (i + 1)), ', sum rew:', rews_cbrl, ', regret: %.4f' % np.sum(regret_hist_cbrl[r, :i, j]))
                    print('UCB1 stats, avg_rew: %.5f' % (rews_ucb1 / (i + 1)), ', sum rew:', rews_ucb1, ', regret: %.4f' % np.sum(regret_hist_ucb1[r, :i, j]))
                    print('XARM stats, avg_rew: %.5f' % (rews_xarm / (i + 1)), ', sum rew:', rews_xarm, ', regret: %.4f' % np.sum(regret_hist_xarm[r, :i, j]))
                    print('Rand stats, avg_rew: %.5f' % (rews_random / (i + 1)), ', sum rew: %.0f' % rews_random,
                          ', regret: %.4f' % np.sum(regret_hist_random[r, :i]))
                    print('')

            print('-------------------------------------------------')

    if npz_str:
        npz_str_uniq = npz_str + '_{}'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime(time.time())))
        np.savez(npz_str_uniq, dx=dx, da=da, dx_bar=dx_bar, da_bar=da_bar, regret_hist_cbrl=regret_hist_cbrl, regret_hist_xarm=regret_hist_xarm,
                 regret_hist_ucb1=regret_hist_ucb1, regret_hist_random=regret_hist_random, Ts=Ts)