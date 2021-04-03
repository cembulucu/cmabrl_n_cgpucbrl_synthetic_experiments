import time

from gp_env import InfiniteArmedGPEnvironmentWithIrrelevantDimensions
from helper_functions import calculate_beta_t, calculate_context_arm_grid, rbf_kernel_w_lin_coeffs, calculate_posterior_mean_var, \
    calculate_discrete_best_ard_method_unknown_rel_dims, calculate_highest_ucb_index
import numpy as np


def main_fn():

    # set parameters for the experiment
    T = 100
    fit_ard_params_every = 10
    dx, da = 10, 2
    noise_sigma = 1
    reps = 20
    conf_scales = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
    cs_n = conf_scales.shape[0]
    kernel_fn = rbf_kernel_w_lin_coeffs
    radius = 10
    lims = (0, radius)
    granularity = radius/20
    # npz_str = 'gp_r' + str(radius)
    npz_str = 'gp_extra'

    # storage for rewards, regrets and played points
    cgp_points_hist, cgp_rews_hist, cgp_regret_hist = np.zeros(shape=(cs_n, reps, T, dx + da)), np.zeros(shape=(cs_n, reps, T)), np.zeros(shape=(cs_n, reps, T))
    rgp_points_hist, rgp_rews_hist, rgp_regret_hist = np.zeros(shape=(cs_n, reps, T, dx + da)), np.zeros(shape=(cs_n, reps, T)), np.zeros(shape=(cs_n, reps, T))
    random_rews_hist, random_regret_hist = np.zeros(shape=(cs_n, reps, T)), np.zeros(shape=(cs_n, reps, T))

    # pre generate contexts so that in each repetition, different confidence scales encounter same contexts
    contexts = radius * np.random.rand(reps, T, dx)
    for r in range(reps):
        # create environment
        bandit_env = InfiniteArmedGPEnvironmentWithIrrelevantDimensions(granularity=granularity, lims=lims, noise_sigma=noise_sigma)
        start = time.time()
        for i, cs in enumerate(conf_scales):
            # set initial ard parameters
            best_ard_params = np.array([1] * (dx + da))
            end = time.time()
            print('time elapsed: ', end - start)
            for t in range(T):
                # get context, noise and find optimal exp rew for this context
                c = contexts[r, t]
                noise_t = np.random.normal(loc=0, scale=noise_sigma)
                opt_exp_rew, opt_arm = bandit_env.find_best_exp_rew_for(c, return_opt_arm=True)

                # play arms arbitrarily in the first round, uniform random in this case
                if t == 0:
                    context_arm_grid = calculate_context_arm_grid(arm_granularity=granularity, da=da, context=c, lims=lims)

                    random_arm_cgp = context_arm_grid[np.random.randint(low=0, high=context_arm_grid.shape[0]), dx:]
                    random_arm_rgp = context_arm_grid[np.random.randint(low=0, high=context_arm_grid.shape[0]), dx:]
                    random_arm = context_arm_grid[np.random.randint(low=0, high=context_arm_grid.shape[0]), dx:]

                    _, true_exp_rew_cgp = bandit_env.get_reward_at(c, random_arm_cgp, return_true_exp_rew=True)
                    _, true_exp_rew_rgp = bandit_env.get_reward_at(c, random_arm_rgp, return_true_exp_rew=True)
                    _, true_exp_rew_ran = bandit_env.get_reward_at(c, random_arm, return_true_exp_rew=True)

                    r_cgp = true_exp_rew_cgp + noise_t
                    r_rgp = true_exp_rew_rgp + noise_t
                    r_ran = true_exp_rew_ran + noise_t

                    reg_cgp = opt_exp_rew - true_exp_rew_cgp
                    reg_rgp = opt_exp_rew - true_exp_rew_rgp
                    reg_ran = opt_exp_rew - true_exp_rew_ran

                    cgp_points_hist[i, r, 0] = np.concatenate((c, random_arm_cgp))[np.newaxis, :]
                    cgp_rews_hist[i, r, 0] = r_cgp
                    cgp_regret_hist[i, r, 0] = reg_cgp

                    rgp_points_hist[i, r, 0] = np.concatenate((c, random_arm_rgp))[np.newaxis, :]
                    rgp_rews_hist[i, r, 0] = r_rgp
                    rgp_regret_hist[i, r, 0] = reg_rgp

                    random_rews_hist[i, r, 0] = r_ran
                    random_regret_hist[i, r, 0] = reg_ran
                    continue

                # calcualte beta_t
                beta_t = calculate_beta_t(t=t, delta=0.01, d=dx + da, r=radius)
                # calculate all possible points that the current context and all arms can produce
                context_arm_grid = calculate_context_arm_grid(arm_granularity=granularity, da=da, context=c, lims=lims)

                # calculate posterior distribution stats for CGP-UCB
                cgp_mean_est, cgp_var_est = calculate_posterior_mean_var(kernel_fn, cgp_points_hist[i, r, :t, :dx], cgp_points_hist[i, r, :t, dx:],
                                                                         cgp_rews_hist[i, r, :t], context_arm_grid, noise_sigma=noise_sigma,
                                                                         ard_coeffs=np.ones(shape=(dx+da)))

                # perform optimization and calculate best hyperparameters that fit the data
                if t % fit_ard_params_every == 0:
                    best_ard_params, _ = calculate_discrete_best_ard_method_unknown_rel_dims(rgp_points_hist[i, r, :t, :dx],
                                                                                             rgp_points_hist[i, r, :t, dx:],
                                                                                             rgp_rews_hist[i, r, :t],
                                                                                             noise_sigma=noise_sigma)

                # calculate posterior distribution stats for CGP-UCB-RL
                rgp_mean_est, rgp_var_est = calculate_posterior_mean_var(kernel_fn, rgp_points_hist[i, r, :t, :dx], rgp_points_hist[i, r, :t, dx:],
                                                                         rgp_rews_hist[i, r, :t], context_arm_grid, noise_sigma=noise_sigma,
                                                                         ard_coeffs=best_ard_params)

                # calculate UCBs for each approach, find best arms(multiple arms are best for CGP-UCB-RL)
                cgp_arm_ind, cgp_high_ucb = calculate_highest_ucb_index(beta_t, cgp_mean_est, cgp_var_est, conf_scale=cs)
                rgp_arm_ind, rgp_high_ucb = calculate_highest_ucb_index(1, rgp_mean_est, rgp_var_est, conf_scale=cs, return_multiple=True)

                # for CGP-UCB-RL calculate the arm with the highest variance calculated according to all dimensions
                _, rgp_var_est2 = calculate_posterior_mean_var(kernel_fn, rgp_points_hist[i, r, :t, :dx], rgp_points_hist[i, r, :t, dx:],
                                                              rgp_rews_hist[i, r, :t], context_arm_grid, noise_sigma=noise_sigma,
                                                              ard_coeffs=np.ones(shape=(dx+da)))

                rgp_var_est2_sels = rgp_var_est2[rgp_arm_ind]
                highest_var_ind = np.argmax(rgp_var_est2_sels)
                rgp_arm_ind = np.squeeze(rgp_arm_ind[highest_var_ind])

                # get arms from context-arm grid
                a_cgp = context_arm_grid[cgp_arm_ind, dx:]
                a_rgp = context_arm_grid[rgp_arm_ind, dx:]
                a_ran = context_arm_grid[np.random.randint(low=0, high=context_arm_grid.shape[0]), dx:]

                # get expected reward ass. with each arm
                _, true_exp_rew_cgp = bandit_env.get_reward_at(c, a_cgp, return_true_exp_rew=True)
                _, true_exp_rew_rgp = bandit_env.get_reward_at(c, a_rgp, return_true_exp_rew=True)
                _, true_exp_rew_ran = bandit_env.get_reward_at(c, a_ran, return_true_exp_rew=True)

                # get rewards
                r_cgp = true_exp_rew_cgp + noise_t
                r_rgp = true_exp_rew_rgp + noise_t
                r_ran = true_exp_rew_ran + noise_t

                # calculate regrets
                reg_cgp = opt_exp_rew - true_exp_rew_cgp
                reg_rgp = opt_exp_rew - true_exp_rew_rgp
                reg_ran = opt_exp_rew - true_exp_rew_ran

                # store the played point
                cgp_points_hist[i, r, t] = context_arm_grid[cgp_arm_ind]
                rgp_points_hist[i, r, t] = context_arm_grid[rgp_arm_ind]

                # store rewards, regrets for all approaches
                cgp_rews_hist[i, r, t] = r_cgp
                cgp_regret_hist[i, r, t] = reg_cgp

                rgp_rews_hist[i, r, t] = r_rgp
                rgp_regret_hist[i, r, t] = reg_rgp

                random_rews_hist[i, r, t] = r_ran
                random_regret_hist[i, r, t] = reg_ran

                # print some info about how the experiment is going
                if t % 33 == 0 or t == T-1:
                    with np.printoptions(precision=2, suppress=True):
                        print('cs = ', cs, ', r = ', r, ', t = ', t)
                        print('rew sums = ', np.sum(cgp_rews_hist[i, r, :t]), np.sum(rgp_rews_hist[i, r, :t]), np.sum(random_rews_hist[i, r, :t]))
                        print('rew avgs = ', np.sum(cgp_rews_hist[i, r, :t])/t, np.sum(rgp_rews_hist[i, r, :t])/t, np.sum(random_rews_hist[i, r, :t])/t)
                        print('regret sums = ', np.sum(cgp_regret_hist[i, r, :t]), np.sum(rgp_regret_hist[i, r, :t]), np.sum(random_regret_hist[i, r, :t]))
                        print('rew est= ', cgp_mean_est[cgp_arm_ind], rgp_mean_est[rgp_arm_ind])
                        print('var est= ', cgp_var_est[cgp_arm_ind], rgp_var_est[rgp_arm_ind])
                        print('high ucb= ', cgp_high_ucb, rgp_high_ucb[highest_var_ind])
                        print('beta t = ', beta_t)
                        print(best_ard_params)
                        print('---***---')

    # save results with unique time stamp
    npz_str_uniq = npz_str + '_{}'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime(time.time())))
    np.savez(npz_str_uniq, dx=dx, da=da, dx_bar=1, da_bar=1, cgp_rews_hist=cgp_rews_hist, cgp_regret_hist=cgp_regret_hist,
             rgp_rews_hist=rgp_rews_hist, rgp_regret_hist=rgp_regret_hist, conf_scales=conf_scales, noise_sigma=noise_sigma,
             random_rews_hist=random_rews_hist, random_regret_hist=random_regret_hist,
             fit_ard_params_every=fit_ard_params_every)
#
if __name__ == '__main__':
   main_fn()