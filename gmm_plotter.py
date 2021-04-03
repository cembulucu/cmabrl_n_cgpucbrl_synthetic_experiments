import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

if __name__ == '__main__':
    load_str = 'E:/cem_files/New folder (2)/code_back2/bandits/contextual_bandit_with_relevance_learning/result_npzs/CBRL_vs_UCB1_vs_Xarmed_reps20_2019_04_08_12_06_04.npz'
    npz_file = np.load(load_str)

    dx, da, dx_bar, da_bar, rew_hist_cbrl, rew_hist_ucb1, rew_hist_xarm, rew_hist_random,\
        regret_hist_cbrl, regret_hist_xarm, regret_hist_ucb1, regret_hist_random\
        = npz_file['dx'], npz_file['da'], npz_file['dx_bar'], npz_file['da_bar'], \
          npz_file['rew_hist_cbrl'], npz_file['rew_hist_ucb1'], npz_file['rew_hist_xarm'], npz_file['rew_hist_random'], \
          npz_file['regret_hist_cbrl'], npz_file['regret_hist_xarm'], npz_file['regret_hist_ucb1'], npz_file['regret_hist_random']
    print(r'$d_x$ = %d, $\bar{d_x}$ = %d, $d_a$ = %d, $\bar{d_a}$ = %d' % (dx, dx_bar, da, da_bar))

    only_cumsum_cbrl = np.cumsum(rew_hist_cbrl, axis=1)
    only_cumsum_xarm = np.cumsum(rew_hist_xarm, axis=1)
    only_cumsum_ucb1 = np.cumsum(rew_hist_ucb1, axis=1)
    only_cumsum_rand = np.cumsum(rew_hist_random, axis=1)

    avg_cumsum_cbrl = np.mean(only_cumsum_cbrl, axis=0)
    avg_cumsum_xarm = np.mean(only_cumsum_xarm, axis=0)
    avg_cumsum_ucb1 = np.mean(only_cumsum_ucb1, axis=0)
    avg_cumsum_rand = np.mean(only_cumsum_rand, axis=0)

    std_cumsum_cbrl = np.std(only_cumsum_cbrl, axis=0)
    std_cumsum_xarm = np.std(only_cumsum_xarm, axis=0)
    std_cumsum_ucb1 = np.std(only_cumsum_ucb1, axis=0)
    std_cumsum_rand = np.std(only_cumsum_rand, axis=0)

    print('Fİnal CBLR: ', avg_cumsum_cbrl[-1])
    print('Fİnal XARAM: ', avg_cumsum_xarm[-1])
    print('Fİnal UB1: ', avg_cumsum_ucb1[-1])
    print('Fİnal rand: ', avg_cumsum_rand[-1])
    print('num reps: ', rew_hist_cbrl.shape)

    print('Fİnal CBLR/XARM ratio: ', avg_cumsum_cbrl[-1] / avg_cumsum_xarm[-1])
    print('Fİnal CBLR/UB1 ratio: ', avg_cumsum_cbrl[-1] / avg_cumsum_ucb1[-1])
    print('Fİnal CBLR/rand ratio: ', avg_cumsum_cbrl[-1] / avg_cumsum_rand[-1])

    print('CBRL final std:', np.std(only_cumsum_cbrl[:, -1]))
    print('XARM final std:', np.std(only_cumsum_xarm[:, -1]))
    print('UCB1 final std:', np.std(only_cumsum_ucb1[:, -1]))

    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0, 0))
    marker_every = 9999
    y_tick_range = np.arange(0, 56000, step=10000)

    n = 4
    colors = pyl.cm.jet(np.linspace(0, 1, n))

    conf_mul = 1.96 / np.sqrt(20)

    plt.figure()
    # a = plt.plot(times, avg_rgp_rews_hist, marker='*', markevery=marker_every, label='CGP-UCB-RL')
    # plt.fill_between(times, avg_rgp_rews_hist + std_rgp_rews_hist * conf_mul, avg_rgp_rews_hist - std_rgp_rews_hist * conf_mul, color=a[0].get_color(),
    #                  edgecolor="r", alpha=.1)

    a = plt.plot(np.arange(1, 100001, 1), avg_cumsum_cbrl, marker='*', markevery=marker_every, label='CMAB-RL', color=colors[0])
    plt.fill_between(np.arange(1, 100001, 1), avg_cumsum_cbrl + std_cumsum_cbrl * conf_mul, avg_cumsum_cbrl - std_cumsum_cbrl * conf_mul, color=a[0].get_color(),
                     edgecolor="r", alpha=.1)
    a = plt.plot(np.arange(1, 100001, 1), avg_cumsum_xarm, marker='o', markevery=marker_every, label='C-HOO', color=colors[1])
    plt.fill_between(np.arange(1, 100001, 1), avg_cumsum_xarm + std_cumsum_xarm * conf_mul, avg_cumsum_xarm - std_cumsum_xarm * conf_mul,
                     color=a[0].get_color(),
                     edgecolor="r", alpha=.1)
    a = plt.plot(np.arange(1, 100001, 1), avg_cumsum_ucb1, marker='d', markevery=marker_every, label='IUP', color=colors[2])
    plt.fill_between(np.arange(1, 100001, 1), avg_cumsum_ucb1 + std_cumsum_ucb1 * conf_mul, avg_cumsum_ucb1 - std_cumsum_ucb1 * conf_mul,
                     color=a[0].get_color(),
                     edgecolor="r", alpha=.1)
    plt.plot(avg_cumsum_rand, '--', label='Uniform Random', color=colors[3])
    plt.legend(loc='upper left')
    plt.xlabel('Rounds')
    plt.ylabel('Total Reward')
    plt.yticks(y_tick_range)
    plt.gca().yaxis.grid(True)
    plt.gca().set_axisbelow(True)
    plt.gca().xaxis.set_major_formatter(xfmt)
    plt.gca().yaxis.set_major_formatter(xfmt)

    plt.figure()

    only_cumsum_cbrl = np.cumsum(regret_hist_cbrl, axis=1)
    only_cumsum_xarm = np.cumsum(regret_hist_xarm, axis=1)
    only_cumsum_ucb1 = np.cumsum(regret_hist_ucb1, axis=1)
    only_cumsum_rand = np.cumsum(regret_hist_random, axis=1)

    avg_cumsum_cbrl = np.mean(only_cumsum_cbrl, axis=0)
    avg_cumsum_xarm = np.mean(only_cumsum_xarm, axis=0)
    avg_cumsum_ucb1 = np.mean(only_cumsum_ucb1, axis=0)
    avg_cumsum_rand = np.mean(only_cumsum_rand, axis=0)

    std_cumsum_cbrl = np.std(only_cumsum_cbrl, axis=0)
    std_cumsum_xarm = np.std(only_cumsum_xarm, axis=0)
    std_cumsum_ucb1 = np.std(only_cumsum_ucb1, axis=0)
    std_cumsum_rand = np.std(only_cumsum_rand, axis=0)

    print('Fİnal CBLR: ', avg_cumsum_cbrl[-1])
    print('Fİnal XARAM: ', avg_cumsum_xarm[-1])
    print('Fİnal UB1: ', avg_cumsum_ucb1[-1])
    print('Fİnal rand: ', avg_cumsum_rand[-1])
    print('num reps: ', rew_hist_cbrl.shape)

    print('CBRL final regret std:', np.std(only_cumsum_cbrl[:, -1]))
    print('XARM final regret std:', np.std(only_cumsum_xarm[:, -1]))
    print('UCB1 final regret std:', np.std(only_cumsum_ucb1[:, -1]))

    # xfmt = ScalarFormatter(useMathText=True)
    # xfmt.set_powerlimits((0, 0))
    # marker_every = 9999
    # y_tick_range = np.arange(0, 56000, step=10000)
    a = plt.plot(np.arange(1, 100001, 1), avg_cumsum_cbrl, marker='*', markevery=marker_every, label='CMAB-RL', color=colors[0])
    plt.fill_between(np.arange(1, 100001, 1), avg_cumsum_cbrl + std_cumsum_cbrl * conf_mul, avg_cumsum_cbrl - std_cumsum_cbrl * conf_mul,
                     color=a[0].get_color(),
                     edgecolor="r", alpha=.1)
    a = plt.plot(np.arange(1, 100001, 1), avg_cumsum_xarm, marker='o', markevery=marker_every, label='C-HOO', color=colors[1])
    plt.fill_between(np.arange(1, 100001, 1), avg_cumsum_xarm + std_cumsum_xarm * conf_mul, avg_cumsum_xarm - std_cumsum_xarm * conf_mul,
                     color=a[0].get_color(),
                     edgecolor="r", alpha=.1)
    a = plt.plot(np.arange(1, 100001, 1), avg_cumsum_ucb1, marker='d', markevery=marker_every, label='IUP', color=colors[2])
    plt.fill_between(np.arange(1, 100001, 1), avg_cumsum_ucb1 + std_cumsum_ucb1 * conf_mul, avg_cumsum_ucb1 - std_cumsum_ucb1 * conf_mul,
                     color=a[0].get_color(),
                     edgecolor="r", alpha=.1)
    plt.plot(avg_cumsum_rand, '--', label='Uniform Random', color=colors[3])
    plt.legend(loc='upper left')
    plt.xlabel('Rounds')
    plt.ylabel('Total Regret')
    plt.yticks(y_tick_range)
    plt.gca().yaxis.grid(True)
    plt.gca().set_axisbelow(True)
    plt.gca().xaxis.set_major_formatter(xfmt)
    plt.gca().yaxis.set_major_formatter(xfmt)

    # plt.figure()
    # print(np.atleast_2d(np.diff(cumsum_cbrl[10000:100001:1000])).T)
    # plt.plot(np.diff(cumsum_xarm[0:100001:100]), 'r-', marker='*', markevery=marker_every, label='CBRL')


    plt.show()