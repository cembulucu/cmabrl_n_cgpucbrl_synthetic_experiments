
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import matplotlib.pylab as pyl

if __name__ == '__main__':
    load_root = ''
    npz_file = np.load(load_root + 'sublinearity_CBRL_vs_UCB1_vs_Xarmed_reps20_2019_04_09_09_38_33.npz')
    regret_hist_cbrl, regret_hist_xarm, regret_hist_ucb1, regret_hist_random, Ts = npz_file['regret_hist_cbrl'], npz_file['regret_hist_xarm'], \
                                                                                   npz_file['regret_hist_ucb1'], npz_file['regret_hist_random'], \
                                                                                   npz_file['Ts']

    regret_hist_cbrl_repavg, regret_hist_xarm_repavg, regret_hist_ucb1_repavg, regret_hist_random_repavg = \
        np.mean(regret_hist_cbrl, axis=0), np.mean(regret_hist_xarm, axis=0), np.mean(regret_hist_ucb1, axis=0), np.mean(regret_hist_random, axis=0)

    reg_plot_cbrl, reg_plot_xarm, reg_plot_ucb1, reg_plot_random = np.zeros_like(Ts, dtype=np.float32), np.zeros_like(Ts, dtype=np.float32), \
                                                                   np.zeros_like(Ts, dtype=np.float32), np.zeros_like(Ts, dtype=np.float32)

    for j, T in enumerate(Ts):
        reg_plot_cbrl[j], reg_plot_xarm[j], reg_plot_ucb1[j], reg_plot_random[j] = \
            np.sum(regret_hist_cbrl_repavg[:T, j]), np.sum(regret_hist_xarm_repavg[:T, j]), \
            np.sum(regret_hist_ucb1_repavg[:T, j]), np.sum(regret_hist_random_repavg[:T, j])

    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0, 0))

    n = 4
    colors = pyl.cm.jet(np.linspace(0, 1, n))

    plt.plot(Ts, reg_plot_cbrl, 'r-', marker='*', markevery=1, label='CMAB-RL', color=colors[0])
    plt.plot(Ts, reg_plot_xarm, 'g-', marker='o', markevery=1, label='C-HOO', color=colors[1])
    plt.plot(Ts, reg_plot_ucb1, 'b-', marker='d', markevery=1, label='IUP', color=colors[2])
    plt.plot(Ts, reg_plot_random, 'm--', label='Uniform Random', color=colors[3])
    plt.xlabel('Horizon Values')
    plt.ylabel('Total Regret')
    plt.gca().yaxis.grid(True)
    plt.gca().set_axisbelow(True)
    plt.gca().xaxis.set_major_formatter(xfmt)
    plt.gca().yaxis.set_major_formatter(xfmt)
    plt.legend()

    plt.show()


