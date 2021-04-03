import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # load the results
    npz_path = 'gp_extra_2019_08_06_11_32_56.npz'
    npz_file = np.load(npz_path)

    cgp_rews_hist, cgp_regret_hist = npz_file['cgp_rews_hist'], npz_file['cgp_regret_hist']
    rgp_rews_hist, rgp_regret_hist = npz_file['rgp_rews_hist'], npz_file['rgp_regret_hist']
    ran_rews_hist, ran_regret_hist = npz_file['random_rews_hist'], npz_file['random_regret_hist']

    times, reps = np.arange(1, cgp_rews_hist.shape[-1] + 1), cgp_rews_hist.shape[1]
    conf_scales = npz_file['conf_scales']
    print(conf_scales)

    # calcualte cumulative reward and regrets
    cgp_rews_hist, cgp_regret_hist = np.cumsum(cgp_rews_hist, axis=-1), np.cumsum(cgp_regret_hist, axis=-1)
    rgp_rews_hist, rgp_regret_hist = np.cumsum(rgp_rews_hist, axis=-1), np.cumsum(rgp_regret_hist, axis=-1)
    ran_rews_hist, ran_regret_hist = np.cumsum(ran_rews_hist, axis=-1), np.cumsum(ran_regret_hist, axis=-1)

    # average over repetitions
    avg_cgp_rews_hist, avg_cgp_regret_hist = np.mean(cgp_rews_hist, axis=1), np.mean(cgp_regret_hist, axis=1)
    avg_rgp_rews_hist, avg_rgp_regret_hist = np.mean(rgp_rews_hist, axis=1), np.mean(rgp_regret_hist, axis=1)
    # as these are random arm selections take mean over all confidence scale and repetitions
    avg_ran_rews_hist, avg_ran_regret_hist = np.mean(ran_rews_hist, axis=(0, 1)), np.mean(ran_regret_hist, axis=(0, 1))

    # std over repetitions
    std_cgp_rews_hist, std_cgp_regret_hist = np.std(cgp_rews_hist, axis=1), np.std(cgp_regret_hist, axis=1)
    std_rgp_rews_hist, std_rgp_regret_hist = np.std(rgp_rews_hist, axis=1), np.std(rgp_regret_hist, axis=1)
    # as these are random arm selections take mean over all confidence scale and repetitions
    std_ran_rews_hist, std_ran_regret_hist = np.std(ran_rews_hist, axis=(0, 1)), np.std(ran_regret_hist, axis=(0, 1))

    best_cgp_scale_ind = np.argmin(avg_cgp_regret_hist[:, -1])
    best_rgp_scale_ind = np.argmin(avg_rgp_regret_hist[:, -1])

    print(avg_cgp_regret_hist[:, -1])
    print(avg_rgp_regret_hist[:, -1])

    print('Best conf scales, CGP-UCB: ', conf_scales[best_cgp_scale_ind], ', CGP-UCB-RL: ', conf_scales[best_rgp_scale_ind])

    avg_cgp_rews_hist, avg_cgp_regret_hist = avg_cgp_rews_hist[best_cgp_scale_ind], avg_cgp_regret_hist[best_cgp_scale_ind]
    avg_rgp_rews_hist, avg_rgp_regret_hist = avg_rgp_rews_hist[best_rgp_scale_ind], avg_rgp_regret_hist[best_rgp_scale_ind]

    std_cgp_rews_hist, std_cgp_regret_hist = std_cgp_rews_hist[best_cgp_scale_ind], std_cgp_regret_hist[best_cgp_scale_ind]
    std_rgp_rews_hist, std_rgp_regret_hist = std_rgp_rews_hist[best_rgp_scale_ind], std_rgp_regret_hist[best_rgp_scale_ind]

    marker_every = 33
    conf_mul = 2.093/np.sqrt(reps) # confidence multiplier according to student t distribution for 95% conf interval

    fig = plt.figure()
    fig.set_rasterized(True)
    a = plt.plot(times, avg_rgp_rews_hist, marker='*', markevery=marker_every, label='CGP-UCB-RL')
    plt.fill_between(times, avg_rgp_rews_hist + std_rgp_rews_hist*conf_mul, avg_rgp_rews_hist - std_rgp_rews_hist*conf_mul, color=a[0].get_color(),
                    edgecolor="r", alpha=.1)
    a = plt.plot(times, avg_cgp_rews_hist, marker='o', markevery=marker_every, label='CGP-UCB')
    plt.fill_between(times, avg_cgp_rews_hist + std_cgp_rews_hist*conf_mul, avg_cgp_rews_hist - std_cgp_rews_hist*conf_mul, color=a[0].get_color(),
                    edgecolor="", alpha=.1)
    a = plt.plot(times, avg_ran_rews_hist, marker='d', markevery=marker_every, label='Uniform Random')
    plt.fill_between(times, avg_ran_rews_hist + std_ran_rews_hist*conf_mul, avg_ran_rews_hist - std_ran_rews_hist*conf_mul, color=a[0].get_color(),
                    edgecolor="", alpha=.1)
    plt.legend(loc='upper left')
    plt.xlabel('Rounds')
    plt.ylabel('Cumulative Reward')
    plt.yticks(np.arange(-10, 81, 10))
    plt.gca().yaxis.grid(True)
    plt.gca().set_axisbelow(True)


    fig = plt.figure()
    fig.set_rasterized(True)
    a = plt.plot(times, avg_rgp_regret_hist, marker='*', markevery=marker_every, label='CGP-UCB-RL')
    plt.fill_between(times, avg_rgp_regret_hist + std_rgp_regret_hist*conf_mul, avg_rgp_regret_hist - std_rgp_regret_hist*conf_mul, color=a[0].get_color(),
                     edgecolor="r", alpha=.2)
    a = plt.plot(times, avg_cgp_regret_hist, marker='o', markevery=marker_every, label='CGP-UCB')
    plt.fill_between(times, avg_cgp_regret_hist + std_cgp_regret_hist*conf_mul, avg_cgp_regret_hist - std_cgp_regret_hist*conf_mul, color=a[0].get_color(),
                     edgecolor="r", alpha=.2)
    a = plt.plot(times, avg_ran_regret_hist, marker='d', markevery=marker_every, label='Uniform Random')
    plt.fill_between(times, avg_ran_regret_hist + std_ran_regret_hist*conf_mul, avg_ran_regret_hist - std_ran_regret_hist*conf_mul, color=a[0].get_color(),
                     edgecolor="r", alpha=.2)
    plt.legend(loc='upper left')
    plt.xlabel('Rounds')
    plt.ylabel('Cumulative Regret')
    plt.yticks(np.arange(0, 171, 20))
    plt.gca().yaxis.grid(True)
    plt.gca().set_axisbelow(True)

    plt.show()