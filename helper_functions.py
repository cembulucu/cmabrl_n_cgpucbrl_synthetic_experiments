import pickle
import numpy as np
import sklearn.metrics as skmetrics
import itertools
from t1dm_env import T1DMEnvironmentGP


def rbf_kernel_w_lin_coeffs(data_1, data_2, coeffs):
    """ calculates pairwise kernels according to coeffs, coeffs=all_ones gives Euclidean distance based kernels"""
    data_1n = data_1*coeffs
    data_2n = data_2*coeffs
    dist_matrix = skmetrics.pairwise_distances(data_1n, data_2n)
    kernel_matrix = np.exp(-0.5 * np.square(dist_matrix))
    return kernel_matrix


def calculate_context_arm_grid(arm_granularity, da, context=None, lims=(0, 1.00000001)):
    """For each arm, calculates the points when concatenated with the context"""
    arm_grid_1d = np.arange(lims[0], lims[1], arm_granularity)
    arm_grid = np.array(list(itertools.product(arm_grid_1d, repeat=da)))
    if context is None:
        return arm_grid
    else:
        contexts_repeated = np.repeat(np.expand_dims(context, axis=0), repeats=arm_grid.shape[0], axis=0)
        context_arm_grid = np.concatenate((contexts_repeated, arm_grid), axis=-1)
        return context_arm_grid


def calculate_posterior_mean_var(kernel_fn, contexts, arms, targets, context_arm_grid, noise_sigma, ard_coeffs):
    """Calculates statistics of the posterior distribution of expected reward function, possibly ignoring some dimensions acc. to ard_coeffs"""
    data = np.concatenate((contexts, arms), axis=-1)  # all points
    kernel_vectors = kernel_fn(context_arm_grid, data, ard_coeffs)  # kernels between all possible context-arms and the previous rounds
    kernel_matrix = kernel_fn(data, data, ard_coeffs)  # kernel matrix of data
    c_matrix = kernel_matrix + (noise_sigma**2)*np.eye(data.shape[0])
    c_matrix_inv = np.linalg.inv(c_matrix)
    mu_ests_vector = np.matmul(kernel_vectors, np.matmul(c_matrix_inv, targets)) # mean estimation
    sigma_ests_first_term = np.diag(kernel_fn(context_arm_grid, context_arm_grid, ard_coeffs))
    sigma_ests_second_term = np.diag(np.matmul(kernel_vectors, np.matmul(c_matrix_inv, kernel_vectors.T)))
    sigma_ests_vector = sigma_ests_first_term - sigma_ests_second_term # variance estimation
    return mu_ests_vector, sigma_ests_vector


def calculate_negative_log_likelihood(kernel_fn, contexts, arms, targets, noise_sigma, ard_coeffs):
    """Calculates the negative log marginal likelihood, possibly ignoring some dimensions acc. to ard_coeffs"""
    data = np.concatenate((contexts, arms), axis=-1)
    kernel_matrix = kernel_fn(data, data, ard_coeffs)
    c_matrix = kernel_matrix + (noise_sigma ** 2) * np.eye(data.shape[0])
    c_matrix_inv = np.linalg.inv(c_matrix)
    first_term = np.matmul(targets.T, np.matmul(c_matrix_inv, targets))
    second_term = np.log(np.linalg.det(c_matrix))
    return first_term + second_term


def calculate_discrete_best_ard_method_unknown_rel_dims(contexts, arms, targets, noise_sigma):
    """Optimization of NLL, find which dimensions should be ignored"""
    kernel_fn = rbf_kernel_w_lin_coeffs  # set kernel function to be used
    dx, da = contexts.shape[1], arms.shape[1]  # dims
    dx_powset, da_powset = calculate_powerset(np.arange(0, dx)), calculate_powerset(np.arange(0, da))
    dx_powset, da_powset = dx_powset[1:], da_powset[1:]  # calculate power set dimensions for contexts and arms and remove empty sets
    nlls, ard_params_list = [], []
    # for each combination of context and arm dimension tuples calculate NLL
    for i, dx_r in enumerate(dx_powset):
        dx_r_np = np.array(dx_r)
        context_ard_params = np.zeros(shape=(dx, ))
        context_ard_params[dx_r_np] = 1
        for j, da_r in enumerate(da_powset):
            da_r_np = np.array(da_r)
            arm_ard_params = np.zeros(shape=(da, ))
            arm_ard_params[da_r_np] = 1
            ard_params = np.concatenate((np.expand_dims(context_ard_params, axis=0), np.expand_dims(arm_ard_params, axis=0)), axis=1)
            ard_params = np.squeeze(ard_params)
            nlls.append(np.squeeze(calculate_negative_log_likelihood(kernel_fn, contexts, arms, targets, noise_sigma, ard_params)))
            ard_params_list.append(ard_params)
    nlls = np.array(nlls)
    ard_params_list = np.array(ard_params_list)
    argmin_ind = np.argmin(nlls)  # find best NLL index
    best_ard_params = ard_params_list[argmin_ind] # select coefficient that best ignore the dimensions
    return best_ard_params, nlls[argmin_ind]


def calculate_powerset(iterable):
    """ powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3) """
    s = list(iterable)
    return np.array(list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))))


def calculate_highest_ucb_index(beta, means, stds, conf_scale, return_multiple=False):
    """ calculate highest UCB"""
    ucbs = means + conf_scale*beta*stds
    highest_indices = np.argwhere(ucbs == np.max(ucbs))
    if return_multiple:
        return highest_indices, ucbs[highest_indices]
    else:
        if highest_indices.size > 1:
            highest_index = np.random.choice(np.squeeze(highest_indices))
        else:
            highest_index = np.squeeze(highest_indices)
        return highest_index, ucbs[highest_index]


def extract_data(path):
    """ extract data from the pickle"""
    with open(path, "rb") as f:
        patients_data = pickle.load(f)

    irrel_context_names = ['skin_temps', 'air_temps', 'gsrs', 'steps', 'exercises', 'heart_rates', 'basals', 'meals']
    irrel_arm_names = []
    rel_context_names = ['prev_cgms']
    rel_arm_names = ['boluses']
    rew_var_names = ['next_cgms_mean']

    # simply extract all data
    data_dim = len(patients_data[0].keys())
    data_arr = np.zeros(shape=(0, data_dim))
    var_names = list(patients_data[0].keys())
    patient_ids = []
    for i in range(6):
        data_dict = patients_data[i]
        pat_i = np.concatenate((np.atleast_2d(data_dict['boluses']), np.atleast_2d(data_dict['prev_cgms']), np.atleast_2d(data_dict['next_cgms_mean']),
                                np.atleast_2d(data_dict['next_cgms_max']), np.atleast_2d(data_dict['next_cgms_min']),
                                np.atleast_2d(data_dict['meals']), np.atleast_2d(data_dict['basals']), np.atleast_2d(data_dict['skin_temps']),
                                np.atleast_2d(data_dict['air_temps']), np.atleast_2d(data_dict['gsrs']), np.atleast_2d(data_dict['steps']),
                                np.atleast_2d(data_dict['exercises']), np.atleast_2d(data_dict['heart_rates']))).T
        data_arr = np.concatenate((data_arr, pat_i), axis=0)
        patient_ids.extend(i * np.ones(shape=(pat_i.shape[0]), dtype=np.int))
    patient_ids_og = np.array(patient_ids)

    # rearrange data so that first dimensions of the contexts and arms are relevant, to do that first get corresponding column indices
    rel_context_col_inds = [var_names.index(i) for i in rel_context_names]  # get indices of rel contexts in var_names
    rel_arm_col_inds = [var_names.index(i) for i in rel_arm_names]  # get indices of rel arm in var_names
    irrel_context_col_inds = [var_names.index(i) for i in irrel_context_names]  # get indices of irrel contexts in var_names
    irrel_arm_col_inds = [var_names.index(i) for i in irrel_arm_names]  # get indices of irrel aem in var_names
    rew_var_ind = [var_names.index(i) for i in rew_var_names]  # get index of reward variable

    context_col_inds = rel_context_col_inds + irrel_context_col_inds  # arrange context inds st rel are at the start
    arm_col_inds = rel_arm_col_inds + irrel_arm_col_inds  # arrange arm inds st rel are at the start

    dx, dx_bar, da, da_bar = len(context_col_inds), len(rel_context_col_inds), len(arm_col_inds), len(rel_arm_col_inds)

    contexts_og, arms_og, rew_vars_og = data_arr[:, context_col_inds], data_arr[:, arm_col_inds], data_arr[:, rew_var_ind]

    return contexts_og, arms_og, rew_vars_og, patient_ids_og, dx, dx_bar, da, da_bar


def calculate_beta_t(t, delta, d, r):
    """Calculate beta_t for CGP-UCB"""
    a = delta*np.exp(-1)
    first_term = 2*np.log((t**2)*2*(np.pi**2) / (3 * delta))
    second_term = 2*d*np.log((t**2)*d*r*np.sqrt(np.log(4*d*a/delta)))
    return first_term + second_term


def create_context_array(reps, horizon):
    """ Generate a contexts according to different patients Gaussian distributions"""
    all_data_path = 'D:/python/projects/gp_relevance_learning/patients_data.pkl'
    all_contexts_og, all_arms_og, all_rew_vars_og, all_patient_ids_og, dx, dx_bar, da, da_bar = extract_data(all_data_path)
    bandit_env = T1DMEnvironmentGP(all_contexts_og, all_arms_og, all_rew_vars_og, all_patient_ids_og, dx, dx_bar, da, da_bar)

    generated_contexts, generated_patient_ids = np.zeros(shape=(reps, horizon, dx)),  np.zeros(shape=(reps, horizon), dtype=np.int)
    for r in range(reps):
        for t in range(horizon):
            c, patient_id = bandit_env.get_context_gaussian()
            generated_contexts[r, t] = c
            generated_patient_ids[r, t] = patient_id

    npz_str = 'gp_contexts_n_patient_ids_reps' + str(reps) + '_horizon' + str(horizon)
    np.savez(npz_str, generated_contexts=generated_contexts, generated_patient_ids=generated_patient_ids)
