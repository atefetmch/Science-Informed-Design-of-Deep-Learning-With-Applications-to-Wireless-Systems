import numpy as np
import tensorflow as tf
import os
tf1 = tf.compat.v1


# -------------------------
# Numpy helpers (classical)
# -------------------------

def compute_P(Phi_diag_elements, Sigma_diag_elements, mu):
    mu_array = mu * np.ones(Phi_diag_elements.size)
    result = np.divide(Phi_diag_elements, (Sigma_diag_elements + mu_array) ** 2)
    return np.sum(result)


def compute_norm_of_complex_array(x):
    return np.sqrt(np.sum((np.absolute(x)) ** 2))


def compute_sinr(channel, precoder, noise_power, user_id, selected_users):
    nr_of_users = np.size(channel, 0)
    numerator = (np.absolute(np.matmul(np.conj(channel[user_id, :]), precoder[user_id, :]))) ** 2

    inter_user_interference = 0.0
    for user_index in range(nr_of_users):
        if user_index != user_id and user_index in selected_users:
            inter_user_interference += (np.absolute(np.matmul(np.conj(channel[user_id, :]), precoder[user_index, :]))) ** 2

    denominator = noise_power + inter_user_interference
    return numerator / denominator


def compute_user_weights(nr_of_users, selected_users):
    result = np.ones(nr_of_users)
    for user_index in range(nr_of_users):
        if user_index not in selected_users:
            result[user_index] = 0.0
    return result


def compute_weighted_sum_rate(user_weights, channel, precoder, noise_power, selected_users):
    nr_of_users = np.size(channel, 0)
    wsr = 0.0
    for user_id in range(nr_of_users):
        if user_id in selected_users:
            sinr = compute_sinr(channel, precoder, noise_power, user_id, selected_users)
            wsr += user_weights[user_id] * np.log2(1.0 + sinr)
    return wsr

def compute_channel(
    nr_of_BS_antennas,
    nr_of_users,
    total_power,
    path_loss_option=False,
    path_loss_range=(-5, 5),
    channel_noise_std=0.0,
    channel_noise_relative=True,
    channel_mean_shift=0.0,   # added: shift applied equally to real and imag
):
    """
    Generates Rayleigh channels, with optional:
      1) path loss scaling
      2) fixed mean shift (same added to real and imag): h <- h + (m + j m)
      3) channel estimation noise (complex Gaussian): h <- h + e

    Returns:
      channel_nn: list length K, each (2M x 2) real representation
      init_tp:    list length K, each (2M x 1) real representation
      channel_WMMSE: (K x M) complex
      reg_param: scalar for RZF
    """
    channel_nn = []
    init_tp = []
    channel_WMMSE = np.zeros((nr_of_users, nr_of_BS_antennas), dtype=np.complex128)

    regularization_parameter_for_RZF_solution = 0.0

    for i in range(nr_of_users):
        path_loss = 0.0
        if path_loss_option:
            path_loss = np.random.uniform(path_loss_range[0], path_loss_range[1])
            regularization_parameter_for_RZF_solution = 1.0 / (10 ** (path_loss / 10.0))

        scale = np.sqrt(10 ** (path_loss / 10.0)) * np.sqrt(0.5)

        # Base Rayleigh channel (complex)
        h_real = scale * np.random.normal(size=(nr_of_BS_antennas, 1))
        h_imag = scale * np.random.normal(size=(nr_of_BS_antennas, 1))
        h = np.reshape(h_real, (nr_of_BS_antennas,)) + 1j * np.reshape(h_imag, (nr_of_BS_antennas,))

        # Fixed mean shift applied equally to real and imag: + (m + j m)
        if channel_mean_shift != 0.0:
            h = h + (channel_mean_shift + 1j * channel_mean_shift)

        # Channel estimation noise: h_hat = h + e,  e ~ CN(0, sigma^2 I)
        if channel_noise_std and channel_noise_std > 0.0:
            if channel_noise_relative:
                rms = np.sqrt(np.mean(np.abs(h) ** 2) + 1e-12)
                sigma = channel_noise_std * rms
            else:
                sigma = channel_noise_std

            e = (sigma / np.sqrt(2.0)) * (
                np.random.normal(size=h.shape) + 1j * np.random.normal(size=h.shape)
            )
            h = h + e

        channel_WMMSE[i, :] = h

        # Build 2M x 2 real block representation from (possibly shifted/noisy) h
        result_real = np.real(h).reshape(nr_of_BS_antennas, 1)
        result_imag = np.imag(h).reshape(nr_of_BS_antennas, 1)

        col1 = np.vstack((result_real, result_imag))
        col2 = np.vstack((-result_imag, result_real))
        H_nn_i = np.hstack((col1, col2))  # (2M x 2)

        channel_nn.append(H_nn_i)
        init_tp.append(col1)  # (2M x 1)

    init_tp_arr = np.array(init_tp)  # (K, 2M, 1)
    init_tp_arr = np.sqrt(total_power) * init_tp_arr / (np.linalg.norm(init_tp_arr) + 1e-12)
    init_tp = [init_tp_arr[i] for i in range(nr_of_users)]

    return channel_nn, init_tp, channel_WMMSE, regularization_parameter_for_RZF_solution


def zero_forcing(channel_realization, total_power):
    # channel_realization: (K x M) complex
    # returns precoder: (K x M) complex (as in notebook)
    H = channel_realization
    V = np.matmul(H.T, np.linalg.inv(np.matmul(np.conj(H), H.T)))
    V = V * np.sqrt(total_power) / np.linalg.norm(V)
    return V.T


def regularized_zero_forcing(channel_realization, total_power, reg_param, path_loss_option):
    # basic RZF, consistent with notebook intent
    H = channel_realization
    K, M = H.shape
    alpha = reg_param if path_loss_option else (K / total_power)
    V = np.matmul(np.linalg.inv(np.matmul(H.T, np.conj(H)) + alpha * np.eye(M)), H.T)
    V = V * np.sqrt(total_power) / np.linalg.norm(V)
    return V.T


def run_WMMSE(epsilon, channel, selected_users, total_power, noise_power, user_weights, max_nr_of_iterations,
             power_tolerance, log=False):
    nr_of_users = channel.shape[0]
    nr_of_BS_antennas = channel.shape[1]

    receiver_precoder = np.zeros(nr_of_users, dtype=np.complex128)
    mse_weights = np.ones(nr_of_users, dtype=np.float64)

    transmitter_precoder = np.zeros((nr_of_users, nr_of_BS_antennas), dtype=np.complex128)
    for i in range(nr_of_users):
        if i in selected_users:
            transmitter_precoder[i, :] = channel[i, :]  # v_i <- h_i

    # Normalize to satisfy total power
    transmitter_precoder = transmitter_precoder * np.sqrt(total_power) / (np.linalg.norm(transmitter_precoder) + 1e-12)

    break_condition = epsilon + 1.0
    it = 0
    wsr_prev = -np.inf

    while break_condition >= epsilon and it < max_nr_of_iterations:
        it += 1

        # Update receiver and weights
        for i in range(nr_of_users):
            if i not in selected_users:
                receiver_precoder[i] = 0.0
                mse_weights[i] = 0.0
                continue

            interf = noise_power
            for j in range(nr_of_users):
                if j in selected_users:
                    interf += np.abs(np.vdot(channel[i, :], transmitter_precoder[j, :])) ** 2

            desired = np.vdot(channel[i, :], transmitter_precoder[i, :])  # h_i^H v_i

            # MMSE receive scalar
            receiver_precoder[i] = desired / (interf + 1e-12)

            # Correct MSE (two equivalent options)
            # Option A: general form
            # Since receiver_precoder is MMSE, the MSE simplifies to:
            e_i = 1.0 - (np.abs(desired) ** 2) / (interf + 1e-12)
            mse_weights[i] = 1.0 / max(e_i, 1e-12)



        # Build A and B
        A = np.zeros((nr_of_BS_antennas, nr_of_BS_antennas), dtype=np.complex128)
        B = np.zeros((nr_of_BS_antennas, nr_of_users), dtype=np.complex128)

        for i in range(nr_of_users):
            if i not in selected_users:
                continue
            hi = channel[i, :].reshape(-1, 1)
            A += user_weights[i] * mse_weights[i] * (np.abs(receiver_precoder[i]) ** 2) * (hi @ hi.conj().T)
            B[:, i:i+1] = user_weights[i] * mse_weights[i] * receiver_precoder[i].conj() * hi

        # Bisection for mu
        mu_low = 0.0
        mu_high = 1.0

        for _ in range(50):
            V_try = np.linalg.solve(A + mu_high * np.eye(nr_of_BS_antennas), B)
            p_try = np.linalg.norm(V_try) ** 2
            if p_try <= total_power:
                break
            mu_high *= 2.0

        V_best = None
        for _ in range(60):
            mu_mid = 0.5 * (mu_low + mu_high)
            V_mid = np.linalg.solve(A + mu_mid * np.eye(nr_of_BS_antennas), B)
            p_mid = np.linalg.norm(V_mid) ** 2
            V_best = V_mid

            if abs(p_mid - total_power) <= power_tolerance:
                break
            if p_mid > total_power:
                mu_low = mu_mid
            else:
                mu_high = mu_mid

        transmitter_precoder = V_best.T

        wsr_now = compute_weighted_sum_rate(user_weights, channel, transmitter_precoder, noise_power, selected_users)
        break_condition = abs(wsr_now - wsr_prev)
        wsr_prev = wsr_now

        if log:
            print("it", it, "WSR", wsr_now, "delta", break_condition)

    final_wsr = compute_weighted_sum_rate(user_weights, channel, transmitter_precoder, noise_power, selected_users)
    return receiver_precoder, mse_weights, transmitter_precoder, final_wsr



# -------------------------
# TensorFlow helpers (deep)
# -------------------------

def compute_sinr_nn(user_weights, channel_input, transmitter_precoder, noise_power, nr_of_users, nr_of_samples_per_batch):
    """
    channel_input: (B, K, 2M, 2)
    transmitter_precoder: (B, K, 2M, 1)
    returns SINR: (B, K, 1)
    """
    sinr_all = []
    for b in range(nr_of_samples_per_batch):
        sinr_users = []
        for i in range(nr_of_users):
            num = tf.reduce_sum((tf.matmul(tf.transpose(channel_input[b, i, :, :]), transmitter_precoder[b, i, :, :])) ** 2)
            inter = 0.0
            for j in range(nr_of_users):
                if j != i:
                    inter += tf.reduce_sum((tf.matmul(tf.transpose(channel_input[b, i, :, :]), transmitter_precoder[b, j, :, :])) ** 2)
            den = noise_power + inter
            sinr_users.append(num / den)
        sinr_all.append(tf.reshape(tf.stack(sinr_users), (nr_of_users, 1)))
    return tf.stack(sinr_all)  # (B, K, 1)


def compute_WSR_nn(user_weights, channel_input, transmitter_precoder, noise_power, nr_of_users, nr_of_samples_per_batch):
    sinr = compute_sinr_nn(user_weights, channel_input, transmitter_precoder, noise_power, nr_of_users, nr_of_samples_per_batch)
    rate = tf.math.log(1.0 + sinr) / tf.math.log(tf.constant(2.0, dtype=tf.float64))
    wsr = tf.reduce_sum(user_weights * rate)
    return wsr


def PGD_step(init_step, name, mse_weights, user_weights, receiver_precoder,
             channel_input, transmitter_precoder_in,
             total_power, nr_of_users, nr_of_BS_antennas, nr_of_samples_per_batch):
    """
    This is your notebook PGD_step but parameterized and reusable.
    It performs ONE PGD update and projection, returns (V_out, step_size_var).
    """
    with tf1.variable_scope(name):
        step_size = tf1.Variable(tf.constant(init_step, dtype=tf.float64), name="step_size", dtype=tf.float64)

        # build sum_gradient = sum_i a_i H_i H_i^T
        def a3(i):
            # sum of squares of receiver_precoder real/imag parts inside the 2x1 representation
            return tf.reduce_sum((receiver_precoder[:, i, :, :]) ** 2, axis=-2)

        a1_exp = tf.tile(tf.expand_dims(mse_weights[:, 0, :], -1), [1, 2 * nr_of_BS_antennas, 2 * nr_of_BS_antennas])
        a2_exp = tf.tile(tf.expand_dims(user_weights[:, 0, :], -1), [1, 2 * nr_of_BS_antennas, 2 * nr_of_BS_antennas])
        a3_exp = tf.tile(tf.expand_dims(a3(0), -1), [1, 2 * nr_of_BS_antennas, 2 * nr_of_BS_antennas])
        temp = a1_exp * a2_exp * a3_exp * tf.matmul(channel_input[:, 0, :, :], tf.transpose(channel_input[:, 0, :, :], perm=[0, 2, 1]))

        for i in range(1, nr_of_users):
            a1_exp = tf.tile(tf.expand_dims(mse_weights[:, i, :], -1), [1, 2 * nr_of_BS_antennas, 2 * nr_of_BS_antennas])
            a2_exp = tf.tile(tf.expand_dims(user_weights[:, i, :], -1), [1, 2 * nr_of_BS_antennas, 2 * nr_of_BS_antennas])
            a3_exp = tf.tile(tf.expand_dims(a3(i), -1), [1, 2 * nr_of_BS_antennas, 2 * nr_of_BS_antennas])
            temp = temp + a1_exp * a2_exp * a3_exp * tf.matmul(channel_input[:, i, :, :], tf.transpose(channel_input[:, i, :, :], perm=[0, 2, 1]))

        sum_gradient = temp

        gradient_list = []
        for i in range(nr_of_users):
            a1_exp = tf.tile(tf.expand_dims(mse_weights[:, i, :], -1), [1, 2 * nr_of_BS_antennas, 1])
            a2_exp = tf.tile(tf.expand_dims(user_weights[:, i, :], -1), [1, 2 * nr_of_BS_antennas, 1])
            term1 = -2.0 * a1_exp * a2_exp * tf.matmul(channel_input[:, i, :, :], receiver_precoder[:, i, :, :])
            term2 = 2.0 * tf.matmul(sum_gradient, transmitter_precoder_in[:, i, :, :])
            gradient_list.append(step_size * (term1 + term2))

        gradient = tf.transpose(tf.stack(gradient_list), perm=[1, 0, 2, 3])
        V_temp = transmitter_precoder_in - gradient

        # projection onto total power constraint per sample
        V_out = []
        for b in range(nr_of_samples_per_batch):
            norm_sq = (tf.linalg.norm(V_temp[b]) ** 2)
            V_out.append(tf.cond(
                norm_sq <= total_power,
                lambda: V_temp[b],
                lambda: tf.cast(tf.sqrt(total_power), tf.float64) * V_temp[b] / tf.linalg.norm(V_temp[b])
            ))

        return tf.stack(V_out), step_size


def SupNesterov_acc_PGD_step(init_step_size, init_momentum1, init_momentum2, name,
                            mse_weights, user_weights, receiver_precoder, channel_input,
                            V_present, V_past,
                            total_power, nr_of_users, nr_of_BS_antennas, nr_of_samples_per_batch):
    """
    Super-Nesterov accelerated PGD step, consistent with your second notebook structure.
    Returns (V_out, step_size_var, momentum1_var, momentum2_var)
    """
    with tf1.variable_scope(name):
        step_size = tf1.Variable(tf.constant(init_step_size, dtype=tf.float64), name="step_size", dtype=tf.float64)
        momentum1 = tf1.Variable(tf.constant(init_momentum1, dtype=tf.float64), name="momentum_1", dtype=tf.float64)
        momentum2 = tf1.Variable(tf.constant(init_momentum2, dtype=tf.float64), name="momentum_2", dtype=tf.float64)

        # lookahead point
        V_look = V_present + momentum1 * (V_present - V_past)

        # use same gradient as PGD_step but evaluated at V_look
        # build sum_gradient
        def a3(i):
            return tf.reduce_sum((receiver_precoder[:, i, :, :]) ** 2, axis=-2)

        a1_exp = tf.tile(tf.expand_dims(mse_weights[:, 0, :], -1), [1, 2 * nr_of_BS_antennas, 2 * nr_of_BS_antennas])
        a2_exp = tf.tile(tf.expand_dims(user_weights[:, 0, :], -1), [1, 2 * nr_of_BS_antennas, 2 * nr_of_BS_antennas])
        a3_exp = tf.tile(tf.expand_dims(a3(0), -1), [1, 2 * nr_of_BS_antennas, 2 * nr_of_BS_antennas])
        temp = a1_exp * a2_exp * a3_exp * tf.matmul(channel_input[:, 0, :, :], tf.transpose(channel_input[:, 0, :, :], perm=[0, 2, 1]))

        for i in range(1, nr_of_users):
            a1_exp = tf.tile(tf.expand_dims(mse_weights[:, i, :], -1), [1, 2 * nr_of_BS_antennas, 2 * nr_of_BS_antennas])
            a2_exp = tf.tile(tf.expand_dims(user_weights[:, i, :], -1), [1, 2 * nr_of_BS_antennas, 2 * nr_of_BS_antennas])
            a3_exp = tf.tile(tf.expand_dims(a3(i), -1), [1, 2 * nr_of_BS_antennas, 2 * nr_of_BS_antennas])
            temp = temp + a1_exp * a2_exp * a3_exp * tf.matmul(channel_input[:, i, :, :], tf.transpose(channel_input[:, i, :, :], perm=[0, 2, 1]))

        sum_gradient = temp

        gradient_list = []
        for i in range(nr_of_users):
            a1_exp = tf.tile(tf.expand_dims(mse_weights[:, i, :], -1), [1, 2 * nr_of_BS_antennas, 1])
            a2_exp = tf.tile(tf.expand_dims(user_weights[:, i, :], -1), [1, 2 * nr_of_BS_antennas, 1])
            term1 = -2.0 * a1_exp * a2_exp * tf.matmul(channel_input[:, i, :, :], receiver_precoder[:, i, :, :])
            term2 = 2.0 * tf.matmul(sum_gradient, V_look[:, i, :, :])
            gradient_list.append(step_size * (term1 + term2))

        gradient = tf.transpose(tf.stack(gradient_list), perm=[1, 0, 2, 3])
        V_temp = V_look - gradient

        # second momentum correction (super-nesterov style)
        V_temp = V_temp + momentum2 * (V_temp - V_present)

        V_out = []
        for b in range(nr_of_samples_per_batch):
            norm_sq = (tf.linalg.norm(V_temp[b]) ** 2)
            V_out.append(tf.cond(
                norm_sq <= total_power,
                lambda: V_temp[b],
                lambda: tf.cast(tf.sqrt(total_power), tf.float64) * V_temp[b] / tf.linalg.norm(V_temp[b])
            ))

        return tf.stack(V_out), step_size, momentum1, momentum2



def ckpt_exists(ckpt_path: str) -> bool:
    if ckpt_path is None:
        return False
    return os.path.exists(ckpt_path + ".index")

def ensure_dir_for_ckpt(ckpt_path: str):
    if ckpt_path is None:
        return
    d = os.path.dirname(ckpt_path)
    if d:
        os.makedirs(d, exist_ok=True)



def pad_nn_user_dim(ch_nn, v0, K_model, K_active):
    """
    ch_nn: [K_active, M2, 2] (list or np.ndarray)
    v0:    [K_active, M2, 1] or None (list or np.ndarray)
    """
    ch_nn = np.asarray(ch_nn)

    if v0 is not None:
        v0 = np.asarray(v0)

    if K_active == K_model:
        return ch_nn, v0

    M2 = ch_nn.shape[1]

    ch_pad = np.zeros((K_model, M2, 2), dtype=ch_nn.dtype)
    ch_pad[:K_active, :, :] = ch_nn

    if v0 is None:
        return ch_pad, None

    v_pad = np.zeros((K_model, M2, 1), dtype=v0.dtype)
    v_pad[:K_active, :, :] = v0

    return ch_pad, v_pad

