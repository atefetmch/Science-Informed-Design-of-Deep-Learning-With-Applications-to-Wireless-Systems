import os
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import compute_channel, compute_WSR_nn, PGD_step, ckpt_exists, ensure_dir_for_ckpt

tf1 = tf.compat.v1


def run(cfg, mode, ckpt_path):
    tf1.reset_default_graph()
    ensure_dir_for_ckpt(ckpt_path)

    # Folder where we save logs/plots should match ckpt folder
    ckpt_dir = os.path.dirname(ckpt_path) if ckpt_path is not None else "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    B = cfg.nr_of_samples_per_batch
    K_model = cfg.nr_of_users               # fixed input/output users (eg 8)
    M2 = 2 * cfg.nr_of_BS_antennas

    channel_input = tf1.placeholder(tf.float64, shape=[B, K_model, M2, 2], name="channel_input")
    init_tp = tf1.placeholder(tf.float64, shape=[B, K_model, M2, 1], name="initial_transmitter_precoder")
    user_weights_ph = tf1.placeholder(tf.float64, shape=[B, K_model, 1], name="user_weights")

    V = init_tp
    profit = []
    step_init = 1.0

    # Active users for this run (train or infer). Must be <= K_model.
    K_active = int(getattr(cfg, "active_nr_of_users", K_model))
    if K_active > K_model:
        raise ValueError("active_nr_of_users must be <= nr_of_users (K_model).")

    # Unfolded iterations
    for l in range(cfg.nr_of_iterations_nn):
        user_interference2 = []
        for b in range(B):
            per_user = []
            for i in range(K_model):
                temp = 0.0
                for j in range(K_model):
                    temp += tf.reduce_sum(
                        (tf.matmul(tf.transpose(channel_input[b, i, :, :]), V[b, j, :, :])) ** 2
                    )
                per_user.append(temp + cfg.noise_power)
            user_interference2.append(per_user)

        user_interference2 = tf.stack(user_interference2)
        user_interference_exp2 = tf.tile(
            tf.expand_dims(tf.tile(tf.expand_dims(user_interference2, -1), [1, 1, 2]), -1),
            [1, 1, 1, 1],
        )

        receiver_precoder_temp = tf.matmul(tf.transpose(channel_input, perm=[0, 1, 3, 2]), V)
        receiver_precoder = tf.divide(receiver_precoder_temp, user_interference_exp2)

        self_interference = tf.reduce_sum(
            (tf.matmul(tf.transpose(channel_input, perm=[0, 1, 3, 2]), V)) ** 2,
            axis=2,
        )

        inter_user_interference_total = []
        for b in range(B):
            inter_user = []
            for i in range(K_model):
                temp = 0.0
                for j in range(K_model):
                    if j != i:
                        temp += tf.reduce_sum(
                            (tf.matmul(tf.transpose(channel_input[b, i, :, :]), V[b, j, :, :])) ** 2
                        )
                inter_user.append(temp + cfg.noise_power)
            inter_user_interference_total.append(tf.reshape(tf.stack(inter_user), (K_model, 1)))
        inter_user_interference_total = tf.stack(inter_user_interference_total)

        mse_weights = tf.divide(self_interference, inter_user_interference_total) + 1.0

        for k in range(cfg.pgd_steps):
            V, _ = PGD_step(
                step_init,
                f"PGD_L{l}_K{k}",
                mse_weights,
                user_weights_ph,
                receiver_precoder,
                channel_input,
                V,
                cfg.total_power,
                K_model,
                cfg.nr_of_BS_antennas,
                B,
            )

        profit.append(
            compute_WSR_nn(
                user_weights_ph,
                channel_input,
                V,
                cfg.noise_power,
                K_model,
                B,
            )
        )

    wsr_sum = compute_WSR_nn(
        user_weights_ph,
        channel_input,
        V,
        cfg.noise_power,
        K_model,
        B,
    )

    WSR_final = wsr_sum / B
    loss_final = -WSR_final

    train_op = tf1.train.AdamOptimizer(cfg.learning_rate).minimize(loss_final)
    saver = tf1.train.Saver(max_to_keep=3)

    def make_batch(K_active_local: int):
        """
        Always returns arrays sized to K_model, but only first K_active_local users are nonzero.
        The remaining users are padded with zeros and have zero weights.
        """
        batch_ch = []
        batch_v0 = []
        batch_w = []

        for _ in range(B):
            ch_nn, v0, _, _ = compute_channel(
                cfg.nr_of_BS_antennas,
                K_active_local,
                cfg.total_power,
                cfg.path_loss_option,
                cfg.path_loss_range,
                channel_noise_std=getattr(cfg, "channel_noise_std", 0.0),
                channel_noise_relative=getattr(cfg, "channel_noise_relative", True),
                channel_mean_shift=cfg.channel_mean_shift,
            )

            # Pad channels and v0 to K_model
            if K_active_local < K_model:
                from utils import pad_nn_user_dim
                ch_nn, v0 = pad_nn_user_dim(ch_nn, v0, K_model, K_active_local)

            batch_ch.append(np.array(ch_nn))
            batch_v0.append(np.array(v0))

            w = np.ones((K_model, 1), dtype=np.float64)
            w[K_active_local:, 0] = 0.0
            batch_w.append(w)

        return {
            channel_input: np.array(batch_ch),
            init_tp: np.array(batch_v0),
            user_weights_ph: np.array(batch_w),
        }

    # logging
    train_steps, train_wsr_hist, train_loss_hist = [], [], []
    eval_steps, eval_wsr_hist, eval_loss_hist = [], [], []

    log_every = 50
    eval_every = 200

    np.random.seed(cfg.train_seed)

    # fixed eval batch uses CURRENT cfg.active_nr_of_users
    np.random.seed(cfg.test_seed + 123)
    eval_feed = make_batch(K_active_local=K_active)
    np.random.seed(cfg.train_seed)

    with tf1.Session() as sess:
        sess.run(tf1.global_variables_initializer())

        # Restore if infer or train_then_infer with existing ckpt
        if mode == "infer" or (mode == "train_then_infer" and ckpt_exists(ckpt_path)):
            if ckpt_path is None:
                raise ValueError("ckpt_path is required for infer mode")
            saver.restore(sess, ckpt_path)

        if mode in ["train", "train_then_infer"]:
            for it in range(cfg.nr_of_batches_training):
                
                feed = make_batch(K_active_local=K_active)
                sess.run(train_op, feed_dict=feed)
                step = it + 1

                if step % log_every == 0:
                    w, l = sess.run([WSR_final, loss_final], feed_dict=feed)
                    train_steps.append(step)
                    train_wsr_hist.append(float(w))
                    train_loss_hist.append(float(l))

                if step % eval_every == 0:
                    w_eval, l_eval = sess.run([WSR_final, loss_final], feed_dict=eval_feed)
                    eval_steps.append(step)
                    eval_wsr_hist.append(float(w_eval))
                    eval_loss_hist.append(float(l_eval))
                    print("it", step, "train_WSR", float(train_wsr_hist[-1]), "eval_WSR", float(w_eval))

            if ckpt_path is None:
                raise ValueError("ckpt_path is required to save training result")
            saver.save(sess, ckpt_path)

            np.savez(
                os.path.join(ckpt_dir, "unfolded_training_log.npz"),
                train_steps=np.array(train_steps),
                train_wsr=np.array(train_wsr_hist),
                train_loss=np.array(train_loss_hist),
                eval_steps=np.array(eval_steps),
                eval_wsr=np.array(eval_wsr_hist),
                eval_loss=np.array(eval_loss_hist),
            )

            plt.figure()
            if len(train_steps) > 0:
                plt.plot(train_steps, train_wsr_hist, label="train WSR_final")
            if len(eval_steps) > 0:
                plt.plot(eval_steps, eval_wsr_hist, label="eval WSR_final")
            plt.xlabel("Training batch index")
            plt.ylabel("Avg WSR (WSR_final)")
            plt.title("Deep Unfolded WMMSE training curve")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(ckpt_dir, "unfolded_training_curve.png"), dpi=150)
            plt.close()

        if mode in ["infer", "train_then_infer"]:
            np.random.seed(cfg.test_seed)
            wsr_out = []
            for _ in range(cfg.nr_of_batches_test):
                feed = make_batch(K_active_local=K_active)
                wsr_out.append(sess.run(WSR_final, feed_dict=feed))
            return float(np.mean(wsr_out))

    return None
