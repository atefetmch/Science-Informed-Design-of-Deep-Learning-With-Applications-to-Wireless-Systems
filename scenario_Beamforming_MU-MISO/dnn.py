import os
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    compute_channel,
    compute_WSR_nn,
    ensure_dir_for_ckpt,
    run_WMMSE,
)

tf1 = tf.compat.v1


def _complexV_to_realblock(Vc):  
    
    Vr = np.real(Vc).reshape(Vc.shape[0], Vc.shape[1], 1)
    Vi = np.imag(Vc).reshape(Vc.shape[0], Vc.shape[1], 1)
    return np.concatenate([Vr, Vi], axis=1)  



def _enforce_power_np(V_realblock, total_power):
    p = np.sum(V_realblock ** 2) + 1e-12
    return V_realblock * np.sqrt(total_power / p)


def _realblock_to_complex_tf(Vrb, M):

    vr = Vrb[:, :, :M, 0]
    vi = Vrb[:, :, M:, 0]
    return tf.complex(vr, vi) 


def run(cfg, mode, ckpt_path):
    tf1.reset_default_graph()
    ensure_dir_for_ckpt(ckpt_path)

    ckpt_dir = os.path.dirname(ckpt_path) if ckpt_path is not None else "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    B = cfg.nr_of_samples_per_batch
    K_model = cfg.nr_of_users
    M = cfg.nr_of_BS_antennas
    M2 = 2 * M

  
    channel_input = tf1.placeholder(tf.float64, shape=[B, K_model, M2, 2], name="channel_input")
    user_weights_ph = tf1.placeholder(tf.float64, shape=[B, K_model, 1], name="user_weights")
    v_target_ph = tf1.placeholder(tf.float64, shape=[B, K_model, M2, 1], name="v_target_wmmse")

    # model
    x_ch = tf.reshape(channel_input, [B, K_model * M2 * 2])  # channels flattened

    # per-sample normalization 
    mu = tf.reduce_mean(x_ch, axis=1, keepdims=True)
    sd = tf.math.reduce_std(x_ch, axis=1, keepdims=True) + 1e-12
    x_ch = (x_ch - mu) / sd

    x_m  = tf.reshape(user_weights_ph, [B, K_model])         # mask flattened (0/1)
    x = tf.concat([x_ch, x_m], axis=1)


    print("DNN hidden:", cfg.dnn_hidden, "K_model:", K_model, "M:", M, "M2:", M2)

    with tf1.variable_scope("dnn_baseline"):
        h = x
        for li, width in enumerate(cfg.dnn_hidden):
            h = tf1.keras.layers.Dense(
                width,
                activation="relu",
                dtype="float64",
                kernel_initializer="he_normal",
                name=f"fc{li+1}",
            )(h)

        y = tf1.keras.layers.Dense(
            K_model * M2,
            activation=None,
            dtype="float64",
            kernel_initializer="he_normal",
            name="out",
        )(h)


    V = tf.reshape(y, [B, K_model, M2, 1], name="precoder_raw")

   
    mask = tf.cast(tf.expand_dims(user_weights_ph, axis=-1), tf.float64)  # (B, K, 1, 1)
    V_masked = V * mask
    Vt_masked = v_target_ph * mask

  
    power = tf.reduce_sum(tf.square(V_masked), axis=[1, 2, 3], keepdims=True) + 1e-12
    V_masked = V_masked * tf.sqrt(tf.cast(cfg.total_power, tf.float64) / power)

    # normalize power on masked target too
    power_t = tf.reduce_sum(tf.square(Vt_masked), axis=[1, 2, 3], keepdims=True) + 1e-12
    Vt_masked = Vt_masked * tf.sqrt(tf.cast(cfg.total_power, tf.float64) / power_t)

    # WSR metric for network output
    WSR = compute_WSR_nn(
        user_weights_ph,
        channel_input,
        V_masked,
        cfg.noise_power,
        K_model,
        B,
    )
    WSR_final = WSR / B
    loss_unsup = -WSR_final

    WSR_target = compute_WSR_nn(
        user_weights_ph,
        channel_input,
        Vt_masked,
        cfg.noise_power,
        K_model,
        B,
    ) / B

  
    V_c = _realblock_to_complex_tf(V_masked, M)    # (B, K, M)
    Vt_c = _realblock_to_complex_tf(Vt_masked, M)  # (B, K, M)

    VVH = tf.einsum("bkm,bkn->bkmn", V_c, tf.math.conj(V_c))      # (B,K,M,M)
    VtVtH = tf.einsum("bkm,bkn->bkmn", Vt_c, tf.math.conj(Vt_c))  # (B,K,M,M)

    # weight inactive users out (user_weights_ph is (B,K,1))
    w = tf.cast(user_weights_ph, tf.float64)  # (B,K,1)
    w = tf.reshape(w, [B, K_model, 1, 1])     # (B,K,1,1)  <-- FIX
    diff = VVH - VtVtH                        # (B,K,M,M)
    loss_sup = tf.reduce_mean(w * tf.square(tf.abs(diff)))


    train_supervised = bool(getattr(cfg, "train_supervised", False))
    train_unsupervised = bool(getattr(cfg, "train_unsupervised", False))

    global_step = tf1.train.get_or_create_global_step()

    if train_supervised and not train_unsupervised:
        loss_final = loss_sup
    elif train_unsupervised and not train_supervised:
        loss_final = loss_unsup
    else:
        warmup_steps = float(getattr(cfg, "warmup_steps", 500))
        step_f = tf.cast(global_step, tf.float64)
        alpha = tf.maximum(0.0, 1.0 - step_f / tf.cast(warmup_steps, tf.float64))
        loss_final = alpha * loss_sup + (1.0 - alpha) * loss_unsup

    # optimizer
    opt = tf1.train.AdamOptimizer(cfg.learning_rate)

    grads_vars = opt.compute_gradients(loss_final)
    grads_vars = [(tf.clip_by_norm(g, 1.0), v) for (g, v) in grads_vars if g is not None]
    train_op = opt.apply_gradients(grads_vars, global_step=global_step)

    saver = tf1.train.Saver(max_to_keep=3)

    K_active_cfg = int(getattr(cfg, "active_nr_of_users", K_model))
    if K_active_cfg > K_model:
        raise ValueError("active_nr_of_users must be <= nr_of_users (K_model).")

    def make_batch(K_active_local: int):
        """
        Returns tensors sized to K_model.
        Active users are chosen per sample; targets come from WMMSE on the active set and are scattered into slots.
        """
        batch_ch, batch_w, batch_vt = [], [], []

        for _ in range(B):
            ch_active, _, ch_wmmse_active, _ = compute_channel(
                cfg.nr_of_BS_antennas,
                K_active_local,
                cfg.total_power,
                cfg.path_loss_option,
                cfg.path_loss_range,
                channel_noise_std=getattr(cfg, "channel_noise_std", 0.0),
                channel_noise_relative=getattr(cfg, "channel_noise_relative", True),
                channel_mean_shift=getattr(cfg, "channel_mean_shift", 0.0),
            )
            ch_active = np.array(ch_active)  # (K_active, 2M, 2)

            # choose active indices in the K_model tensor
            if K_active_local == K_model:
                active_idx = np.arange(K_model)
            else:
               # active_idx = np.random.choice(K_model, size=K_active_local, replace=False)
                active_idx = np.arange(K_active_local)


            # scatter channels
            ch_full = np.zeros((K_model, M2, 2), dtype=np.float64)
            ch_full[active_idx, :, :] = ch_active

            # weights mask
            w = np.zeros((K_model, 1), dtype=np.float64)
            w[active_idx, 0] = 1.0

            # WMMSE on the compact active system
            selected_users = list(range(K_active_local))
            user_weights_vec = np.ones(K_active_local, dtype=np.float64)

            _, _, Vc_active, _ = run_WMMSE(
                cfg.epsilon,
                ch_wmmse_active,
                selected_users,
                cfg.total_power,
                cfg.noise_power,
                user_weights_vec,
                cfg.nr_of_iterations_wmmse,
                cfg.power_tolerance,
                log=False,
            )  

            for k in range(Vc_active.shape[0]):
                ref = Vc_active[k, 0]
                if np.abs(ref) > 1e-12:
                    Vc_active[k, :] *= np.exp(-1j * np.angle(ref))

            V0_active = _complexV_to_realblock(Vc_active)
            V0_active = _enforce_power_np(V0_active, cfg.total_power)

            Vt_full = np.zeros((K_model, M2, 1), dtype=np.float64)
            Vt_full[active_idx, :, :] = V0_active

            batch_ch.append(ch_full)
            batch_w.append(w)
            batch_vt.append(Vt_full)

        return {
            channel_input: np.array(batch_ch),
            user_weights_ph: np.array(batch_w),
            v_target_ph: np.array(batch_vt),
        }

    # logging
    train_steps, train_wsr_hist, train_loss_hist = [], [], []
    eval_steps, eval_wsr_hist, eval_loss_hist = [], [], []

    log_every = 50
    eval_every = 200

    np.random.seed(cfg.train_seed)
    np.random.seed(cfg.test_seed + 123)
    eval_feed = make_batch(K_active_local=K_active_cfg)
    np.random.seed(cfg.train_seed)

    with tf1.Session() as sess:
        sess.run(tf1.global_variables_initializer())

        if mode in ["infer", "train_then_infer"]:
            if ckpt_path is None and mode == "infer":
                raise ValueError("ckpt_path is required for infer mode")
            if ckpt_path is not None:
                latest = tf1.train.latest_checkpoint(ckpt_dir)
                if latest is not None:
                    saver.restore(sess, latest)
                    print("Restored:", latest)
                elif mode == "infer":
                    raise ValueError(f"No checkpoint found in {ckpt_dir}.")

        if mode in ["train", "train_then_infer"]:
            for it in range(cfg.nr_of_batches_training):
                feed = make_batch(K_active_local=K_active_cfg)
                sess.run(train_op, feed_dict=feed)
                step = it + 1

                if step % log_every == 0:
                    w_net, w_tgt, l_val = sess.run([WSR_final, WSR_target, loss_final], feed_dict=feed)
                    train_steps.append(step)
                    train_wsr_hist.append(float(w_net))
                    train_loss_hist.append(float(l_val))
                    print("it", step, "WSR_net", float(w_net), "WSR_tgt", float(w_tgt), "loss", float(l_val))

                if step % eval_every == 0:
                    w_eval, w_tgt_eval, l_eval = sess.run([WSR_final, WSR_target, loss_final], feed_dict=eval_feed)
                    eval_steps.append(step)
                    eval_wsr_hist.append(float(w_eval))
                    eval_loss_hist.append(float(l_eval))
                    print("EVAL it", step, "WSR_net", float(w_eval), "WSR_tgt", float(w_tgt_eval))

            if ckpt_path is None:
                raise ValueError("ckpt_path is required to save training result")
            saver.save(sess, ckpt_path)

            np.savez(
                os.path.join(ckpt_dir, "dnn_training_log.npz"),
                train_steps=np.array(train_steps),
                train_wsr=np.array(train_wsr_hist),
                train_loss=np.array(train_loss_hist),
                eval_steps=np.array(eval_steps),
                eval_wsr=np.array(eval_wsr_hist),
                eval_loss=np.array(eval_loss_hist),
            )

            plt.figure()
            if len(train_steps) > 0:
                plt.plot(train_steps, train_wsr_hist, label="train WSR_net")
            if len(eval_steps) > 0:
                plt.plot(eval_steps, eval_wsr_hist, label="eval WSR_net")
            plt.xlabel("Training batch index")
            plt.ylabel("Avg WSR")
            plt.title("DNN training curve")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(ckpt_dir, "dnn_training_curve.png"), dpi=150)
            plt.close()

        if mode in ["infer", "train_then_infer"]:
            np.random.seed(cfg.test_seed)
            wsr_out = []
            for _ in range(cfg.nr_of_batches_test):
                feed = make_batch(K_active_local=K_active_cfg)
                wsr_out.append(sess.run(WSR_final, feed_dict=feed))
            return float(np.mean(wsr_out))

    return None
