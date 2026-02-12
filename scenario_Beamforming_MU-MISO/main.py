import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"         
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"         

import tensorflow as tf

# Turn off TF deprecation warnings (Python side)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Use the explicit compat call to avoid the deprecated symbol warning
tf.compat.v1.disable_eager_execution()


tf.get_logger().setLevel("ERROR")

import argparse
from config import Config


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["wmmse", "truncated", "unfolded", "sn_unfolded", "dnn", "deepsets_dnn"],
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train_then_infer",
        choices=["train", "infer", "train_then_infer"],
    )

    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--pgd_steps", type=int, default=None)
    parser.add_argument("--L", type=int, default=None)
    parser.add_argument("--active_k", type=int, default=None)

    # DNN training objective options
    parser.add_argument(
        "--supervised",
        action="store_true",
        help="Train DNN to imitate RZF/ZF (MSE loss).",
    )
    parser.add_argument(
        "--unsupervised",
        action="store_true",
        help="Train DNN with -WSR (unsupervised). If neither flag is set, hybrid is used.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Hybrid: number of steps to decay from supervised to unsupervised.",
    )
    parser.add_argument(
        "--rzf_alpha",
        type=float,
        default=None,
        help="If set, use RZF with this alpha. If None, use default formula in utils.regularized_zero_forcing.",
    )

    args = parser.parse_args()
    cfg = Config()

    if args.active_k is not None:
        cfg.active_nr_of_users = args.active_k

    if cfg.active_nr_of_users > cfg.nr_of_users:
        raise ValueError("active_k must be <= nr_of_users (K_model)")

    if args.pgd_steps is not None:
        cfg.pgd_steps = args.pgd_steps
    if args.L is not None:
        cfg.nr_of_iterations_nn = args.L

    # Set DNN objective options into cfg (dnn.py reads these)
    cfg.train_supervised = bool(args.supervised)
    cfg.train_unsupervised = bool(args.unsupervised)
    cfg.warmup_steps = int(args.warmup_steps)
    cfg.rzf_alpha = args.rzf_alpha  # can be None

    # Pick checkpoint directory based on algo
    if args.algo in ["unfolded", "sn_unfolded"]:
        ckpt_dir = f"checkpoints_L{cfg.nr_of_iterations_nn}_K{cfg.pgd_steps}"
    elif args.algo == "dnn":
        # for naming consistency you can include pgd_steps in folder name if you want
        if cfg.pgd_steps is not None:
            ckpt_dir = f"checkpoints_L{cfg.nr_of_iterations_nn}_K{cfg.pgd_steps}"
        else:
            ckpt_dir = f"checkpoints_L{cfg.nr_of_iterations_nn}"
    else:
        ckpt_dir = "checkpoints"

    os.makedirs(ckpt_dir, exist_ok=True)


    if args.ckpt is None:
        ckpt_path = {
            "unfolded": os.path.join(ckpt_dir, "unfolded.ckpt"),
            "sn_unfolded": os.path.join(ckpt_dir, "sn_unfolded.ckpt"),
            "dnn": os.path.join(ckpt_dir, "dnn.ckpt"),
        }.get(args.algo, None)
    else:
        ckpt_path = args.ckpt
        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)

    if args.algo == "wmmse":
        from algos.wmmse import evaluate
        print("Avg WSR (WMMSE):", evaluate(cfg))
        return

    if args.algo == "truncated":
        from algos.truncated_wmmse import evaluate
        print("Avg WSR (Truncated WMMSE):", evaluate(cfg))
        return

    if args.algo == "unfolded":
        from algos.unfolded_wmmse import run
        print("Avg WSR (Deep Unfolded WMMSE):", run(cfg, args.mode, ckpt_path))
        return

    if args.algo == "sn_unfolded":
        from algos.sn_unfolded_wmmse import run
        print("Avg WSR (Super-Nesterov Unfolded WMMSE):", run(cfg, args.mode, ckpt_path))
        return

    if args.algo == "dnn":
        from algos.dnn import run
        print("Avg WSR (DNN baseline):", run(cfg, args.mode, ckpt_path))
        return
    
    if args.algo == "deepsets_dnn":
        from algos.deepsets_dnn import run
        ckpt_dir = f"checkpoints_deepsets_L{cfg.nr_of_iterations_nn}_K{cfg.pgd_steps}"
        os.makedirs(ckpt_dir, exist_ok=True)
        if args.ckpt is None:
            ckpt_path = os.path.join(ckpt_dir, "deepsets_dnn.ckpt")
        else:
            ckpt_path = args.ckpt
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)

        print("Avg WSR (DeepSets DNN):", run(cfg, args.mode, ckpt_path))
        return



if __name__ == "__main__":
    main()
