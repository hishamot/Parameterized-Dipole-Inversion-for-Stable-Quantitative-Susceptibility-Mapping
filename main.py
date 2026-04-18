"""
main.py — Command-line entry point for PIN-QSM.

Usage examples
--------------
# Full pipeline (train + test):
python main.py train \
    --train_csv data/train.csv \
    --val_csv   data/val.csv   \
    --test_csv  data/test.csv  \
    --root      /path/to/data  \
    --epochs 50 --lr 1e-4

# Training only:
python main.py train \
    --train_csv data/train.csv --val_csv data/val.csv \
    --root /path/to/data --epochs 50

# Testing only (requires a saved checkpoint):
python main.py test \
    --test_csv data/test.csv \
    --root /path/to/data \
    --model_path best_model.pt \
    --dinv_path  best_dipole_inverse.pt
"""

import argparse
import os
import sys


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pin_qsm",
        description="Physics-Informed Network for Quantitative Susceptibility Mapping (QSM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------ train
    train_p = subparsers.add_parser(
        "train", help="Train (and optionally test) the QSM UNet."
    )
    _add_data_args(train_p, require_test=False)
    _add_train_args(train_p)
    train_p.add_argument(
        "--test_csv",
        type=str,
        default=None,
        help="If provided, run test evaluation after training.",
    )
    train_p.add_argument(
        "--n_vis",
        type=int,
        default=30,
        help="Number of test samples for which to save visualisations.",
    )
    train_p.add_argument(
        "--vis_dir",
        type=str,
        default=".",
        help="Directory to save visualisation PNG files.",
    )
    train_p.add_argument(
        "--plot_loss",
        action="store_true",
        default=False,
        help="Save a loss-curve plot after training.",
    )

    # ------------------------------------------------------------------ test
    test_p = subparsers.add_parser(
        "test", help="Evaluate a saved checkpoint on the test set."
    )
    _add_data_args(test_p, require_test=True)
    test_p.add_argument(
        "--model_path",
        type=str,
        default="best_model.pt",
        help="Path to saved model weights.",
    )
    test_p.add_argument(
        "--dinv_path",
        type=str,
        default="best_dipole_inverse.pt",
        help="Path to saved D_inv tensor.",
    )
    test_p.add_argument(
        "--n_vis",
        type=int,
        default=30,
        help="Number of test samples for which to save visualisations.",
    )
    test_p.add_argument(
        "--vis_dir",
        type=str,
        default=".",
        help="Directory to save visualisation PNG files.",
    )
    test_p.add_argument(
        "--plot_metrics",
        action="store_true",
        default=False,
        help="Save a metrics summary bar chart.",
    )

    return parser


def _add_data_args(p: argparse.ArgumentParser, require_test: bool = False) -> None:
    p.add_argument("--root", type=str, required=True, help="Data root directory.")
    p.add_argument("--train_csv", type=str, default=None, help="Training CSV file.")
    p.add_argument("--val_csv", type=str, default=None, help="Validation CSV file.")
    if require_test:
        p.add_argument("--test_csv", type=str, required=True, help="Test CSV file.")


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device.",
    )
    p.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    p.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
    p.add_argument(
        "--model_path",
        type=str,
        default="best_model.pt",
        help="Output path for best model weights.",
    )
    p.add_argument(
        "--dinv_path",
        type=str,
        default="best_dipole_inverse.pt",
        help="Output path for best D_inv tensor.",
    )


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def run_train(args: argparse.Namespace) -> None:
    import torch
    from training import train

    # Validate required args for training
    for attr, flag in [("train_csv", "--train_csv"), ("val_csv", "--val_csv")]:
        if getattr(args, attr) is None:
            print(f"error: {flag} is required for the train command.", file=sys.stderr)
            sys.exit(1)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠  CUDA not available — falling back to CPU.")
        device = "cpu"

    # Create output directories if needed
    os.makedirs(args.vis_dir, exist_ok=True)

    print("=" * 60)
    print("PIN-QSM  |  TRAINING")
    print(f"  train_csv  : {args.train_csv}")
    print(f"  val_csv    : {args.val_csv}")
    print(f"  root       : {args.root}")
    print(f"  device     : {device}")
    print(f"  epochs     : {args.epochs}")
    print(f"  lr         : {args.lr}")
    print(f"  model_path : {args.model_path}")
    print(f"  dinv_path  : {args.dinv_path}")
    print("=" * 60)

    train_losses, val_losses = train(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        root=args.root,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.model_path,
        dinv_save_path=args.dinv_path,
    )

    if args.plot_loss:
        from visualization import plot_loss_curves
        plot_loss_curves(train_losses, val_losses, out_path="loss_curves.png")
        print("📊 Loss curve saved → loss_curves.png")

    # Optional: run test after training
    if args.test_csv is not None:
        print("\n--- Running test evaluation ---")
        _run_test_internal(
            test_csv=args.test_csv,
            root=args.root,
            model_path=args.model_path,
            dinv_path=args.dinv_path,
            device=device,
            n_vis=args.n_vis,
            vis_dir=args.vis_dir,
            plot_metrics=False,
        )


def run_test(args: argparse.Namespace) -> None:
    import torch
    device = "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠  CUDA not available — falling back to CPU.")
        device = "cpu"

    os.makedirs(args.vis_dir, exist_ok=True)

    print("=" * 60)
    print("PIN-QSM  |  TESTING")
    print(f"  test_csv   : {args.test_csv}")
    print(f"  root       : {args.root}")
    print(f"  model_path : {args.model_path}")
    print(f"  dinv_path  : {args.dinv_path}")
    print(f"  device     : {device}")
    print("=" * 60)

    _run_test_internal(
        test_csv=args.test_csv,
        root=args.root,
        model_path=args.model_path,
        dinv_path=args.dinv_path,
        device=device,
        n_vis=args.n_vis,
        vis_dir=args.vis_dir,
        plot_metrics=args.plot_metrics,
    )


def _run_test_internal(
    test_csv, root, model_path, dinv_path, device, n_vis, vis_dir, plot_metrics
):
    from testing import test
    results = test(
        test_csv=test_csv,
        root=root,
        model_path=model_path,
        dinv_path=dinv_path,
        device=device,
        n_vis=n_vis,
        vis_dir=vis_dir,
    )
    if plot_metrics:
        from visualization import plot_metrics_summary
        plot_metrics_summary(results, out_path="metrics_summary.png")
        print("📊 Metrics chart saved → metrics_summary.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "test":
        run_test(args)


if __name__ == "__main__":
    main()
