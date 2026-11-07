import argparse
import json
from pathlib import Path

from ...nextgen2025.train import train
from ...nextgen2025.eval import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root with input/ and target/")
    parser.add_argument("--workdir", type=str, required=True, help="Working directory for outputs")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--model_variant", type=str, default="dual_stage_base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--augment_blur_dark", action="store_true")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    ckpt_dir = workdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train(
        data_root=args.data_root,
        out_dir=str(ckpt_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        model_variant=args.model_variant,
        device=args.device,
        augment_blur_dark=args.augment_blur_dark,
    )

    # Evaluate last checkpoint
    last_ckpt = sorted(ckpt_dir.glob("model_epoch_*.pt"))[-1]
    metrics = evaluate(
        data_root=args.data_root,
        checkpoint=str(last_ckpt),
        model_variant=args.model_variant,
        device=args.device,
        image_size=args.image_size,
    )
    with open(workdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Experiment complete.")


if __name__ == "__main__":
    main()


