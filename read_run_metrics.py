from pathlib import Path
import sys
import re

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch


def get_best_epoch(run_dir: Path):
    ckpt_dir = run_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("epoch=*-step=*.ckpt"))
    if not ckpts:
        return None, None, None

    ckpt_path = ckpts[0]
    m = re.search(r"epoch=(\d+)-step=(\d+)\.ckpt", ckpt_path.name)
    epoch = int(m.group(1)) if m else None

    ckpt = torch.load(ckpt_path, map_location="cpu")
    callbacks = ckpt.get("callbacks", {})
    best_val_cer = None
    for _, v in callbacks.items():
        if isinstance(v, dict) and "best_model_score" in v:
            score = v["best_model_score"]
            best_val_cer = float(score) if score is not None else None
            break

    return epoch, best_val_cer, ckpt_path


def load_event_accumulator(run_dir: Path):
    tb_dir = run_dir / "lightning_logs" / "version_0"
    if not tb_dir.exists():
        raise FileNotFoundError(f"No lightning_logs/version_0 in {run_dir}")

    ea = EventAccumulator(str(tb_dir))
    ea.Reload()
    return ea


def last_scalar(ea: EventAccumulator, tag: str):
    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        return None
    vals = ea.Scalars(tag)
    if not vals:
        return None
    return vals[-1].value


def main():
    if len(sys.argv) < 2:
        print("Usage: python read_run_metrics.py <run_dir1> [<run_dir2> ...]")
        sys.exit(1)

    for run in sys.argv[1:]:
        run_dir = Path(run)
        print(f"==== {run_dir} ====")

        best_epoch, best_val_cer, ckpt_path = get_best_epoch(run_dir)
        print(f"best_epoch: {best_epoch}")
        print(f"best_ckpt: {ckpt_path}")
        print(f"best_val_CER_from_ckpt: {best_val_cer}")

        ea = load_event_accumulator(run_dir)
        tags = set(ea.Tags().get("scalars", []))

        # Print available tags once in case names differ
        print("available_tags:", sorted(tags))

        for tag in ["val/loss", "val/CER", "test/loss", "test/CER", "val/IER", "val/DER", "val/SER", "test/IER", "test/DER", "test/SER"]:
            print(f"{tag}: {last_scalar(ea, tag)}")

        print()


if __name__ == "__main__":
    main()