from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytorch_lightning as pl
import os

print("START OF SCRIPT", flush=True)

def main():
    with initialize(version_base=None, config_path="config"):
        cfg = compose(
            config_name="base",   # or whatever top-level name worked for you before
            overrides=[
                "model=cnn_transformer",
                "trainer.max_epochs=0",
            ],
        )

    print(cfg, flush=True)

    module = instantiate(cfg.module, _recursive_=False)

    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=False)
    transforms_cfg = OmegaConf.to_container(cfg.transforms, resolve=False)

    dataset_cfg["root"] = os.path.join(os.getcwd(), "data")

    train_sessions = dataset_cfg["train"]
    val_sessions = dataset_cfg["val"]
    test_sessions = dataset_cfg["test"]

    train_transform = [instantiate(cfg.to_tensor), instantiate(cfg.band_rotation), instantiate(cfg.temporal_jitter), instantiate(cfg.logspec), instantiate(cfg.specaug)]
    val_transform = [instantiate(cfg.to_tensor), instantiate(cfg.logspec)]
    test_transform = [instantiate(cfg.to_tensor), instantiate(cfg.logspec)]

    datamodule = instantiate(
        cfg.datamodule,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        train_sessions=train_sessions,
        val_sessions=val_sessions,
        test_sessions=test_sessions,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        root=dataset_cfg["root"],
    )

    ckpt = "/home/eagle/nndl/emg2qwerty/logs/2026-03-07/02-43-55/checkpoints/epoch=57-step=6960.ckpt"

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )

    print("\n=== VALIDATE ===", flush=True)
    trainer.validate(model=module, datamodule=datamodule, ckpt_path=ckpt)

    print("\n=== TEST ===", flush=True)
    trainer.test(model=module, datamodule=datamodule, ckpt_path=ckpt)

if __name__ == "__main__":
    main()