import os
import argparse
from datetime import datetime
from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger

# from ovi.lightning.ovi_module import OviFusionTrainModule
from ovi.lightning.ovi_ref import OviFusionTrainModule
from ovi.data.loader import AspectRatioDataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Optional yaml config for training hyperparams")
    parser.add_argument("--data_paths", type=str, nargs="+", default=None, help="Path(s) to dataset files")
    parser.add_argument("--num_nodes", type=int, default=1, help="num nodes for trainint")
    parser.add_argument("--deepspeed_config", type=str, default=None, help="Path to DeepSpeed config json (optional)")

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = {}
    if args.config:
        try:
            cfg = OmegaConf.load(args.config)
        except Exception as e:
            raise ValueError(f"Error loading config file: {e}")
    else:
        raise ValueError("A config file must be provided with --config argument.")

    # Set global seed for reproducibility
    # Each rank will get a different seed based on rank number for proper randomness
    base_seed = cfg.get("seed", 42)
    pl.seed_everything(base_seed, workers=True)

    train_loader = AspectRatioDataLoader(
        data_paths=args.data_paths,
        batch_size=cfg.get("batch_size", 1),
        num_workers=cfg.get("dataloader_num_workers", 4),
        steps_per_epoch=cfg.get("steps_per_epoch", None),
        num_aspect_buckets=cfg.get("num_aspect_buckets", 8),
        seed=cfg.get("seed", None),
        pin_memory=cfg.get("pin_memory", False),
        description_model=cfg.get("description_model", "qwen2-VL-72B-detail"),
        audio_description_model=cfg.get("audio_description_model", "mimo-audio"),
        num_frames=cfg.get("num_frames", 81),
        frame_tolerance=cfg.get("frame_tolerance", 10),
        max_audio_duration=cfg.get("max_audio_duration", 20.0),
        height=cfg.get("height", 720),
        width=cfg.get("width", 720),
    )

    model = OviFusionTrainModule(cfg)

    strategy = None
    training_strategy = cfg.get("training_strategy", "auto")

    # 如果是单卡训练，使用默认策略
    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
        strategy = "auto"
    else:
        # 多卡训练时使用配置的策略
        if training_strategy.startswith("deepspeed"):
            # Lightning supports passing the deepspeed config path or a string like 'deepspeed_stage_2'
            if args.deepspeed_config:
                strategy = args.deepspeed_config
            else:
                strategy = training_strategy
        elif training_strategy == "ddp":
            strategy = DDPStrategy(find_unused_parameters=False)

    # Setup TensorBoard logger with explicit log directory
    # Get local rank to determine if this is the main process
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Setup directories - all ranks need access to these paths
    base_dir = cfg.get("default_root_dir", "./outputs")
    log_dir = os.getenv('TENSORBOARD_LOG_PATH', base_dir)

    # Ensure all ranks use the same experiment name
    # Use timestamp with hour precision - all processes will get the same value
    timestamp = datetime.now().strftime("%Y%m%d_%H")  # e.g., 20251117_10
    experiment_name = f"run_{timestamp}"

    checkpoint_dir = os.path.join(base_dir, experiment_name)

    # Only create logger on rank 0
    if local_rank == 0:
        tb_logger = TensorBoardLogger(
            save_dir=log_dir,
            name=experiment_name,
            version="",  # Empty version to avoid extra subdirectory
            default_hp_metric=False
        )
        print(f"TensorBoard logs will be saved to: {tb_logger.log_dir}")
        print(f"Checkpoints will be saved to: {checkpoint_dir}")
    else:
        # Other ranks don't need logger
        tb_logger = False  # PyTorch Lightning accepts False to disable logging

    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        strategy=strategy,
        num_nodes=args.num_nodes,
        max_epochs=int(cfg.get("max_epochs", 50)),
        accumulate_grad_batches=int(cfg.get("accumulate_grad_batches", 1)),
        precision=cfg.get("precision", "bf16"),
        default_root_dir=cfg.get("default_root_dir", None),
        logger=tb_logger,
        use_distributed_sampler=False,
        log_every_n_steps=int(cfg.get("log_every_n_steps", 10)),
                
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                # save_weights_only=True,
                dirpath=checkpoint_dir,
                # monitor="train/loss_epoch",  # Monitor epoch-level training loss
                monitor=None,
                save_top_k=int(cfg.get("save_top_k", 0)),
                mode="min",
                save_weights_only=True,
                save_last=True,
                filename="epoch={epoch}-step={step}-loss={train/loss_epoch:.4f}",
                auto_insert_metric_name=False,  # Prevent auto-adding metric name with /
            )
        ],
    )

    trainer.fit(model, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()
