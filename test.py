"""
Test script for OVI Fusion Model with LoRA support.

This script supports:
- Text-to-video (t2v), image-to-video (i2v), and text+image-to-video (t2i2v) generation
- LoRA checkpoint loading for fine-tuned models
- Single prompt or batch processing from CSV
- Distributed inference with sequence parallelism
- Comprehensive logging and error handling
"""

import os
import sys
import logging
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

from ovi.ovi_fusion_engine import OviFusionEngine
from ovi.lightning.ovi_module import OviFusionTrainModule
from ovi.utils.io_utils import save_video
from ovi.utils.processing_utils import (
    format_prompt_for_filename,
    validate_and_process_user_prompt
)
from ovi.distributed_comms.util import (
    get_world_size,
    get_local_rank,
    get_global_rank
)
from ovi.distributed_comms.parallel_states import (
    initialize_sequence_parallel_state,
    get_sequence_parallel_state,
    nccl_info
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test OVI Fusion Model with LoRA support"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="ovi/configs/test/test.yaml",
        help="Path to test configuration YAML file"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA checkpoint (overrides config)"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Path to CSV file with prompts (overrides config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["t2v", "i2v", "t2i2v"],
        default=None,
        help="Generation mode (overrides config)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, overrides config)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["720x720_5s", "960x960_5s", "960x960_10s"],
        default=None,
        help="Model variant to use (overrides config)"
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA weights from checkpoint if available"
    )

    return parser.parse_args()


def setup_logging(rank: int, log_level: str = "INFO"):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    if rank == 0:
        logging.basicConfig(
            level=level,
            format="[%(asctime)s] [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)]
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def load_prompts_from_csv(csv_path: str, mode: str) -> List[Tuple[str, Optional[str]]]:
    """
    Load prompts from CSV file.

    Expected CSV format:
    - For t2v: text_prompt
    - For i2v/t2i2v: text_prompt, image_path

    Args:
        csv_path: Path to CSV file
        mode: Generation mode (t2v, i2v, t2i2v)

    Returns:
        List of (text_prompt, image_path) tuples
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Prompts file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "text_prompt" not in df.columns:
        raise ValueError("CSV must contain 'text_prompt' column")

    prompts = []
    for _, row in df.iterrows():
        text_prompt = row["text_prompt"]
        image_path = row.get("image_path", None) if mode in ["i2v", "t2i2v"] else None
        prompts.append((text_prompt, image_path))

    return prompts


def validate_config(config):
    """Validate test configuration."""
    # Validate mode
    mode = config.get("mode", "t2v")
    if mode not in ["t2v", "i2v", "t2i2v"]:
        raise ValueError(f"Invalid mode: {mode}. Must be one of [t2v, i2v, t2i2v]")

    # Validate image path for i2v/t2i2v modes
    if mode in ["i2v", "t2i2v"]:
        image_path = config.get("image_path")
        prompts_file = config.get("prompts_file")
        if not prompts_file and (not image_path or not os.path.exists(image_path)):
            raise ValueError(f"Mode {mode} requires valid image_path or prompts_file")

    # Validate LoRA path if provided
    lora_path = config.get("lora_path")
    if lora_path and not os.path.exists(lora_path):
        logging.warning(f"LoRA path does not exist: {lora_path}")

    # Validate video dimensions
    hw = config.get("video_frame_height_width", [720, 720])
    # OmegaConf may return a ListConfig, convert to list
    if hasattr(hw, '__iter__') and not isinstance(hw, str):
        hw = list(hw)
        config.video_frame_height_width = hw  # Update config with proper list

    if not isinstance(hw, list) or len(hw) != 2:
        raise ValueError(f"video_frame_height_width must be [height, width], got {hw} (type: {type(hw)})")

    return True


def setup_distributed(config):
    """Setup distributed environment."""
    world_size = get_world_size()
    global_rank = get_global_rank()
    local_rank = get_local_rank()

    torch.cuda.set_device(local_rank)

    sp_size = config.get("sp_size", 1)
    if sp_size > world_size or world_size % sp_size != 0:
        raise ValueError(
            f"sp_size ({sp_size}) must be <= world_size ({world_size}) "
            f"and world_size must be divisible by sp_size"
        )

    if world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=global_rank,
            world_size=world_size
        )
    else:
        if sp_size != 1:
            raise ValueError(f"When world_size is 1, sp_size must be 1, got {sp_size}")

    initialize_sequence_parallel_state(sp_size)

    return local_rank, global_rank, world_size, sp_size


def load_model_with_lora(config, device):
    """
    Load OVI Fusion Engine with optional LoRA weights.

    Args:
        config: Test configuration
        device: Device to load model on

    Returns:
        OviFusionEngine instance
    """
    target_dtype = torch.bfloat16 if config.get("dtype", "bf16") == "bf16" else torch.float16

    logging.info("Loading OVI Fusion Engine...")

    # Create engine
    ovi_engine = OviFusionEngine(
        config=config,
        device=device,
        target_dtype=target_dtype
    )

    # Load LoRA if specified
    lora_path = config.get("lora_path", None)
    lora_alpha = config.get("lora_alpha", 1.0)
    use_ema = config.get("use_ema", False)

    if lora_path and os.path.exists(lora_path):
        logging.info(f"Loading LoRA checkpoint from: {lora_path}")

        try:
            # Load checkpoint
            if lora_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                checkpoint = load_file(lora_path, device="cpu")
            else:
                checkpoint = torch.load(lora_path, map_location="cpu")

            # Check for EMA weights and select which to use
            # Note: Both regular and EMA state_dict contain LoRA parameters and fusion layer weights
            # The base model is already loaded in OviFusionEngine, weights are applied on top
            if use_ema and "ema_state_dict" in checkpoint:
                logging.info("[EMA] Loading EMA weights from checkpoint for better quality")
                state_dict = checkpoint["ema_state_dict"]
            elif use_ema:
                logging.warning("[EMA] use_ema=True but no EMA weights found in checkpoint, using regular weights")
                state_dict = checkpoint.get("state_dict", checkpoint)
            else:
                state_dict = checkpoint.get("state_dict", checkpoint)

            # Separate LoRA parameters and fusion layer weights
            # Fusion layers (k_fusion, v_fusion, pre_attn_norm_fusion, norm_k_fusion) are trained directly
            # Audio/Video model parameters use LoRA (lora_A, lora_B)
            lora_state = {}
            fusion_state = {}
            fusion_layer_names = ("k_fusion", "v_fusion", "pre_attn_norm_fusion", "norm_k_fusion")

            for key, value in state_dict.items():
                if "lora_A" in key or "lora_B" in key:
                    lora_state[key] = value
                elif any(fname in key for fname in fusion_layer_names):
                    fusion_state[key] = value

            # Apply LoRA weights to fusion model (for audio/video models)
            if lora_state:
                from ovi.data.lora import GeneralLoRALoader
                # Use CPU for computing LoRA weights to save GPU memory
                # The loader will automatically move weights to the correct device when applying
                lora_loader = GeneralLoRALoader(device="cpu", torch_dtype=target_dtype)
                num_lora = lora_loader.load(ovi_engine.model, lora_state, alpha=lora_alpha)
                logging.info(f"LoRA weights loaded: {num_lora} modules updated with alpha={lora_alpha}")

            # Load fusion layer weights directly
            if fusion_state:
                model_state = ovi_engine.model.state_dict()
                fusion_loaded = 0
                for key, value in fusion_state.items():
                    if key in model_state:
                        model_state[key] = value.to(dtype=model_state[key].dtype)
                        fusion_loaded += 1
                    else:
                        logging.warning(f"Fusion layer key not found in model: {key}")
                ovi_engine.model.load_state_dict(model_state)
                logging.info(f"Fusion layer weights loaded: {fusion_loaded} parameters")

            weight_type = "EMA" if (use_ema and "ema_state_dict" in checkpoint) else "regular"
            total_params = len(lora_state) + len(fusion_state)
            logging.info(f"{weight_type} weights loaded successfully: {total_params} parameters total")

        except Exception as e:
            logging.error(f"Failed to load LoRA weights: {e}")
            raise
    else:
        logging.info("No LoRA weights specified, using base model")

    logging.info("OVI Fusion Engine loaded!")

    return ovi_engine


def distribute_data(all_data, global_rank, world_size, sp_size):
    """
    Distribute data across GPUs/SP groups.

    Args:
        all_data: List of all data samples
        global_rank: Global rank of current process
        world_size: Total number of processes
        sp_size: Sequence parallel size

    Returns:
        List of data samples for current rank
    """
    use_sp = get_sequence_parallel_state()

    if use_sp:
        sp_rank = nccl_info.rank_within_group
        sp_group_id = global_rank // sp_size
        num_sp_groups = world_size // sp_size
    else:
        sp_rank = 0
        sp_group_id = global_rank
        num_sp_groups = world_size

    # Distribute across SP groups
    this_rank_data = all_data[sp_group_id::num_sp_groups]

    return this_rank_data, sp_rank


def main():
    """Main test function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = OmegaConf.load(args.config)

    # Override config with command line arguments
    if args.lora_path:
        config.lora_path = args.lora_path
    if args.prompts_file:
        config.prompts_file = args.prompts_file
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.mode:
        config.mode = args.mode
    if args.seed is not None:
        config.seed = args.seed
    if args.device:
        config.device = args.device
    if args.model_name:
        config.model_name = args.model_name
    if args.use_ema:
        config.use_ema = True

    # Setup distributed
    local_rank, global_rank, world_size, sp_size = setup_distributed(config)

    # Setup logging
    setup_logging(global_rank, config.get("log_level", "INFO"))

    # Validate configuration
    validate_config(config)

    logging.info(f"Model: {config.get('model_name', 'default')}")
    logging.info(f"Using SP: {get_sequence_parallel_state()}, SP_SIZE: {sp_size}")
    logging.info(f"World size: {world_size}, Global rank: {global_rank}, Local rank: {local_rank}")

    # Load prompts
    prompts_file = config.get("prompts_file")
    if prompts_file:
        logging.info(f"Loading prompts from: {prompts_file}")
        all_prompts = load_prompts_from_csv(prompts_file, config.mode)
    else:
        text_prompt = config.get("text_prompt")
        image_path = config.get("image_path")
        all_prompts = [(text_prompt, image_path)]

    logging.info(f"Total prompts to process: {len(all_prompts)}")

    # Distribute data across ranks
    this_rank_prompts, sp_rank = distribute_data(
        all_prompts, global_rank, world_size, sp_size
    )

    logging.info(f"Rank {global_rank} processing {len(this_rank_prompts)} prompts")

    # Create output directory
    output_dir = config.get("output_dir", "./test_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    device = local_rank if config.get("device") == "cuda" else config.get("device")
    ovi_engine = load_model_with_lora(config, device)

    # Process each prompt
    for prompt_idx, (text_prompt, image_path) in enumerate(tqdm(
        this_rank_prompts,
        desc=f"Rank {global_rank}",
        disable=(global_rank != 0)
    )):
        # Validate image path for i2v mode
        if config.mode in ["i2v", "t2i2v"]:
            if not image_path or not os.path.exists(image_path):
                logging.warning(f"Skipping prompt {prompt_idx}: invalid image_path")
                continue

        # Generate multiple samples if requested
        for sample_idx in range(config.get("each_example_n_times", 1)):
            try:
                seed = config.get("seed", 100) + sample_idx

                # Generate
                generated_video, generated_audio, generated_image = ovi_engine.generate(
                    text_prompt=text_prompt,
                    image_path=image_path,
                    video_frame_height_width=config.get("video_frame_height_width", [720, 720]),
                    seed=seed,
                    solver_name=config.get("solver_name", "unipc"),
                    sample_steps=config.get("sample_steps", 50),
                    shift=config.get("shift", 5.0),
                    video_guidance_scale=config.get("video_guidance_scale", 4.0),
                    audio_guidance_scale=config.get("audio_guidance_scale", 3.0),
                    slg_layer=config.get("slg_layer", 11),
                    video_negative_prompt=config.get("video_negative_prompt", ""),
                    audio_negative_prompt=config.get("audio_negative_prompt", "")
                )

                # Save results (only on SP rank 0)
                if sp_rank == 0:
                    formatted_prompt = format_prompt_for_filename(text_prompt)
                    hw_str = "x".join(map(str, config.video_frame_height_width))
                    output_filename = f"{formatted_prompt}_{hw_str}_seed{seed}_rank{global_rank}"

                    if config.get("save_video", True):
                        output_path = os.path.join(
                            output_dir,
                            f"{output_filename}.{config.get('output_format', 'mp4')}"
                        )
                        save_video(
                            output_path,
                            generated_video,
                            generated_audio if config.get("save_audio", True) else None,
                            fps=config.get("fps", 24),
                            sample_rate=config.get("sample_rate", 16000)
                        )
                        logging.info(f"Saved: {output_path}")

                    if generated_image is not None and config.get("save_image", True):
                        image_path = os.path.join(output_dir, f"{output_filename}.png")
                        generated_image.save(image_path)
                        logging.info(f"Saved image: {image_path}")

            except Exception as e:
                logging.error(f"Error processing prompt {prompt_idx}, sample {sample_idx}: {e}")
                if config.get("verbose", True):
                    import traceback
                    traceback.print_exc()
                continue

    # Cleanup
    if world_size > 1:
        torch.distributed.barrier()

    logging.info(f"Rank {global_rank} completed processing!")


if __name__ == "__main__":
    main()
