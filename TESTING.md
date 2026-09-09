# OVI Fusion Model Testing Guide

This guide explains how to use the testing scripts for the OVI Fusion model with LoRA support.

## Files Overview

- `test.py` - Main testing script with LoRA support
- `test.sh` - Bash launcher script for easy testing
- `ovi/configs/test/test.yaml` - Default test configuration
- `ovi/configs/test/test_i2v.yaml` - Image-to-video test config
- `ovi/configs/test/test_with_lora.yaml` - LoRA testing config example

## Quick Start

### 1. Basic Text-to-Video Test

```bash
# Single prompt test
bash test.sh
```

### 2. Test with LoRA Checkpoint

```bash
# Set LoRA path and run
LORA_PATH=./outputs/checkpoints/lora_epoch_10.ckpt bash test.sh
```

### 3. Batch Testing from CSV

```bash
# Process multiple prompts from CSV file
PROMPTS_FILE=./example_prompts/test_t2v_simple.csv \
OUTPUT_DIR=./batch_results \
bash test.sh
```

### 4. Image-to-Video Test

```bash
# Use i2v config
MODE=i2v \
CONFIG_FILE=ovi/configs/test/test_i2v.yaml \
bash test.sh
```

### 5. Multi-GPU Testing

```bash
# Use 4 GPUs with sequence parallelism
GPU_NUM=4 \
LORA_PATH=./outputs/checkpoints/lora.safetensors \
bash test.sh
```

## Environment Variables

### Required
- `CONFIG_FILE` - Path to test YAML config (default: `ovi/configs/test/test.yaml`)

### Optional
- `LORA_PATH` - Path to LoRA checkpoint file (.ckpt or .safetensors)
- `PROMPTS_FILE` - CSV file with prompts (overrides single prompt)
- `OUTPUT_DIR` - Where to save generated videos (default: `./test_outputs`)
- `MODE` - Generation mode: `t2v`, `i2v`, or `t2i2v` (default: `t2v`)
- `SEED` - Random seed for reproducibility (default: `100`)

### Distributed Settings
- `GPU_NUM` - Number of GPUs per node (default: `1`)
- `NUM_NODES` - Number of nodes for multi-node testing (default: `1`)
- `MASTER_ADDR` - Master node address (default: `127.0.0.1`)
- `MASTER_PORT` - Master port (default: `9998`)
- `CUDA_VISIBLE_DEVICES` - GPU devices to use (e.g., `0,1,2,3`)

## Configuration File Format

### Basic Structure

```yaml
# Model paths
ckpt_dir: ./ckpts
lora_path: ./path/to/lora.ckpt  # Optional

# Test mode
mode: t2v  # t2v, i2v, or t2i2v

# Input
text_prompt: "Your prompt here"
image_path: null  # For i2v/t2i2v mode
prompts_file: null  # Or use CSV batch file

# Video parameters
num_frames: 81
video_frame_height_width: [720, 720]
fps: 24

# Generation settings
seed: 100
sample_steps: 50
video_guidance_scale: 4.0
audio_guidance_scale: 3.0

# Output
output_dir: ./test_outputs
```

## CSV Format for Batch Testing

### Text-to-Video (t2v)
```csv
text_prompt,image_path
"A serene beach at sunset",
"A bustling city at night",
```

### Image-to-Video (i2v)
```csv
text_prompt,image_path
"Add motion to this scene",./images/scene1.jpg
"Bring this to life",./images/scene2.jpg
```

## Advanced Usage Examples

### 1. Testing LoRA with Custom Settings

```bash
GPU_NUM=2 \
LORA_PATH=./outputs/my_lora.safetensors \
CONFIG_FILE=./my_test_config.yaml \
OUTPUT_DIR=./my_results \
SEED=42 \
bash test.sh
```

### 2. High-Resolution Testing

Create a custom config:
```yaml
video_frame_height_width: [1080, 1920]
sample_steps: 100
video_guidance_scale: 5.0
```

### 3. Multi-Sample Generation

Set in config:
```yaml
each_example_n_times: 5  # Generate 5 samples per prompt
seed: 100  # Seeds will be 100, 101, 102, 103, 104
```

### 4. Sequence Parallelism

For very high resolution or long videos:
```yaml
sp_size: 4  # Use 4 GPUs for sequence parallelism
```

Then run with 4 GPUs:
```bash
GPU_NUM=4 bash test.sh
```

## Output Structure

Generated files will be saved as:
```
{output_dir}/
  ├── {formatted_prompt}_{width}x{height}_seed{seed}_rank{rank}.mp4
  ├── {formatted_prompt}_{width}x{height}_seed{seed}_rank{rank}.png  # If save_image=true
  └── ...
```

## Troubleshooting

### LoRA Not Loading
- Check that `lora_path` points to a valid checkpoint
- Ensure the checkpoint contains LoRA weights (not full model)
- Verify LoRA rank/alpha match your training config

### Out of Memory
- Reduce `video_frame_height_width`
- Reduce `num_frames`
- Set `sp_size > 1` for sequence parallelism
- Use fewer `sample_steps`

### Distributed Issues
- Ensure `MASTER_ADDR` and `MASTER_PORT` are correct
- Check that `GPU_NUM` matches available GPUs
- Verify `sp_size` divides `GPU_NUM` evenly

### Image Not Found (i2v mode)
- Check `image_path` in config or CSV
- Use absolute paths or paths relative to script directory
- Verify image file exists and is readable

## Performance Tips

1. **Use bf16 precision** for faster inference
2. **Enable sequence parallelism** for high-resolution videos
3. **Batch multiple prompts** in CSV for efficiency
4. **Reduce sample_steps** for faster (but lower quality) results
5. **Use persistent workers** by setting `num_workers > 0`

## Examples

See `example_prompts/` directory for:
- `test_t2v_simple.csv` - Basic text-to-video prompts
- `test_i2v_simple.csv` - Image-to-video prompts
- `gpt_examples_t2v.csv` - Complex text-to-video scenarios
- `gpt_examples_i2v.csv` - Complex image-to-video scenarios
