import torch
import os
import json
from safetensors.torch import load_file

from ovi.modules.fusion import FusionModel
from ovi.modules.t5 import T5EncoderModel
from ovi.modules.vae2_2 import Wan2_2_VAE
from ovi.modules.mmaudio.features_utils import FeaturesUtils

def init_wan_vae_2_2(ckpt_dir, rank=0):
    vae_config = {}
    vae_config['device'] = rank
    vae_pth = os.path.join(ckpt_dir, "Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    vae_config['vae_pth'] = vae_pth
    vae_model = Wan2_2_VAE(**vae_config)

    return vae_model

def init_mmaudio_vae(ckpt_dir, rank=0):
    vae_config = {}
    vae_config['mode'] = '16k'
    vae_config['need_vae_encoder'] = True

    tod_vae_ckpt = os.path.join(ckpt_dir, "MMAudio/ext_weights/v1-16.pth")
    bigvgan_vocoder_ckpt = os.path.join(ckpt_dir, "MMAudio/ext_weights/best_netG.pt")

    vae_config['tod_vae_ckpt'] = tod_vae_ckpt
    vae_config['bigvgan_vocoder_ckpt'] = bigvgan_vocoder_ckpt

    vae = FeaturesUtils(**vae_config).to(rank)

    return vae

def init_fusion_score_model_ovi(rank: int = 0, meta_init=False, gradient_checkpoint=False):
    video_config = "ovi/configs/model/dit/video.json"
    audio_config = "ovi/configs/model/dit/audio.json"
    assert os.path.exists(video_config), f"{video_config} does not exist"
    assert os.path.exists(audio_config), f"{audio_config} does not exist"

    with open(video_config) as f:
        video_config = json.load(f)

    with open(audio_config) as f:
        audio_config = json.load(f)

    if meta_init:
        with torch.device("meta"):
            fusion_model = FusionModel(video_config, audio_config, gradient_checkpoint)
    else:
        fusion_model = FusionModel(video_config, audio_config, gradient_checkpoint)

    params_all = sum(p.numel() for p in fusion_model.parameters())

    if rank == 0:
        print(
            f"Score model (Fusion) all parameters:{params_all}"
        )

    return fusion_model, video_config, audio_config

def init_text_model(ckpt_dir, rank, cpu_offload=False):
    wan_dir = os.path.join(ckpt_dir, "Wan2.2-TI2V-5B")
    text_encoder_path = os.path.join(wan_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    text_tokenizer_path = os.path.join(wan_dir, "google/umt5-xxl")

    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=rank,
        checkpoint_path=text_encoder_path,
        tokenizer_path=text_tokenizer_path,
        cpu_offload=cpu_offload,
        shard_fn=None)


    return text_encoder

import gc
from safetensors.torch import load_file 

# def load_fusion_checkpoint(model, checkpoint_path, from_meta=False, is_base=True):
#     if checkpoint_path and os.path.exists(checkpoint_path):
#         if checkpoint_path.endswith(".safetensors"):
#             df = load_file(checkpoint_path, device="cpu")
#         elif checkpoint_path.endswith(".pt"):
#             try:
#                 df = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
#                 df = df['module'] if 'module' in df else df
#             except Exception as e:
#                 df = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
#                 df = df['app']['model']
#         else:
#             raise RuntimeError("We only support .safetensors and .pt checkpoints")

#         missing, unexpected = model.load_state_dict(df, strict=False, assign=from_meta)
#         print(f"Missing keys: {missing}")
#         print(f"Unexpected keys: {unexpected}")
#         del df
#         import gc
#         gc.collect()
#         print(f"Successfully loaded fusion checkpoint from {checkpoint_path}")
#     else:
#         raise RuntimeError("{checkpoint=} does not exists'")


def load_fusion_checkpoint(model, checkpoint_path, from_meta=False, is_base=True):
    """
    智能加载 Checkpoint。
    支持双向映射：
    1. Standard Checkpoint -> LoRA Model (weight -> base_layer.weight)
    2. Finetuned/LoRA Checkpoint -> Standard Model (base_layer.weight -> weight) [新增]
    """
    if not (checkpoint_path and os.path.exists(checkpoint_path)):
        raise RuntimeError(f"Checkpoint path does not exist: {checkpoint_path}")

    print(f"Loading {'BASE ' if is_base else 'FULL/LORA '}checkpoint from {checkpoint_path}...")

    # 1. 加载 State Dict
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path, device="cpu")
    else:
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # 2. 解包 State Dict
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "module" in state_dict:
        state_dict = state_dict["module"]
    elif "app" in state_dict and "model" in state_dict["app"]:
        state_dict = state_dict["app"]["model"]

    new_state_dict = {}
    model_keys = set(model.state_dict().keys())
    
    # 需要剥离的前缀
    prefixes_to_strip = ["base_model.model.", "model.", "module."]

    for k, v in state_dict.items():
        clean_key = k
        # A. 剥离前缀
        for prefix in prefixes_to_strip:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
                break 
        
        # B. 基础权重过滤 (仅在明确只加载 base 时生效)
        if is_base:
            # 这里需要注意：如果你想把微调过的 base_layer 加载回 base 模型，
            # 这里的过滤条件不能太激进把 .base_layer.weight 给过滤掉了
            # 如果你的微调权重里包含 lora_A/lora_B，那些是要过滤的，但 base_layer 不是
            if "lora_A" in clean_key or "lora_B" in clean_key:
                continue

        # C. 智能双向映射
        if clean_key not in model_keys:
            # 尝试 1: Standard -> LoRA (.weight -> .base_layer.weight)
            potential_base_key = clean_key.replace(".weight", ".base_layer.weight").replace(".bias", ".base_layer.bias")
            
            # 尝试 2: LoRA -> Standard (.base_layer.weight -> .weight) [新增逻辑]
            potential_std_key = clean_key.replace(".base_layer.weight", ".weight").replace(".base_layer.bias", ".bias")

            if potential_base_key in model_keys:
                # 命中：说明模型是 LoRA 结构，CKPT 是普通结构
                clean_key = potential_base_key
            elif potential_std_key in model_keys:
                # 命中：说明模型是普通结构，CKPT 是 LoRA 结构
                clean_key = potential_std_key
            
            # 否则保持原样，让它报 Missing 方便排查

        new_state_dict[clean_key] = v

    # 3. 处理 Meta Tensor (assign=True)
    # 只要涉及到初始化加载 (is_base) 或者源模型是 meta (from_meta)，就开启 assign
    use_assign = from_meta or is_base 
    
    print(f"Start loading state dict (strict=False, assign={use_assign})...")
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False, assign=use_assign)

    # 4. 日志打印 (过滤掉不重要的 Warning)

    if missing:
        print(f"⚠️ Warning: Missing keys: {missing[:3]}... (Total {len(missing)})")

    if unexpected:
        real_unexpected = [k for k in unexpected if not any(x in k for x in ["vae", "text_encoder", "tod"])]
        if real_unexpected:
            print(f"⚠️ Warning: Unexpected keys: {real_unexpected[:3]}... (Total {len(real_unexpected)})")

    # 5. 内存清理
    del state_dict
    del new_state_dict
    gc.collect()
    
    # 6. 安全检查：确认是否真的加载进去了 (检查 Meta)
    try:
        first_param = next(model.parameters())
        if first_param.device.type == 'meta':
            print("❌ FATAL ERROR: Model parameters are still on META device! Load failed.")
        else:
            print(f"✅ Successfully loaded {'BASE' if is_base else 'LORA/FULL'} checkpoint. Model device: {first_param.device}")
    except StopIteration:
        pass