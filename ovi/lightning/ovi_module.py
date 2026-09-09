import os
import re
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import Any, Dict
from ovi.utils.model_loading_utils import (
    init_fusion_score_model_ovi,
    load_fusion_checkpoint,
    init_text_model,
    init_wan_vae_2_2,
    init_mmaudio_vae,
)
from ovi.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ovi.data.lora import GeneralLoRALoader

from deepspeed.ops.adam import DeepSpeedCPUAdam
from peft import LoraConfig, inject_adapter_in_model

# Model specifications mapping
NAME_TO_MODEL_SPECS_MAP = {
    "720x720_5s": {
        "path": "model.safetensors",
        "video_latent_length": 31,
        "audio_latent_length": 157,
        "video_area": 720 * 720,
        "formatter": lambda text: re.sub(r"Audio:\s*(.*)", r"<AUDCAP>\1<ENDAUDCAP>", text, flags=re.S)
    },
    "960x960_5s": {
        "path": "model_960x960.safetensors",
        "video_latent_length": 31,
        "audio_latent_length": 157,
        "video_area": 960 * 960,
        "formatter": lambda text: re.sub(r"<AUDCAP>(.*?)<ENDAUDCAP>", r"Audio: \1", text, flags=re.S)
    },
    "960x960_10s": {
        "path": "model_960x960_10s.safetensors",
        "video_latent_length": 61,
        "audio_latent_length": 314,
        "video_area": 960 * 960,
        "formatter": lambda text: re.sub(r"<AUDCAP>(.*?)<ENDAUDCAP>", r"Audio: \1", text, flags=re.S)
    }
}


class OviFusionTrainModule(pl.LightningModule):
    """
    Training module for OVI fusion model using a Flow-matching noise-prediction objective.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        # initialize fusion model
        self.ckpt_dir = cfg.get("ckpt_dir", "ckpts")
        self.gradient_checkpoint = cfg.get("gradient_checkpoint", False)

        # Get model name and specs
        self.model_name = cfg.get("model_name", "720x720_5s")
        if self.model_name not in NAME_TO_MODEL_SPECS_MAP:
            raise ValueError(
                f"Model name '{self.model_name}' not found in predefined model specs. "
                f"Available models: {list(NAME_TO_MODEL_SPECS_MAP.keys())}"
            )
        model_specs = NAME_TO_MODEL_SPECS_MAP[self.model_name]

        # Initialize model
        model, video_cfg, audio_cfg = init_fusion_score_model_ovi(
            rank=self.local_rank, meta_init=False, gradient_checkpoint=self.gradient_checkpoint
        )

        # Load checkpoint using model_name specs
        model_ckpt_path = os.path.join(self.ckpt_dir, "Ovi", model_specs["path"])
        if not os.path.exists(model_ckpt_path):
            raise RuntimeError(
                f"REQUIRED fusion checkpoint not found at {model_ckpt_path}. "
                f"Please ensure the checkpoint for model '{self.model_name}' is downloaded."
            )
        else:
            print(f"Loading fusion checkpoint from {model_ckpt_path}...")
        load_fusion_checkpoint(model, model_ckpt_path, from_meta=False)

        self.text_formatter = model_specs["formatter"]

        self.model = model
        self.model.requires_grad_(False)
        self.model.train()

        # Initialize encoding models for data preprocessing
        # All encoder models are already frozen and in eval mode:
        # - text_model (T5EncoderModel): wrapper class, internal model is eval+frozen
        # - video_vae_model (Wan2_2_VAE): wrapper class, internal model is eval+frozen
        # - audio_vae_model (FeaturesUtils): nn.Module with train() overridden to always eval

        # T5 text encoder optimization: enable CPU offload to save GPU memory
        self.text_encoder_cpu_offload = cfg.get("text_encoder_cpu_offload", True)
        # Initialize encoders as regular attributes (not nn.Module children)
        # This prevents them from being saved in checkpoints
        # Note: We manually assign them as non-module attributes to avoid checkpoint bloat
        text_model = init_text_model(self.ckpt_dir, rank=self.local_rank, cpu_offload=True)
        video_vae = init_wan_vae_2_2(self.ckpt_dir, rank=self.local_rank)
        audio_vae = init_mmaudio_vae(self.ckpt_dir, rank=self.local_rank)
        # Freeze audio_vae parameters (FeaturesUtils is nn.Module, needs explicit freezing)
        audio_vae.eval()
        audio_vae.requires_grad_(False)

        # Store as non-module attributes to exclude from state_dict
        object.__setattr__(self, 'text_model', text_model)
        object.__setattr__(self, 'video_vae_model', video_vae)
        object.__setattr__(self, 'audio_vae_model', audio_vae)

        if self.text_encoder_cpu_offload:
            print("T5 text encoder CPU offload enabled - model will be moved to GPU only during encoding")

        # Negative prompts for classifier-free guidance
        self.video_negative_prompt = cfg.get("video_negative_prompt", "")
        self.audio_negative_prompt = cfg.get("audio_negative_prompt", "")

        # diffusion schedule and FlowUniPC scheduler
        self.num_train_timesteps = int(cfg.get("num_train_timesteps", 1000))
        self.shift = float(cfg.get("shift", 5.0))
        self.video_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=self.num_train_timesteps)
        self.video_scheduler.set_timesteps(self.num_train_timesteps, shift=self.shift)
        self.audio_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=self.num_train_timesteps)
        self.audio_scheduler.set_timesteps(self.num_train_timesteps, shift=self.shift)

        # training hyperparams
        self.learning_rate = float(cfg.get("lr", 5e-5))
        self.train_cfg_prob = float(cfg.get("train_cfg_prob", 0.0))
        self.weight_decay = float(cfg.get("weight_decay", 0.0))
        self.video_loss_weight = float(cfg.get("video_loss_weight", 0.85))
        self.warmup_steps = int(cfg.get("warmup_steps", 0))
        self.deepspeed_offload = "offload" in cfg.get("training_strategy", "auto")
        self.train_i2v = bool(cfg.get("train_i2v", False))

        # Mixed training: support for audio-only data
        self.enable_mixed_training = bool(cfg.get("enable_mixed_training", False))
        self.audio_only_loss_weight = float(cfg.get("audio_only_loss_weight", 1.0))
        if self.enable_mixed_training:
            print(f"[Mixed Training] Enabled - audio_only_loss_weight={self.audio_only_loss_weight}")
            print(f"[Mixed Training] Multimodal loss weight: video={self.video_loss_weight}, audio={1-self.video_loss_weight}")

        # Fine-tuning options: which parts of the model to train
        # train_fusion: train the fusion layers (k_fusion, v_fusion, pre_attn_norm_fusion, norm_k_fusion)
        # train_audio: train the audio_model parameters (excluding fusion layers)
        # train_video: train the video_model parameters (excluding fusion layers)
        self.train_fusion = bool(cfg.get("train_fusion", True))
        self.train_audio = bool(cfg.get("train_audio", False))
        self.train_video = bool(cfg.get("train_video", False))

        # LoRA options
        self.train_lora = bool(cfg.get("train_lora", True))
        self.lora_rank = int(cfg.get("lora_rank", 32))
        self.lora_alpha = float(cfg.get("lora_alpha", 32))
        self.lora_target_modules = cfg.get("lora_target_modules", "q,k,v,o")
        self.lora_path = cfg.get("lora_path", None)
        self.pretrain_lora_path = cfg.get("pretrain_lora_path", None)

        # LoRA injection/loading logic (order-sensitive):
        # 1) If lora_path provided -> apply/merge LoRA into base model AND load fusion layer weights
        # 2) If train_lora True -> inject adapters; if pretrain_lora_path provided -> load into adapters (continue training)
        if self.lora_path is not None:
            try:
                # Load checkpoint (contains both LoRA params and fusion layer params)
                lora_state = self.load_lora_state_dict(self.lora_path)

                # Apply LoRA weights (lora_A, lora_B) to base model
                lora_loader = GeneralLoRALoader(device="cpu", torch_dtype=torch.float32)
                lora_loader.load(self.model, lora_state, alpha=1.0)

                # Load fusion layer weights directly (k_fusion, v_fusion, etc.)
                fusion_keys = {k: v for k, v in lora_state.items() if "fusion" in k}
                if fusion_keys:
                    missing, unexpected = self.model.load_state_dict(fusion_keys, strict=False)
                    loaded_fusion = len(fusion_keys) - len(missing)
                    print(f"Loaded {loaded_fusion} fusion layer parameters from {self.lora_path}")
            except Exception as e:
                print(f"Warning: failed to apply LoRA checkpoint {self.lora_path}: {e}")

        if self.train_lora:
            # inject adapters for training (either fresh or to continue training)
            self.add_lora_to_model(
                self.model,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_target_modules=self.lora_target_modules,
                pretrained_lora_path=self.pretrain_lora_path,
                train_fusion=self.train_fusion,
                train_audio=self.train_audio,
                train_video=self.train_video,
            )
        else:
            # Full fine-tuning mode: selectively enable training based on config
            self._setup_trainable_parameters()

        # loss
        self.criterion = nn.MSELoss()

        # EMA config
        self.ema_decay = float(cfg.get("ema_decay", 0.999))
        self.use_ema = bool(cfg.get("use_ema", True))
        self.auto_ema = bool(cfg.get("auto_ema", False))
        self.ema_state = None

    def _setup_trainable_parameters(self):
        """Setup which parameters are trainable based on train_fusion, train_audio, train_video config.

        Fusion layers are identified by 'fusion' in their name (k_fusion, v_fusion, pre_attn_norm_fusion, norm_k_fusion).
        Audio model layers are identified by 'audio_model' prefix.
        Video model layers are identified by 'video_model' prefix.
        """
        # First, freeze all parameters
        self.model.requires_grad_(False)

        trainable_params = 0
        fusion_params = 0
        audio_params = 0
        video_params = 0
        frozen_params = 0

        for name, param in self.model.named_parameters():
            should_train = False

            is_fusion_layer = "fusion" in name

            if is_fusion_layer:
                if self.train_fusion:
                    should_train = True
                    fusion_params += param.numel()
            elif name.startswith("audio_model."):
                if self.train_audio:
                    should_train = True
                    audio_params += param.numel()
            elif name.startswith("video_model."):
                if self.train_video:
                    should_train = True
                    video_params += param.numel()

            if should_train:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()

        total_params = trainable_params + frozen_params
        print(f"[Training Setup] Trainable parameters breakdown:")
        print(f"  - Fusion layers:    {fusion_params:>12,} params ({fusion_params/1e6:.2f}M) (train_fusion={self.train_fusion})")
        print(f"  - Audio model:      {audio_params:>12,} params ({audio_params/1e6:.2f}M) (train_audio={self.train_audio})")
        print(f"  - Video model:      {video_params:>12,} params ({video_params/1e6:.2f}M) (train_video={self.train_video})")
        print(f"  - Frozen:           {frozen_params:>12,} params ({frozen_params/1e6:.2f}M)")
        print(f"  - Total trainable:  {trainable_params:>12,} params ({trainable_params/1e6:.2f}M)")
        print(f"  - Total model:      {total_params:>12,} params ({total_params/1e6:.2f}M)")
        print(f"  - Trainable ratio:  {trainable_params/total_params*100:.2f}%")

    def setup(self, stage: str):
        """Called at the beginning of fit and test. Move VAE models to correct device."""
        # Move VAE models to the same device as the main model
        # This is important for multi-GPU training where each rank has a different device
        if hasattr(self, 'video_vae_model') and hasattr(self.video_vae_model, 'model'):
            self.video_vae_model.model = self.video_vae_model.model.to(self.device)
            # Move scale parameters (mean and std tensors) to the correct device
            if hasattr(self.video_vae_model, 'scale') and isinstance(self.video_vae_model.scale, list):
                self.video_vae_model.scale = [s.to(self.device) if isinstance(s, torch.Tensor) else s
                                               for s in self.video_vae_model.scale]
        if hasattr(self, 'audio_vae_model'):
            self.audio_vae_model = self.audio_vae_model.to(self.device)

        if hasattr(self, 'text_model') and not self.text_encoder_cpu_offload:
            print(f"[Rank {self.global_rank}] Moving T5 text encoder to {self.device}")
            self.text_model.model = self.text_model.model.to(self.device)
            self.text_model.device = self.device

    def on_train_epoch_start(self):
        """Called at the start of each training epoch. Auto-enable EMA if conditions are met."""
        if self.use_ema and self.ema_state is None:
            # Initialize EMA state dict with only trainable parameters
            # This saves ~50GB GPU memory compared to copying the entire model
            self.ema_state = {}
            current_device = next(self.model.parameters()).device
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.ema_state[name] = param.data.clone().detach().to(current_device)
            print(f"[EMA] Initialized EMA for {len(self.ema_state)} trainable parameters")

        if self.auto_ema and not self.use_ema and self.ema_state is None:
            # Calculate 50% threshold of max_epochs
            max_epochs = self.trainer.max_epochs
            ema_start_epoch = int(max_epochs * 0.5)

            if self.current_epoch >= ema_start_epoch:
                print(f"\n{'='*60}")
                print(f"[Auto EMA] Epoch {self.current_epoch}/{max_epochs} (>= 50% threshold)")
                print("[Auto EMA] Enabling EMA for model stabilization and better final quality")
                print(f"{'='*60}\n")

                self.use_ema = True
                self.ema_state = {}
                current_device = next(self.model.parameters()).device
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        self.ema_state[name] = param.data.clone().detach().to(current_device)
                print(f"[Auto EMA] Initialized EMA for {len(self.ema_state)} trainable parameters")

    def get_noisy_latents_and_target(self, scheduler, latents, noise, t):
        """Sample noisy latents and target flow for a given timestep."""
        noise_latents = scheduler.add_noise(latents, noise, t)
        target_flow = noise - latents
        return noise_latents, target_flow

    def training_step(self, batch, batch_idx):
        # Encode raw data from batch
        # Batch now contains raw 'video', 'audio', 'text' instead of pre-encoded latents
        # For mixed training, video can be optional (audio-only samples)
        # Note: modality_type is collated into a list/tuple by default_collate, take first element
        modality_type_raw = batch.get("modality_type", ["multimodal"])
        modality_type = modality_type_raw[0] if isinstance(modality_type_raw, (list, tuple)) else modality_type_raw
        is_audio_only = (modality_type == "audio_only") if self.enable_mixed_training else False

        required_raw = ["audio", "text"]
        if not is_audio_only:
            required_raw.append("video")

        missing = [k for k in required_raw if k not in batch]
        if missing:
            error_msg = (
                f"Batch missing required keys: {missing}. "
                f"Modality type: {modality_type}, "
                f"Available keys: {list(batch.keys())}. "
                f"This should not happen as data loader validates samples. "
                f"Please check your data preprocessing pipeline."
            )
            raise RuntimeError(error_msg)

        # Get model dtype once for reuse (typically BFloat16 for training)
        model_dtype = next(self.model.parameters()).dtype
        current_device = next(self.model.parameters()).device

        # Encode text embeddings
        with torch.no_grad():
            text_batch = batch["text"]  # List of text strings
            # Create text prompts: positive (from batch), negative for video and audio
            # Apply text formatter based on model_name
            text_prompts = []
            for text in text_batch:
                formatted_text = self.text_formatter(text)
                text_prompts.append(formatted_text)

            # Add negative prompts for the whole batch
            video_neg_prompts = [self.video_negative_prompt] * len(text_batch)
            audio_neg_prompts = [self.audio_negative_prompt] * len(text_batch)

            if self.text_encoder_cpu_offload:
                self.text_model.model = self.text_model.model.to(current_device)

            all_prompts = text_prompts + video_neg_prompts + audio_neg_prompts
            all_embeddings = self.text_model(all_prompts, current_device)

            # Move model back to CPU to free GPU memory
            if self.text_encoder_cpu_offload:
                self.text_model.model = self.text_model.model.cpu()
                torch.cuda.empty_cache()

            # Split embeddings and convert to model dtype
            batch_size = len(text_batch)
            text_embeddings_video_pos = [emb.to(dtype=model_dtype) for emb in all_embeddings[:batch_size]]
            text_embeddings_audio_pos = [emb.to(dtype=model_dtype) for emb in all_embeddings[:batch_size]]
            text_embeddings_video_neg = [emb.to(dtype=model_dtype) for emb in all_embeddings[batch_size:2*batch_size]]
            text_embeddings_audio_neg = [emb.to(dtype=model_dtype) for emb in all_embeddings[2*batch_size:]]

        # Encode video latents - batch encode for better performance
        # For audio-only data, skip video encoding entirely
        if not is_audio_only:
            with torch.no_grad():
                video_data = batch["video"].to(current_device)  # B C T H W

                # Batch encode all videos at once
                video_latents = self.video_vae_model.wrapped_encode(video_data)  # B C T H W

                # Batch encode first frames as images
                image_data = video_data[:, :, 0:1, :, :]  # B C 1 H W
                image_latents = self.video_vae_model.wrapped_encode(image_data)  # B C 1 H W

                # Convert to model dtype
                video_latents = video_latents.to(dtype=model_dtype)
                image_latents = image_latents.to(dtype=model_dtype)

        

        # Encode audio latents - batch encode for better performance
        with torch.no_grad():
            # Audio must be float32 for STFT operation (cuFFT doesn't support BFloat16)
            audio_data = batch["audio"].to(current_device, dtype=torch.float32)  # B C L

            # Batch encode all audio at once
            audio_latents = self.audio_vae_model.wrapped_encode(audio_data)  # B C L

            # Transpose to L C format expected by model: B C L -> B L C
            audio_latents = audio_latents.transpose(1, 2)  # B L C

            # Convert to model dtype
            audio_latents = audio_latents.to(dtype=model_dtype)


        # Now proceed with the original training logic
        audio_noise = torch.randn_like(audio_latents)
        timestep_id = torch.randint(0, self.num_train_timesteps, (1,))
        audio_timestep = self.audio_scheduler.timesteps[timestep_id].to(dtype=audio_latents.dtype, device=current_device)
        audio_noise_latents, audio_target_flow = self.get_noisy_latents_and_target(
            self.audio_scheduler, audio_latents, audio_noise, audio_timestep
        )
        max_seq_len_audio = audio_noise.shape[1]  # L dimension from latents_audios shape [B, L, D]

        # Process video latents only for multimodal data
        if not is_audio_only:
            video_noise = torch.randn_like(video_latents)
            video_timestep = self.video_scheduler.timesteps[timestep_id].to(dtype=video_latents.dtype, device=current_device)
            video_noise_latents, video_target_flow = self.get_noisy_latents_and_target(
                self.video_scheduler, video_latents, video_noise, video_timestep
            )
            _patch_size_h, _patch_size_w = self.model.video_model.patch_size[1], self.model.video_model.patch_size[2]
            max_seq_len_video = (
                video_noise.shape[-1] * video_noise.shape[-2] * video_noise.shape[-3] // (_patch_size_h * _patch_size_w)
            )
        else:
            max_seq_len_video = None

        if torch.rand(1).item() < self.train_cfg_prob:
            text_emb_video = text_embeddings_video_neg
            text_emb_audio = text_embeddings_audio_neg
        else:
            text_emb_video = text_embeddings_video_pos
            text_emb_audio = text_embeddings_audio_pos

        if self.train_i2v and not is_audio_only:
            # image_latents: [B, C, 1, H, W] -> take frame 0 -> [B, C, H, W]
            video_noise_latents[:, :, 0] = image_latents[:, :, 0]

        # For audio-only data, pass None for video inputs to trigger audio-only mode
        if is_audio_only:
            video_for_model = None
            text_emb_video_for_model = None
        else:
            video_for_model = [vid for vid in video_noise_latents]
            text_emb_video_for_model = [tev for tev in text_emb_video]

        audio_for_model = [aud for aud in audio_noise_latents]
        text_emb_audio_for_model = [tea for tea in text_emb_audio]

        # call fusion model -- wrap tensors in lists like inference path
        # Model's forward already supports audio-only mode (returns None, audio_output)
        pred_vid, pred_audio = self.model(
            vid=video_for_model,
            audio=audio_for_model,
            t=audio_timestep if is_audio_only else video_timestep,
            vid_context=text_emb_video_for_model,
            audio_context=text_emb_audio_for_model,
            vid_seq_len=max_seq_len_video,
            audio_seq_len=max_seq_len_audio,
            first_frame_is_clean=self.train_i2v and not is_audio_only,
        )

        # pred_vid, pred_audio: lists or None, transform to tensors
        # For audio-only mode, pred_vid will be None
        batch_size = audio_latents.shape[0]

        if is_audio_only:
            # Audio-only training: only compute audio loss
            pred_audio = torch.stack([p for p in pred_audio], dim=0)
            audio_loss = self.criterion(pred_audio.float(), audio_target_flow.float())

            # Apply audio-only loss weight (can be > 1.0 to compensate for smaller magnitude)
            loss = audio_loss * self.audio_only_loss_weight
            video_loss = torch.zeros_like(audio_loss)  # For logging only

            # Log with modality tag
            self.log("train/audio_only_loss", audio_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)
            self.log("train/video_loss", video_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
            self.log("train/audio_loss", audio_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
            # Log modality type for monitoring (1 = audio_only)
            self.log("train/is_audio_only", 1.0, on_step=True, on_epoch=False, prog_bar=False, batch_size=batch_size, sync_dist=True)
        else:
            # Multimodal training: compute both losses
            pred_vid = torch.stack([p for p in pred_vid], dim=0)
            pred_audio = torch.stack([p for p in pred_audio], dim=0)

            # compute FlowMatch loss vs target_flow
            if self.train_i2v:
                video_target_flow[:, :, 0] = 0.0
                pred_vid[:, :, 0] = 0.0
            video_loss = self.criterion(pred_vid.float(), video_target_flow.float())
            audio_loss = self.criterion(pred_audio.float(), audio_target_flow.float())
            loss = video_loss * self.video_loss_weight + audio_loss * (1 - self.video_loss_weight)

            # Log multimodal losses
            self.log("train/multimodal_loss", loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)
            self.log("train/video_loss", video_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
            self.log("train/audio_loss", audio_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
            # Log modality type for monitoring (0 = multimodal)
            self.log("train/is_audio_only", 0.0, on_step=True, on_epoch=False, prog_bar=False, batch_size=batch_size, sync_dist=True)

        # EMA update
        if self.use_ema and self.ema_state is not None:
            self.update_ema()
        sys.stdout.flush()
        return loss

    def update_ema(self):
        """Update EMA weights for trainable parameters only (memory efficient)."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.ema_state:
                    # EMA update: ema = decay * ema + (1 - decay) * current
                    self.ema_state[name].mul_(self.ema_decay).add_(param.data, alpha=1.0 - self.ema_decay)

    def configure_optimizers(self):
        # choose optimizer; if deepspeed cpu offload requested, use DeepSpeedCPUAdam if available
        params = filter(lambda p: p.requires_grad, self.model.parameters())

        # Scale learning rate with square root of world size (more conservative than linear scaling)
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        scaled_lr = self.learning_rate * (world_size ** 0.5)

        if world_size > 1:
            print(f"[Rank {self.global_rank}] Scaling learning rate (sqrt): {self.learning_rate} -> {scaled_lr} (world_size={world_size})")

        if self.deepspeed_offload:
            optimizer = DeepSpeedCPUAdam(params, lr=scaled_lr)
        else:
            optimizer = torch.optim.AdamW(params, lr=scaled_lr, weight_decay=self.weight_decay)

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_steps - max(self.warmup_steps, 0), eta_min=1e-6
        )

        if self.warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: min(step / self.warmup_steps, 1.0)
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[self.warmup_steps]
            )
        else:
            scheduler = cosine_scheduler
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def load_lora_state_dict(self, path: str):
        """Load LoRA state dict into model (non-strict)."""
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        # load checkpoint dict
        lora_state = None
        try:
            if path.endswith(".safetensors"):
                from safetensors.torch import load_file as load_safetensors

                lora_state = load_safetensors(path, device="cpu")
            else:
                loaded = torch.load(path, map_location="cpu")
                # attempt to unwrap common wrappers
                if isinstance(loaded, dict) and ("state_dict" in loaded or "model" in loaded or "module" in loaded):
                    # prefer direct state_dict keys if present
                    if "state_dict" in loaded:
                        lora_state = loaded["state_dict"]
                    elif "model" in loaded:
                        lora_state = loaded["model"]
                    elif "module" in loaded:
                        lora_state = loaded["module"]
                    else:
                        lora_state = loaded
                else:
                    lora_state = loaded
        except Exception as e:
            raise RuntimeError(f"Failed to load lora checkpoint {path}: {e}")

        if not isinstance(lora_state, dict):
            raise RuntimeError("Loaded LoRA checkpoint is not a dict-like state")

        prefixes = ["model.", "diffusion_model.", "module."]
        for prefix in prefixes:
            lora_state = {k.replace(prefix, ""): v for k, v in lora_state.items()}

        return lora_state

    def add_lora_to_model(
        self,
        model,
        lora_rank=16,
        lora_alpha=16,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        init_lora_weights="kaiming",
        pretrained_lora_path=None,
        lora_dtype="fp32",
        train_fusion=True,
        train_audio=False,
        train_video=False,
    ):
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=(init_lora_weights == "kaiming"),
            target_modules=lora_target_modules.split(","),
        )
        # inject LoRA into entire fusion model
        model = inject_adapter_in_model(lora_config, model)

        # Selectively freeze/unfreeze parameters based on train_fusion/train_audio/train_video
        # - Fusion layers (k_fusion, v_fusion, etc.) are NEW layers, not LoRA-adapted
        # - Audio/Video model layers get LoRA adapters injected
        lora_audio_params = 0
        lora_video_params = 0
        fusion_params = 0
        frozen_params = 0

        target_dtype = torch.float32 if lora_dtype == "fp32" else torch.bfloat16

        for name, param in model.named_parameters():
            is_fusion_layer = "fusion" in name
            is_lora_param = "lora_" in name

            if not param.requires_grad:
                # Already frozen (base model params)
                # Check if this is a fusion layer that should be trained
                if is_fusion_layer and train_fusion:
                    param.requires_grad = True
                    param.data = param.data.to(target_dtype)
                    fusion_params += param.numel()
                else:
                    frozen_params += param.numel()
                continue

            # This parameter has requires_grad=True (LoRA param after injection)
            if not is_lora_param:
                # Non-LoRA trainable param - check if it's a fusion layer
                if is_fusion_layer and train_fusion:
                    fusion_params += param.numel()
                    param.data = param.data.to(target_dtype)
                else:
                    param.requires_grad = False
                    frozen_params += param.numel()
                continue

            # This is a LoRA parameter
            should_train = False

            if name.startswith("audio_model."):
                if train_audio:
                    should_train = True
                    lora_audio_params += param.numel()
            elif name.startswith("video_model."):
                if train_video:
                    should_train = True
                    lora_video_params += param.numel()

            if not should_train:
                param.requires_grad = False
                frozen_params += param.numel()
            else:
                param.data = param.data.to(target_dtype)

        total_trainable = fusion_params + lora_audio_params + lora_video_params
        total_params = total_trainable + frozen_params
        print(f"[LoRA Setup] Trainable parameters breakdown:")
        print(f"  - Fusion (FULL Params):  {fusion_params:>12,} params ({fusion_params/1e6:.2f}M) (train_fusion={train_fusion})")
        print(f"  - Audio (LoRA Params):   {lora_audio_params:>12,} params ({lora_audio_params/1e6:.2f}M) (train_audio={train_audio})")
        print(f"  - Video (LoRA Params):   {lora_video_params:>12,} params ({lora_video_params/1e6:.2f}M) (train_video={train_video})")
        print(f"  - Frozen:                {frozen_params:>12,} params ({frozen_params/1e6:.2f}M)")
        print(f"  - Total trainable:       {total_trainable:>12,} params ({total_trainable/1e6:.2f}M)")
        print(f"  - Total model:           {total_params:>12,} params ({total_params/1e6:.2f}M)")
        print(f"  - Trainable ratio:       {total_trainable/total_params*100:.2f}%")

        if pretrained_lora_path is not None:
            try:
                lora_state_pretrain = self.load_lora_state_dict(pretrained_lora_path)
                missing_keys, unexpected_keys = model.load_state_dict(lora_state_pretrain, strict=False)
                all_keys = [i for i, _ in model.named_parameters()]
                num_updated_keys = len(all_keys) - len(missing_keys)
                num_unexpected_keys = len(unexpected_keys)
                print(
                    f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected."
                )
            except Exception as e:
                print(f"Warning: failed to load pretrain LoRA checkpoint {pretrained_lora_path}: {e}")

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(
            filter(lambda named_param: named_param[1].requires_grad, self.model.named_parameters())
        )
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.model.state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)

        # Save EMA weights if enabled (only trainable parameters)
        if self.use_ema and self.ema_state is not None:
            checkpoint["ema_state_dict"] = {k: v.cpu() for k, v in self.ema_state.items()}
            print(f"[Checkpoint] Saved {len(self.ema_state)} EMA LoRA parameters")
