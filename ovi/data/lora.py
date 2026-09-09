import torch
from tqdm import tqdm


class GeneralLoRALoader:
    def __init__(self, device="cpu", torch_dtype=torch.float32):
        self.device = device
        self.torch_dtype = torch_dtype

    def get_name_dict(self, lora_state_dict):
        lora_name_dict = {}
        for key in lora_state_dict:
            if ".lora_B." not in key:
                continue
            keys = key.split(".")
            if len(keys) > keys.index("lora_B") + 2:
                keys.pop(keys.index("lora_B") + 1)
            keys.pop(keys.index("lora_B"))
            if keys[0] == "diffusion_model":
                keys.pop(0)
            keys.pop(-1)
            target_name = ".".join(keys)
            lora_name_dict[target_name] = (key, key.replace(".lora_B.", ".lora_A."))
        return lora_name_dict

    def load(self, model: torch.nn.Module, state_dict_lora, alpha=1.0):
        """
        Load LoRA weights into model (optimized version).

        Args:
            model: Target model to load LoRA into
            state_dict_lora: LoRA state dict
            alpha: LoRA scaling factor
        """
        updated_num = 0
        lora_name_dict = self.get_name_dict(state_dict_lora)

        # Pre-compute all LoRA weights to avoid redundant operations
        print(f"Computing LoRA weights for {len(lora_name_dict)} modules...")
        lora_weights = {}

        for name, (key_up, key_down) in tqdm(lora_name_dict.items(), desc="Computing LoRA"):
            weight_up = state_dict_lora[key_up].to(device=self.device, dtype=self.torch_dtype)
            weight_down = state_dict_lora[key_down].to(device=self.device, dtype=self.torch_dtype)

            # Handle conv layers (4D tensors)
            if len(weight_up.shape) == 4:
                weight_up = weight_up.squeeze(3).squeeze(2)
                weight_down = weight_down.squeeze(3).squeeze(2)
                weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
            else:
                weight_lora = alpha * torch.mm(weight_up, weight_down)

            lora_weights[name] = weight_lora

        # Apply pre-computed LoRA weights
        print("Applying LoRA weights to model...")
        model_dict = dict(model.named_modules())

        for name, weight_lora in tqdm(lora_weights.items(), desc="Applying LoRA"):
            if name in model_dict:
                module = model_dict[name]
                # Get the original device and dtype of the module's weight
                original_device = module.weight.device
                original_dtype = module.weight.dtype

                # Direct parameter update (faster than load_state_dict)
                with torch.no_grad():
                    # Move LoRA weight to the same device as the module
                    weight_lora_device = weight_lora.to(device=original_device, dtype=original_dtype)
                    # Update weight in-place
                    module.weight.data.add_(weight_lora_device)
                updated_num += 1

        print(f"{updated_num} tensors are updated by LoRA.")
        return updated_num
