import os
import json
import random
import torch
import imageio
import torchaudio
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List, Dict, Optional, Tuple, Union

def safe_collate(batch):
    """Custom collate that handles variable-length audio and ensures tensor resizability."""
    from torch.utils.data._utils.collate import default_collate
    # Find max audio length in batch for ANY audio key
    max_audio_lens = {}
    audio_keys = ['audio', 'mismatched_audio'] # 定义所有需要 padding 的音频 key
    for key in audio_keys:
        max_len = 0
        for item in batch:
            if key in item and isinstance(item[key], torch.Tensor):
                max_len = max(max_len, item[key].shape[0])
        max_audio_lens[key] = max_len
    safe_batch = []
    for item in batch:
        safe_item = {}
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                cloned_value = value.detach().clone()
                # Pad audio to batch max length if needed
                # 修改：支持 audio 和 mismatched_audio 以及其他潜在的音频 key
                if key in max_audio_lens and max_audio_lens[key] > 0 and cloned_value.dim() == 1:
                    target_len = max_audio_lens[key]
                    if cloned_value.shape[0] < target_len:
                        pad_len = target_len - cloned_value.shape[0]
                        cloned_value = torch.cat([
                            cloned_value,
                            torch.zeros(pad_len, dtype=cloned_value.dtype, device=cloned_value.device)
                        ], dim=0)
                safe_item[key] = cloned_value
            else:
                safe_item[key] = value
        safe_batch.append(safe_item)
        
    batch_keys = [set(item.keys()) for item in safe_batch]
    if len(set(frozenset(keys) for keys in batch_keys)) > 1:
        # Mixed modality in batch - log warning
        print("Warning: Batch contains samples with different keys. This may cause issues.")
        print(f"  Keys variations: {batch_keys}")
        # Continue anyway - default_collate will handle or fail gracefully

    return default_collate(safe_batch)



def get_aspect_buckets(
    aspect_ratios: List[float], target_area: int, num_buckets: int = 8
) -> Tuple[List[str], Dict[float, str], Dict[str, Tuple[int, int]]]:
    """Group aspect ratios into buckets for more efficient batching and calculate dimensions for each bucket.

    Args:
        aspect_ratios: List of width/height ratios
        target_area: Target area in pixels for resizing
        num_buckets: Number of buckets to create

    Returns:
        Tuple containing:
        - List of bucket names
        - Mapping of aspect ratios to bucket names
        - Mapping of bucket names to (height, width) dimensions
    """
    if not aspect_ratios:
        return [], {}, {}

    # Convert to numpy array for easier manipulation
    ratios = np.array(aspect_ratios)

    # Use log scale for better distribution
    log_ratios = np.log(ratios)

    # Calculate bucket boundaries using percentiles
    percentiles = np.linspace(0, 100, num_buckets + 1)
    boundaries = np.percentile(log_ratios, percentiles)

    # Create bucket names, mapping and dimensions
    bucket_names = []
    ratio_to_bucket = {}
    bucket_dimensions = {}

    height_factor = 64
    width_factor = 64

    for i in range(num_buckets):
        min_ratio = np.exp(boundaries[i])
        max_ratio = np.exp(boundaries[i + 1])

        bucket_ratio = np.exp((boundaries[i] + boundaries[i + 1]) / 2)
        height = int(np.sqrt(target_area / bucket_ratio))
        width = int(height * bucket_ratio)

        # check size alignment
        height = (height + height_factor - 1) // height_factor * height_factor
        width = (width + width_factor - 1) // width_factor * width_factor

        bucket_name = f"aspect_{min_ratio:.2f}_{max_ratio:.2f}"
        bucket_names.append(bucket_name)
        bucket_dimensions[bucket_name] = (height, width)

    # Assign each ratio to its closest bucket based on log distance
    bucket_centers = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(num_buckets)]
    for ratio in ratios:
        log_ratio = np.log(ratio)
        # Find the bucket with the closest center
        distances = [abs(log_ratio - center) for center in bucket_centers]
        closest_bucket_idx = np.argmin(distances)
        ratio_to_bucket[ratio] = bucket_names[closest_bucket_idx]

    return bucket_names, ratio_to_bucket, bucket_dimensions


class RealTimeDataset(Dataset):
    """A dataset that processes data on-the-fly with aspect ratio bucketing."""

    def __init__(
        self,
        data_root: List[str],
        steps_per_epoch: Optional[int] = None,
        num_aspect_buckets: int = 8,
        description_model: str = "qwen2-VL-72B-detail",
        audio_description_model: Optional[str] = "mimo-audio",  # Separate model for audio-only samples
        num_frames: int = 81,
        frame_tolerance: int = 10,  # Allow ±10 frames variance for video length
        max_audio_duration: float = 20.0,  # Max audio duration in seconds (for audio-only samples)
        height: int = 720,
        width: int = 720,
        sample_rate: int = 16000,
        normalize_audio: bool = False,
    ):
        super().__init__()
        self.data_paths = data_root
        self.steps_per_epoch = steps_per_epoch

        # Support multiple description models (can be string or list)
        # Convert to list for uniform handling
        if isinstance(description_model, str):
            self.description_models = [description_model]
        else:
            self.description_models = list(description_model)

        # Audio-only samples can use different description models
        # If not specified, use the same as multimodal
        if audio_description_model is not None:
            if isinstance(audio_description_model, str):
                self.audio_description_models = [audio_description_model]
            else:
                self.audio_description_models = list(audio_description_model)
        else:
            self.audio_description_models = self.description_models

        # Keep first model name for backward compatibility
        self.description_model = self.description_models[0]

        self.num_frames = num_frames
        self.frame_tolerance = frame_tolerance  # For video length filtering
        self.max_audio_duration = max_audio_duration  # For audio-only duration filtering
        self.bucket_dimensions: Dict[str, Tuple[int, int]] = {}
        self.to_tensor = transforms.ToTensor()
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.resampler = {}

        # Load and combine metadata
        self.metadata = []
        self.bucket_to_indices: Dict[str, List[int]] = {}

        self.stats = {
            "original_total": 0,
            "retained_total": 0,
            "filtered_missing_audio": 0,
            "filtered_missing_video": 0,
            "filtered_missing_description": 0,
            "filtered_frame_tolerance": 0,
            "filtered_audio_duration": 0
        }

        # Note: Encoding models are NOT loaded here to support multi-worker data loading
        # They will be loaded in the training module and encoding will happen in training_step

        # Collect metadata and aspect ratios
        aspect_ratios = []
        for data_idx, path in enumerate(self.data_paths):
            meta_file = path
            if os.path.exists(meta_file):
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                    for item in tqdm(meta):
                        # Determine if this is an audio-only sample
                        self.stats["original_total"] += 1

                        is_audio_only = item.get("audio_only", False) or "video_path" not in item or not item.get("video_path")

                        # Check required fields based on modality type
                        has_audio = "audio_path" in item and os.path.exists(item["audio_path"])
                        
                        if not has_audio:
                            self.stats["filtered_missing_audio"] += 1
                            continue
                        
                        
                        # Check description availability - try multiple models in priority order
                        # Use different model lists for audio-only vs multimodal samples
                        desc_models_to_try = self.audio_description_models if is_audio_only else self.description_models
                        found_desc_model = None
                        if "description" in item and isinstance(item["description"], dict):
                            for model_name in desc_models_to_try:
                                if item["description"].get(model_name):
                                    found_desc_model = model_name
                                    break
                        has_desc = found_desc_model is not None
                        

                        if found_desc_model is None:
                            self.stats["filtered_missing_description"] += 1
                            continue

                        if is_audio_only:
                            # Audio-only samples only need audio and description
                            if not (has_audio and has_desc):
                                continue
                            # Store which description model was found for this sample
                            item["_desc_model_used"] = found_desc_model

                            # Filter by audio duration: only keep samples within max_audio_duration
                            if "quality" in item and "duration" in item["quality"]:
                                audio_duration = float(item["quality"]["duration"])
                                if audio_duration > max_audio_duration:
                                    self.stats["filtered_audio_duration"] += 1
                                    continue
                        else:
                            # Multimodal samples need video, audio and description
                            has_video = "video_path" in item and os.path.exists(item["video_path"])
                            if not (has_video and has_audio and has_desc):
                                self.stats["filtered_missing_video"] += 1
                                continue
                            # Store which description model was found for this sample
                            item["_desc_model_used"] = found_desc_model

                            # Filter video by frame count: keep only if within num_frames ± frame_tolerance
                            if "quality" in item and "frameNum" in item["quality"]:
                                frame_count = int(item["quality"]["frameNum"])
                                min_frames = num_frames - frame_tolerance
                                max_frames = num_frames + frame_tolerance

                                # Skip if outside tolerance range
                                if frame_count < min_frames or frame_count > max_frames:
                                    self.stats["filtered_frame_tolerance"] += 1
                                    continue
                            else:
                                self.stats["filtered_frame_tolerance"] += 1
                                continue

                        item["data_idx"] = data_idx
                        self.metadata.append(item)

                        # Get aspect ratio from metadata
                        # Use -1 for audio-only samples (will be filtered out in bucket calculation)
                        if not is_audio_only and "quality" in item and "width" in item["quality"] and "height" in item["quality"]:
                            aspect_ratio = item["quality"]["width"] / item["quality"]["height"]
                        elif not is_audio_only:
                            # Multimodal sample without quality info, default to 1.0
                            aspect_ratio = 1.0
                        else:
                            # Audio-only sample: use -1 as special marker
                            aspect_ratio = -1.0
                        aspect_ratios.append(aspect_ratio)
        self.stats["retained_total"] = len(self.metadata) # 2. 最后保留的样本数
        self.print_filtering_report()


        if not self.metadata:
            raise ValueError(
                f"No valid samples found in paths: {data_root}. "
                f"Check that your data contains video_path, audio_path, and description fields."
            )

        # Create aspect ratio buckets with dimensions
        # Filter out audio-only samples (aspect_ratio = -1) before calculating buckets
        multimodal_aspect_ratios = [ar for ar in aspect_ratios if ar > 0]
        if multimodal_aspect_ratios:
            bucket_names, ratio_to_bucket, self.bucket_dimensions = get_aspect_buckets(
                multimodal_aspect_ratios, height * width, num_aspect_buckets
            )
        else:
            # No multimodal samples, create empty bucket info
            bucket_names, ratio_to_bucket, self.bucket_dimensions = [], {}, {}

        # Assign samples to buckets
        # IMPORTANT: Separate audio-only samples into dedicated buckets to avoid mixing modalities in the same batch
        for idx, (meta_item, aspect_ratio) in enumerate(zip(self.metadata, aspect_ratios)):
            is_audio_only = meta_item.get("audio_only", False) or "video_path" not in meta_item or not meta_item.get("video_path")

            if is_audio_only:
                # Audio-only samples go to a dedicated "audio_only" bucket
                bucket = "audio_only"
                # Use default dimensions for audio-only (not actually used, but kept for consistency)
                if bucket not in self.bucket_dimensions:
                    self.bucket_dimensions[bucket] = (height, width)
            else:
                # Multimodal samples use aspect ratio buckets
                bucket = ratio_to_bucket[aspect_ratio]

            if bucket not in self.bucket_to_indices:
                self.bucket_to_indices[bucket] = []
            self.bucket_to_indices[bucket].append(idx)
            meta_item["bucket"] = bucket

        # Print dataset statistics
        num_audio_only = len(self.bucket_to_indices.get("audio_only", []))
        num_multimodal = len(self.metadata) - num_audio_only
        print("\n[Dataset Statistics]")
        print(f"  Total samples: {len(self.metadata)}")
        print(f"  Multimodal samples: {num_multimodal} ({num_multimodal/len(self.metadata)*100:.1f}%)")
        print(f"  Audio-only samples: {num_audio_only} ({num_audio_only/len(self.metadata)*100:.1f}%)")
        print(f"  Number of buckets: {len(self.bucket_to_indices)}")
        if num_audio_only > 0:
            print("  Note: Audio-only samples are in dedicated 'audio_only' bucket")

    def __len__(self):
        if self.steps_per_epoch:
            return self.steps_per_epoch
        return len(self.metadata)

    def print_filtering_report(self):
        print("\n" + "="*50)
        print("  [ 数据集样本加载详细报告 ]")
        print("-" * 50)
        print(f"  1. 原始总样本数 (JSON记录):    {self.stats['original_total']}")
        print(f"  2. 最终保留样本数:            {self.stats['retained_total']}")
        print(f"  3. 总计过滤样本数:            {self.stats['original_total'] - self.stats['retained_total']}")
        print("-" * 50)
        print("  [ 具体过滤原因分类 ]:")
        print(f"     - 缺失音频文件:           {self.stats['filtered_missing_audio']}")
        print(f"     - 缺失视频文件 (多模态):    {self.stats['filtered_missing_video']}")
        print(f"     - 缺失指定描述词:          {self.stats['filtered_missing_description']}")
        print(f"     - 视频帧数不在范围:        {self.stats['filtered_frame_tolerance']}")
        print(f"     - 音频时长超过限制:        {self.stats['filtered_audio_duration']}")
        
        retained_pct = (self.stats['retained_total'] / self.stats['original_total'] * 100) if self.stats['original_total'] > 0 else 0
        print("-" * 50)
        print(f"  数据留存率: {retained_pct:.2f}%")
        print("="*50 + "\n")

    def __getitem__(self, idx):
        # If steps_per_epoch is set, use modulo to cycle through the dataset
        if self.steps_per_epoch:
            idx = idx % len(self.metadata)

        # Get the bucket of the requested sample
        original_idx = idx
        original_bucket = self.metadata[idx].get("bucket", None)

        # Try to get a valid sample, skip if process_data returns None
        max_attempts = 20
        for attempt in range(max_attempts):
            meta = self.metadata[idx]
            data = self.process_data(meta)

            # Validate data based on modality type
            if data is not None:
                modality_type = data.get("modality_type", "multimodal")

                # For audio-only samples, only need audio and text
                if modality_type == "audio_only":
                    if "audio" in data and "text" in data:
                        return data
                # For multimodal samples, need video, audio and text with correct num_frames
                else:
                    if "video" in data and "audio" in data and "text" in data:
                        if data["video"].shape[1] == self.num_frames:
                            return data

            # If this sample failed, try the next one in the SAME bucket
            if original_bucket and original_bucket in self.bucket_to_indices:
                bucket_indices = self.bucket_to_indices[original_bucket]
                # Find the position of current idx in the bucket
                try:
                    pos = bucket_indices.index(idx)
                    # Get next index in the same bucket (with wraparound)
                    next_pos = (pos + 1) % len(bucket_indices)
                    idx = bucket_indices[next_pos]
                except ValueError:
                    # If idx not in bucket (shouldn't happen), fall back to sequential
                    idx = (idx + 1) % len(self.metadata)
            else:
                # No bucket info, fall back to sequential search
                idx = (idx + 1) % len(self.metadata)

        # If all attempts failed, log warning and try a random sample from the dataset
        # This prevents training from stopping due to corrupted data
        print(f"Warning: Failed to load valid sample after {max_attempts} attempts starting from index {original_idx}")
        print("         Trying a random sample instead to avoid stopping training...")

        # Try random samples from the entire dataset (up to 5 attempts)
        for _ in range(5):
            random_idx = random.randint(0, len(self.metadata) - 1)
            meta = self.metadata[random_idx]
            data = self.process_data(meta)

            # Validate random sample the same way as normal samples
            if data is not None:
                modality_type = data.get("modality_type", "multimodal")
                if modality_type == "audio_only":
                    if "audio" in data and "text" in data:
                        return data
                else:
                    if "video" in data and "audio" in data and "text" in data:
                        if data["video"].shape[1] == self.num_frames:
                            return data

        # If all random samples also fail, raise error (dataset might be completely corrupted)
        raise RuntimeError(
            f"Failed to load any valid samples. Dataset at index {original_idx} and 5 random indices "
            f"all failed. Please check your data quality."
        )

    def get_bucket_names(self):
        """Returns list of bucket names found in the dataset."""
        return list(self.bucket_to_indices.keys())

    def process_data(self, meta):
        """Process video, audio, and text data in real-time."""
        data = {}

        # Determine modality type from metadata
        # If video_path is missing or marked as audio_only, treat as audio-only sample
        is_audio_only = meta.get("audio_only", False) or "video_path" not in meta or not meta.get("video_path")
        data["modality_type"] = "audio_only" if is_audio_only else "multimodal"

        # Set height/width for bucket dimensions (used for logging/debugging and video resizing)
        bucket = meta.get("bucket", "audio_only")
        if bucket in self.bucket_dimensions:
            target_height, target_width = self.bucket_dimensions[bucket]
        else:
            # Fallback to default dimensions if bucket not found
            target_height, target_width = 720, 720
        data["height"] = int(target_height)
        data["width"] = int(target_width)

        # Process video data (skip for audio-only samples)
        if not is_audio_only and "video_path" in meta:
            video_path = str(meta["video_path"]).strip()
            if os.path.exists(video_path):
                try:
                    with imageio.get_reader(video_path) as reader:
                        total_frames = reader.count_frames()

                        # Safety check: verify frame count is within acceptable range
                        # This protects against samples that passed filtering without quality metadata
                        min_frames = self.num_frames - self.frame_tolerance
                        max_frames = self.num_frames + self.frame_tolerance

                        if total_frames < min_frames or total_frames > max_frames:
                            # print(f"Warning: Video {video_path} has {total_frames} frames, outside acceptable range [{min_frames}, {max_frames}]")
                            return None

                        # Determine how many frames to read
                        frames_to_read = min(total_frames, self.num_frames)

                        # For videos with enough frames, randomly select start position
                        # For shorter videos, read from the beginning
                        if total_frames >= self.num_frames:
                            max_start_frame = total_frames - self.num_frames
                            start_frame = torch.randint(0, max_start_frame, (1,)).item() if max_start_frame > 0 else 0
                        else:
                            start_frame = 0

                        frames = []
                        # Read available frames
                        for i in range(start_frame, start_frame + frames_to_read):
                            try:
                                frame = reader.get_data(i)
                                frames.append(frame)
                            except (IndexError, Exception) as e:
                                # If we can't read a frame, this video is invalid
                                print(f"Warning: Failed to read frame {i} from {video_path}: {e}")
                                return None

                        # If video has fewer frames than num_frames, pad by repeating the last frame
                        if len(frames) < self.num_frames:
                            last_frame = frames[-1]
                            pad_count = self.num_frames - len(frames)
                            frames.extend([last_frame] * pad_count)

                        # Verify we now have exactly num_frames
                        if len(frames) != self.num_frames:
                            return None

                        # frames = [
                        #     self.to_tensor(
                        #         Image.fromarray(frame).resize((target_width, target_height), Image.Resampling.LANCZOS)
                        #     )
                        #     for frame in frames
                        # ]
                        # frames = torch.stack(frames, dim=1)  # (C,T,H,W)

                        processed_frames = []
                        for frame in frames:
                            # 1. Numpy (H, W, C) -> Tensor (C, H, W), 值范围变为 [0.0, 1.0]
                            tensor_frame = self.to_tensor(frame)

                            # 2. 归一化到 [-1, 1] 
                            tensor_frame = tensor_frame * 2.0 - 1.0

                            # 3. 使用 Bicubic 插值调整尺寸
                            # interpolate 需要 (Batch, C, H, W) 输入，所以需要 unsqueeze(0)
                            if tensor_frame.shape[1] != target_height or tensor_frame.shape[2] != target_width:
                                tensor_frame = tensor_frame.unsqueeze(0)  # (1, C, H, W)
                                tensor_frame = torch.nn.functional.interpolate(
                                    tensor_frame,
                                    size=(target_height, target_width),
                                    mode='bicubic',
                                    align_corners=False
                                )
                                tensor_frame = tensor_frame.squeeze(0)  # 变回 (C, H, W)

                            processed_frames.append(tensor_frame)

                        # 4. 堆叠帧: (C, T, H, W)
                        frames = torch.stack(processed_frames, dim=1)

                        try:
                            meta_data = reader.get_meta_data()
                            fps = meta_data.get("fps") or meta_data.get("video_fps") or meta_data.get("frame_rate")
                        except Exception:
                            fps = None

                        data["video"] = frames
                        data["total_frames"] = int(total_frames)
                        data["start_frame"] = int(start_frame)
                        data["video_fps"] = float(fps) if fps is not None else 30.0
                except Exception as e:
                    print(f"Warning: Failed to read video from {video_path}: {e}")
                    return None
            else:
                if not is_audio_only:
                    print(f"Warning: Video file not found: {video_path}")
                    
                    print(f"  -> Debug Raw Path: {repr(video_path)}")
                    return None
        elif not is_audio_only:
            print(f"Warning: No video_path in metadata for multimodal sample: {meta}")
            return None

        # Process audio data
        if "audio_path" in meta:
            audio_path = str(meta["audio_path"]).strip()
            if os.path.exists(audio_path):
                try:
                    waveform, sample_rate = torchaudio.load(audio_path)  # waveform: [channels, samples]
                except Exception as e:
                    print(f"Warning: Failed to load audio from {audio_path}: {e}")
                    return None

                if waveform is not None:
                    # convert to mono keeping channel dim for Resample
                    mono = waveform.mean(dim=0, keepdim=True)  # [1, samples]

                    # resample if necessary
                    if sample_rate and sample_rate != self.sample_rate:
                        sr_key = int(sample_rate)
                        if sr_key not in self.resampler:
                            self.resampler[sr_key] = torchaudio.transforms.Resample(
                                orig_freq=sample_rate,
                                new_freq=self.sample_rate,
                                lowpass_filter_width=64,
                                rolloff=0.9475937167399596,
                                resampling_method="sinc_interp_kaiser",
                                beta=14.769656459379492,
                            )
                        mono = self.resampler[sr_key](mono)
                        sample_rate = self.sample_rate

                    audio_chunk = mono.squeeze(0)  # [samples]

                    # Crop audio to match video segment duration (or use full audio for audio-only)
                    # Audio length varies by video FPS; batch consistency handled by collate padding
                    if not is_audio_only and "start_frame" in data and "video_fps" in data:
                        video_fps = data["video_fps"]
                        start_frame = int(data["start_frame"])

                        start_time = start_frame / float(video_fps)
                        duration = self.num_frames / float(video_fps)
                        expected_audio_samples = int(round(duration * float(self.sample_rate)))

                        start_sample = int(round(start_time * float(self.sample_rate)))
                        start_sample = max(0, start_sample)
                        end_sample = start_sample + expected_audio_samples

                        # Extract or pad audio to expected length
                        if end_sample <= audio_chunk.size(0):
                            audio_chunk = audio_chunk[start_sample:end_sample]
                        elif start_sample < audio_chunk.size(0):
                            available = audio_chunk[start_sample:]
                            pad_len = expected_audio_samples - available.size(0)
                            audio_chunk = torch.cat([available, torch.zeros(pad_len, dtype=audio_chunk.dtype)], dim=0)
                        else:
                            # Start is beyond audio length, return all zeros
                            audio_chunk = torch.zeros(expected_audio_samples, dtype=audio_chunk.dtype)
                    elif is_audio_only:
                        actual_duration = audio_chunk.size(0) / float(self.sample_rate)
                        if actual_duration > self.max_audio_duration:
                            print(f"Warning: Audio {audio_path} duration {actual_duration:.2f}s exceeds max {self.max_audio_duration}s")
                            return None

                    # normalization (after cropping)
                    abs_max = audio_chunk.abs().max() if audio_chunk.numel() > 0 else 0.0
                    if self.normalize_audio and abs_max > 0:
                        audio_chunk = audio_chunk / abs_max * 0.95  # Normalize to avoid clipping

                    data["audio"] = audio_chunk
                    data["audio_sr"] = int(sample_rate) if sample_rate else self.sample_rate
                else:
                    return None
            else:
                print(f"Warning: Audio file not found: {audio_path}")
                print(f"  -> Debug Raw Path: {repr(audio_path)}")
                return None
        else:
            print(f"Warning: No audio_path in metadata: {meta}")
            return None

        # Process audio data
        if "mismatched_audio_path" in meta:
            mismatched_audio_path = str(meta["mismatched_audio_path"]).strip()
            if os.path.exists(mismatched_audio_path):
                try:
                    waveform, sample_rate = torchaudio.load(mismatched_audio_path)  # waveform: [channels, samples]
                except Exception as e:
                    print(f"Warning: Failed to load audio from {mismatched_audio_path}: {e}")
                    return None

                if waveform is not None:
                    # convert to mono keeping channel dim for Resample
                    mono = waveform.mean(dim=0, keepdim=True)  # [1, samples]

                    # resample if necessary
                    if sample_rate and sample_rate != self.sample_rate:
                        sr_key = int(sample_rate)
                        if sr_key not in self.resampler:
                            self.resampler[sr_key] = torchaudio.transforms.Resample(
                                orig_freq=sample_rate,
                                new_freq=self.sample_rate,
                                lowpass_filter_width=64,
                                rolloff=0.9475937167399596,
                                resampling_method="sinc_interp_kaiser",
                                beta=14.769656459379492,
                            )
                        mono = self.resampler[sr_key](mono)
                        sample_rate = self.sample_rate

                    audio_chunk = mono.squeeze(0)  # [samples]

                    # Crop audio to match video segment duration (or use full audio for audio-only)
                    # Audio length varies by video FPS; batch consistency handled by collate padding
                    if not is_audio_only and "start_frame" in data and "video_fps" in data:
                        video_fps = data["video_fps"]
                        start_frame = int(data["start_frame"])

                        start_time = start_frame / float(video_fps)
                        duration = self.num_frames / float(video_fps)
                        expected_audio_samples = int(round(duration * float(self.sample_rate)))

                        start_sample = int(round(start_time * float(self.sample_rate)))
                        start_sample = max(0, start_sample)
                        end_sample = start_sample + expected_audio_samples

                        # Extract or pad audio to expected length
                        if end_sample <= audio_chunk.size(0):
                            audio_chunk = audio_chunk[start_sample:end_sample]
                        elif start_sample < audio_chunk.size(0):
                            available = audio_chunk[start_sample:]
                            pad_len = expected_audio_samples - available.size(0)
                            audio_chunk = torch.cat([available, torch.zeros(pad_len, dtype=audio_chunk.dtype)], dim=0)
                        else:
                            # Start is beyond audio length, return all zeros
                            audio_chunk = torch.zeros(expected_audio_samples, dtype=audio_chunk.dtype)
                    elif is_audio_only:
                        actual_duration = audio_chunk.size(0) / float(self.sample_rate)
                        if actual_duration > self.max_audio_duration:
                            print(f"Warning: Audio {audio_path} duration {actual_duration:.2f}s exceeds max {self.max_audio_duration}s")
                            return None

                    # normalization (after cropping)
                    abs_max = audio_chunk.abs().max() if audio_chunk.numel() > 0 else 0.0
                    if self.normalize_audio and abs_max > 0:
                        audio_chunk = audio_chunk / abs_max * 0.95  # Normalize to avoid clipping

                    data["mismatched_audio"] = audio_chunk
                    data["mismatched_audio_sr"] = int(sample_rate) if sample_rate else self.sample_rate
                else:
                    return None
            else:
                # print(f"Warning: Audio file not found: {mismatched_audio_path}")
                # print(f"  -> Debug Raw Path: {repr(mismatched_audio_path)}")
                return None
        else:
            print(f"Warning: No audio_path in metadata: {meta}")
            return None

        # Process text description
        if "description" in meta:
            if isinstance(meta["description"], dict):
                # Use the description model that was found during dataset init
                # If not available, try all models in priority order
                desc_model_used = meta.get("_desc_model_used")
                description = None

                if desc_model_used and meta["description"].get(desc_model_used):
                    description = meta["description"].get(desc_model_used)
                else:
                    # Fallback: try all models in priority order
                    desc_models_to_try = self.audio_description_models if is_audio_only else self.description_models
                    for model_name in desc_models_to_try:
                        if meta["description"].get(model_name):
                            description = meta["description"].get(model_name)
                            break

                if description:
                    # Return raw description without formatting
                    # Text formatting will be applied in the training module
                    data["text"] = description
                else:
                    models_tried = self.audio_description_models if is_audio_only else self.description_models
                    print(f"Warning: No description found for models {models_tried} in metadata: {list(meta['description'].keys())}")
                    return None
            else:
                print(f"Warning: Description is not a dict in metadata: {meta['description']}")
                return None
        else:
            print(f"Warning: No description in metadata: {meta}")
            return None

        return data


class BucketBatchSampler(Sampler):
    """A batch sampler that creates batches by sampling from similar buckets."""

    def __init__(
        self,
        sampler: Sampler,
        dataset: Dataset,
        batch_size: int,
        bucket_names: List[str],
        drop_last: bool = True,
        seed: Optional[int] = None,
        in_order: bool = False,
    ):
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.in_order = in_order

        if seed is not None:
            self.seed = seed
        else:
            self.seed = torch.randint(0, 2**32 - 1, (1,)).item()

        # Get distributed info directly from sampler (DistributedSampler has these attributes)
        self.num_replicas = getattr(sampler, 'num_replicas', 1)
        self.rank = getattr(sampler, 'rank', 0)

        # Debug: print distributed info (only on rank 0)
        if self.rank == 0:
            print(f"[BucketBatchSampler] num_replicas={self.num_replicas}, rank={self.rank}")
            print(f"[BucketBatchSampler] Sampler type: {type(self.sampler).__name__}")

        # Get indices for each bucket
        self.bucket_indices = {}
        for bucket in bucket_names:
            if hasattr(dataset, "bucket_to_indices"):
                self.bucket_indices[bucket] = dataset.bucket_to_indices[bucket]
            else:
                self.bucket_indices[bucket] = []

        # Remove empty buckets
        self.bucket_names = [b for b in bucket_names if len(self.bucket_indices[b]) > 0]

        if not self.bucket_names:
            raise ValueError("No valid buckets found in dataset")

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        self.seed = torch.randint(0, 2**32 - 1, (1,)).item()  # Update seed for next epoch

        indices = []
        for bucket in self.bucket_names:
            bucket_indices = self.bucket_indices[bucket]
            if self.in_order:
                bucket_indices_ordered = list(bucket_indices)
            else:
                bucket_indices_ordered = torch.tensor(bucket_indices)[
                    torch.randperm(len(bucket_indices), generator=g)
                ].tolist()

            # Create batches
            for i in range(0, len(bucket_indices_ordered), self.batch_size):
                if self.drop_last and i + self.batch_size > len(bucket_indices_ordered):
                    break
                batch = bucket_indices_ordered[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    indices.append(batch)

        # Shuffle batches for better randomness (but keep each batch's samples from same bucket)
        if not self.in_order:
            indices = [indices[i] for i in torch.randperm(len(indices), generator=g).tolist()]

        # Debug: print total batches before distribution (only rank 0)
        if self.rank == 0:
            print(f"[BucketBatchSampler] Total batches before distribution: {len(indices)}")

        # Apply distributed sampling: each rank gets a subset of batches
        if self.num_replicas > 1:
            # Distribute batches evenly across ranks
            indices = indices[self.rank::self.num_replicas]
            # Debug: print after distribution
            if self.rank == 0:
                print(f"[BucketBatchSampler] Batches after distribution (rank {self.rank}): {len(indices)}")

        # Truncate batches to respect steps_per_epoch
        expected_batches = len(self)
        if len(indices) > expected_batches:
            indices = indices[:expected_batches]
            if self.rank == 0:
                print(f"[BucketBatchSampler] Truncated to {expected_batches} batches")

        return iter(indices)

    def __len__(self):
        if self.drop_last:
            total_batches = sum(len(self.bucket_indices[bucket]) // self.batch_size for bucket in self.bucket_names)
        else:
            total_batches = sum(
                (len(self.bucket_indices[bucket]) + self.batch_size - 1) // self.batch_size
                for bucket in self.bucket_names
            )

        total_batches = min(total_batches, len(self.dataset) // self.batch_size)

        # Adjust for distributed training: each rank gets 1/num_replicas of the batches
        if self.num_replicas > 1:
            total_batches = total_batches // self.num_replicas

        # Debug: print on rank 0
        if self.rank == 0:
            print(f"[BucketBatchSampler] Total batches per rank: {total_batches}")

        return total_batches


class AspectRatioDataLoader(DataLoader):
    """DataLoader that handles aspect ratio bucketing for efficient training."""

    def __init__(
        self,
        data_paths: list,
        batch_size: int,
        num_workers: int = 4,
        steps_per_epoch: Optional[int] = None,
        num_aspect_buckets: int = 8,
        seed: Optional[int] = None,
        pin_memory: bool = False,
        description_model: Union[str, List[str]] = "qwen2-VL-72B-detail",
        audio_description_model: Optional[Union[str, List[str]]] = "mimo-audio",
        num_frames: int = 81,
        frame_tolerance: int = 10,
        max_audio_duration: float = 20.0,
        height: int = 720,
        width: int = 720,
        **kwargs,
    ):
        # Initialize dataset
        dataset = RealTimeDataset(
            data_root=data_paths,
            steps_per_epoch=steps_per_epoch,
            num_aspect_buckets=num_aspect_buckets,
            description_model=description_model,
            audio_description_model=audio_description_model,
            num_frames=num_frames,
            frame_tolerance=frame_tolerance,
            max_audio_duration=max_audio_duration,
            height=height,
            width=width,
        )

        # Get bucket names
        bucket_names = dataset.get_bucket_names()

        # Create appropriate sampler based on environment
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        rank = int(os.environ.get('RANK', '0'))

        if world_size > 1:
            # Multi-GPU training: use DistributedSampler
            sampler = torch.utils.data.DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False  # We handle shuffling in BucketBatchSampler
            )
        else:
            # Single GPU training: use RandomSampler
            sampler = torch.utils.data.RandomSampler(dataset)

        batch_sampler = BucketBatchSampler(
            sampler=sampler,
            dataset=dataset,
            batch_size=batch_size,
            bucket_names=bucket_names,
            drop_last=True,
            seed=seed,
            in_order=False,  # Shuffle buckets for better training
        )

        # Initialize DataLoader with custom batch sampler
        super().__init__(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=safe_collate,  # Use custom collate function
            **kwargs,
        )

    def get_dataset(self) -> RealTimeDataset:
        """Returns the underlying dataset."""
        return self.dataset

    def set_epoch(self, epoch: int):
        """Updates epoch for distributed sampler if using distributed training."""
        if isinstance(self.batch_sampler.sampler, torch.utils.data.DistributedSampler):
            self.batch_sampler.sampler.set_epoch(epoch)
