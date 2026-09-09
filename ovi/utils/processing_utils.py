import torch
import re
import numpy as np
import torch
import cv2
import os
import math
from typing import Tuple
import pandas as pd
import io
from pydub import AudioSegment
from PIL import Image
import torchaudio

def preprocess_image_tensor(image_path, device, target_dtype, h_w_multiple_of=32, resize_total_area=720*720):
    """Preprocess video data into standardized tensor format and (optionally) resize area."""
    def _parse_area(val):
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, (tuple, list)) and len(val) == 2:
            return int(val[0]) * int(val[1])
        if isinstance(val, str):
            m = re.match(r"\s*(\d+)\s*[x\*\s]\s*(\d+)\s*$", val, flags=re.IGNORECASE)
            if m:
                return int(m.group(1)) * int(m.group(2))
            if val.strip().isdigit():
                return int(val.strip())
        raise ValueError(f"resize_total_area={val!r} is not a valid area or WxH.")

    def _best_hw_for_area(h, w, area_target, multiple):
        if area_target <= 0:
            return h, w
        ratio_wh = w / float(h)
        area_unit = multiple * multiple
        tgt_units = max(1, area_target // area_unit)
        p0 = max(1, int(round(np.sqrt(tgt_units / max(ratio_wh, 1e-8)))))
        candidates = []
        for dp in range(-3, 4):
            p = max(1, p0 + dp)
            q = max(1, int(round(p * ratio_wh)))
            H = p * multiple
            W = q * multiple
            candidates.append((H, W))
        scale = np.sqrt(area_target / (h * float(w)))
        H_sc = max(multiple, int(round(h * scale / multiple)) * multiple)
        W_sc = max(multiple, int(round(w * scale / multiple)) * multiple)
        candidates.append((H_sc, W_sc))
        def score(HW):
            H, W = HW
            area = H * W
            return (abs(area - area_target), abs((W / max(H, 1e-8)) - ratio_wh))
        H_best, W_best = min(candidates, key=score)
        return H_best, W_best

    def is_video(path):
        video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg']
        return os.path.splitext(path)[1].lower() in video_exts

    if isinstance(image_path, str):
        if is_video(image_path):
            # 使用OpenCV读取视频的第一帧
            cap = cv2.VideoCapture(image_path)
            ret, image = cap.read()
            if ret:
                # OpenCV读取的是BGR格式，需要转换为RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cap.release()
            # 确保在读取失败时有适当的错误处理
            if not ret:
                raise ValueError(f"Failed to read frame from video: {image_path}")
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        assert isinstance(image_path, Image.Image)
        if image_path.mode != "RGB":
            image_path = image_path.convert("RGB")
        image = np.array(image_path)

    image = image.transpose(2, 0, 1)
    image = image.astype(np.float32) / 255.0

    image_tensor = torch.from_numpy(image).float().to(device, dtype=target_dtype).unsqueeze(0) ## b c h w
    image_tensor = image_tensor * 2.0 - 1.0 ## -1 to 1

    _, c, h, w = image_tensor.shape
    area_target = _parse_area(resize_total_area)
    if area_target is not None:
        target_h, target_w = _best_hw_for_area(h, w, area_target, h_w_multiple_of)
    else:
        target_h = (h // h_w_multiple_of) * h_w_multiple_of
        target_w = (w // h_w_multiple_of) * h_w_multiple_of

    target_h = max(h_w_multiple_of, int(target_h))
    target_w = max(h_w_multiple_of, int(target_w))

    if (h != target_h) or (w != target_w):
        image_tensor = torch.nn.functional.interpolate(
            image_tensor,
            size=(target_h, target_w),
            mode='bicubic',
            align_corners=False
        )

    return image_tensor

def preprocess_audio_tensor(audio, device):
    """Preprocess audio data into standardized tensor format."""
    if isinstance(audio, np.ndarray):
        audio_tensor = torch.from_numpy(audio).float().squeeze().unsqueeze(0).to(device)
    else:
        audio_tensor = audio.squeeze().unsqueeze(0).to(device)
    return audio_tensor


def calc_dims_from_area(
    aspect_ratio: str,
    total_area: int = 720*720,
    divisible_by: int = 32
) -> Tuple[int, int]:
    """
    Calculate width and height given an aspect ratio (h:w), total area,
    and divisibility constraint.

    Args:
        aspect_ratio (str): Aspect ratio string in format "h:w" (e.g., "9:16").
        total_area (int): Target maximum area (width * height ≤ total_area).
        divisible_by (int): Force width and height to be divisible by this value.

    Returns:
        (width, height): Tuple of integers that satisfy constraints.
    """
    # Parse aspect ratio string
    h_ratio, w_ratio = map(int, aspect_ratio.split(":"))

    # Reduce ratio
    gcd = math.gcd(h_ratio, w_ratio)
    h_ratio //= gcd
    w_ratio //= gcd

    # Scaling factor
    k = math.sqrt(total_area / (h_ratio * w_ratio))

    # Floor to multiples of divisible_by
    height = (int(k * h_ratio) // divisible_by) * divisible_by
    width  = (int(k * w_ratio) // divisible_by) * divisible_by

    # Safety check: avoid 0
    height = max(height, divisible_by)
    width  = max(width, divisible_by)

    return height, width


def snap_hw_to_multiple_of_32(h: int, w: int, area = 720 * 720) -> tuple[int, int]:
    """
    Scale (h, w) to match a target area if provided, then snap both
    dimensions to the nearest multiple of 32 (min 32).

    Args:
        h (int): original height
        w (int): original width
        area (int, optional): target area to scale to. If None, no scaling is applied.

    Returns:
        (new_h, new_w): dimensions adjusted
    """
    if h <= 0 or w <= 0:
        raise ValueError(f"h and w must be positive, got {(h, w)}")

    # If a target area is provided, rescale h, w proportionally
    if area is not None and area > 0:
        current_area = h * w
        scale = math.sqrt(area / float(current_area))
        h = int(round(h * scale))
        w = int(round(w * scale))

    # Snap to nearest multiple of 32
    def _n32(x: int) -> int:
        return max(32, int(round(x / 32)) * 32)

    return _n32(h), _n32(w)
def scale_hw_to_area_divisible(h, w, area=1024*1024, n=16):
    """
    Scale (h, w) so that area ≈ A, while keeping aspect ratio,
    and then round so both are divisible by n.

    Args:
        h (int): original height
        w (int): original width
        A (int or float): target area
        n (int): divisibility requirement

    Returns:
        (new_h, new_w): scaled and adjusted dimensions
    """
    # Current area
    current_area = h * w

    if current_area == 0:
        raise ValueError("Height and width must be positive")

    # Scale factor to match target area
    scale = math.sqrt(area / current_area)

    # Apply scaling while preserving aspect ratio
    new_h = h * scale
    new_w = w * scale

    # Round to nearest multiple of n
    new_h = int(round(new_h / n) * n)
    new_w = int(round(new_w / n) * n)

    # Ensure non-zero
    new_h = max(new_h, n)
    new_w = max(new_w, n)

    return new_h, new_w

# def validate_and_process_user_prompt(text_prompt: str, image_path: str = None, mode: str = "t2v") -> str:
#     if not isinstance(text_prompt, str):
#         raise ValueError("User input must be a string")

#     # Normalize whitespace
#     text_prompt = text_prompt.strip()

#     # Check if it's a file path that exists
#     if os.path.isfile(text_prompt):
#         _, ext = os.path.splitext(text_prompt.lower())

#         if ext == ".csv":
#             df = pd.read_csv(text_prompt)
#             df = df.fillna("")
#         elif ext == ".tsv":
#             df = pd.read_csv(text_prompt, sep="\t")
#             df = df.fillna("")
#         else:
#             raise ValueError(f"Unsupported file type: {ext}. Only .csv and .tsv are allowed.")

#         assert "text_prompt" in df.keys(), f"Missing required columns in TSV file."
#         text_prompts = list(df["text_prompt"])
#         if mode == "i2v" and 'image_path' in df.keys():
#             image_paths = list(df["image_path"])
#             assert all(p is None or len(p) == 0 or os.path.isfile(p) for p in image_paths), "One or more image paths in the TSV file do not exist."
#         else:
#             print("Warning: image_path was not found, assuming t2v or t2i2v mode...")
#             image_paths = [None] * len(text_prompts)

#     else:
#         assert image_path is None or os.path.isfile(image_path), f"Image path is not None but {image_path} does not exist."
#         text_prompts = [text_prompt]
#         image_paths = [image_path]

#     return text_prompts, image_paths


def validate_and_process_user_prompt(text_prompt: str, image_path: str = None, audio_path: str = None, mode: str = "t2v"):
    if not isinstance(text_prompt, str):
        raise ValueError("User input must be a string")

    # 标准化空格
    text_prompt = text_prompt.strip()

    # 初始化返回值
    text_prompts = []
    image_paths = []
    audio_paths = []

    # 检查是否为存在的 CSV/TSV 文件路径
    if os.path.isfile(text_prompt):
        _, ext = os.path.splitext(text_prompt.lower())

        if ext == ".csv":
            df = pd.read_csv(text_prompt)
        elif ext == ".tsv":
            df = pd.read_csv(text_prompt, sep="\t")
        else:
            raise ValueError(f"Unsupported file type: {ext}. Only .csv and .tsv are allowed.")

        df = df.fillna("")
        
        # 1. 处理文本列 (必需)
        assert "text_prompt" in df.keys(), "Missing required column 'text_prompt' in file."
        text_prompts = [str(p).strip().strip('"') for p in df["text_prompt"]]

        # 2. 处理图片列
        if 'image_path' in df.keys():
            # 清理路径中的空格和多余引号
            image_paths = [str(p).strip().strip('"') for p in df["image_path"]]
            # 如果是 i2v 模式，检查图片是否存在
            if mode == "i2v":
                for p in image_paths:
                    if not p or not os.path.isfile(p):
                        raise FileNotFoundError(f"Image path in file does not exist: {p}")
        else:
            print("Warning: 'image_path' column not found, setting to None.")
            image_paths = [None] * len(text_prompts)

        # 3. 处理音频列 (新增)
        if 'ref_audio_path' in df.keys():
            audio_paths = [str(p).strip().strip('"') for p in df["ref_audio_path"]]
            # 过滤掉空字符串，转为 None
            audio_paths = [p if (p and len(p) > 0) else None for p in audio_paths]
            # 验证存在的音频文件路径
            for p in audio_paths:
                if p and not os.path.isfile(p):
                    raise FileNotFoundError(f"Audio path in file does not exist: {p}")
        else:
            print("Warning: 'ref_audio_path' column not found, setting to None.")
            audio_paths = [None] * len(text_prompts)

    else:
        # 如果 text_prompt 不是文件，而是单条字符串
        text_prompts = [text_prompt]
        
        # 处理传入的单条图片路径
        if image_path:
            image_path = image_path.strip().strip('"')
            assert os.path.isfile(image_path), f"Image path does not exist: {image_path}"
        image_paths = [image_path]

        # 处理传入的单条音频路径
        if audio_path:
            audio_path = audio_path.strip().strip('"')
            assert os.path.isfile(audio_path), f"Audio path does not exist: {audio_path}"
        audio_paths = [audio_path]

    return text_prompts, image_paths, audio_paths


def format_prompt_for_filename(text: str) -> str:
    # remove anything inside <...>
    no_tags = re.sub(r"<.*?>", "", text)
    # replace spaces and slashes with underscores
    safe = no_tags.replace(" ", "_").replace("/", "_")
    # truncate to 50 chars
    return safe[:50]



def audio_bytes_to_tensor(audio_bytes, target_sr=16000):
    """
    Convert audio bytes to a 16kHz mono torch tensor in [-1, 1].

    Args:
        audio_bytes (bytes): Raw audio bytes
        target_sr (int): Target sample rate

    Returns:
        torch.Tensor: shape (num_samples,)
        int: sample rate
    """
    # Load audio from bytes
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")

    # Convert to mono if needed
    if audio.channels != 1:
        audio = audio.set_channels(1)

    # Resample if needed
    if audio.frame_rate != target_sr:
        audio = audio.set_frame_rate(target_sr)

    # Convert to numpy
    samples = np.array(audio.get_array_of_samples())
    samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max

    # Convert to torch tensor
    tensor = torch.from_numpy(samples)  # shape: (num_samples,)

    return tensor, target_sr

def audio_path_to_tensor(path, target_sr=16000):
    with open(path, "rb") as f:
        audio_bytes = f.read()
    return audio_bytes_to_tensor(audio_bytes, target_sr=target_sr)

def clean_text(text: str) -> str:
    """
    Remove all text between <S>...</E> and <AUDCAP>...</ENDAUDCAP> tags,
    as well as any 'Audio: ...' lines. Also trims excess whitespace.
    """
    # Remove <S> ... <E> dialogue tags
    text = re.sub(r"<S>.*?<E>", "", text, flags=re.DOTALL)

    # Remove <AUDCAP> ... <ENDAUDCAP> audio caption tags
    text = re.sub(r"<AUDCAP>.*?<ENDAUDCAP>", "", text, flags=re.DOTALL)

    # Remove 'Audio: ...' lines (including multiline ones)
    text = re.sub(r"Audio:\s*.*", "", text, flags=re.DOTALL)

    # Strip leading/trailing whitespace
    return text.strip()

def preprocess_ref_audio(audio_path, target_duration_sec=None, sample_rate=16000):
    """
    处理推理用的参考音频，逻辑复刻 RealTimeDataset
    """
    # 1. 加载音频
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Reference audio not found: {audio_path}")
    
    waveform, sr = torchaudio.load(audio_path)
    
    # 2. 转单声道 (Mean across channels)
    # 训练代码: mono = waveform.mean(dim=0, keepdim=True)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # 3. 重采样到 16000Hz 
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    
    audio_chunk = waveform.squeeze(0)
    
    # 4. 长度处理 (Crop or Pad)
    # 推理时，你需要根据想要生成的视频时长来截取音频
    if target_duration_sec is not None:
        target_samples = int(target_duration_sec * sample_rate)
        
        if audio_chunk.shape[0] > target_samples:
            # 截取前 target_samples (或者你可以选择中间段)
            audio_chunk = audio_chunk[:target_samples]
        elif audio_chunk.shape[0] < target_samples:
            # 填充 0
            pad_len = target_samples - audio_chunk.shape[0]
            padding = torch.zeros(pad_len, dtype=audio_chunk.dtype, device=audio_chunk.device)
            audio_chunk = torch.cat([audio_chunk, padding], dim=0)
            
    # 5. 归一化 (Normalization)
    # 训练代码: audio_chunk = audio_chunk / abs_max * 0.95
    abs_max = audio_chunk.abs().max()
    if abs_max > 0:
        audio_chunk = audio_chunk / abs_max * 0.95
        
    # 增加 Batch 和 Channel 维度 -> [1, 1, Samples]
    # 训练中是 collate 堆叠的，推理通常只有一条，所以要手动 unsqueeze
    return audio_chunk.unsqueeze(0).unsqueeze(0)