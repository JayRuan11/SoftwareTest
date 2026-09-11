
import json
import math
import os
import sys
import wave

import numpy as np
import pytest
import torch

CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from ovi.data.loader import RealTimeDataset

def write_wav(path, seconds=1.0, sr=16000, channels=1, freq=440.0):
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float32) / sr
    sig = 0.5 * np.sin(2 * math.pi * freq * t)
    data = np.repeat(sig[None, :], channels, axis=0).reshape(-1)
    pcm = (np.clip(data, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return str(path)

def write_video(path, n_frames=8, size=64, fps=8.0):
    import imageio

    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (size, size, 3), dtype=np.uint8) for _ in range(n_frames)]
    imageio.mimwrite(
        str(path), frames, fps=fps, codec="libx264", quality=7, macro_block_size=None
    )
    return str(path)

def write_dummy(path):
    with open(path, "wb") as f:
        f.write(b"\x00")
    return str(path)

def write_meta(tmp_path, items, name="meta.json"):
    p = tmp_path / name
    p.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return str(p)

def make_dataset(meta_paths, **kw):
    params = dict(
        data_root=meta_paths,
        num_frames=8,
        frame_tolerance=2,
        height=64,
        width=64,
        description_model="descA",
        audio_description_model="descA",
        num_aspect_buckets=1,
    )
    params.update(kw)
    return RealTimeDataset(**params)

def dummy_item(
    tmp_path,
    idx=0,
    *,
    mode="multimodal",
    frame_num=8,
    desc_key="descA",
    desc_text="dummy",
    duration=1.0,
    height=64,
    width=64,
    drop_audio=False,
    drop_audio_file=False,
    drop_video_file=False,
    drop_desc=False,
    drop_quality=False,
):
    item = {}
    if mode == "audio_only":
        item["audio_only"] = True
    else:
        vp = tmp_path / f"v{idx}.mp4"
        if not drop_video_file:
            write_dummy(vp)
        item["video_path"] = str(vp)
    if not drop_audio:
        ap = tmp_path / f"a{idx}.wav"
        if not drop_audio_file:
            write_dummy(ap)
        item["audio_path"] = str(ap)
    if not drop_desc:
        item["description"] = {desc_key: desc_text}
    if not drop_quality:
        if mode == "audio_only":
            item["quality"] = {"duration": duration}
        else:
            item["quality"] = {
                "frameNum": frame_num,
                "width": width,
                "height": height,
                "duration": duration,
            }
    return item

def valid_item(
    tmp_path,
    idx=0,
    *,
    n_frames=8,
    fps=8.0,
    size=64,
    sr=16000,
    audio_secs=1.5,
    channels=1,
    with_mismatched=True,
    desc_key="descA",
    desc_text=None,
    audio_only=False,
    duration=None,
):
    item = {}
    if audio_only:
        item["audio_only"] = True
    else:
        item["video_path"] = write_video(
            tmp_path / f"v{idx}.mp4", n_frames=n_frames, size=size, fps=fps
        )
    item["audio_path"] = write_wav(
        tmp_path / f"a{idx}.wav", seconds=audio_secs, sr=sr, channels=channels
    )
    if with_mismatched:
        item["mismatched_audio_path"] = write_wav(
            tmp_path / f"m{idx}.wav", seconds=audio_secs, sr=sr
        )
    item["description"] = {desc_key: desc_text or f"sample {idx} description"}
    if audio_only:
        item["quality"] = {"duration": audio_secs if duration is None else duration}
    else:
        item["quality"] = {
            "frameNum": n_frames,
            "width": size,
            "height": size,
            "duration": audio_secs if duration is None else duration,
        }
    return item

class FakeDataset:
    def __init__(self, bucket_to_indices, length=None):
        self.bucket_to_indices = bucket_to_indices
        self._length = (
            length
            if length is not None
            else sum(len(v) for v in bucket_to_indices.values())
        )

    def __len__(self):
        return self._length

class FakeSampler:
    def __init__(self, num_replicas=1, rank=0):
        self.num_replicas = num_replicas
        self.rank = rank

@pytest.fixture(autouse=True)
def stub_torchaudio_load(monkeypatch):
    import ovi.data.loader as loader_mod

    def fake_load(path, *args, **kwargs):
        with wave.open(str(path), "rb") as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            raw = w.readframes(w.getnframes())
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            data = data.reshape(-1, ch).T
        else:
            data = data.reshape(1, -1)
        return torch.from_numpy(data.copy()), sr

    monkeypatch.setattr(loader_mod.torchaudio, "load", fake_load)
