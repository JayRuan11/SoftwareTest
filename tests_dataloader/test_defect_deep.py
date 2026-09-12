
import json

import numpy as np
import pytest
import torch

from conftest import dummy_item, make_dataset, valid_item, write_meta
import ovi.data.loader as loader_mod
import imageio

def _write_video(path, h, w, n=8, fps=8):
    frames = [np.zeros((h, w, 3), np.uint8) for _ in range(n)]
    imageio.mimwrite(
        str(path), frames, fps=fps, codec="libx264", quality=7, macro_block_size=None
    )
    return str(path)

def _multi_item(tmp_path, idx, video_path, w, h, desc):
    from conftest import write_wav

    return {
        "video_path": video_path,
        "audio_path": write_wav(tmp_path / f"a{idx}.wav", 1.5),
        "mismatched_audio_path": write_wav(tmp_path / f"m{idx}.wav", 1.5),
        "description": {"descA": desc},
        "quality": {"frameNum": 8, "width": w, "height": h, "duration": 1.5},
    }

class _MetaWrapReader:

    def __init__(self, real, meta):
        self._real = real
        self._meta = meta

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, *a):
        return self._real.__exit__(*a)

    def count_frames(self):
        return self._real.count_frames()

    def get_data(self, i):
        return self._real.get_data(i)

    def get_meta_data(self):
        return self._meta

@pytest.mark.xfail(
    strict=True,
    reason="BUG-004: 同桶全部损坏后随机回退可跨桶返回样本，破坏同桶同形状不变量，"
    "下游 collate 形状不一致崩溃或静默混入错误分辨率数据",
)
def test_UT033_random_fallback_stays_within_bucket(tmp_path, monkeypatch):
    v0 = _write_video(tmp_path / "v0.mp4", 256, 256)
    v1 = _write_video(tmp_path / "v1.mp4", 256, 512)
    items = [
        _multi_item(tmp_path, 0, v0, 256, 256, "BROKEN-SQUARE"),
        _multi_item(tmp_path, 1, v1, 512, 256, "VALID-WIDE"),
    ]
    ds = make_dataset([write_meta(tmp_path, items)], num_aspect_buckets=2,
                      height=256, width=256)
    orig_bucket = ds.metadata[0]["bucket"]
    other_bucket = ds.metadata[1]["bucket"]
    assert orig_bucket != other_bucket, "前提：两样本应属不同桶"
    with open(v0, "wb") as f:
        f.write(b"corrupt")

    import random as _random
    monkeypatch.setattr(_random, "randint", lambda a, b: 1)

    with pytest.raises(RuntimeError):
        _ = ds[0]

@pytest.mark.xfail(
    strict=True,
    reason="BUG-004b: 多模态桶样本损坏后随机回退可返回纯音频样本，"
    "批次内缺 video 键 → safe_collate/default_collate KeyError，训练中断",
)
def test_UT034_random_fallback_not_cross_modality(tmp_path, monkeypatch):
    from conftest import write_wav

    v0 = _write_video(tmp_path / "v0.mp4", 64, 64)
    items = [
        _multi_item(tmp_path, 0, v0, 64, 64, "BROKEN-MULTI"),
        {
            "audio_only": True,
            "audio_path": write_wav(tmp_path / "a1.wav", 1.0),
            "mismatched_audio_path": write_wav(tmp_path / "m1.wav", 1.0),
            "description": {"descA": "VALID-AUDIO-ONLY"},
            "quality": {"duration": 1.0},
        },
    ]
    ds = make_dataset([write_meta(tmp_path, items)], num_aspect_buckets=1)
    assert ds.metadata[0]["bucket"] != "audio_only"
    with open(v0, "wb") as f:
        f.write(b"corrupt")

    import random as _random
    monkeypatch.setattr(_random, "randint", lambda a, b: 1)

    with pytest.raises(RuntimeError):
        _ = ds[0]

@pytest.mark.xfail(
    strict=True,
    reason="BUG-005: fps 元数据缺失或为 0 时静默回退 30fps，"
    "真实低 fps 视频的音频裁剪窗错误 → 训练数据音画错位且无告警",
)
@pytest.mark.parametrize("bad_meta", [{}, {"fps": 0}], ids=["fps-missing", "fps-zero"])
def test_UT035_missing_fps_not_silently_30(tmp_path, monkeypatch, bad_meta):

    item = valid_item(tmp_path, 0, n_frames=8, fps=8, audio_secs=1.5)
    ds = make_dataset([write_meta(tmp_path, [item])])
    orig_get_reader = imageio.get_reader

    def fake_get_reader(path, **kw):
        return _MetaWrapReader(orig_get_reader(path), bad_meta)

    monkeypatch.setattr(loader_mod.imageio, "get_reader", fake_get_reader)
    data = ds.process_data(ds.metadata[0])

    if data is None:
        return
    assert data["audio"].shape[0] == 16000, (
        f"BUG-005: video_fps={data['video_fps']}，"
        f"音频窗 {data['audio'].shape[0]} 采样（真实 8fps 应为 16000）"
    )
