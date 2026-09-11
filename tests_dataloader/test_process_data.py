
import os

import pytest
import torch

from conftest import make_dataset, valid_item, write_meta

def _single_ds(tmp_path, item=None, **kw):
    item = item if item is not None else valid_item(tmp_path, 0)
    return make_dataset([write_meta(tmp_path, [item])], **kw)

def test_UT016_multimodal_getitem_shapes(tmp_path):
    ds = _single_ds(tmp_path)
    data = ds[0]
    assert data["modality_type"] == "multimodal"
    assert data["video"].shape == (3, 8, 64, 64)
    assert data["video"].min() >= -1.0 - 1e-6
    assert data["video"].max() <= 1.0 + 1e-6
    assert data["audio"].dim() == 1
    assert data["audio"].shape[0] == 16000
    assert data["text"] == "sample 0 description"

def test_UT017_short_video_padded_by_repeating_last_frame(tmp_path):
    item = valid_item(tmp_path, 0, n_frames=6)
    ds = make_dataset([write_meta(tmp_path, [item])], frame_tolerance=3)
    data = ds[0]
    assert data["total_frames"] == 6
    assert data["video"].shape[1] == 8
    assert torch.equal(data["video"][:, 5], data["video"][:, 6])
    assert torch.equal(data["video"][:, 5], data["video"][:, 7])

def test_UT018_short_audio_zero_padded_to_video_duration(tmp_path):
    item = valid_item(tmp_path, 0, audio_secs=0.5)
    ds = _single_ds(tmp_path, item)
    audio = ds[0]["audio"]
    assert audio.shape[0] == 16000
    assert torch.all(audio[8000:] == 0)

def test_UT019_mismatched_audio_loaded(tmp_path):
    ds = _single_ds(tmp_path)
    data = ds[0]
    assert "mismatched_audio" in data
    assert data["mismatched_audio"].dim() == 1
    assert data["mismatched_audio"].shape[0] == 16000
    assert data["mismatched_audio_sr"] == 16000

def test_UT020_getitem_falls_back_to_next_sample_in_bucket(tmp_path):
    items = [valid_item(tmp_path, 0), valid_item(tmp_path, 1)]
    ds = make_dataset([write_meta(tmp_path, items)])
    with open(items[0]["video_path"], "wb") as f:
        f.write(b"corrupt")
    data = ds[0]
    assert data["text"] == "sample 1 description"

def test_UT021_all_samples_corrupt_raises_runtime_error(tmp_path):
    item = valid_item(tmp_path, 0)
    ds = make_dataset([write_meta(tmp_path, [item])])
    with open(item["video_path"], "wb") as f:
        f.write(b"corrupt")
    with pytest.raises(RuntimeError, match="Failed to load any valid samples"):
        _ = ds[0]

def test_UT022_audio_only_getitem_keys(tmp_path):
    item = valid_item(tmp_path, 0, audio_only=True, audio_secs=1.0)
    ds = _single_ds(tmp_path, item)
    data = ds[0]
    assert data["modality_type"] == "audio_only"
    assert "video" not in data
    assert "audio" in data and "text" in data
    assert "audio_only" in ds.get_bucket_names()

def test_UT032_video_resized_to_bucket_dimensions(tmp_path):
    item = valid_item(tmp_path, 0, size=96)
    ds = _single_ds(tmp_path, item)
    data = ds[0]
    assert data["video"].shape == (3, 8, 64, 64)

@pytest.mark.xfail(
    strict=True, reason="BUG-001: process_data 强制要求 mismatched_audio_path，缺字段样本初始化可通过但取数必失败"
)
def test_UT023_sample_without_mismatched_audio_should_load(tmp_path):
    item = valid_item(tmp_path, 0, with_mismatched=False)
    ds = make_dataset([write_meta(tmp_path, [item])])
    assert ds.stats["retained_total"] == 1
    data = ds[0]
    assert data["text"] == "sample 0 description"
    assert "mismatched_audio" not in data

@pytest.mark.xfail(
    strict=True, reason="BUG-003: torch.randint(0, max_start_frame) 上界开区间，最后一个合法起始帧永不被采样"
)
def test_UT024_last_valid_start_frame_is_reachable(tmp_path):
    item = valid_item(tmp_path, 0, n_frames=12)
    ds = make_dataset([write_meta(tmp_path, [item])], frame_tolerance=4)
    starts = {ds.process_data(ds.metadata[0])["start_frame"] for _ in range(40)}
    assert 4 in starts
