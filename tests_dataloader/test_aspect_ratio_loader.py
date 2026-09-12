
import torch

from conftest import valid_item, write_meta
from ovi.data.loader import AspectRatioDataLoader, BucketBatchSampler, RealTimeDataset

def _build_loader(tmp_path, n_samples=4, **kw):
    items = [valid_item(tmp_path, i) for i in range(n_samples)]
    meta = write_meta(tmp_path, items)
    params = dict(
        data_paths=[meta],
        batch_size=2,
        num_workers=0,
        num_aspect_buckets=1,
        description_model="descA",
        audio_description_model="descA",
        num_frames=8,
        frame_tolerance=2,
        height=64,
        width=64,
    )
    params.update(kw)
    return AspectRatioDataLoader(**params)

def test_UT029_loader_yields_collated_batches(tmp_path):
    loader = _build_loader(tmp_path)
    batch = next(iter(loader))
    assert batch["video"].shape == (2, 3, 8, 64, 64)
    assert batch["audio"].dim() == 2
    assert batch["audio"].shape[0] == 2
    assert len(batch["text"]) == 2
    assert isinstance(loader.dataset, RealTimeDataset)
    assert isinstance(loader.batch_sampler, BucketBatchSampler)

def test_UT030_batch_audio_padded_to_max_in_batch(tmp_path):
    items = [
        valid_item(tmp_path, 0, audio_secs=0.5),
        valid_item(tmp_path, 1, audio_secs=1.5),
    ]
    meta = write_meta(tmp_path, items)
    loader = AspectRatioDataLoader(
        data_paths=[meta], batch_size=2, num_workers=0, num_aspect_buckets=1,
        description_model="descA", audio_description_model="descA",
        num_frames=8, frame_tolerance=2, height=64, width=64,
    )
    batch = next(iter(loader))

    assert batch["audio"].shape == (2, 16000)
    short_row = batch["text"].index("sample 0 description")
    assert torch.all(batch["audio"][short_row, 8000:] == 0)

def test_UT031_mixed_modality_buckets_separated(tmp_path):
    items = [valid_item(tmp_path, i) for i in range(2)]
    items += [valid_item(tmp_path, 10 + i, audio_only=True, audio_secs=1.0) for i in range(2)]
    meta = write_meta(tmp_path, items)
    loader = AspectRatioDataLoader(
        data_paths=[meta], batch_size=2, num_workers=0, num_aspect_buckets=1,
        description_model="descA", audio_description_model="descA",
        num_frames=8, frame_tolerance=2, height=64, width=64,
    )
    ds = loader.get_dataset()
    names = ds.get_bucket_names()
    assert "audio_only" in names
    for batch in loader:
        types = set(batch["modality_type"])
        assert len(types) == 1
