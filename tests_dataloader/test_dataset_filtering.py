
import pytest

from conftest import dummy_item, make_dataset, write_meta

def test_UT008_valid_multimodal_sample_retained(tmp_path):
    items = [dummy_item(tmp_path, 0)]
    ds = make_dataset([write_meta(tmp_path, items)])
    assert ds.stats["retained_total"] == 1
    assert len(ds.metadata) == 1
    assert ds.metadata[0]["bucket"] in ds.bucket_to_indices

def test_UT009_missing_audio_path_filtered(tmp_path):
    items = [
        dummy_item(tmp_path, 0, drop_audio=True),
        dummy_item(tmp_path, 1, drop_audio_file=True),
        dummy_item(tmp_path, 2),
    ]
    ds = make_dataset([write_meta(tmp_path, items)])
    assert ds.stats["filtered_missing_audio"] == 2
    assert ds.stats["retained_total"] == 1

def test_UT010_missing_description_filtered(tmp_path):
    items = [dummy_item(tmp_path, 0, drop_desc=True), dummy_item(tmp_path, 1)]
    ds = make_dataset([write_meta(tmp_path, items)])
    assert ds.stats["filtered_missing_description"] == 1
    assert ds.stats["retained_total"] == 1

def test_UT011_multimodal_missing_video_file_filtered(tmp_path):
    items = [dummy_item(tmp_path, 0, drop_video_file=True), dummy_item(tmp_path, 1)]
    ds = make_dataset([write_meta(tmp_path, items)])
    assert ds.stats["filtered_missing_video"] == 1
    assert ds.stats["retained_total"] == 1

def test_UT012_frame_count_below_lower_boundary_filtered(tmp_path):
    items = [
        dummy_item(tmp_path, 0, frame_num=5),
        dummy_item(tmp_path, 1, frame_num=6),
        dummy_item(tmp_path, 2),
    ]
    ds = make_dataset([write_meta(tmp_path, items)])
    assert ds.stats["filtered_frame_tolerance"] == 1
    assert ds.stats["retained_total"] == 2

def test_UT013_frame_count_above_upper_boundary_filtered(tmp_path):
    items = [
        dummy_item(tmp_path, 0, frame_num=11),
        dummy_item(tmp_path, 1, frame_num=10),
        dummy_item(tmp_path, 2),
    ]
    ds = make_dataset([write_meta(tmp_path, items)])
    assert ds.stats["filtered_frame_tolerance"] == 1
    assert ds.stats["retained_total"] == 2

def test_UT014_audio_only_duration_boundary(tmp_path):
    items = [
        dummy_item(tmp_path, 0, mode="audio_only", duration=20.0),
        dummy_item(tmp_path, 1, mode="audio_only", duration=20.1),
        dummy_item(tmp_path, 2),
    ]
    ds = make_dataset([write_meta(tmp_path, items)], max_audio_duration=20.0)
    assert ds.stats["filtered_audio_duration"] == 1
    assert ds.stats["retained_total"] == 2

def test_UT015_no_valid_samples_raises(tmp_path):
    items = [dummy_item(tmp_path, 0, drop_audio=True)]
    with pytest.raises(ValueError, match="No valid samples"):
        make_dataset([write_meta(tmp_path, items)])

    with pytest.raises(ValueError, match="No valid samples"):
        make_dataset([str(tmp_path / "not_exist.json")])
