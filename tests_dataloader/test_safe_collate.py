
import pytest
import torch

from ovi.data.loader import safe_collate

def test_UT001_equal_length_audio_collated():
    batch = [{"audio": torch.ones(100)}, {"audio": torch.zeros(100)}]
    out = safe_collate(batch)
    assert out["audio"].shape == (2, 100)
    assert torch.equal(out["audio"][0], torch.ones(100))
    assert torch.equal(out["audio"][1], torch.zeros(100))

def test_UT002_variable_length_audio_padded_with_zeros():
    batch = [{"audio": torch.ones(100)}, {"audio": torch.ones(60)}]
    out = safe_collate(batch)
    assert out["audio"].shape == (2, 100)
    assert torch.equal(out["audio"][1, :60], torch.ones(60))
    assert torch.equal(out["audio"][1, 60:], torch.zeros(40))

def test_UT003_non_tensor_values_pass_through():
    batch = [{"text": "hello", "audio": torch.ones(3)}, {"text": "world", "audio": torch.ones(3)}]
    out = safe_collate(batch)
    assert out["text"] == ["hello", "world"]
    assert out["audio"].shape == (2, 3)

def test_UT004_mixed_keys_warns_then_raises(capsys):
    batch = [{"audio": torch.ones(3)}, {"text": "x"}]
    with pytest.raises(KeyError):
        safe_collate(batch)
    captured = capsys.readouterr()
    assert "different keys" in captured.out
