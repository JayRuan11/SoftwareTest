
import pytest

from conftest import FakeDataset, FakeSampler
from ovi.data.loader import BucketBatchSampler

def _make_sampler(bucket_to_indices, batch_size, *, drop_last=True, seed=1234,
                  in_order=False, num_replicas=1, rank=0, length=None):
    ds = FakeDataset(bucket_to_indices, length=length)
    names = list(bucket_to_indices.keys())
    return BucketBatchSampler(
        sampler=FakeSampler(num_replicas, rank),
        dataset=ds,
        batch_size=batch_size,
        bucket_names=names,
        drop_last=drop_last,
        seed=seed,
        in_order=in_order,
    )

def test_UT025_batches_stay_within_one_bucket():
    b2i = {"b1": [0, 1, 2, 3], "b2": [4, 5, 6, 7]}
    s = _make_sampler(b2i, 2)
    for batch in s:
        home = "b1" if batch[0] < 4 else "b2"
        pool = set(b2i[home])
        assert set(batch) <= pool

def test_UT026_drop_last_discards_partial_batch():
    b2i = {"b1": [0, 1, 2, 3, 4]}
    s = _make_sampler(b2i, 2, drop_last=True, in_order=True)
    batches = list(s)
    assert len(batches) == 2
    assert all(len(b) == 2 for b in batches)

def test_UT027_distributed_ranks_get_disjoint_batches():
    b2i = {"b1": list(range(8)), "b2": list(range(8, 16))}
    r0 = list(_make_sampler(b2i, 2, seed=7, num_replicas=2, rank=0))
    r1 = list(_make_sampler(b2i, 2, seed=7, num_replicas=2, rank=1))
    assert len(r0) == len(r1)
    f0 = {frozenset(b) for b in r0}
    f1 = {frozenset(b) for b in r1}
    assert f0.isdisjoint(f1)

@pytest.mark.xfail(
    strict=True, reason="BUG-002: drop_last=False 时 __len__ 仍用整除截断，尾批被 __iter__ 丢弃"
)
def test_UT028_drop_last_false_keeps_partial_batch():
    b2i = {"b1": [0, 1, 2]}
    s = _make_sampler(b2i, 2, drop_last=False, in_order=True)
    batches = list(s)
    assert len(batches) == 2
    assert batches[-1] == [2]
