
from ovi.data.loader import get_aspect_buckets

def test_UT005_empty_input_returns_empty():
    names, r2b, dims = get_aspect_buckets([], 640 * 640, 4)
    assert names == [] and r2b == {} and dims == {}

def test_UT006_bucket_count_and_mapping_consistency():
    ratios = [0.5, 0.8, 1.0, 1.25, 2.0]
    names, r2b, dims = get_aspect_buckets(ratios, 640 * 640, 4)
    assert len(names) == 4
    assert set(dims.keys()) == set(names)
    for r in ratios:
        assert r in r2b
        assert r2b[r] in dims

def test_UT007_dimensions_aligned_to_64():
    names, _, dims = get_aspect_buckets([0.3, 0.7, 1.0, 1.5, 2.5], 720 * 720, 5)
    for name in names:
        h, w = dims[name]
        assert h % 64 == 0 and w % 64 == 0
        assert h > 0 and w > 0
