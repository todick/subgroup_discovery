import itertools
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, "./") 

from src.sampling import sample_random, sample_twise


def _twise_interactions(data, t):
    n_features = data.shape[1]
    idxs = range(n_features)
    tuples = []
    for row in data:
        row_ints = {tuple((i, row[i]) for i in combo) for combo in itertools.combinations(idxs, t)}
        tuples.append(row_ints)
    return set().union(*tuples)


def test_sample_random_size_and_seed():
    data = np.arange(20).reshape(10, 2)
    dataset = {"data": data, "target": np.zeros(10)}

    out = sample_random(dataset, 5, seed=123)
    assert out["data"].shape == (5, 2)
    assert out["target"].shape == (5,)

    # deterministic with same seed
    out2 = sample_random(dataset, 5, seed=123)
    np.testing.assert_array_equal(out["data"], out2["data"])
    np.testing.assert_array_equal(out["target"], out2["target"])


def test_sample_random_raises_on_oversample():
    data = np.arange(6).reshape(3, 2)
    dataset = {"data": data, "target": np.zeros(3)}
    with pytest.raises(ValueError):
        sample_random(dataset, 4)


def test_sample_twise_invalid_t():
    data = np.arange(6).reshape(3, 2)
    dataset = {"data": data, "target": np.zeros(3)}
    with pytest.raises(ValueError):
        sample_twise(dataset, t=0)
    with pytest.raises(ValueError):
        sample_twise(dataset, t=3)


def test_sample_twise_covers_t1():
    """Test t-wise coverage for t=1 with multiple random large datasets."""
    np.random.seed(42)
    for trial in range(5):
        # Generate random dataset: 100-200 rows, 5-10 features, values 0-4
        n_rows = np.random.randint(100, 201)
        n_features = np.random.randint(5, 11)
        n_values = np.random.randint(3, 6)
        data = np.random.randint(0, n_values, size=(n_rows, n_features))
        dataset = {"data": data, "target": np.random.rand(n_rows)}

        out = sample_twise(dataset, t=1, seed=42 + trial)
        # every feature/value pair present in full dataset must appear in sample
        full_ints = _twise_interactions(data, 1)
        sample_ints = _twise_interactions(out["data"], 1)
        assert full_ints.issubset(sample_ints), f"Trial {trial}: Not all 1-wise interactions covered"
        assert len(out["data"]) <= len(data), f"Trial {trial}: Sample larger than original"


def test_sample_twise_covers_t2():
    """Test t-wise coverage for t=2 with multiple random large datasets."""
    np.random.seed(7)
    for trial in range(5):
        # Generate random dataset: 80-150 rows, 6-10 features, values 0-3
        n_rows = np.random.randint(80, 151)
        n_features = np.random.randint(6, 11)
        n_values = np.random.randint(2, 5)
        data = np.random.randint(0, n_values, size=(n_rows, n_features))
        dataset = {"data": data, "target": np.random.rand(n_rows)}

        out = sample_twise(dataset, t=2, seed=7 + trial)
        full_ints = _twise_interactions(data, 2)
        sample_ints = _twise_interactions(out["data"], 2)
        assert full_ints.issubset(sample_ints), f"Trial {trial}: Not all 2-wise interactions covered"
        # selected rows should be fewer or equal to total
        assert len(out["data"]) <= len(data), f"Trial {trial}: Sample larger than original"

def test_sample_twise_deterministic():
    data = np.random.randint(0, 5, size=(100, 4))
    dataset = {"data": data, "target": np.zeros(len(data))}

    out1 = sample_twise(dataset, t=3, seed=99)
    out2 = sample_twise(dataset, t=3, seed=99)

    np.testing.assert_array_equal(out1["data"], out2["data"])
    np.testing.assert_array_equal(out1["target"], out2["target"])

def test_sample_twise_covers_t3():
    """Test t-wise coverage for t=3 with multiple random large datasets."""
    np.random.seed(21)
    for trial in range(5):
        # Generate random dataset: 50-100 rows, 5-8 features, values 0-2
        n_rows = np.random.randint(50, 101)
        n_features = np.random.randint(5, 9)
        n_values = np.random.randint(2, 4)
        data = np.random.randint(0, n_values, size=(n_rows, n_features))
        dataset = {"data": data, "target": np.random.rand(n_rows)}

        out = sample_twise(dataset, t=3, seed=21 + trial)
        full_ints = _twise_interactions(data, 3)
        print(full_ints)
        sample_ints = _twise_interactions(out["data"], 3)
        assert full_ints.issubset(sample_ints), f"Trial {trial}: Not all 3-wise interactions covered"
        # selected rows should be fewer or equal to total
        assert len(out["data"]) <= len(data), f"Trial {trial}: Sample larger than original"