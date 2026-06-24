import numpy as np
from itertools import combinations


def sample_random(dataset, n_samples, seed=None):
    """Simple random sampler."""

    if "data" not in dataset or "target" not in dataset:
        raise ValueError("dataset must contain 'data' and 'target' keys")

    total = len(dataset["data"])
    if n_samples > total:
        raise ValueError(f"Requested {n_samples} samples, but dataset has {total}")

    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=n_samples, replace=False)

    return _package(dataset, indices)


def sample_twise(dataset, t=2, seed=None):
    """Greedy t-wise sampler."""

    if "data" not in dataset or "target" not in dataset:
        raise ValueError("dataset must contain 'data' and 'target' keys")

    data = dataset["data"]
    n_rows, n_features = data.shape
    if t < 1 or t > n_features:
        raise ValueError(f"t must be between 1 and {n_features}")
    if n_rows == 0:
        raise ValueError("dataset must contain at least one sample")

    rng = np.random.default_rng(seed)

    # Fast path for t == 1 (cover each feature/value at least once).
    if t == 1:
        perm = rng.permutation(n_rows)
        first_hits = []
        for feat in range(n_features):
            _, first_rel = np.unique(data[perm, feat], return_index=True)
            first_hits.append(perm[first_rel])
        selected = np.unique(np.concatenate(first_hits))
        rng.shuffle(selected)
        return _package(dataset, selected.astype(int, copy=False))

    # Precompute unique values for each feature as Python lists
    feature_domains = [np.unique(data[:, f]).tolist() for f in range(n_features)]

    selected_mask = np.zeros(n_rows, dtype=bool)

    for cols in combinations(range(n_features), t):
        slice_ = data[:, cols]

        # Build index: value_tuple -> row indices, and row -> tuple
        tuple_to_rows = {}
        row_tuple = []  # row_tuple[i] = the value tuple for row i
        for row_idx, row in enumerate(slice_):
            key = tuple(row.tolist())
            tuple_to_rows.setdefault(key, []).append(row_idx)
            row_tuple.append(key)

        # Tuples already covered by selected rows
        covered = {row_tuple[i] for i in np.flatnonzero(selected_mask)}
        uncovered = set(tuple_to_rows.keys()) - covered

        # Greedy: pick rows covering uncovered tuples
        while uncovered:
            # Find candidate rows (those with uncovered tuples)
            candidates = [i for i in range(n_rows) 
                          if not selected_mask[i] and row_tuple[i] in uncovered]
            if not candidates:
                break
            selected_mask[rng.choice(candidates)] = True
            covered = {row_tuple[i] for i in np.flatnonzero(selected_mask)}
            uncovered -= covered

    return _package(dataset, np.flatnonzero(selected_mask))


def _package(dataset, indices):
    """Helper to slice dataset and keep metadata and indices."""

    sampled = {"data": dataset["data"][indices], "target": dataset["target"][indices]}
    sampled["indices"] = indices
    if "feature_names" in dataset:
        sampled["feature_names"] = dataset["feature_names"]
    if "target_name" in dataset:
        sampled["target_name"] = dataset["target_name"]
    return sampled
