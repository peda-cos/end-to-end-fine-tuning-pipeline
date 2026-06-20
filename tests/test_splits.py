from __future__ import annotations

from ftpipe.data.splits import split_records


def _records(count: int):
    return [
        {
            "id": f"r-{index}",
            "instruction": f"Ticket: unique body number {index} about order {1000 + index}",
            "content": f"unique body number {index} about order {1000 + index}",
        }
        for index in range(count)
    ]


def test_split_is_seeded_and_disjoint():
    first = split_records(_records(100), 0.1, seed=7)
    second = split_records(_records(100), 0.1, seed=7)
    assert [row["id"] for row in first[0]] == [row["id"] for row in second[0]]
    train_ids = {row["id"] for row in first[0]}
    val_ids = {row["id"] for row in first[1]}
    assert not (train_ids & val_ids)
    assert first[2]["train"] == 90
    assert first[2]["val"] == 10


def test_cross_split_duplicates_are_rejected():
    base = _records(30)
    records = []
    for row in base:
        records.append(row)
        records.append(
            {
                "id": row["id"] + "-copy",
                "instruction": row["instruction"] + " extra",
                "content": row["content"] + " extra",
            }
        )
    train, val, _ = split_records(records, 0.5, seed=5)
    from ftpipe.data.dedup import MinHasher

    hasher = MinHasher(ngram=3, num_hashes=16, bands=4, rows=4, seed=5)
    for val_row in val:
        for train_row in train:
            assert hasher.jaccard(hasher.signature(val_row["content"]), hasher.signature(train_row["content"])) < 0.95
    assert len(val) > 0
