from __future__ import annotations


def test_normalize_text_collapses_line_endings_and_strips_nuls() -> None:
    from src.data.split_assign import normalize_text

    noisy = "alpha\r\nbeta\rgamma\x00"
    assert normalize_text(noisy) == "alpha\nbeta\ngamma"


def test_assign_split_is_stable_for_equivalent_normalized_text() -> None:
    from src.data.split_assign import assign_split, normalize_text

    text_a = normalize_text("stable\r\ntext")
    text_b = normalize_text("stable\ntext\x00")
    split_a = assign_split(text_a, modulus=1000, ranges={"train": [40, 999], "val": [20, 39], "test": [0, 19]})
    split_b = assign_split(text_b, modulus=1000, ranges={"train": [40, 999], "val": [20, 39], "test": [0, 19]})
    assert split_a == split_b
    assert split_a in {"train", "val", "test"}
