from __future__ import annotations

import pytest

from icd_coder.catalogue import Catalogue, normalise_code


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" 1f40.z ", "1F40.Z"),
        ("NC72.2&XK8G", "NC72.2"),
        ("1F40.0/XS25", "1F40.0"),
        ("", None),
        (42, None),
        (None, None),
    ],
)
def test_normalise_code_matches_the_scorer(raw: object, expected: str | None) -> None:
    assert normalise_code(raw) == expected


def test_validate_rejects_codes_outside_the_release(catalogue: Catalogue) -> None:
    assert catalogue.validate("5a11") == "5A11"
    assert catalogue.validate("5A11&XA123") == "5A11"
    assert catalogue.validate("5A99") is None


def test_default_child_prefers_the_unspecified_residual(catalogue: Catalogue) -> None:
    child = catalogue.default_child("1F40")
    assert child is not None
    assert child.code == "1F40.Z"


def test_default_child_falls_back_to_unspecified_title_then_uncomplicated(
    catalogue: Catalogue,
) -> None:
    unspecified = catalogue.default_child("CA23")
    uncomplicated = catalogue.default_child("1D80")
    assert unspecified is not None
    assert unspecified.code == "CA23.3"
    assert uncomplicated is not None
    assert uncomplicated.code == "1D80.0"


def test_default_child_is_none_without_a_convention(catalogue: Catalogue) -> None:
    assert catalogue.default_child("1B12") is None
    assert catalogue.default_child("5A11") is None


def test_most_specific_default_walks_down_until_no_rule_applies(catalogue: Catalogue) -> None:
    assert catalogue.most_specific_default("1F40") == "1F40.Z"
    assert catalogue.most_specific_default("1B12") == "1B12"
    assert catalogue.most_specific_default("5A11") == "5A11"


def test_hierarchy_navigation(catalogue: Catalogue) -> None:
    assert catalogue.stem("CA23.30") == "CA23"
    assert [category.code for category in catalogue.descendants("CA23")] == [
        "CA23.3",
        "CA23.30",
        "CA23.32",
    ]
    assert [category.code for category in catalogue.family("1D80.1")] == [
        "1D80",
        "1D80.0",
        "1D80.1",
    ]
