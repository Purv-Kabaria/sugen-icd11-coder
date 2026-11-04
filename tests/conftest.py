from __future__ import annotations

import pytest

from icd_coder.catalogue import Catalogue, Category


def make_category(
    code: str,
    title: str,
    *,
    parent: str | None = None,
    leaf: bool = True,
    index_terms: tuple[str, ...] = (),
    chapter: str = "01",
) -> Category:
    return Category(
        code=code,
        title=title,
        chapter=chapter,
        is_leaf=leaf,
        is_residual=code[-1] in "YZ" and "." in code,
        depth=1 + len(code.partition(".")[2]),
        parent=parent,
        ancestors=(),
        definition=None,
        coding_note=None,
        index_terms=index_terms,
        inclusions=(),
        exclusions=(),
        translations={},
    )


@pytest.fixture
def catalogue() -> Catalogue:
    return Catalogue(
        [
            make_category("1F40", "Malaria due to Plasmodium falciparum", leaf=False),
            make_category(
                "1F40.0", "Plasmodium falciparum malaria with cerebral complications", parent="1F40"
            ),
            make_category(
                "1F40.Y", "Other severe or complicated Plasmodium falciparum malaria", parent="1F40"
            ),
            make_category(
                "1F40.Z",
                "Malaria due to Plasmodium falciparum, unspecified",
                parent="1F40",
                index_terms=("falciparum malaria",),
            ),
            make_category("1D80", "Mumps", leaf=False),
            make_category("1D80.0", "Mumps without complication", parent="1D80"),
            make_category("1D80.1", "Mumps orchitis", parent="1D80"),
            make_category("CA23", "Asthma", leaf=False, chapter="12"),
            make_category("CA23.3", "Unspecified asthma", parent="CA23", leaf=False, chapter="12"),
            make_category(
                "CA23.30", "Unspecified asthma with exacerbation", parent="CA23.3", chapter="12"
            ),
            make_category(
                "CA23.32", "Unspecified asthma, uncomplicated", parent="CA23.3", chapter="12"
            ),
            make_category("1B12", "Tuberculosis of other systems or organs", leaf=False),
            make_category("1B12.0", "Tuberculosis of the nervous system", parent="1B12"),
            make_category("1B12.1", "Tuberculosis of the digestive system", parent="1B12"),
            make_category("5A11", "Type 2 diabetes mellitus", chapter="05"),
        ]
    )
