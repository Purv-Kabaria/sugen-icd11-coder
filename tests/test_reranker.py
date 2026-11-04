from pathlib import Path

import lightgbm
import numpy as np
import pytest

from icd_coder.catalogue import Catalogue
from icd_coder.features import FEATURES, FeatureExtractor
from icd_coder.lexical import LabelIndex, catalogue_labels
from icd_coder.offline import OfflineCoder
from icd_coder.reranker import Reranker

MALARIA_NOTE = "S: fever x3/7.\nIx: BS for MPS positive.\nA: Uncomplicated falciparum malaria."


def train_toy_model(feature_names: list[str], relevant: str) -> lightgbm.Booster:
    """A ranker that has learned to put candidates with feature `relevant` set first."""
    rng = np.random.default_rng(0)
    features = rng.random((400, len(feature_names)))
    column = feature_names.index(relevant)
    features[:, column] = rng.integers(0, 2, 400)
    data = lightgbm.Dataset(
        features, label=features[:, column], group=[10] * 40, feature_name=feature_names
    )
    parameters = {"objective": "lambdarank", "min_data_in_leaf": 5, "verbose": -1}
    return lightgbm.train(parameters, data, num_boost_round=20)


@pytest.fixture
def extractor(catalogue: Catalogue) -> FeatureExtractor:
    return FeatureExtractor(catalogue, LabelIndex.build(catalogue_labels(catalogue)))


def test_missing_models_mean_no_reranker(tmp_path: Path, extractor: FeatureExtractor) -> None:
    assert Reranker.load(tmp_path, extractor) is None


def test_unreadable_models_mean_no_reranker(tmp_path: Path, extractor: FeatureExtractor) -> None:
    (tmp_path / "reranker-1.txt").write_text("not a model", encoding="utf-8")
    assert Reranker.load(tmp_path, extractor) is None


def test_models_with_other_features_are_rejected(
    tmp_path: Path, extractor: FeatureExtractor
) -> None:
    names = [f"other_{number}" for number in range(len(FEATURES))]
    train_toy_model(names, names[0]).save_model(str(tmp_path / "reranker-1.txt"))
    assert Reranker.load(tmp_path, extractor) is None


def test_reranker_orders_candidates_by_its_ensemble(
    tmp_path: Path, extractor: FeatureExtractor, catalogue: Catalogue
) -> None:
    for number in (1, 2):
        model = train_toy_model(list(FEATURES), "is_unspecified")
        model.save_model(str(tmp_path / f"reranker-{number}.txt"))
    reranker = Reranker.load(tmp_path, extractor)
    assert reranker is not None

    ranked = reranker.rank(MALARIA_NOTE)
    scores = [scored.score for scored in ranked]
    assert scores == sorted(scores, reverse=True)
    assert catalogue[ranked[0].code].is_unspecified


def test_offline_coder_prefers_the_reranker_and_fills_from_rules(
    tmp_path: Path, extractor: FeatureExtractor, catalogue: Catalogue
) -> None:
    train_toy_model(list(FEATURES), "is_unspecified").save_model(str(tmp_path / "reranker-1.txt"))
    index = LabelIndex.build(catalogue_labels(catalogue))
    coder = OfflineCoder(catalogue, index, Reranker.load(tmp_path, extractor))
    codes = coder.predict(MALARIA_NOTE)
    assert 1 <= len(codes) <= 3
    assert len(set(codes)) == len(codes)
    assert all(code in catalogue for code in codes)
    assert catalogue[codes[0]].is_unspecified


def test_reranker_answers_an_empty_note(tmp_path: Path, extractor: FeatureExtractor) -> None:
    train_toy_model(list(FEATURES), "is_leaf").save_model(str(tmp_path / "reranker-1.txt"))
    reranker = Reranker.load(tmp_path, extractor)
    assert reranker is not None
    assert reranker.rank("") == []
