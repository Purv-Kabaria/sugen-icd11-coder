"""A learned ranker over the candidate codes of a note.

An ensemble of LightGBM LambdaRank models, trained by `tools/train_reranker.py` on
synthetic notes, scores the candidates from `FeatureExtractor`; the ensemble score
is the mean of its members. Without model files the offline coder keeps its rule-based
ranking, so the reranker can only add accuracy, never availability.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import lightgbm
import numpy as np
import numpy.typing as npt

from icd_coder.features import FEATURES, FeatureExtractor
from icd_coder.lexical import ScoredCode

LOG = logging.getLogger(__name__)
MODEL_GLOB = "reranker-*.txt"


class Reranker:
    def __init__(self, models: Sequence[lightgbm.Booster], extractor: FeatureExtractor) -> None:
        self._models = list(models)
        self._extractor = extractor

    @classmethod
    def load(cls, model_dir: Path, extractor: FeatureExtractor) -> Reranker | None:
        """The ensemble in `model_dir`, or None when there is none or it does not match."""
        paths = sorted(model_dir.glob(MODEL_GLOB))
        if not paths:
            return None
        try:
            models = [lightgbm.Booster(model_file=str(path)) for path in paths]
        except (OSError, lightgbm.basic.LightGBMError) as error:
            LOG.warning("cannot load reranker from %s (%s); using rules", model_dir, error)
            return None
        if any(model.feature_name() != list(FEATURES) for model in models):
            LOG.warning("reranker features differ from this code's; using rules")
            return None
        LOG.info("loaded reranker ensemble of %d models", len(models))
        return cls(models, extractor)

    def rank(self, text: str) -> list[ScoredCode]:
        candidates = self._extractor.extract(text)
        if not candidates.codes:
            return []
        scores = ensemble_scores(self._models, candidates.features)
        order = np.argsort(-scores, kind="stable")
        return [ScoredCode(candidates.codes[index], float(scores[index])) for index in order]


def ensemble_scores(
    models: Sequence[lightgbm.Booster], features: npt.NDArray[np.float32]
) -> npt.NDArray[np.float64]:
    """The mean score of the ensemble's members for each candidate row."""
    return np.mean(
        [np.asarray(model.predict(features), dtype=np.float64) for model in models], axis=0
    )
