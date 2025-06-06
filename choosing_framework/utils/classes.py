#THIS CODE OR THE BIGGER PART OF IT CAME FROM THE REPOSITORY  https://github.com/HLasse/data-centric-synthetic-data

import pandas as pd 
import numpy as np
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Extra


class StratifiedIndices(BaseModel):
    easy: np.ndarray
    ambiguous: Optional[np.ndarray]
    hard: np.ndarray

    class Config:
        arbitrary_types_allowed = True





class DataSegments(BaseModel, extra=Extra.allow):
    n_easy: int
    n_ambiguous: int
    n_hard: int

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # calculate proportions
        total = sum(
            [
                self.n_easy,
                self.n_ambiguous,
                self.n_hard,
            ],
        )
        if total != 0:
            self.proportion_easy = self.n_easy / total
            if self.n_ambiguous != 0:
                self.proportion_ambiguous = self.n_ambiguous / total
            else:
                self.proportion_ambiguous = 0
            self.proportion_hard = self.n_hard / total


class AlphaPrecisionMetrics(BaseModel):
    delta_precision_alpha_OC: float
    delta_coverage_beta_OC: float
    authenticity_OC: float
    delta_precision_alpha_naive: float
    delta_coverage_beta_naive: float
    authenticity_naive: float



class SculptedData(BaseModel):
    X_easy: pd.DataFrame
    X_ambiguous: Optional[pd.DataFrame]
    X_hard: pd.DataFrame
    y_easy: pd.Series
    y_ambiguous: Optional[pd.Series]
    y_hard: pd.Series
    indices: StratifiedIndices
    percentile_threshold: Optional[int]
    data_centric_threshold: Optional[float]

    class Config:
        arbitrary_types_allowed = True


class ProcessedData(BaseModel):
    dataset: pd.DataFrame
    data_segments: DataSegments
    statistical_likeness: Optional[AlphaPrecisionMetrics]
    detection_auc: Optional[float]

    class Config:
        arbitrary_types_allowed = True

