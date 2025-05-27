
#THIS CODE OR THE BIGGER PART OF IT CAME FROM THE REPOSITORY  https://github.com/HLasse/data-centric-synthetic-data

import pandas as pd
from typing import Optional
import numpy as np
from pydantic import BaseModel, Extra
import cleanlab
from cleanlab.regression.learn import CleanLearning
from sklearn.linear_model import LinearRegression
from typing import List, Literal, Optional, Union
from torch import nn

from classes import *
from dataiqreg import *


def add_target_col(X: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    df = X.copy()
    df["target"] = target
    return df


def get_cleanlab_label_issue_df(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    model =  LinearRegression()
    cl = CleanLearning(model)
    _ = cl.fit(X, y)
    label_issues = cl.get_label_issues()
    if label_issues is None:
        raise ValueError("Model not fit?")
    return label_issues

def get_sculpted_data(
    X: pd.DataFrame,
    y: pd.Series,
    stratified_indices: StratifiedIndices,
    percentile_threshold: Optional[int],
    data_centric_threshold: Optional[float],
):
    """Get the stratified samples from the dataset."""
    X_easy = X.iloc[stratified_indices.easy]
    y_easy = y.iloc[stratified_indices.easy]
    if stratified_indices.ambiguous is not None:
        X_ambigious = X.loc[stratified_indices.ambiguous]
        y_ambigious = y[stratified_indices.ambiguous]
    else:
        X_ambigious = None
        y_ambigious = None
    X_hard = X.iloc[stratified_indices.hard]
    y_hard = y.iloc[stratified_indices.hard]
    return SculptedData(
        X_easy=X_easy,
        X_ambiguous=X_ambigious,
        X_hard=X_hard,
        y_easy=y_easy,
        y_ambiguous=y_ambigious,
        y_hard=y_hard,
        indices=stratified_indices,
        percentile_threshold=percentile_threshold,
        data_centric_threshold=data_centric_threshold,
    )


def get_datacentric_segments_from_sculpted_data(
    sculpted_data: SculptedData,
    subsets: List[Literal["easy", "ambi", "hard"]],
) -> DataSegments:
    """Get the number of samples in each subset of the sculpted data.
    The number of samples is set to 0 if the subset is not in `subsets`.

    Args:
        sculpted_data (SculptedData): The sculpted data.
        subsets (List[Literal["easy", "ambi", "hard"]]): The subsets of the
            sculpted data to get the number of samples for.

    Returns:
        DataIQSegments: The number of samples in each subset of the sculpted data.
    """
    n_easy = sculpted_data.indices.easy.size if "easy" in subsets else 0
    if sculpted_data.indices.ambiguous is None:
        n_ambi = 0
    else:
        n_ambi = sculpted_data.indices.ambiguous.size if "ambi" in subsets else 0
    n_hard = sculpted_data.indices.hard.size if "hard" in subsets else 0
    return DataSegments(
        n_easy=n_easy,
        n_ambiguous=n_ambi,
        n_hard=n_hard,
    )


def sculpt_with_cleanlab(
    X: pd.DataFrame,
    y: pd.Series,
    data_centric_threshold: Optional[float],
) :
    """Sculpt the given data with Cleanlab. Uses the Cleanlab model to stratify the
    data into easy and hard examples. Note that cleanlab does not have a notion of
    ambiguous examples.

    Args:
        X (pd.DataFrame): The training features.
        y (pd.Series): The training labels.
        data_centric_threshold (float): The data centric threshold for making the
            easy/ambiguous/hard split.
    """

    label_issues = get_cleanlab_label_issue_df(X=X, y=y)
    if data_centric_threshold is None:
        easy_indices = np.where(~label_issues["is_label_issue"])[0]
        hard_indices = np.where(label_issues["is_label_issue"])[0]
    else:
        easy_indices = np.where(label_issues["label_quality"] > data_centric_threshold)[
            0
        ]
        hard_indices = np.where(
            label_issues["label_quality"] <= data_centric_threshold,
        )[0]

    return get_sculpted_data(
        X=X,
        y=y,
        stratified_indices=StratifiedIndices(
            easy=easy_indices,
            ambiguous=None,
            hard=hard_indices,
        ),
        percentile_threshold=None,
        data_centric_threshold=data_centric_threshold,
    )








def sculpt_with_dataiq(
    X: pd.DataFrame,
    y: pd.Series,
    percentile_threshold: Optional[int],
    data_centric_threshold: Optional[float],
    method: Literal["dataiq", "datamaps"],
    device='cuda',
    epochs=10
) -> SculptedData:
    """Sculpt the given data with DataIQ. Fits an XGBoost model and uses the model
    with DataIQ to stratify the data into easy, ambiguous, and hard examples.

    Args:
        X (pd.DataFrame): The training features.
        y (pd.Series): The training labels.
        percentile_threshold (int): The percentile threshold for aleatoric uncertainty
            used for making the easy/ambiguous/hard split.
        data_centric_threshold (float): The data centric threshold for making the
            easy/ambiguous/hard split.
        method (Literal["dataiq", "datamaps"]): The method to use for sculpting.

        Returns:
            SculptedData: The sculpted data.
    """
    linear_reg = LinearRegression()
    dataiq = fit_dataiq_sk(X=X, y=y,device=device,epochs=epochs, clf=linear_reg)
    return stratify_and_sculpt_with_dataiq(
        X=X,
        y=y,
        dataiq=dataiq,
        percentile_threshold=percentile_threshold,
        data_centric_threshold=data_centric_threshold,
        method=method,
    )



def stratify_and_sculpt_with_dataiq(
    X: pd.DataFrame,
    y: pd.Series,
    dataiq,
    percentile_threshold: Optional[int],
    data_centric_threshold: Optional[float],
    method: Literal["dataiq", "datamaps"],
) -> SculptedData:
    """Stratify the dataset into easy, ambiguous and hard examples
    based on the confidence and aleatoric uncertainty

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Labels
        dataiq (Union[DataIQGradientBoosting, DataIQBatch]): DataIQ object
        percentile_threshold (int): Percentile threshold for aleatoric uncertainty
            to use for stratifying
        data_centric_threshold (float): Data centric threshold for aleatoric
            or epistemic uncertainty to use for stratifying
        method (Literal["dataiq", "datamaps"]): Method to use for data centric
            uncertainty

    Returns:
        SculptedData: Stratified dataset
    """
    aleatoric_uncertainty,conficence = dataiq
    if method == "dataiq":
        
        data_uncertainty = aleatoric_uncertainty  # type: ignore
    #elif method == "datamaps":
        #data_uncertainty = dataiq.epistemic_variability  # type: ignore
    else:
        raise ValueError(f"Unknown data centric method {method}")

    stratified_indices = stratify_samples(
        label_probabilities=conficence,  # type: ignore
        data_uncertainty=data_uncertainty,
        percentile_threshold=percentile_threshold,
        data_centric_threshold=data_centric_threshold,
    )
    return get_sculpted_data(
        X=X,
        y=y,
        stratified_indices=stratified_indices,
        percentile_threshold=percentile_threshold,
        data_centric_threshold=data_centric_threshold,
    )
def stratify_samples(
    label_probabilities: np.ndarray,
    data_uncertainty: np.ndarray,
    percentile_threshold: Optional[int],
    data_centric_threshold: Optional[float],
) -> StratifiedIndices:
    """Stratify samples into easy, ambigious, and hard examples based on the
    confidence and aleatoric uncertainty of the model/data. The percentile
    threshold determines how many samples are in each category.

    Args:
        label_probabilities (np.ndarray): Mean probabilities of the ground truth
            label for each sample.
        data_uncertainty (np.ndarray): Aleatoric uncertainty epistemic variability
            of the data for each sample.
        percentile_threshold (int): Percentile threshold to use for stratifying
            the samples. If `percentile_threshold` is not None, then the
            `data_centric_threshold` is ignored.
        data_centric_threshold (float): Data centric threshold to use for
            stratifying the samples. If `data_centric_threshold` is not None, then
            the `percentile_threshold` is ignored.

    Returns:
        StratifiedIndices: Indices of the stratified samples
    """
    if percentile_threshold is not None and data_centric_threshold is not None:
        raise ValueError(
            "Must provide only one of percentile_threshold or data_centric_threshold",
        )

    if percentile_threshold is not None:
        data_uncertainty_below_threshold = data_uncertainty <= np.percentile(
            data_uncertainty,
            percentile_threshold,
        )
    elif data_centric_threshold is not None:
        data_uncertainty_below_threshold = data_uncertainty <= data_centric_threshold
    else:
        raise ValueError(
            "Must provide either percentile_threshold or data_centric_threshold",
        )
    confidence_treshold_width = 0.5
    confidence_threshold_lower = 0.5 - confidence_treshold_width / 2
    confidence_threshold_upper = 0.5 + confidence_treshold_width / 2

    hard_idx = np.where(
        (label_probabilities <= confidence_threshold_lower)
        & (data_uncertainty_below_threshold),
    )[0]
    easy_idx = np.where(
        (label_probabilities >= confidence_threshold_upper)
        & (data_uncertainty_below_threshold),
    )[0]

    # ambigious ids are those not in hard_idx or easy_idx
    all_ids = np.arange(len(label_probabilities))
    ambigious_idx = np.setdiff1d(all_ids, np.concatenate((hard_idx, easy_idx)))
    return StratifiedIndices(easy=easy_idx, ambiguous=ambigious_idx, hard=hard_idx)