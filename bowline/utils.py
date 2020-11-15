"""Utility functions for processors."""

import pandas as pd


def detect_series_type(series: pd.Series) -> str:
    """Detect the data type of a pandas series.

    This function attempts to detect the type of a given series between four different values
    (categorical, binary, id, and numerical). This is done using a mix of inference from pandas
    as well as some heuristics rules.

    NOTE: This method is experimental and has no claim to high accuracy so use with caution.

    Args:
        series (pd.Series): Pandas data series to check.

    Returns:
        str: Data type of the given series.

    """
    series = series.infer_objects()
    dtype = str(series.dtype)

    unique_values = series.nunique()

    # If a binary variable
    if unique_values == 2:
        return "binary"

    # If an ID column
    if len(series) == unique_values:
        return "id"

    if not any(word in dtype for word in ["float", "int"]):
        return "category"

    if unique_values / len(series) > 0.01:
        return "number"

    # Check if the mode appears more than 20% of all observations
    percent_of_all_obs = series.value_counts(normalize=True)
    if percent_of_all_obs.max() > 0.20:
        # Make sure it wasn't a large number of zeros causing the 20%
        if percent_of_all_obs.idxmax() != 0:
            return "category"

        # If it was mostly 0s, remove them and check again
        percent_of_all_obs_wo_zero = series[series != 0].value_counts(normalize=True)
        if percent_of_all_obs_wo_zero.max() > 0.20:
            return "category"

    # If nothing is detected, select 'number'
    return "number"
