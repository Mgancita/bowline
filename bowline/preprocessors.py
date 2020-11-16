"""Preprocessors within the bowline package.

AUTHORS: Marco Gancitano
"""

from typing import Callable, Dict, List, Optional

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from .utils import detect_series_type


class StandardPreprocessor:
    """Preprocess data.

    This class is a configurable tool which allows to pick and chose what operations should be
    applied to your data. The available tools are Imputing, Scaling, label-encoing, Multicolinearity
    check, and one-hot encoding. In addition, there's an experimental method to detect the data
    type of the variable (numeric, categoric, binary, id) so you don't have to.

    Args:
        data (pd.DataFrame): DataFrame to preprocess for downstream applications.
        numeric_features (List[str], optional): List of numeric columns within 'data'.
                Defaults to [].
        categoric_features (List[str], optional): List of categoric columns within 'data'. These
                are columns which have more than 2 options. Defaults to [].
        binary_features (List[str], optional): List of binary columns within 'data'. These are
                columns which have 2 options. Defaults to [].
        auto_detect_variable (bool, optional): Whether to use the experimental auto variable
                detection. If set to 'True' all feature lists will be ignored. Defaults to False.

    Raises:
        TypeError: If 'data' isn't a pandas.DataFrame.
        TypeError: If 'auto_detect_variable' isn't a boolean.
        TypeError: If any feature list is not a list.
        ValueError: If any value in a feature list is not a column in 'data'.

    """

    def __init__(
        self,
        data: pd.DataFrame,
        numeric_features: List[str] = [],
        categoric_features: List[str] = [],
        binary_features: List[str] = [],
        auto_detect_variable: bool = False,
    ) -> None:
        """Instantiate class."""
        # Check data type
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"'data' must be of type pandas.DataFrame. {type(data)} given.")

        # Check auto_detect_variable type
        if not isinstance(auto_detect_variable, bool):
            raise TypeError(
                f"'auto_detect_variable' must be a boolean. {type(auto_detect_variable)} given."
            )

        # Set data
        self.data = data

        # Set feature types automatically
        if auto_detect_variable:
            self._detect_column_types()
        else:
            if numeric_features + categoric_features + binary_features == []:
                raise ValueError("All feature lists empty. Must supply at least 1 feature.")
            self._manually_set_column_types(numeric_features, categoric_features, binary_features)

    def process(
        self,
        target: str,
        train_test_splitter: Optional[Callable[..., List[pd.DataFrame]]] = train_test_split,
        test_size: float = 0.25,
        imputer: Optional[BaseEstimator] = SimpleImputer(),
        scaler: Optional[BaseEstimator] = StandardScaler(),
        label_encoder: Optional[BaseEstimator] = LabelEncoder(),
        one_hot_encoder: Optional[BaseEstimator] = OneHotEncoder(),
        remove_nans: bool = False,
        scale_target: bool = True,
        random_state: Optional[int] = None,
    ) -> List[pd.DataFrame]:
        """Process data attribute for downstream applications.

        TODO: Add over/under sampling

        Args:
            target (str): Column name of the target variable you plan to predict downstream.
            train_test_splitter (Optional[Callable[..., List[pd.DataFrame]]], optional): Function
                    for splitting the final data. Set to None to skip train-test splitting.
                    Defaults to train_test_split().
            test_size (float): Percent of the data that should go to the testing set.
                    Defaults to 0.25.
            imputer (Optional[BaseEstimator], optional): Class instance of an imputer, must have a
                    valid 'fit_transform' method. Set to None to skip imputing.
                    Defaults to SimpleImputer().
            scaler (Optional[BaseEstimator], optional): Class instance of a scaler, must have a
                    valid 'fit_transform' method. Set to None to skip scaling.
                    Defaults to StandardScaler().
            label_encoder (Optional[BaseEstimator], optional): Class instance of a label encoder,
                    must have a valid 'fit_transform' method. Set to None to skip label encoding.
                    Defaults to LabelEncoder().
            one_hot_encoder (Optional[BaseEstimator], optional): Class instance of a one-hot
                    encoder, must have a valid 'fit_transform' method. Set to None to skip one-hot
                    encoding. Defaults to OneHotEncoder().
            remove_nans (bool, optional): Whether to remove any found NaNs. Defaults to False.
            scale_target (bool, optional): Whether to scale the 'target' variable. Target data is
                    typically not scaled for non-parametric models. Defaults to True.
            random_state (Optional[int], optional): A seed for random processes to make them
                    reproducible. Defaults to None.

        Raises:
            ValueError: If 'target' is not within a the feature lists.
            ValueError: If there are NaNs in 'data' and 'remove_nans' isn't True.

        Returns:
            List[pd.DataFrame]: List of processed data. Length 2 (X and y) if splitter is
                    None. Length 4 (X_train, X_test, y_train, y_test) if splitter isn't None.

        """
        if target not in self.features_to_check:
            raise ValueError(
                "'target' must be in 'numeric_features', 'categoric_features', or "
                "'binary_features'."
            )

        self.processed_data = self.data

        if imputer:
            self._impute(imputer, target)

        self._check_columns_for_nans(remove_nans)

        if label_encoder:
            self._label_encode(label_encoder)

        if one_hot_encoder:
            self._one_hot_encode(one_hot_encoder, target)

        if scaler:
            self._scale_data(scaler, target, scale_target)

        return self._split_data(train_test_splitter, test_size, target, random_state)

    def _check_columns_for_nans(self, remove_nans: bool) -> None:
        """Check if columns have NaNs and remove them if requested.

        Args:
            remove_nans (bool): Whether to remove found NaNs or not.

        Raises:
            ValueError: If NaNs are found and aren't requested to be removed.

        """
        columns_with_a_nan = self._detect_columns_with_nan()
        if columns_with_a_nan != []:
            if remove_nans:
                self.processed_data.dropna(subset=columns_with_a_nan, inplace=True)
            else:
                raise ValueError(
                    f"NaNs within columns {columns_with_a_nan}. Can either remove manually, use "
                    "'imputer' if the column is numeric (except for 'target'), or set "
                    "'remove_nans' to True."
                )

    def _detect_column_types(self) -> None:
        """Detect the type of each column in a given DataFrame."""
        series_of_types = self.data.apply(detect_series_type)
        variable_name_series = series_of_types.index

        self.numeric_features = list(variable_name_series[series_of_types == "number"])
        self.categoric_features = list(variable_name_series[series_of_types == "category"])
        self.binary_features = list(variable_name_series[series_of_types == "binary"])

    def _detect_columns_with_nan(self) -> List[str]:
        """Detect the columns which contain NaNs within 'processed_data'.

        Returns:
            List[str]: Columns which contain NaNs.

        """
        null_in_columns = self.processed_data.loc[:, self.features_to_check].isnull().apply(any)
        return list(self.processed_data.loc[:, self.features_to_check].columns[null_in_columns])

    def _impute(self, imputer: BaseEstimator, target: str) -> None:
        """Impute any missing data within 'numeric_features'.

        This method skips imputing the target variable.

        Args:
            imputer (BaseEstimator): Class instance to impute the data. Must have valid
                    'fit_transform' method.
            target (str): Column name for the target variable.

        """
        numeric_features_wo_target = list(set(self.numeric_features) - set([target]))
        if numeric_features_wo_target:
            self.processed_data.loc[:, numeric_features_wo_target] = imputer.fit_transform(
                self.processed_data.loc[:, numeric_features_wo_target]
            )

    def _label_encode(self, encoder: BaseEstimator) -> None:
        """Encode binary features to a numeric series (0s and 1s).

        Args:
            encoder (BaseEstimator): Class instance to encode the data. Must have valid
                    'fit_transform' method.

        """
        if self.binary_features:
            self.processed_data.loc[:, self.binary_features] = self.processed_data.loc[
                :, self.binary_features
            ].apply(encoder.fit_transform)

    def _manually_set_column_types(
        self, numeric_features: List[str], categoric_features: List[str], binary_features: List[str]
    ) -> None:
        """Manually set column types.

        Args:
            numeric_features (List[str]): List of numeric columns names.
            categoric_features (List[str]): List of categoric columns names.
            binary_features (List[str]): List of binary columns names.

        Raises:
            TypeError: If any feature list is not of type List.
            ValueError: If any feature list contains values not in 'data.columns'.

        """
        data_column_names_set = set(self.data.columns)
        features_dict: Dict[str, List[str]] = {
            "numeric_features": numeric_features,
            "categorical_features": categoric_features,
            "binary_features": binary_features,
        }
        for arg_name, arg in features_dict.items():
            # Check feature list types
            if not isinstance(arg, List):
                raise TypeError(f"'{arg_name}' must be a boolean. {type(arg)} given.")
            # Check feature list values
            if not set(arg).issubset(data_column_names_set):
                raise ValueError(
                    f"'data' must contain all the columns in {arg_name}. Given columns not in "
                    "dataframe: {data_column_names_set - set(arg)}"
                )

        self.numeric_features = numeric_features
        self.categoric_features = categoric_features
        self.binary_features = binary_features

    def _one_hot_encode(self, encoder: BaseEstimator, target: str) -> None:
        """One-hot encode categoric features.

        Args:
            encoder (BaseEstimator): Class instance to encode the data. Must have valid
                    'fit_transform' method.
            target (str): Column name of target variable.

        """
        categoric_features_wo_target = list(set(self.categoric_features) - set([target]))
        for feature in categoric_features_wo_target:
            one_hot = pd.get_dummies(self.processed_data.loc[:, feature], prefix=feature)
            self.processed_data = self.processed_data.drop(feature, axis=1)
            self.processed_data = self.processed_data.join(one_hot)

    def _scale_data(self, scaler: BaseEstimator, target: str, scale_target: bool) -> None:
        """Scale numeric features.

        This method can either be used to scale the target variable or not.

        Args:
            scaler (BaseEstimator): Class instance to scale the data. Must have valid
                    'fit_transform' method.
            target (str): Column name of target variable.
            scale_target (bool): Whether to scale the target variable or not.

        """
        if scale_target:
            features_to_scale = self.numeric_features
        else:
            features_to_scale = list(set(self.numeric_features) - set([target]))

        if features_to_scale:
            self.processed_data.loc[:, features_to_scale] = scaler.fit_transform(
                self.processed_data.loc[:, features_to_scale]
            )

    def _split_data(
        self,
        splitter: Optional[Callable[..., List[pd.DataFrame]]],
        test_size: float,
        target: str,
        random_state: Optional[int],
    ) -> List[pd.DataFrame]:
        """Split the data.

        Args:
            splitter (Optional[Callable[..., List[pd.DataFrame]]]): Function to split the
                    processed data.
            test_size (float): Percent of the data that should go to the testing set.
            target (str): Column name of target variable.
            random_state (Optional[int]): A seed to make splitting reproducible.

        Returns:
            List[pd.DataFrame]: List of processed data. Length 2 (X and y) if splitter is
                    None. Length is determined by splitter if splitter isn't None.

        """
        x = self.processed_data.drop(target, axis=1)
        y = self.processed_data.loc[:, target]
        if splitter:
            return splitter(x, y, random_state=random_state, test_size=test_size)

        return [x, y]

    @property
    def features_to_check(self) -> List[str]:
        """List of features to use on most aggregate functions.

        Returns:
            List[str]: Concatenation of numeric_features, categoric_features, and binary_features.

        """
        return self.numeric_features + self.categoric_features + self.binary_features
