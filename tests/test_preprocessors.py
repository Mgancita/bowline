"""Module to test bowline.preprocessors."""

from copy import deepcopy

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest

from bowline.preprocessors import StandardPreprocessor


@pytest.mark.parametrize(
    "data, numeric_features, categoric_features, binary_features, auto_detect_variable",
    [
        ({}, [], [], [], False),
        (pd.DataFrame(), [], [], [], "yes"),
    ],
)
def test_standard_preprocessor_bad_init(
    data, numeric_features, categoric_features, binary_features, auto_detect_variable
):
    """Test StandardPreprocessor.__init__ with bad arguments."""
    with pytest.raises(TypeError):
        StandardPreprocessor(
            data, numeric_features, categoric_features, binary_features, auto_detect_variable
        )


class TestStandardPreprocessor:
    """Test StandardPreprocessor."""

    def setup(self):
        """Set generic variables for test case."""
        self.data = pd.read_csv(
            "tests/test_files/standard_preprocessor/all_estimators_run/raw_data.csv"
        )
        self.preprocessor = StandardPreprocessor(
            data=self.data,
            numeric_features=["age", "capital-gain"],
            categoric_features=["workclass", "education", "marital-status", "occupation", "race"],
            binary_features=["sex", "Salary"],
        )

    def test_init(self):
        """Test __init__."""
        assert self.preprocessor.data.equals(self.data)
        assert self.preprocessor.numeric_features == ["age", "capital-gain"]
        assert self.preprocessor.categoric_features == [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "race",
        ]
        assert self.preprocessor.binary_features == ["sex", "Salary"]

    def test_init_w_non_list_features(self):
        """Test __init__."""
        with pytest.raises(TypeError):
            StandardPreprocessor(data=self.data, numeric_features="features")

    def test_init_w_empty_list_features(self):
        """Test __init__."""
        with pytest.raises(ValueError):
            StandardPreprocessor(data=self.data)

    def test_init_w_non_data_columns_feature(self):
        """Test __init__."""
        with pytest.raises(ValueError):
            StandardPreprocessor(data=self.data, numeric_features=["non-feature"])

    def test_init_w_auto_detection(self):
        """Test __init__ with auto detection."""
        self.preprocessor = StandardPreprocessor(data=self.data, auto_detect_variable=True)
        assert self.preprocessor.data.equals(self.data)
        assert self.preprocessor.numeric_features == ["age", "capital-gain"]
        assert self.preprocessor.categoric_features == [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "race",
        ]
        assert self.preprocessor.binary_features == ["sex", "Salary"]

    def test_process(self):
        """Test process."""
        x_train, x_test, y_train, y_test = self.preprocessor.process(
            "Salary", remove_nans=True, random_state=2020
        )

        x_train_expected = pd.read_csv(
            "tests/test_files/standard_preprocessor/all_estimators_run/x_train.csv"
        )
        x_test_expected = pd.read_csv(
            "tests/test_files/standard_preprocessor/all_estimators_run/x_test.csv"
        )
        y_train_expected = pd.read_csv(
            "tests/test_files/standard_preprocessor/all_estimators_run/y_train.csv", squeeze=True
        )
        y_test_expected = pd.read_csv(
            "tests/test_files/standard_preprocessor/all_estimators_run/y_test.csv", squeeze=True
        )

        assert_frame_equal(
            x_train.reset_index(drop=True).sort_index(axis=1),
            x_train_expected.sort_index(axis=1),
            check_dtype=False,
        )
        assert_frame_equal(
            x_test.reset_index(drop=True).sort_index(axis=1),
            x_test_expected.sort_index(axis=1),
            check_dtype=False,
        )
        assert_series_equal(y_train.reset_index(drop=True), y_train_expected, check_dtype=False)
        assert_series_equal(y_test.reset_index(drop=True), y_test_expected, check_dtype=False)

    def test_process_with_invalid_target(self):
        """Test process with target not in features_to_check."""
        with pytest.raises(ValueError):
            self.preprocessor.process("Fake target")

    def test_process_no_estimators(self):
        """Test process with no estimators given."""
        x, y = self.preprocessor.process(
            "Salary",
            train_test_splitter=None,
            imputer=None,
            scaler=None,
            label_encoder=None,
            one_hot_encode=False,
            remove_nans=True,
            random_state=2020,
        )

        x.to_csv("tests/test_files/standard_preprocessor/no_estimators_run/x.csv", index=False)
        y.to_csv("tests/test_files/standard_preprocessor/no_estimators_run/y.csv", index=False)

        x_expected = pd.read_csv("tests/test_files/standard_preprocessor/no_estimators_run/x.csv")
        y_expected = pd.read_csv(
            "tests/test_files/standard_preprocessor/no_estimators_run/y.csv", squeeze=True
        )

        assert_frame_equal(
            x.reset_index(drop=True).sort_index(axis=1),
            x_expected.sort_index(axis=1),
            check_dtype=False,
        )
        assert_series_equal(y.reset_index(drop=True), y_expected, check_dtype=False)

    def test_check_columns_for_nans_no_remove_nans(self):
        """Test _check_columns_for_nans with remove_nans set to False."""
        self.preprocessor.processed_data = self.data

        with pytest.raises(ValueError):
            self.preprocessor._check_columns_for_nans(remove_nans=False)

    def test_check_columns_for_nans_no_nans(self):
        """Test _check_columns_for_nans with no nans in dataset."""
        save_data = self.data.dropna()
        self.preprocessor.processed_data = deepcopy(save_data)

        self.preprocessor._check_columns_for_nans(remove_nans=False)

        assert_frame_equal(save_data, self.preprocessor.processed_data)

    def test_impute_no_numeric_features(self):
        """Test _impute with no numeric_features set."""
        self.preprocessor.numeric_features = []

        self.preprocessor._impute("", "")

    def test_label_encode_no_binary_features(self):
        """Test _label_encode with no binary_features set."""
        self.preprocessor.binary_features = []

        self.preprocessor._label_encode("")

    def test_manually_set_column_types(self):
        """Test _manually_set_column_types with non list."""
        with pytest.raises(TypeError):
            self.preprocessor._manually_set_column_types("not-list", [], [])

    def test_manually_set_column_with_not_subset(self):
        """Test _manually_set_column_types with a feature list not being a subset."""
        with pytest.raises(ValueError):
            self.preprocessor._manually_set_column_types(["not-subset-feature"], [], [])

    def test_scale_data_with_scale_target_as_false(self):
        """Test _scale_data with 'scale_target' as False."""
        self.preprocessor = StandardPreprocessor(data=self.data, binary_features=["Salary"])
        self.preprocessor._scale_data("", target="Salary", scale_target=False)

    def test_features_to_check(self):
        """Test features_to_check property."""
        assert self.preprocessor.features_to_check == [
            "age",
            "capital-gain",
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "race",
            "sex",
            "Salary",
        ]
