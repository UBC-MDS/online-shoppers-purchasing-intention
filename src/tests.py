# author: See individual functions
# created: 2021-12-03
# last updated on: 2021-12-03
# last updated by: Nico Van den Hooff

"""
Test functions for all scripts.

Usage: src/tests/tests.py [--eda=<eda_path>] [--train=<train>] [--test=<test>]

Options:
--eda=<eda>                     File path of the eda data [default: data/processed/train-eda.csv]
--train=<train>                 File path of the train data [default: data/processed/train.csv]
--test=<test>                   File path of the test data [default: data/processed/test.csv]
"""


import pandas as pd
from docopt import docopt
from matplotlib import figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from model_selection import read_cleaned_data, get_X_y
from data_preprocess import train_test_split, clean_data, feat_engineer
from model_selection import get_models, cross_validate_models, get_confusion_matrices
from tune_model import (
    create_model_and_params,
    perform_random_search,
    get_search_results,
    get_final_predictions,
)


opt = docopt(__doc__)


def eda_tests(input_path):
    # Author: Arijeet Chatterjee
    """Tests the type of data for EDA charts

    Parameters
    ----------
    input_path : str
        Path of the data file to carry out the EDA
    """

    df = pd.read_csv(input_path)
    assert isinstance(df, pd.DataFrame), "Error in df type"


def data_process_tests():
    # Author: Ting Zhe Yan
    """Test functions for data_preprocess.py"""
    # train_test_split
    df1 = pd.DataFrame(
        {
            "row": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Month": [
                "Jan",
                "Feb",
                "Dec",
                "Dec",
                "Mar",
                "Jan",
                "Jan",
                "Jan",
                "Jan",
                "Jan",
            ],
        }
    )
    assert len(train_test_split(df1, 0.2)[1]) == 2, "Error in train_test_split"
    assert train_test_split(df1, 0.2)[1].iloc[0, 0] == 3, "Error in train_test_split"

    # clean_data
    df2 = pd.DataFrame({"Month": ["June"], "Revenue": 1})
    assert clean_data(df2).iloc[0, 0] == "Jun", "Error in clean_data"

    # feat_engineer
    df3 = pd.DataFrame(
        [[0, 0, 0, 0, 0, 0, 0, 0, 100, 1, "Feb", 1, 1, 1, 1, "TRUE", "FALSE", "FALSE"]],
        columns=[
            "Administrative",
            "Administrative_Duration",
            "Informational",
            "Informational_Duration",
            "ProductRelated",
            "ProductRelated_Duration",
            "BounceRates",
            "ExitRates",
            "PageValues",
            "SpecialDay",
            "Month",
            "OperatingSystems",
            "Browser",
            "Region",
            "TrafficType",
            "VisitorType",
            "Weekend",
            "Revenue",
        ],
    )
    assert (
        feat_engineer(df3).isnull().values.any() == False
    ), "Error in fill na in feat_engineer"


def model_selection_tests(train, test):
    # Author: Nico Van den Hooff
    """Test function for model_selection.py

    Parameters
    ----------
    train_df : pandas DataFrame
        The train DataFrame
    test_df : pandas DataFrame
        The test DataFrame
    """

    # test 1: test type returned by function that reads data
    train_df, test_df = read_cleaned_data(train, test)
    assert isinstance(train_df, pd.DataFrame), "Error in train_df type"
    assert isinstance(test_df, pd.DataFrame), "Error in test_df type"

    # test 2: test type returned by function splits data into arrays
    X_train, X_test, y_train, y_test = get_X_y(train_df, test_df)
    assert pd.concat(objs=[X_train, y_train], axis=1).equals(
        train_df
    ), "Error in Xy train split"
    assert pd.concat(objs=[X_train, y_train], axis=1).equals(
        train_df
    ), "Error in Xy test split"

    # test 3: test type returned by model generation function
    models = get_models()
    assert isinstance(models, dict), "Error in model dictionary"

    # test 3: test type returned by cross validation function
    results_df = cross_validate_models(models, X_train, y_train)
    assert isinstance(results_df, pd.DataFrame), "Error in results df type"

    # test 4: test type returned by cross validation function
    cm_figures = get_confusion_matrices(models, X_train, y_train)
    assert isinstance(cm_figures, dict), "Error in cm_figures dictionary"


def tune_model_tests(train, test):
    # Author: Nico Van den Hooff
    """Test function for tune_model.py

    Parameters
    ----------
    train_df : pandas DataFrame
        The train DataFrame
    test_df : pandas DataFrame
        The test DataFrame
    """
    train_df, test_df = read_cleaned_data(train, test)
    X_train, X_test, y_train, y_test = get_X_y(train_df, test_df)

    # test 1: test types of models and search space
    model, search_space = create_model_and_params()
    assert isinstance(model, RandomForestClassifier), "Incorrect model type"
    assert isinstance(search_space, dict), "Incorrect search space type"

    # test 2: test type of randomsearchcv
    random_search = perform_random_search(X_train, y_train, model, search_space)
    assert isinstance(
        random_search, RandomizedSearchCV
    ), "Random search error in return type"

    # test 3: test types of results
    results = get_search_results(random_search)
    assert isinstance(results, dict), "Incorrect type for results"

    # test 4: test types of plots and classification report
    cm_plot, cr_df = get_final_predictions(
        results["best_estimator"], X_train, y_train, X_test, y_test
    )
    assert isinstance(cm_plot, figure.Figure), "Error in CM plot type"
    assert isinstance(cr_df, pd.DataFrame), "Error in results df type"


def main(eda, train, test):
    """Calls the test functions

    Parameters
    ----------
    eda : str
        File path of the eda data
    train : str
        File path of the train data
    test : str
        File path of the test data
    """
    print("Testing EDA functions")
    eda_tests(eda)

    print("Testing Data Preprocessing functions")
    data_process_tests()

    print("Testing Model Selection functions")
    model_selection_tests(train, test)

    print("Testing Model Tuning functions")
    tune_model_tests(train, test)


if __name__ == "__main__":
    main(opt["--eda"], opt["--train"], opt["--test"])
