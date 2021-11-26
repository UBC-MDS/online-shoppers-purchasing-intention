# author: Nico Van den Hooff
# created: 2021-11-2
# last updated on: 2021-11-26
# last updated by: Nico Van den Hooff

"""
Tunes the hyperparameters for our best model, in this case Random Forest.
Outputs the best performing model.

Usage: src/tune_model.py [--train=<train>] [--test=<test>] [--output_path=<output_path>]

Options:
--train=<train>                 File path of the train data [default: data/processed/train.csv]
--test=<test>                   File path of the test data [default: data/processed/test.csv]
--output_path=<output_path>     Folder path where to write results [default: results/]
"""


import numpy as np
import pandas as pd
from docopt import docopt
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from model_selection import (
    read_cleaned_data,
    get_X_y,
    get_styled_df_html,
    write_df_html,
)


opt = docopt(__doc__)


def create_model_and_params():
    """Creates the model to tune along with the hyperparameter search space.

    Returns
    -------
    RandomForestClassifier, dict
        The model to tune and a dictionary of the hyperparameter search space
    """
    model = RandomForestClassifier()

    search_space = {
        "n_estimators": randint(100, 1000),
        "criterion": ["gini", "entropy"],
        "max_depth": np.arange(10, 100, 5),
        "max_features": ["auto", "log2"],
        "min_samples_split": [2, 4, 8],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced", "balanced_subsample", None],
    }

    return model, search_space


def perform_random_search(
    X_train, y_train, model, search_space, n_iter=100, scoring="recall"
):
    """Performs random search cross validation.

    Parameters
    ----------
    X_train : numpy ndarray
        The feature matrix
    y_train : numpy ndarray
        The target labels
    model : sklearn estimator
        The model to perform a random search for
    search_space : dict
        A dictionary of hyperparameters
    n_iter : int, optional
        Number of parameter settings that are sampled, by default 10
    scoring : str, optional
        Strategy to evaluate the performance of the model, by default "recall"

    Returns
    -------
    RandomizedSearchCV
        The fit sklearn RandomizedSearchCV object
    """
    random_search = RandomizedSearchCV(
        model,
        search_space,
        n_iter=n_iter,
        scoring=scoring,
        n_jobs=-1,
        random_state=42,
        return_train_score=True,
    )

    random_search.fit(X_train, y_train)

    return random_search


def get_search_results(random_search):
    """Gets the best estimator, parameters, score, and results from a RandomizedSearchCV.

    Parameters
    ----------
    random_search : RandomizedSearchCV
        A fit sklearn RandomizedSearchCV object

    Returns
    -------
    dict
        A dictionary of results
    """
    results = {
        "best_estimator": random_search.best_estimator_,
        "best_params": random_search.best_params_,
        "best_score": random_search.best_score_,
        "cv_results_": random_search.cv_results_,
    }

    return results


def get_final_predictions(model, X_train, y_train, X_test, y_test):
    """Fits a final model, predicts, and returns a confusion matrix and classification report.

    Parameters
    ----------
    model : sklearn estimator
        The model to perform a random search for
    X_train : numpy ndarray
        The feature matrix (train)
    y_train : numpy ndarray
        The target labels (train)
    X_test : numpy ndarray
        The feature matrix (train)
    y_test : numpy ndarray
        The target labels (train)

    Returns
    -------
    matplotlib fig, pandas DataFrame
        The confusion matrix plot and classification report
    """

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    labels = ["No Purchase", "Purchase"]
    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=labels)
    cm.ax_.set_title(f"Final Random Forest Confusion Matrix")
    cm_plot = cm.figure_

    cr = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    cr_df = pd.DataFrame(cr).T

    return cm_plot, cr_df


def main(train, test, output_path):
    """Main function that performs hyperparameter tuning and outputs the results.

    Parameters
    ----------
    train : str
        Input path for train data from docopt
    test : str
        Input path for train data from docopt
    output_path : str
        Output folder path to save results
    """

    print("-- Reading in clean data")
    train_df, test_df = read_cleaned_data(train, test)

    print("-- Generating X and y array")
    X_train, X_test, y_train, y_test = get_X_y(train_df, test_df)

    print("-- Creating random forest and search space")
    model, search_space = create_model_and_params()

    print("-- Performing random search")
    random_search = perform_random_search(X_train, y_train, model, search_space)
    results = get_search_results(random_search)

    print("-- Getting final predictions")
    cm_plot, cr_df = get_final_predictions(
        results["best_estimator"], X_train, y_train, X_test, y_test
    )

    print("-- Output results (html) and images")
    cm_plot.savefig(f"{output_path}Final_RandomForest_cm.png", bbox_inches="tight")

    styled_results_df_html = get_styled_df_html(
        cr_df,
        round_num=True,
        digits=3,
    )

    write_df_html(
        styled_results_df_html,
        f"{output_path}Final_Classification_Report",
    )


# TODO: move tests into a seperate file
def tests(train, test):
    """Tests for all functions except main function.

    Parameters
    ----------
    train_df : pandas DataFrame
        The train DataFrame
    test_df : pandas DataFrame
        The test DataFrame
    """

    # these are tested in model_selection.py
    train_df, test_df = read_cleaned_data(train, test)
    X_train, X_test, y_train, y_test = get_X_y(train_df, test_df)

    # take small slices to make below tests run faster
    X_train = X_train[:10]
    X_test = X_test[:10]
    y_train = y_train[:10]
    y_test = y_test[:10]

    # test 1: test types of models and search space
    model, search_space = create_model_and_params()
    assert (isinstance(model, RandomForestClassifier), "Incorrect model type")
    assert (isinstance(search_space, dict), "Incorrect search space type")

    # test 2: test type of randomsearchcv
    random_search = perform_random_search(X_train, y_train, model, search_space)
    assert (
        isinstance(random_search, RandomizedSearchCV),
        "Random search error in return type",
    )

    # test 3: test types of results
    results = get_search_results(random_search)
    assert (isinstance(results, dict), "Incorrect type for results")

    # test 4: test types of plots and classification report
    cm_plot, cr_df = get_final_predictions(
        results["best_estimator"], X_train, y_train, X_test, y_test
    )
    assert isinstance(cm_plot, matplotlib.pyplot.figure, "Error in CM plot type")
    assert isinstance(cr_df, pd.DataFrame, "Error in results df type")


if __name__ == "__main__":
    tests(opt["--train"], opt["--test"])
    main(opt["--train"], opt["--test"], opt["--output_path"])
