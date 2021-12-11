# author: Nico Van den Hooff
# created: 2021-11-2
# last updated on: 2021-11-26
# last updated by: Nico Van den Hooff

"""
Tunes the hyperparameters for our best model, in this case Random Forest.
Outputs the best performing model.

Usage: src/tune_model.py [--data_path=<data_path>] [--output_path=<output_path>]

Options:
--data_path=<data_path>         Input path of the preprocessed data [default: data/processed/]
--output_path=<output_path>     Output path of where to write results [default: results/model_tuning/]
"""


import numpy as np
import pandas as pd
from docopt import docopt
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from model_selection import read_cleaned_data, get_X_y
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from model_selection import get_transformer, get_feat_type


opt = docopt(__doc__)


def create_model_and_params():
    """Creates the model to tune along with the hyperparameter search space.

    Returns
    -------
    RandomForestClassifier, dict
        The model to tune and a dictionary of the hyperparameter search space
    """
    ct = get_transformer()

    model = make_pipeline(ct, RandomForestClassifier())

    search_space = {
        "randomforestclassifier__n_estimators": randint(100, 1000),
        "randomforestclassifier__criterion": ["gini", "entropy"],
        "randomforestclassifier__max_depth": np.arange(10, 100, 5),
        "randomforestclassifier__max_features": ["auto", "log2"],
        "randomforestclassifier__min_samples_split": [2, 4, 8],
        "randomforestclassifier__min_samples_leaf": [1, 2, 4],
        "randomforestclassifier__class_weight": [
            "balanced",
            "balanced_subsample",
            None,
        ],
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


def main(data_path, output_path):
    """Main function that performs hyperparameter tuning and outputs the results.

    Parameters
    ----------
    data_path : str
        Input path of the preprocessed data
    output_path : str
        Output path of where to write result tables
    """

    print("-- Reading in clean data")
    train_df, test_df = read_cleaned_data(data_path)

    print("-- Generating X and y array")
    X_train, X_test, y_train, y_test = get_X_y(train_df, test_df)

    print("-- Creating random forest and search space")
    model, search_space = create_model_and_params()

    print("-- Performing random search")
    random_search = perform_random_search(X_train, y_train, model, search_space)
    results = get_search_results(random_search)

    best_hyper_params = results["best_params"]

    # cleans up for indices in dataframe
    index = []
    for key in best_hyper_params.keys():
        index.append(key[24:])

    best_hyper_params_df = pd.DataFrame(
        data=best_hyper_params.values(),
        index=index,
        columns=["value"],
    )

    best_recall = results["best_score"]
    print(f"-- Search complete, best recall score: {best_recall}")

    print("-- Getting final predictions")
    cm_plot, cr_df = get_final_predictions(
        results["best_estimator"], X_train, y_train, X_test, y_test
    )

    print("-- Output results and images")
    best_hyper_params_df.to_csv((f"{output_path}best_hyperparameters.csv"))
    cr_df.to_csv(f"{output_path}classification_report.csv")
    cm_plot.savefig(f"{output_path}final_cm.png", bbox_inches="tight")


if __name__ == "__main__":
    main(opt["--data_path"], opt["--output_path"])
