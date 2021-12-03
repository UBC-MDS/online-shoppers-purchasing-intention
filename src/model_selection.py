# author: Nico Van den Hooff
# created: 2021-11-23
# last updated on: 2021-12-03
# last updated by: Nico Van den Hooff

"""
Reads in the pre-processed train and test data.  Splits this data into X and y arrays.
Cross validates a selection of machine learning models and outputs the results of 
cross validation along with confusion matrices.

Usage: src/ml_modelling.py [--train=<train>] [--test=<test>] [--output_path=<output_path>]

Options:
--train=<train>                 File path of the train data [default: data/processed/train.csv]
--test=<test>                   File path of the test data [default: data/processed/test.csv]
--output_path=<output_path>     Folder path where to write results [default: results/]
"""


import warnings
import pandas as pd
import xgboost as xgb
from docopt import docopt
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.exceptions import UndefinedMetricWarning

opt = docopt(__doc__)

# turn off warnings for zero division calculations in DummyClassifier CV
warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)


def read_cleaned_data(train, test):
    """Reads in train and test data and returns as pandas DataFrames.

    Parameters
    ----------
    train : str
        File path of the training data
    test : str
        File path of the test data

    Returns
    -------
    tuple of pandas DataFrames
        The train and test pandas DataFrames
    """
    train_df = pd.read_csv(train)
    test_df = pd.read_csv(test)

    return train_df, test_df


def get_X_y(train_df, test_df, target="Revenue"):
    """Splits the train and test data into X and y arrays.

    Parameters
    ----------
    train_df : pandas DataFrame
        The train DataFrame
    test_df : pandas DataFrame
        The test DataFrame
    target : str, optional
        The target label, by default "Revenue"

    Returns
    -------
    tuple of pandas DataFrames
        The train and test X and y arrays
    """
    X_train, X_test = (train_df.drop(columns=[target]), test_df.drop(columns=[target]))
    y_train, y_test = (train_df[target], test_df[target])

    return X_train, X_test, y_train, y_test


def get_models():
    """Creates the machine learning model objects.

    Returns
    -------
    dict :
        Dictionary of model instances.
    """
    models = {
        "DummyClassifier": DummyClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=1500),  # helps convergence
        "SVC": SVC(),
        "RandomForest": RandomForestClassifier(),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    }

    return models


def get_mean_cv_scores(model, X_train, y_train, **kwargs):
    """Calculates and returns the mean cross validation score for a model.

    Parameters
    ----------
    model : sklearn estimator or xgb model
        The model to cross validate
    X_train : numpy ndarray
        The feature matrix
    y_train : numpy ndarray
        The target labels

    Returns
    -------
    pandas Series
        The mean cross validation scores with standard deviations
    """
    output = []

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()

    # present scores as score (+/- sd)
    for i in range(len(mean_scores)):
        output.append(f"{mean_scores[i]:.2f} (+/- {std_scores[i]:.2f})")

    return pd.Series(data=output, index=mean_scores.index)


def cross_validate_models(
    models, X_train, y_train, cv=5, metrics=["accuracy", "precision", "recall", "f1"]
):
    """Performs cross validation for a set of models and returns results.

    Parameters
    ----------
    models : list
        A list of sklearn estimators (also accepts xgb model)
    X_train : numpy ndarray
        The feature matrix
    y_train : numpy ndarray
        The target labels
    cv : int, optional
        Number of folds to perform, by default 5
    metrics : list, optional
        The scoring metrics, by default ["precision", "recall"]

    Returns
    -------
    pandas DataFrame
        The results of cross validation for the given models
    """
    results = {}

    for name, model in models.items():
        results[name] = get_mean_cv_scores(
            model, X_train, y_train, cv=cv, return_train_score=True, scoring=metrics
        )

    results_df = pd.DataFrame(results)

    return results_df


def get_confusion_matrices(models, X_train, y_train):
    """Calculates and returns the confusion matrices for a set of models.

    Parameters
    ----------
    models : list
        A list of sklearn estimators (also accepts xgb model)
    X_train : numpy ndarray
        The feature matrix
    y_train : numpy ndarray
        The target labels

    Returns
    -------
    dict
        A dictionary of confusion matrices in matplot lib figure form
    """
    cm_figures = {}

    for name, model in models.items():
        labels = ["No Purchase", "Purchase"]

        y_pred = cross_val_predict(model, X_train, y_train)

        # creates base confusion matrix plot
        cm = ConfusionMatrixDisplay.from_predictions(
            y_train, y_pred, display_labels=labels
        )

        # sets the title of the confusion matrix
        cm.ax_.set_title(f"{name} Confusion Matrix")

        # extracts and adds matplotlib figure to dictionary
        cm_figures[name] = cm.figure_

    return cm_figures


def get_styled_df_html(df, round_num=False, digits=None):
    """Returns the html code for a styled pandas DataFrame

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame to obtain the styled html code for
    round_num : bool, optional
        Whether or not to round the numbers in the DataFrame, by default False
    digits : int or None, optional
        The number of digits to round to, by default None

    Returns
    -------
    str
        The html code for the styled dataframe
    """

    # css formatters for cell highlights, index and header colours
    cell_hover = {"selector": "td:hover", "props": [("background-color", "#ffffb3")]}

    index_names = {
        "selector": ".index_name",
        "props": "font-style: italic; color: darkgrey; font-weight:normal;",
    }

    headers = {
        "selector": "th:not(.index_name)",
        "props": "background-color: #000066; color: white;",
    }

    if round_num:
        fmt = "{:." + str(digits) + "f}"
        styled_df = df.style.format(fmt)
    else:
        styled_df = df.style.format()

    # apply css and get html code
    styled_df.set_table_styles([cell_hover, index_names, headers])
    styled_df_html = styled_df.to_html()

    return styled_df_html


def write_df_html(styled_df_html, filename):
    """Writes the html code for a style Pandas Dataframe to a .html file.

    Parameters
    ----------
    styled_df_html : str
        The html code to write to a file
    filename : str
        The name of the .html file (do not include .html in this)
    """
    file = open(f"{filename}.html", "w")
    file.write(styled_df_html)
    file.close()


def main(train, test, output_path):
    """Main function that performs model selection and outputs the results.

    Parameters
    ----------
    train : str
        Input path for train data from docopt
    test : str
        Input path for train data from docopt
    output_path : str
        Output folder path to save results
    """

    # train and test data
    print("-- Reading in clean data")
    train_df, test_df = read_cleaned_data(train, test)

    print("-- Generating X and y array")
    X_train, _, y_train, _ = get_X_y(train_df, test_df)

    # create model instances
    print("-- Generating base models")
    models = get_models()

    # cross validate and write results to csv
    print("-- Cross validating models")
    results_df = cross_validate_models(models, X_train, y_train)

    # create confusion matrices
    print("-- Creating confusion matrices")
    cm_figures = get_confusion_matrices(models, X_train, y_train)

    print("-- Output results (html) and images")
    # output results
    styled_results_df_html = get_styled_df_html(results_df)
    write_df_html(styled_results_df_html, f"{output_path}model_selection_results")

    # output cm images
    for model, figure in cm_figures.items():
        name = f"{output_path}{model}_cm.png"
        figure.savefig(name, bbox_inches="tight")


if __name__ == "__main__":
    main(opt["--train"], opt["--test"], opt["--output_path"])
