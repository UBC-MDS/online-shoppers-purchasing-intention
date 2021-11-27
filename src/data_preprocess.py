# author: Ting Zhe Yan
# created: 2021-11-23
# last updated on: 2021-11-23
# last updated by: Nico Van den Hooff

"""Reads the raw data and performs and data cleaning/pre-processing, 
transforming, and/or paritionting that needs to happen before exploratory data analysis 
or modeling takes place

Usage: src/data_preprocess.py [--input_path=<input_path>] [--output_path=<output_path>] [--test_size=<test_size>]

Options:
--input_path=<input_path>    Input file path  [default: data/raw/online_shoppers_intention.csv].
--output_path=<output_path>  Folder path (exclude filename) of where to locally write the file [default: data/processed/].
--test_size=<test_size>      Proportion of dataset to be included in test split [default: 0.2].
"""

import math
import pandas as pd
from docopt import docopt
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

opt = docopt(__doc__)


def read_data(input_path):
    """Reads raw data and return as Pandas dataframe

    Parameters
    ----------
    input_path : string
        Input file path

    Returns
    -------
    pandas.DataFrame
        Input as a pandas dataframe
    """
    print("-- Read data..")
    df = pd.read_csv(input_path)
    return df


def clean_data(df):
    """Perform data cleaning

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe
    """
    print("-- Clean data")
    # 'Jun' is spelt as 'June' in raw data
    df["Month"] = df["Month"].replace("June", "Jun")

    df["Revenue"] = df["Revenue"].astype("int64")

    return df


def train_test_split(df, test_size):
    """Split dataframe into train and test. Assumes df contains 1 year
    of data. Test will always start from the end of the data, in
    chronological order.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    test_size : float
        Should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the test split

    Returns
    -------
    list
        List containing train & test split
    """
    print("-- Split data into train/test")
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    # Sort by month
    df["Month"] = pd.Categorical(df["Month"], categories=months, ordered=True)
    df = df.sort_values("Month")

    # Split Train/Test
    n_test = math.ceil(test_size * len(df))
    n_train = len(df) - n_test

    test = df.tail(n_test)
    train = df.head(n_train)

    return [train, test]


def feat_engineer(df):
    """Feature engineering - creating new features

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Dataframe with new features
    """
    print("-- Feature Engineering")
    df["total_page_view"] = (
        df["Administrative"] + df["Informational"] + df["ProductRelated"]
    )
    df["total_duration"] = (
        df["Administrative_Duration"]
        + df["Informational_Duration"]
        + df["ProductRelated_Duration"]
    )
    df["product_view_percent"] = df["ProductRelated"] / df["total_page_view"]
    df["product_dur_percent"] = df["ProductRelated_Duration"] / df["total_duration"]
    df["ave_product_duration"] = df["ProductRelated_Duration"] / df["ProductRelated"]
    df["page_values_x_bounce_rate"] = df["PageValues"] * (1 - df["BounceRates"])
    df["page_values_per_product_view"] = df["PageValues"] / df["ProductRelated"]
    df["page_values_per_product_dur"] = df["PageValues"] / df["ProductRelated_Duration"]

    # some nans are generated in the new features due to zero division caused by
    # totals that are equal to zero (e.g. total_page_view = 0)
    nan_replacements = {
        "product_view_percent": 0,
        "product_dur_percent": 0,
        "ave_product_duration": 0,
        "page_values_x_bounce_rate": 0,
        "page_values_per_product_view": 0,
        "page_values_per_product_dur": 0,
    }

    # correct values should be zero, updated here
    df = df.fillna(nan_replacements)

    return df


# Dictionary of features type for transformation
feat_type = {
    "numeric": [
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
        "total_page_view",
        "total_duration",
        "product_view_percent",
        "product_dur_percent",
        "ave_product_duration",
        "page_values_x_bounce_rate",
        "page_values_per_product_view",
        "page_values_per_product_dur",
    ],
    "category": ["OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"],
    "binary": ["Weekend"],
    "drop": ["Month"],
    "target": ["Revenue"],
}


def get_transformer():
    """Get Column Transformer for feature transformation

    Returns
    -------
    ColumnTransformer
        Returns a ColumnTransformer object
    """
    print("-- Get Column Transformer")

    ct = make_column_transformer(
        (StandardScaler(), feat_type["numeric"]),
        (OneHotEncoder(sparse=False), feat_type["category"]),
        (OneHotEncoder(sparse=False, drop="if_binary"), feat_type["binary"]),
        ("drop", feat_type["drop"]),
        remainder="passthrough",
    )

    return ct


# TODO: move tests into a separate files
def test_function():
    """Test functions in data_preprocess.py"""
    print("====Test===")
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

    print("====Test Ends===")


def main(input_path, output_path, test_size):
    """Main function for data preprocessing. Includes reading of data, data
    cleaning, train/test split, feature engineering, and feature
    transformation. Output 2 sets of clean data, one before transformation
    for EDA, one after transformation for machine learning.

    Parameters
    ----------
    input_path : string
        Input path for input data from docopt
    output_path : string
        Output folder path from docopt
    test_size : string
        Test data proportion from docopt
    """
    # Run test function
    test_function()

    test_size = float(test_size)

    # Read raw data
    df = read_data(input_path)

    # Data cleaning
    df = clean_data(df)

    # Train / Test Split
    train, test = train_test_split(df, test_size=test_size)

    # Feature Engineering
    train = feat_engineer(train)
    test = feat_engineer(test)

    # Output pre-transformed data for EDA
    print("-- Output pre-transformed data for EDA")
    train.to_csv(output_path + "train-eda.csv", index=False)
    test.to_csv(output_path + "test-eda.csv", index=False)

    # Transformation
    # TODO: what to do with outliers?
    ct = get_transformer()
    train_np = ct.fit_transform(train)
    test_np = ct.transform(test)
    col_name = (
        feat_type["numeric"]
        + ct.named_transformers_["onehotencoder-1"].get_feature_names_out().tolist()
        + ct.named_transformers_["onehotencoder-2"].get_feature_names_out().tolist()
        + feat_type["target"]
    )
    train = pd.DataFrame(train_np, columns=col_name)
    test = pd.DataFrame(test_np, columns=col_name)

    # Output
    print("-- Output clean data")
    train.to_csv(output_path + "train.csv", index=False)
    test.to_csv(output_path + "test.csv", index=False)


if __name__ == "__main__":
    main(opt["--input_path"], opt["--output_path"], opt["--test_size"])
