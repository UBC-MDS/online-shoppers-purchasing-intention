# author: Arijeet Chatterjee
# date: 2021-11-23
# last updated on: 2021-12-03
# last updated by: Nico Van den Hooff

"""
Reads the pre-processed data and creates the charts for exploratory data analysis

Usage: src/eda_charts.py [--input_path=<input_path>] [--output_path=<output_path>]

Options:
--input_path=<input_path>          Input file path  [default: data/processed/train-eda.csv].
--output_path=<output_path>        Folder path (exclude filename) of where to locally write the EDA charts in png format [default: results/].
"""

from docopt import docopt
import pandas as pd
import altair as alt
import numpy as np
import os

alt.renderers.enable("mimetype")
alt.data_transformers.enable("data_server")

opt = docopt(__doc__)


def chart_target_distribution(data, target_var, output_path):
    """
    Create and save distribution of the target variable

    Parameters
    ----------
    data : pandas DataFrame
        Data frame for carrying out the EDA
    target_var : str
        Name of the target variable
    output_path : str
        Path of the folder to save the chart

    Returns
    -------

    """
    ch_target_dist = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            alt.X("count()", title="Count"),
            alt.Y(target_var, type="nominal"),
            alt.Color("Revenue"),
        )
        .properties(width=450, height=200)
    )
    ch_target_dist.save(os.path.join(output_path, "chart_target_distribution.png"))


def chart_numeric_var_distribution(data, numeric_vars, output_path):
    """
    Create and save distributions of the numeric variables in the data

    Parameters
    ----------
    data : pandas DataFrame
        Data frame for carrying out the EDA
    numeric_vars : list
        List of numeric variables
    output_path : str
        Path of the folder to save the chart

    Returns
    -------

    """
    ch_numeric_vars_dist = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            alt.X(alt.repeat(), type="quantitative", bin=alt.Bin(maxbins=100)),
            alt.Y("count()", title="Count"),
            alt.Color("Revenue"),
        )
        .properties(width=250, height=200)
        .repeat(numeric_vars, columns=2)
    )
    ch_numeric_vars_dist.save(
        os.path.join(output_path, "chart_numeric_var_distribution.png")
    )


def chart_categorical_var_count(data, categorical_vars, output_path):
    """
    Create and save categorical variable counts

    Parameters
    ----------
    data : pandas DataFrame
        Data frame for carrying out the EDA
    categorical_vars : list
        List of categorical variables
    output_path : str
        Path of the folder to save the chart

    Returns
    -------

    """
    ch_categorical_vars_count = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            alt.X("count()", title="Count"),
            alt.Y(alt.repeat(), type="nominal"),
        )
        .properties(width=450, height=200)
        .repeat(categorical_vars, columns=1)
    )
    ch_categorical_vars_count.save(
        os.path.join(output_path, "chart_categorical_var_count.png")
    )


def density_plot(data, output_path):
    """
    Create and save density plot

    Parameters
    ----------
    data : pandas DataFrame
        Data frame for carrying out the EDA
    output_path : str
        Path of the folder to save the chart

    Returns
    -------

    """
    ch_density = (
        alt.Chart(data.query("PageValues < 40"))
        .transform_density(
            "PageValues", groupby=["Revenue"], as_=["PageValues", "density"]
        )
        .mark_area(fill=None, strokeWidth=2)
        .encode(alt.X("PageValues"), y="density:Q", stroke="Revenue")
    )

    ch_density.save(os.path.join(output_path, "chart_density.png"))


def chart_correlation(data, output_path):
    """
    Create and save correlation plot of all the variables in the data

    Parameters
    ----------
    data : pandas DataFrame
        Data frame for carrying out the EDA
    output_path : str
        Path of the folder to save the chart

    Returns
    -------

    """
    corr_df = data.select_dtypes("number").corr("spearman").reset_index().melt("index")
    corr_df.columns = ["X1", "X2", "correlation"]
    base = (
        alt.Chart(corr_df, title="Correlation Heatmap")
        .transform_filter(alt.datum.X1 < alt.datum.X2)
        .encode(
            alt.X("X1", title=None),
            alt.Y("X2", title=None),
        )
        .properties(width=alt.Step(50), height=alt.Step(50))
    )
    rects = base.mark_rect().encode(
        alt.Color(
            "correlation",
            scale=alt.Scale(scheme="redblue", reverse=True, domain=(-1, 1)),
        )
    )
    text = base.mark_text(size=15).encode(
        text=alt.Text("correlation", format=".2f"),
        color=alt.condition(
            "datum.correlation > 0.5", alt.value("white"), alt.value("black")
        ),
    )
    ch_correlation = rects + text
    ch_correlation.save(os.path.join(output_path, "chart_correlation.png"))


def main(input_path, output_path):
    """
    Main function which orchestrates creation of the EDA charts

    Parameters
    ----------
    input_path : str
        Path of the data file to carry out the EDA
    output_path : str
        Path of the folder to save the charts

    """
    df = pd.read_csv(input_path)

    # Change target_var back to categorical for EDA
    target_var = "Revenue"
    df[target_var] = np.where(df[target_var] == 1, "True", "False")

    numeric_cols = df.select_dtypes("number").columns.tolist()
    category_cols = ["Month", "VisitorType", "Weekend"]

    # Create chart to visualize distribution of target variable
    print(f"-- Creating chart for distribution of the target variable")
    chart_target_distribution(data=df, target_var=target_var, output_path=output_path)

    # Create chart to visualize distribution of numeric variables
    print(f"-- Creating chart for distribution of numeric variables")
    chart_numeric_var_distribution(
        data=df, numeric_vars=numeric_cols, output_path=output_path
    )

    # Create chart to visualize categorical variables
    print(f"-- Creating chart for categorical variables")
    chart_categorical_var_count(
        data=df, categorical_vars=category_cols, output_path=output_path
    )

    # Create correlation plot
    print(f"-- Creating correlation chart")
    chart_correlation(data=df, output_path=output_path)

    # Create Density plot
    print(f"-- Creating density chart")
    density_plot(data=df, output_path=output_path)


if __name__ == "__main__":
    main(opt["--input_path"], opt["--output_path"])
