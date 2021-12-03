
# Online Shoppers Purchasing Intention

Contributors: Nico Van den Hooff, Ting Zhe (TZ) Yan, Arijeet Chatterjee

## Summary

This repository contains our work for our project that is focused on applying machine learning classification models to e-commerce data. Specifically, we are building a classification model with the intention to determine if a website visitor will complete a purchase or not.

## Introduction

### Research question

The research question that we are attempting to answer with our analysis is a predictive question, and is stated as follows:
> Given clickstream and session data of a user who visits an e-commerce website, can we predict whether or not that visitor will make a purchase?

Nowadays, it is common for companies to sell their products online, with little to no physical presence such as a traditional brick and mortar store. Answering this question is critical for these types of companies in order to ensure that they are able to remain profitable. This information can be used to nudge a potential customer in real-time to complete an online purchase, increasing overall purchase conversion rates. Examples of nudges include highlighting popular products through social proof, and exit intent overlay on webpages.

### Data set

The dataset used in our analysis was obtained from the [UC Irvine Machine Learning Repository](https://archive-beta.ics.uci.edu/), a popular website with hundreds of datasets available for analysis. The creators of the dataset are C. Sakar and Yomi Kastro, and the original dataset can be obtained at this [link](https://archive-beta.ics.uci.edu/ml/datasets/online+shoppers+purchasing+intention+dataset).

Each row in the dataset contains a feature vector that contains data corresponding to a visit "session" (period of time spent) of a user on an e-commerce website. The dataset was specifically formed so that each session would belong to a unique user over a 1-year period. The total number of sessions in the dataset is 12,330.

Examples of features included in the dataset are:

- Duration (time) spent on Administrative, Informational, or Product related sections of the website.
- Data from Google Analytics such as bounce rate, exit rate, and page value.
- What month the session took place it, and whether or not the day of the session falls on a weekend.
- The operating system and browser used by the website visitor.

The target feature and class label in the dataset is called `Revenue` and contains either a `True` or `False` value, which correspond to whether or not the user made a purchase on the website during their visit, respectively. It is worth noting that the dataset is unbalanced, as 85% of the sessions contain a `False` class label, with the remaining 15% containing a `True` label.

We will include a detailed table that describes each feature and its type (e.g. numerical vs. categorical) in our analysis.

### Project plan

In order to answer the predictive question above, we will use python, and specifically `sk-learn`, to build a Machine Learning classification model that classifies whether or not a user will make a purchase based on a set of input features.  We will also use `XGBoost`, a non-sklearn library.

Before building the model, we will perform a train test split of the data, retaining 20% of the data for our test set. We will then perform exploratory data analysis ("EDA") with `altair` in order to identify any high level trends in the data, and to analyze the features and their relationship(s) with the target label further. Specifically, examples of EDA that we will perform are: an analysis of the distributions of each numeric feature faceted by the target label, a similar analysis for categorical features, a visual plot of the distribution of our target label, and a heat map correlation matrix of our features. Please note that this is not an exhaustive list of the EDA that we will perform, but rather a few examples.

Once we have performed EDA, we will perform feature engineering. The purpose of this will be to deal with scenarios such as skewed features and features that contain outliers. We will also consider the creation of any new features that we deem will be useful for our model. We note that the dataset does not contain any missing data, so we will not need to perform any data imputation techniques. We will however perform research and consider varying methods available to deal with the unbalance in our datasets target label. At a high level, potential solutions may be synthesizing new training data from the existing data, and tuning appropriate model hyper parameters.

Next, we will begin the process of model selection. We will train a baseline model with `DummyClassifier` from `sk-learn`. The list of models that we will consider are: Logistic Regression, Support Vector Classifier, Random Forest, and XGBoost. In determining the metric we use for our model, we need to first consider the context of Type I and Type II errors for our problem:

- A type I (false positive) error would be predicting that a customer will make a purchase, when they in fact do not.
- A type II (false negative) error would be predicting that a customer will not make a purchase, when in fact they do.

In determining the metric we use for our model, we need to consider the business objective of the e-commerce company. We assume that the company will have the following objectives: 

1. Maximize revenue by increasing purchase conversion rate
2. Minimize disruption to customer experience from targeted nudges

Based on this, we can use precision, recall, and average precision as our model metrics. To be more specific, we want to maximize recall, while keeping a minimum threshold of 60% precision (threshold based on business requirement and tolerance). Average precision will be a secondary metric to monitor, since we are interested in both precision and recall.

We will use cross-validation to select the model that we will focus on in our analysis. Once we have performed cross validation, we will pick the model with the best performance, and perform a round of hyper parameter tuning. We will then iteratively update our project as needed to obtain the best model performance that we can.

Once we have finalized the hyper parameters we will train a final model on the entire training set, and evaluate its performance on the test set. We will include the confusion matrix and classification report along with our models final precision score in the final report of our project.

To date, we have:

- Performed exploratory data analysis, which can be found [here](https://github.com/UBC-MDS/online-shoppers-purchasing-intention/tree/main/eda)
- Written scripts to download data, preprocess data, produce EDA charts, for model selection, and for tuning our model, which can be found [here](https://github.com/UBC-MDS/online-shoppers-purchasing-intention/tree/main/src)
- Completed a first draft of our report, in the form of a Jupyter Book, which can be found [here](https://github.com/UBC-MDS/online-shoppers-purchasing-intention/tree/main/reports)

## Statement of future direction

In milestone 3 and forward we will focus on:

- Automating our pipeline
- Dealing with the class imbalance in our data
- Automating our jupyter book images further (right now the html is directly embedded)

## Analysis

We have published a copy of our analysis in the form of a [Jupyter Book](https://ubc-mds.github.io/online-shoppers-purchasing-intention/intro.html)

## Usage

To replicate this analysis, clone this GitHub repository, install the dependencies listed below, and run the following command at the command line/terminal from the root directory of this project:

```
make all
```

To restore the repo to a clean state, with no intermediate or result files, run the following command at the command line/terminal from the root directory of this project:

```
make clean
```

<!-- 
```
The suggested way to run this analysis is as below:
# download data
python src/download_data.py --url=https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv --out_path=data/raw/online_shoppers_intention.csv

# pre-process data
python src/data_preprocess.py --input_path=data/raw/online_shoppers_intention.csv --output_path=data/processed/ --test_size=0.2

# create explanatory data analysis figures and write to file
python src/eda_charts.py --input_path=data/processed/train-eda.csv --output_path=results/

# model selection
python src/model_selection.py --train=data/processed/train.csv --test=data/processed/test.csv --output_path=results/

# tune model
python src/tune_model.py --train=data/processed/train.csv --test=data/processed/test.csv --output_path=results/

# render final report
jupyter-book build -all reports/
``` -->

## Dependencies

Python 3.9 and Python packages

- numpy=1.21.2
- pandas=1.3.3
- scikit-learn=1.0
- scipy=1.7.1
- docopt=0.6.2
- xgboost=1.5.0
- altair=4.1.0
- altair_saver
- altair-data-server==0.4.1
- ipykernel
- jupyter-book

## License

The source code for the site is licensed under the MIT license, which you can find [here](https://github.com/UBC-MDS/online-shoppers-purchasing-intention/blob/main/LICENSE).

## References

Sakar, C., and Kasto, Yomi. 2018. “UCI Machine Learning Repository.” University of California, Irvine, School of Information; Computer Sciences. [https://archive-beta.ics.uci.edu/](https://archive-beta.ics.uci.edu/).
