# Description

This is a a copy of our original project plan.

## Project plan

In order to our research question, we will use python, and specifically `sk-learn`, to build a Machine Learning classification model that classifies whether or not a user will make a purchase based on a set of input features.  We will also use `XGBoost`, a non-sklearn library.

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