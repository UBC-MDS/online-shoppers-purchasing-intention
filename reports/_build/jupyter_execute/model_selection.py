#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# # Model selection

# ## Metrics

# In determining the metric we use for our model, we need to first consider the context of Type I and Type II errors for our problem:
# 
# - A type I (false positive) error would be predicting that a customer will make a purchase, when they in fact do not.
# - A type II (false negative) error would be predicting that a customer will not make a purchase, when in fact they do.
# 
# In determining the metric we use for our model, we need to consider the business objective of the e-commerce company. We assume that the company will have the following objectives: 1) maximize revenue by increasing purchase conversion rate, and 2) minimize disruption to customer experience from targeted nudges.
# 
# Based on this, we can use precision, recall, and average precision. To be more specific, we want to maximize recall, while keeping a minimum threshold of 60% precision (threshold based on business requirement and tolerance). Average precision will be a secondary metric to monitor, since we are interested in both precision and recall.

# ## Base model

# Our based model will be the `DummyClassifier` model from `sklearn` with the default parameters.  From the `sklearn` [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html), the default `strategy` parameter is `prior` which always predicts the class that maximizes the class prior (like “most_frequent”) and predict_proba returns the class prior.

# ## Additional models tested

# In performing model selection, we will fit the following models to our training data, and use cross validation to assess which model to select for further hyperparameter tuning:
# 
# - Logistic Regression
# - Support Vector Machine w/ RBF kernl
# - Random Forest Classifier
# - XGBoost Classifier

# ## Cross validation results

# We performed 5 fold cross validation for the above models on our training data and observed the following metrics.  Please note that the following scores are the mean values over the 5 folds of cross validation:

# In[2]:


model_selection_results_df = pd.read_csv("../results/model_selection_results.csv", index_col=0)
model_selection_results_df


# From the above we can see that:
# 
# - The logistic regression model had the best precision scores on the validation tests during cross validation.  However, the recall scores of this model were quite poor.
# - The support vector machine had similar precision scores to the logistic regression model, and slightly better recall scores.
# - The random forest model and XGBoost models both severely overfit the training set, which can be seen by perfrect accuracy and precision scores.  However, the recall scores on the test set of these models are still higher than the logistic regression and support vector machine models.
# - The F1 scores of the models are consistent with the analysis above.

# ## Preliminary confusion matrices

# In addition we used sklearns `cross_val_predict` to generate the following preliminary confusion matrices:
# 
# ![DummyClassifier](images/DummyClassifier_cm.png)
# 
# The dummy classifier confusion matrix simply serves as a baseline.
# 
# ![LogisticRegression](images/LogisticRegression_cm.png)
# 
# The logistic regression model is outputing 220 false positives, and 927 false negatives.
# 
# ![SVC](images/SVC_cm.png)
# 
# The support vector machine is outputing 286 false positives, and 777 false negatives.
# 
# ![RandomForest](images/RandomForest_cm.png)
# 
# The random forest is outputing 388 false positives, and 666 false negatives.
# 
# ![XGBoost](images/XGBoost_cm.png)
# 
# The XGBoost model is outputing 478 false positives, and 657 false negatives.

# ## Model selection

# We note that the above results were obtained with no hyperparameter tuning.  Based on these results alone, we can see that the random forest appears to be a promising model for our problem.  We will therefore select this model to tune further.  Random forests tend to be a great model for classification problems as they inject randomness into a probelm in the form of bagging and random features {cite}`breiman2001random`.
