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
# Further, we need to consider the business objective of an e-commerce company who would potentially be using this model. We assume that the company will have the following objectives: 
# 
# 1. Maximize revenue by increasing purchase conversion rate
# 2. Minimize disruption to customer experience from targeted nudges
# 
# Based on this, we note that relevant metrics include precision, recall, and average precision. To be more specific, we want to maximize recall, while keeping a minimum threshold of 60% precision (threshold based on business requirement and tolerance). Average precision will be a secondary metric to monitor, since we are interested in both precision and recall.

# ## Base model

# Our baseline model will be the `DummyClassifier` model from `sklearn` with the default parameters.  From the `sklearn` [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html), the default `strategy` parameter is `prior` which always predicts the class that maximizes the class prior (like “most_frequent”) and predict_proba returns the class prior.

# ## Additional models tested

# In performing model selection, we will also consider and fit the following models to our training data:
# 
# - Logistic Regression
# - Support Vector Machine w/ RBF kernl
# - Random Forest Classifier
# - XGBoost Classifier
# 
# In assessing these models, we will also consider:
# 
# 1. Five fold cross validation results
# 2. Confuision matrices generated with cross validated predictions on the train set
# 3. Precision recall curves generated with cross validated predictions on the train set

# ## Cross validation

# We performed 5 fold cross validation for the above models on our training data and observed the following metrics:
# 
# _Please note that the following metrics are the mean values over the 5 folds of cross validation_

# In[2]:


model_selection_results_df = pd.read_csv("../results/model_selection/model_selection_results.csv", index_col=0)

# set pandas table styles
s = model_selection_results_df.style

cell_hover = { 
    'selector': 'td:hover',
    'props': [('background-color', '#ffffb3')]
}

index_names = {
    'selector': '.index_name',
    'props': 'font-style: italic; color: darkgrey; font-weight:normal;'
}
headers = {
    'selector': 'th:not(.index_name)',
    'props': 'background-color: #000066; color: white;'
}

s.set_table_styles([cell_hover, index_names, headers])


# From the above we can see that the:
# 
# - Logistic regression model had the best precision scores on the validation tests during cross validation.  However, the recall scores of this model were quite poor.
# - Support vector machine had similar precision scores to the logistic regression model, and slightly better recall scores.
# - Random forest model and XGBoost models both severely overfit the training set, which can be seen by perfrect accuracy and precision scores.  However, the recall scores on the test set of these models are still higher than the logistic regression and support vector machine models.  We note that it is common for ensemble models to overfit without hyperparameter tuning.
# - The F1 scores of the models are consistent with the analysis above.
# - The Random forest model has the highest average precision on the validation sets.
# 
# Based on the above results, the Random Forest model or XGBoost model look promising.

# ## Confusion matrices

# We used sklearn's `cross_val_predict` with `ConfusionMatrixDisplay.from_predictions` to generate the following confusion matrices:

# ![ConfusionMatrix](images/model_cm.png)

# From the confusion matrices above, we can see that the:
# - Logistic regression model is outputing 220 false positives, and 927 false negatives.
# - Support vector machine is outputing 286 false positives, and 777 false negatives.
# - Random forest is outputing 388 false positives, and 669 false negatives.
# - XGBoost model is outputing 478 false positives, and 657 false negatives.
# 
# Considering our goals of maximizing recall, with a budget of of 60% for precision, the random forest and XGBoost model both look promising based on the confusion matrices.

# ## Precision Recall Curves

# We used sklearn's `cross_val_predict` with `PrecisionRecallDisplay.from_predictions` to generate the following plot:

# ![PRCurves](images/model_pr_curves.png)

# With precision recall curves, a high area under the curve represnts both high precision and recall.  The precision recall curves appear to show the same trend as the CV results and confusion matrices, which is that the Random forest and XGBoost models look the most promising.

# ## Model selection

# Above, we analyzed what can be considered "quick and dirty" models, where we simply analyze models results on a feature matrix with no hyperparameter tuning.  The main purpose of training such models is to identify a promising model(s) for our problem to refine further.
# 
# Based on these results alone, it appears that the random forest and XGBoost models did the best with no tuning at all.  We note that both these models are ensemble models, and share some similarities.  As our dataset is not that large, we will select the Random forest model to tune further.  
# 
# Finally, we note that Random forests are often a great model for classification problems as they inject randomness into a problem in the form of bagging and random features {cite}`breiman2001random`.
