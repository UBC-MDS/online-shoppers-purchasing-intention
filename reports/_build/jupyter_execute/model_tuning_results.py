#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# # Model Tuning and Results

# ## Hyperparameter tuning

# We tuned the following hyperparameters of the Random Forest classifier:
# 
# (The below is a summary of the detailed discussion of the hyperparameters on sk-learns website {cite}`scikit-learn`)

# ### `n_estimators`

# This is the number of trees (models) to include in our forest (ensemble of models).  We note that a higher number of trees increases model complexity and can introduce overfitting.

# ### `criterion`

# This is the function that is used to measure the quality of a split at a node in the tree.  `sklearn` allows for either `gini` or `entropy` functions to be used.

# ### `max_depth`

# This is the maximum depth of each tree in the forest.  Higher values increase model complexity and can introduce overfitting.

# ### `max_features`

# This is the number of features that are considered when making the best split at a node in the tree.  We used the following values:
# 
# - `auto` which sets `max_features=sqrt(n_features)`
# - `log2` which sets `max_features=log2(n_features)`

# ### `min_samples_split`

# This is the minimum number of samples required to split an internal node of the tree.

# ### `min_samples_leaf`

# This is the minimum number of samples required to be at a leaf node.

# ### `class_weight`

# This hyperparameter can be used to deal with the imbalance in our training data.  Specifically, this hyper parameter controls the weights associated with each class.  We tested `balanced` which uses the values of our target label to automatically adjust weights inversely proportional to class frequencies as `n_samples / (n_classes * np.bincount(y))`

# ### Randomized search cross validation

# In order to tune the hyperparameters of our model we used randomized search cross validation.  We set a budget of 100 models to train.

# ## Final results (test set)

# ### Confusion matrix

# ![RandomForestFinal](images/Final_RandomForest_cm.png)

# ### Classification report

# In[2]:


cr = pd.read_csv("../results/classification_report.csv", index_col=0)
cr


# ## Discussion of results

# The tuned random forest is outputing 268 false positives, and 88 false negatives.  The macro average recall score is 0.827 and the macro average precision score is 0.748, which is above our budget of 0.60 that we set at the beginning of our project.
