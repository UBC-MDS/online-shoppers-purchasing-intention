# Data Analysis

The data consists of numeric and categorical features. Amongst these features, "BounceRates", "ExitRates" and "PageValues" features represent the metrics measured by "Google Analytics" for each page in the e-commerce site. We have mentioned below the key observations from studying the correlation in the data:

- "BounceRates" and "ExitRates" are similar to each other.
- Sessions with high BounceRates have less webpage interaction and less sales
- "PageValues" could be an important feature. It has positive correlation with "Revenue", and is the most direct feature linking to Revenue.

## Correlation plot of the dataset

![correlation_plot](images/chart_correlation.png)

We looked at the distribution of the numeric features, but this analysis proved inconclusive.

## Distribution of the numerical variables

![distribution_numerical_vars_plot](images/chart_numeric_var_distribution.png)

Finally, we have highlighted below the imbalance in the taget variable. As mentioned in the data set introduction, 85% of the sessions contain a `False` class label, with the remaining 15% containing a `True` label

## Target distribution

![distribution_target_plot](images/chart_target_distribution.png)
