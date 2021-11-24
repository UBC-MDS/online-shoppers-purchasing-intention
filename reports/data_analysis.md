# Data Analysis

## Data set

The dataset used in our analysis was obtained from the [UC Irvine Machine Learning Repository](https://archive-beta.ics.uci.edu/), a popular website with hundreds of datasets available for analysis. The creators of the dataset are C. Sakar and Yomi Kastro, and the original dataset can be obtained at this [link](https://archive-beta.ics.uci.edu/ml/datasets/online+shoppers+purchasing+intention+dataset).

Each row in the dataset contains a feature vector that contains data corresponding to a visit "session" (period of time spent) of a user on an e-commerce website. The dataset was specifically formed so that each session would belong to a unique user over a 1-year period. The total number of sessions in the dataset is 12,330.

Examples of features included in the dataset are:

- Duration (time) spent on Administrative, Informational, or Product related sections of the website.
- Data from Google Analytics such as bounce rate, exit rate, and page value.
- What month the session took place it, and whether or not the day of the session falls on a weekend.
- The operating system and browser used by the website visitor.

The target feature and class label in the dataset is called `Revenue` and contains either a `True` or `False` value, which correspond to whether or not the user made a purchase on the website during their visit, respectively. It is worth noting that the dataset is unbalanced, as 85% of the sessions contain a `False` class label, with the remaining 15% containing a `True` label.

Distribution of the numerical variables:

![distribution_numerical_vars_plot](../../../results/chart_numeric_var_distribution.png)

Correlation plot of the dataset:

![correlation_plot](../../../results/chart_correlation.png)

Target distribution:

![distribution_target_plot](../../../results/chart_target_distribution.png)
