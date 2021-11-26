# Introduction

## Research question

The research question that we are attempting to answer with our analysis is a predictive question, and is stated as follows:
> Given clickstream and session data of a user who visits an e-commerce website, can we predict whether or not that visitor will make a purchase?

Nowadays, it is common for companies to sell their products online, with little to no physical presence such as a traditional brick and mortar store. Answering this question is critical for these types of companies in order to ensure that they are able to remain profitable. This information can be used to nudge a potential customer in real-time to complete an online purchase, increasing overall purchase conversion rates. Examples of nudges include highlighting popular products through social proof, and exit intent overlay on webpages.

## Data set

The dataset used in our analysis was obtained from the [UC Irvine Machine Learning Repository](https://archive-beta.ics.uci.edu/), a popular website with hundreds of datasets available for analysis. The creators of the dataset are C. Sakar and Yomi Kastro, and the original dataset can be obtained at {cite}`sakar2018uci` [link](https://archive-beta.ics.uci.edu/ml/datasets/online+shoppers+purchasing+intention+dataset).

Each row in the dataset contains a feature vector that contains data corresponding to a visit "session" (period of time spent) of a user on an e-commerce website. The dataset was specifically formed so that each session would belong to a unique user over a 1-year period. The total number of sessions in the dataset is 12,330.

Examples of features included in the dataset are:

- Duration (time) spent on Administrative, Informational, or Product related sections of the website.
- Data from Google Analytics such as bounce rate, exit rate, and page value.
- What month the session took place it, and whether or not the day of the session falls on a weekend.
- The operating system and browser used by the website visitor.

The target feature and class label in the dataset is called `Revenue` and contains either a `True` or `False` value, which correspond to whether or not the user made a purchase on the website during their visit, respectively. It is worth noting that the dataset is unbalanced, as 85% of the sessions contain a `False` class label, with the remaining 15% containing a `True` label.

## Data Cleaning & Feature Engineering

The dataset has good data quality in general. It is in a tidy format {cite}`wickham2014tidy`, and does not contain `NULL` values. There are some minor issues with outliers, but we have decided not to handle them.

Feature engineering is done to capture feature interactions, and aggregating common features. The table below list some of our new features:

| Feature Name          |      Description      |
|-----------------------|-----------------------|
| `total_page_view`     | Sum of page views from `Administrative`, `Informational`, `ProductRelated` pages |
| `total_page_duration` | Sum of page durations from `Administrative`, `Informational`, `ProductRelated` pages |
| `product_view_percent`| Percentage of `ProductRelated` page views |
| `product_dur_percent`| Percentage of `ProductRelated` page durations |
| `ave_product_duration`| Average duration per page for `ProductRelated` pages |
| `page_values_x_bounce_rate`| `PageValues` $\times$ (1 - `BounceRates`) |
| `page_values_per_product_view`| `PageValues` per `ProductRelated` page views |
| `page_values_per_product_dur`| `PageValues` per `ProductRelated` page durations |
## Bibliography

```{bibliography}
```

<!-- Sample content

:::{note}
Here is a note!
:::

And here is a code block:

```
e = mc^2
```

EOF -->
