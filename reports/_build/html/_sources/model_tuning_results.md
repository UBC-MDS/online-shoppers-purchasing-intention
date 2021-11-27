# Model Tuning and Results

## Hyperparameter tuning

We tuned the following hyperparameters of the Random Forest classifier:

(The below is a summary of the detailed discussion of the hyperparameters on sk-learns website {cite}`scikit-learn`)

### `n_estimators`

This is the number of trees (models) to include in our forest (ensemble of models).  We note that a higher number of trees increases model complexity and can introduce overfitting.

### `criterion`

This is the function that is used to measure the quality of a split at a node in the tree.  `sklearn` allows for either `gini` or `entropy` functions to be used.

### `max_depth`

This is the maximum depth of each tree in the forest.  Higher values increase model complexity and can introduce overfitting.

### `max_features`

This is the number of features that are considered when making the best split at a node in the tree.  We used the following values:

- `auto` which sets `max_features=sqrt(n_features)`
- `log2` which sets `max_features=log2(n_features)`

### `min_samples_split`

This is the minimum number of samples required to split an internal node of the tree.

### `min_samples_leaf`

This is the minimum number of samples required to be at a leaf node.

### `class_weight`

This hyperparameter can be used to deal with the imbalance in our training data.  Specifically, this hyper parameter controls the weights associated with each class.  We tested `balanced` which uses the values of our target label to automatically adjust weights inversely proportional to class frequencies as `n_samples / (n_classes * np.bincount(y))`


### Randomized search cross validation

In order to tune the hyperparameters of our model we used randomized search cross validation.  We set a budget of 100 models to train.

## Final results (test set)

### Confusion matrix

![RandomForestFinal](../../../results/Final_RandomForest_cm.png)

### Classification report

<style type="text/css">
#T_00242_ td:hover {
  background-color: #ffffb3;
}
#T_00242_ .index_name {
  font-style: italic;
  color: darkgrey;
  font-weight: normal;
}
#T_00242_ th:not(.index_name) {
  background-color: #000066;
  color: white;
}
</style>
<table id="T_00242_">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th class="col_heading level0 col0" >precision</th>
      <th class="col_heading level0 col1" >recall</th>
      <th class="col_heading level0 col2" >f1-score</th>
      <th class="col_heading level0 col3" >support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_00242_level0_row0" class="row_heading level0 row0" >No Purchase</th>
      <td id="T_00242_row0_col0" class="data row0 col0" >0.953</td>
      <td id="T_00242_row0_col1" class="data row0 col1" >0.869</td>
      <td id="T_00242_row0_col2" class="data row0 col2" >0.909</td>
      <td id="T_00242_row0_col3" class="data row0 col3" >2058.000</td>
    </tr>
    <tr>
      <th id="T_00242_level0_row1" class="row_heading level0 row1" >Purchase</th>
      <td id="T_00242_row1_col0" class="data row1 col0" >0.543</td>
      <td id="T_00242_row1_col1" class="data row1 col1" >0.784</td>
      <td id="T_00242_row1_col2" class="data row1 col2" >0.642</td>
      <td id="T_00242_row1_col3" class="data row1 col3" >408.000</td>
    </tr>
    <tr>
      <th id="T_00242_level0_row2" class="row_heading level0 row2" >accuracy</th>
      <td id="T_00242_row2_col0" class="data row2 col0" >0.855</td>
      <td id="T_00242_row2_col1" class="data row2 col1" >0.855</td>
      <td id="T_00242_row2_col2" class="data row2 col2" >0.855</td>
      <td id="T_00242_row2_col3" class="data row2 col3" >0.855</td>
    </tr>
    <tr>
      <th id="T_00242_level0_row3" class="row_heading level0 row3" >macro avg</th>
      <td id="T_00242_row3_col0" class="data row3 col0" >0.748</td>
      <td id="T_00242_row3_col1" class="data row3 col1" >0.827</td>
      <td id="T_00242_row3_col2" class="data row3 col2" >0.776</td>
      <td id="T_00242_row3_col3" class="data row3 col3" >2466.000</td>
    </tr>
    <tr>
      <th id="T_00242_level0_row4" class="row_heading level0 row4" >weighted avg</th>
      <td id="T_00242_row4_col0" class="data row4 col0" >0.885</td>
      <td id="T_00242_row4_col1" class="data row4 col1" >0.855</td>
      <td id="T_00242_row4_col2" class="data row4 col2" >0.865</td>
      <td id="T_00242_row4_col3" class="data row4 col3" >2466.000</td>
    </tr>
  </tbody>
</table>


## Discussion of results

The tuned random forest is outputing 268 false positives, and 88 false negatives.  The macro average recall score is 0.827 and the macro average precision score is 0.748, which is above our budget of 0.60 that we set at the beginning of our project.