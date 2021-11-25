# Model Tuning and Results

## Hyperparameter tuning

We tuned the following hyperparameters of the Random Forest classifier:

- `n_estimators`
- `criterion`
- `max_depth`
- `max_features`
- `min_samples_split`
- `min_samples_leaf`
- `class_weight`

Specifically, we used Randomized Search with cross validation to select the best parameters in the search space above.

## Final results

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
