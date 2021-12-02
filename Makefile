all : reports/_build/ #results/Final_Classification_Report.html results/Final_RandomForest_cm.png 

# download data
data/raw/online_shoppers_intention.csv : src/download_data.py
	python src/download_data.py --url=https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv --out_path=data/raw/online_shoppers_intention.csv

# preprocess data
data/processed/train-eda.csv data/processed/test-eda.csv data/processed/train.csv data/processed/test.csv : src/data_preprocess.py data/raw/online_shoppers_intention.csv
	python src/data_preprocess.py --input_path=data/raw/online_shoppers_intention.csv --output_path=data/processed/ --test_size=0.2

# create eda charts and save to file
reports/images/chart_target_distribution.png reports/images/chart_numeric_var_distribution.png reports/images/chart_categorical_var_count.png reports/images/chart_correlation.png : src/eda_charts.py data/processed/train-eda.csv data/processed/test-eda.csv
	python src/eda_charts.py --input_path=data/processed/train-eda.csv --output_path=reports/images/

# model selection
reports/images/model_selection_results.html reports/images/DummyClassifier_cm.png reports/images/RandomForest_cm.png reports/images/LogisticRegression_cm.png reports/images/SVC_cm.png reports/images/XGBoost_cm.png : src/model_selection.py data/processed/train.csv data/processed/test.csv
	python src/model_selection.py --train=data/processed/train.csv --test=data/processed/test.csv --output_path=reports/images/

# tune model
reports/images/Final_Classification_Report.html reports/images/Final_RandomForest_cm.png : src/tune_model.py data/processed/train.csv data/processed/test.csv
	python src/tune_model.py --train=data/processed/train.csv --test=data/processed/test.csv --output_path=reports/images/

# generate jupyter book
reports/_build/ : reports/images/model_selection_results.html reports/images/DummyClassifier_cm.png reports/images/RandomForest_cm.png reports/images/LogisticRegression_cm.png reports/images/SVC_cm.png reports/images/XGBoost_cm.png reports/images/Final_Classification_Report.html reports/images/Final_RandomForest_cm.png
	jupyter-book build --all reports/

# clean up intermediate and results files
clean:
	rm -f data/raw/*.csv
	rm -f data/processed/*.csv
	rm -f reports/*.png
	rm -f reports/*.html
	rm -r reports/_build/