all : results/Final_Classification_Report.html results/Final_RandomForest_cm.png results/model_selection_results.html results/DummyClassifier_cm.png results/RandomForest_cm.png results/LogisticRegression_cm.png results/SVC_cm.png results/XGBoost_cm.png

# download data
data/raw/online_shoppers_intention.csv : src/download_data.py
	python src/download_data.py --url=https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv --out_path=data/raw/online_shoppers_intention.csv

# preprocess data
data/processed/train-eda.csv data/processed/test-eda.csv data/processed/train.csv data/processed/test.csv : src/data_preprocess.py data/raw/online_shoppers_intention.csv
	python src/data_preprocess.py --input_path=data/raw/online_shoppers_intention.csv --output_path=data/processed/ --test_size=0.2

# create eda charts and save to file
results/chart_target_distribution.png results/chart_numeric_var_distribution.png results/chart_categorical_var_count.png results/chart_correlation.png : src/eda_charts.py data/processed/train-eda.csv data/processed/test-eda.csv
	python src/eda_charts.py --input_path=data/processed/train-eda.csv --output_path=results/

# model selection
results/model_selection_results.html results/DummyClassifier_cm.png results/RandomForest_cm.png results/LogisticRegression_cm.png results/SVC_cm.png results/XGBoost_cm.png : src/model_selection.py data/processed/train.csv data/processed/test.csv
	python src/ml_modelling.py --train=data/processed/train.csv --test=data/processed/test.csv --output_path=results/

# tune model
results/Final_Classification_Report.html results/Final_RandomForest_cm.png : src/tune_model.py data/processed/train.csv data/processed/test.csv
	python src/tune_model.py --train=data/processed/train.csv --test=data/processed/test.csv --output_path=results/

# generate jupyter book


# clean up intermediate and results files
clean:
	rm -f data/raw/*.csv
	rm -f data/processed/*.csv
	rm -f results/*.png
	rm -f results/*.html