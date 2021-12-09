# online shoppers purchasing intention Makefile
# Authors: Nico Van den Hooff, TZ Yan, Arijeet Chatterjee
# Date: December 1, 2021
# Last updated: December 8, 2021

all : docs/

# download raw data
data/raw/online_shoppers_intention.csv : src/download_data.py
	python src/download_data.py --url=https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv --output_path=data/raw/

# preprocess raw data
data/processed/ : src/data_preprocess.py data/raw/online_shoppers_intention.csv
	python src/data_preprocess.py --input_path=data/raw/ --output_path=data/processed/ --test_size=0.2

# create eda charts and save to directory
eda/figures/ : src/eda_charts.py data/processed/
	python src/eda_charts.py --input_path=data/processed/ --output_path=eda/figures/

# perform model selection and save results
results/model_selection/ : src/model_selection.py data/processed/
	python src/model_selection.py --data_path=data/processed/ --output_path=results/model_selection/

# perform model tuning and save results
results/model_tuning/ : src/tune_model.py data/processed/
	python src/tune_model.py --data_path=data/processed/ --output_path=results/model_tuning/

# build jupyter book
reports/_build/ : eda/figures/ results/model_selection/ results/model_tuning/
	cp -a eda/figures/* reports/images
	cp -a results/model_selection/*.png reports/images
	cp -a results/model_tuning/*.png reports/images
	jupyter-book build --all reports/

# copy jupyter book files for github pages build
docs/ : reports/_build/
	cp -a reports/_build/html/. docs/

# clean up intermediate and results files
clean:
	rm -f data/raw/*
	rm -f data/processed/*
	rm -f eda/figures/*
	rm -f results/model_selection/*
	rm -f results/model_tuning/*
	rm -rf reports/_build/*
	rm -f reports/images/*
	rm -rf docs/*