# author: Ting Zhe Yan
# date: 2021-11-17

"""
Downloads data from web to a local filepath

Usage: src/download_data.py [--url=<url>] [--output_path=<output_path>]

Options:
--url=<url>                  URL of dataset  [default: https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv].
--output_path=<output_path>  Output path of where to locally write the file [default: data/raw/].
"""

from docopt import docopt
from urllib import request

opt = docopt(__doc__)


def main(url, output_path):
    """Main function to download dataset from the internet

    Parameters
    ----------
    url : string
        Direct URL to download dataset
    out_path : string
        File path (including filename) of where to locally write the file
    """
    try:
        request.urlretrieve(url, f"{output_path}online_shoppers_intention.csv")
    except Exception as e:
        print("Could not download file from {}".format(url))
        print(e)


if __name__ == "__main__":
    print("-- Downloading data")
    main(opt["--url"], opt["--output_path"])
