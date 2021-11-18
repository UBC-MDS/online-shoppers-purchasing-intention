# author: Ting Zhe Yan
# date: 2021-11-17

'''Downloads data from web to a local filepath

Usage: src/download_data.py [--url=<url>] [--out_path=<out_path>]

Options:
--url=<url>            URL of dataset  [default: https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv].
--out_path=<out_path>  File path (including filename) of where to locally write the file [default: data/raw/online_shoppers_intention.csv].
'''

from docopt import docopt
from urllib import request

opt = docopt(__doc__)

def main(url, out_path):
    try:
        request.urlretrieve(url, out_path)
    except Exception as e:
        print("Could not download file from {}".format(url))
        print(e)
        
if __name__ == "__main__":
    main(opt["--url"], opt["--out_path"])
