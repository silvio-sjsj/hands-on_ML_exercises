import os
import tarfile
import urllib
import pandas as pd

# Local variables
LOCAL_PATH = "handson_ml3/my_folder/"
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets" , "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def download_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH, local_path=LOCAL_PATH):
    """Download the housing data and extract it to a csv"""
    if not os.path.isdir(local_path + housing_path):
        os.makedirs(local_path + housing_path)
    tgz_path = os.path.join(local_path + housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=local_path + housing_path)
    housing_tgz.close()
    return pd.read_csv(local_path + housing_path + '/housing.csv')