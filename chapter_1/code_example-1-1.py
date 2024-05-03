"""."""
import os
import sklearn
import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model

# Local variables
local_path = "handson-ml3/my_folder/"
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
datapath = os.path.join("datasets", "lifesat", "")
os.makedirs(local_path + datapath, exist_ok=True)

#This function merges the OECD's life satisfaction data and the IMF's GDP per capita
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

# Download the data
for filename in ("oecd_bli_2015.csv", "gdp_per_capita.csv"):
    print("Downloading", filename)
    url = DOWNLOAD_ROOT + "datasets/lifesat/" + filename
    urllib.request.urlretrieve(url, local_path + datapath + filename)

# Code example_1-1

# Load the data

oecd_bli = pd.read_csv(local_path + datapath + "oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv(local_path + datapath + "gdp_per_capita.csv",
            thousands=',', delimiter='\t',
            encoding='latin1', na_values="n/a")

# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()

# Select a linear model
model = sklearn.linear_model.LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[22587]]  # Cyprus' GDP per capita
print(model.predict(X_new)) # outputs [[ 5.96242338]]

#####################################################################################
#####################################################################################

"""
If you had used an instance-based learning algorithm instead, you would have found
that Slovenia has the closest GDP per capita to that of Cyprus ($20,732), and since
the OECD data tells us that Slovenians’ life satisfaction is 5.7, you would have
predicted a life satisfaction of 5.7 for Cyprus. If you zoom out a bit and look at the
two next-closest countries, you will find Portugal and Spain with life satisfactions of
5.1 and 6.5, respectively. Averaging these three values, you get 5.77, which is pretty
close to your model-based prediction. This simple algorithm is called k-Nearest
Neighbors regression (in this example, k = 3).
"""

import sklearn.neighbors

model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[22587]]  # Cyprus' GDP per capita
print(model.predict(X_new)) # outputs [[ 5.76666666667]]

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "fundamentals"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)