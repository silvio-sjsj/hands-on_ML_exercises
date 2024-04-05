"""This code is to implement the suggestion at the end of Exercise 1 from chapter 2"""
import sklearn
import pandas as pd
import numpy as np
from packaging import version
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from scipy import stats

from handson_ml3.my_folder.chapter_2.downloading_the_data import download_housing_data
from handson_ml3.my_folder.chapter_2.preprocessing import Preprocessor

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

housing = download_housing_data()

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

preprocessor = Preprocessor()
X_train = preprocessor.fit_transform(housing) # Just for test

param_grid = [
        {'svr__kernel': ['linear'], 'svr__C': [30000.0, 35000.0, 41000.0, 44000.0, 55000.0,
                                               62000.0, 70000.0, 83000.0]},
        {'svr__kernel': ['rbf'], 'svr__C': [1000.0, 25000.0, 32000.0, 40000.0, 50000.0,
                                            60000.0, 82000.0, 90000.0, 101000.0],
                                 'svr__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]

svr_pipeline = Pipeline([("preprocessing", preprocessor), ("svr", SVR())])
grid_search = GridSearchCV(svr_pipeline, param_grid, cv=3,
                           scoring='neg_root_mean_squared_error')

grid_search.fit(housing.iloc[:5000], housing_labels.iloc[:5000])
#grid_search.fit(housing, housing_labels)

svr_grid_search_rmse = -grid_search.best_score_
print(grid_search.best_params_)

print(svr_grid_search_rmse)

final_model = grid_search.best_estimator_

# Testing:

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)
final_rmse = mean_squared_error(y_test, final_predictions, squared=False)
print(final_rmse)

"""OR"""
final_rmse = root_mean_squared_error(y_test, final_predictions)
print(final_rmse)

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                        loc=squared_errors.mean(),
                        scale=stats.sem(squared_errors)))