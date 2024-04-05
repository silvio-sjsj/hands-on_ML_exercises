"""This code is to implement the suggestion at the end of Exercise 1 from chapter 2"""
import sklearn
import pandas as pd
import numpy as np
from packaging import version
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import uniform
from scipy.stats import expon, loguniform
from sklearn.model_selection import cross_val_score

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

# Define the parameter distributions including the threshold
param_distribs = {
    'selector__threshold': uniform(0, 0.05),
    'svr__C': loguniform(20, 200_000),
    'svr__gamma': expon(scale=1.0),
    'svr__kernel': ['linear', 'rbf']
}

# Create the pipeline with a placeholder threshold value
selector_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('selector', SelectFromModel(RandomForestRegressor(random_state=42))),
    ('svr', SVR()),
])

# Define the random search
rnd_search = RandomizedSearchCV(selector_pipeline,
                                param_distributions=param_distribs,
                                n_iter=50, cv=3,
                                scoring='neg_root_mean_squared_error',
                                verbose=2,
                                random_state=42)

rnd_search.fit(housing.iloc[:5000], housing_labels.iloc[:5000])

# Get the best parameters
best_threshold = rnd_search.best_params_['selector__threshold']
best_svr_C = rnd_search.best_params_['svr__C']
best_svr_gamma = rnd_search.best_params_['svr__gamma']
best_svr_kernel = rnd_search.best_params_['svr__kernel']

print(best_threshold)
print(best_svr_C)
print(best_svr_gamma)
print(best_svr_kernel)

# Update the pipeline with the best parameters
selector_pipeline.set_params(
    selector__threshold=best_threshold,
    svr__C=best_svr_C,
    svr__gamma=best_svr_gamma,
    svr__kernel=best_svr_kernel
)

selector_rmses = -cross_val_score(selector_pipeline,
                                  housing.iloc[:5000],
                                  housing_labels.iloc[:5000],
                                  scoring="neg_root_mean_squared_error",
                                  cv=3)
print(pd.Series(selector_rmses).describe())

final_model = rnd_search.best_estimator_

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
print(np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                        loc=squared_errors.mean(),
                        scale=stats.sem(squared_errors))))