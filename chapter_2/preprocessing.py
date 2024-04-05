import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
#from sklearn import set_config

#set_config(transform_output='pandas')

class Preprocessor(TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        preprocessing = self._preprocessing()
        return preprocessing.fit_transform(X)

    #def _column_ratio(self, X):
    #    ratio = X.iloc[:, 0] / X.iloc[:, 1]
    #    return np.reshape(ratio.to_numpy(), (-1, 1))

    def _column_ratio(self, X):
        return X[:, [0]] / X[:, [1]]

    def _ratio_name(self, function_transformer, feature_names_in):
        return ["ratio"]

    def _ratio_pipeline(self):
        return make_pipeline(
            SimpleImputer(strategy="median"),
            FunctionTransformer(self._column_ratio, feature_names_out=self._ratio_name),
            StandardScaler()
        )

    def _log_pipeline(self):
        return make_pipeline(
            SimpleImputer(strategy="median"),
            FunctionTransformer(np.log, feature_names_out="one-to-one"),
            StandardScaler()
        )

    def _cat_pipeline(self):
        return make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore",
                          categories=[['<1H OCEAN',
                                      'INLAND',
                                      'ISLAND',
                                      'NEAR BAY',
                                      'NEAR OCEAN']]
        ))

    def _default_num_pipeline(self):
        return make_pipeline(SimpleImputer(strategy="median"),
                             StandardScaler()
        )
        
    def cluster_simil(self):
        self._cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
        return self._cluster_simil

    def _preprocessing(self):

        self.cluster_simil()
        
        return ColumnTransformer([
            ("bedrooms", self._ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
            ("rooms_per_house", self._ratio_pipeline(), ["total_rooms", "households"]),
            ("people_per_house", self._ratio_pipeline(), ["population", "households"]),
            ("log", self._log_pipeline(), ["total_bedrooms", "total_rooms", "population",
                                           "households", "median_income"]),
            ("geo", self._cluster_simil, ["latitude", "longitude"]),
            ("cat", self._cat_pipeline(), make_column_selector(dtype_include=object)),
    ],
    remainder=self._default_num_pipeline())  # one column remaining: housing_median_age

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]