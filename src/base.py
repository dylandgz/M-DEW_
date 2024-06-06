
import inspect
from itertools import combinations
from typing import Iterable
import warnings
warnings.filterwarnings("ignore")

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute._base import _BaseImputer
from tqdm import tqdm


__all__ = [
    'BaseInheritance', 
    'BaseInheritanceImpute', 
    'EstimatorWithImputation',
    'InheritanceCompatibleEstimator',
    'Test'
]


class Test:
    def __init__(self, name: str='', filename: str='', features=[], cost=0) -> None:
        self.name = name
        self.filename = filename
        self.features = features
        self.cost = cost

    def get_test_features(self):
        return self.features

    def set_test_features(self, features):
        self.features = list(features)

    def add_test_features(self, features):
        self.features += list(features)

    def build_data(self, df):
        return df[list(self.features)]

    def get_cost(self):
        return self.cost

    def set_cost(self, cost):
        self.cost = cost


class BaseInheritance():
    def __init__(
        self,
        base_estimator: BaseEstimator,
        data: pd.DataFrame,
        target_col: str,
        tests: Iterable[Test],
        base_features: Iterable[str],
        feature_to_test_map: dict,
        base_estimator_params: dict = {},
        prediction_method: str = 'auto'
    ) -> None:
        if not inspect.isclass(base_estimator):
            # if base_estimator is instance, get its class...
            # ... since we instantiate later in `build_dag()`
            self.base_estimator = type(base_estimator)
        else:
            self.base_estimator = base_estimator
        self.data = data
        self.target_col = target_col
        self.tests = tests
        self.base_features = base_features
        self.feature_to_test_map = feature_to_test_map
        self.base_estimator_params = base_estimator_params
        base_estimator_instance = base_estimator()
        prediction_methods = ['predict_proba', 'predict', 'decision_function']
        if prediction_method == 'auto':
            if hasattr(base_estimator_instance, 'predict_proba'):
                self.prediction_method = 'predict_proba'
            elif hasattr(base_estimator_instance, 'decision_function'):
                self.prediction_method = 'decision_function'
            else:
                self.prediction_method = 'predict'
        else:
            assert prediction_method in prediction_methods, \
            "please choose one of {}, {}, {}".format(*prediction_methods)
            self.prediction_method = prediction_method
        # hard-code prediction_method for now.
        self.prediction_method = 'predict'
        del base_estimator_instance
        self.dag = nx.DiGraph()
        self.build_dag()

    def build_dag(self) -> None:
        df = self.data.drop(columns=[self.target_col])

        test_powerset = []
        
        for i in range(1, len(self.tests) + 1):
            test_powerset += combinations(self.tests, i)
        
        levels = [
            [x for x in test_powerset if len(x) == L + 1] 
            for L in range(len(self.tests))
        ]
        
        # node attribute schema: {(level, Tuple[tests]): {'features': feature_list}}
        base_indices = tuple(
            sorted([
                df.columns.get_loc(c) 
                for c in self.base_features
                if c != self.target_col
            ])
        )
        self.dag.add_node(
            node_for_adding=(0,base_indices),
            tests={},
            features=self.base_features, 
            predictions = {}, 
            errors=[], 
            ground_truth = {}
        )
        
        for i, level in enumerate(levels):
        
            for j, test_indices in enumerate(level):
        
                features_ = []
                features_ += self.base_features
                
                test_names = {}
                for t in test_indices:
                    features_ += self.tests[t].get_test_features()
                    test_names[t] = self.tests[t]

                col_indices = tuple(
                    sorted(
                        [df.columns.get_loc(c) for c in features_]
                    )
                )
                current_node = (i + 1, tuple(col_indices))
                self.dag.add_node(
                    node_for_adding=current_node, 
                    tests=test_names, 
                    features=features_, 
                    predictions = {}, 
                    errors=[], 
                    ground_truth = {}
                )
                nodes_one_level_up = [
                    node for node in self.dag.nodes if node[0] == i
                ]
                
                for node in nodes_one_level_up:
                    if all(idx in col_indices for idx in node[1]):
                        self.dag.add_edge(node, current_node)

    def fit_node_estimators(self):
        feature_set = nx.get_node_attributes(self.dag, 'features')

        with tqdm(total=len(self.dag.nodes)) as pbar:
            
            for node in self.dag.nodes:
                
                features = list(set(feature_set[node]))
                df = self.data.copy()
                targets = np.array(df[self.target_col])
                df = df[[c for c in df.columns if c != self.target_col]]

                for i, f in enumerate(df.columns):
                    if f == self.target_col:
                        continue
                    elif f not in features:
                        df[f] = [np.NaN] * len(df)

                model = self.base_estimator(**self.base_estimator_params)

                model.fit(df, targets)
                model = InheritanceCompatibleEstimator(model, node)
                self.prediction_method = model.prediction_method
                self.dag.nodes[node]['model'] = model
                self.dag.nodes[node]['predictions'] = {}
                self.dag.nodes[node]['passthrough_predictions'] = {}
                self.dag.nodes[node]['ground_truth'] = {}
                self.dag.nodes[node]['errors'] = []
                self.dag.nodes[node]['n_samples'] = len(df)
                
                pbar.update(1)


class BaseInheritanceImpute():
    def __init__(
        self,
        base_estimator: BaseEstimator,
        imputer: _BaseImputer,
        data: pd.DataFrame,
        target_col: str,
        tests: Iterable[Test],
        base_features: Iterable[str],
        feature_to_test_map: dict,
        base_estimator_params: dict = {},
        imputer_params: dict = {},
        prediction_method: str = 'auto'
    ) -> None:
        if not inspect.isclass(base_estimator):
            # if base_estimator is instance, get its class...
            # ... since we instantiate later in `build_dag()`
            self.base_estimator = type(base_estimator)
        else:
            self.base_estimator = base_estimator
        if not inspect.isclass(imputer):
            self.imputer = imputer
            # warning: need to supply imputer params in original instance?
        else:
            self.imputer = imputer(**imputer_params)
        X = data[[c for c in data.columns if c != target_col]]
        targets = data[target_col]
        X_cols = X.columns
        X = pd.DataFrame(data=self.imputer.fit_transform(X), columns=X_cols)
        X[target_col] = targets
        self.data = X
        self.target_col = target_col
        self.tests = tests
        self.base_features = base_features
        self.feature_to_test_map = feature_to_test_map
        self.base_estimator_params = base_estimator_params
        base_estimator_instance = base_estimator(**base_estimator_params)
        prediction_methods = ['predict_proba', 'predict', 'decision_function']
        if prediction_method == 'auto':
            if hasattr(base_estimator_instance, 'predict_proba'):
                self.prediction_method = 'predict_proba'
            elif hasattr(base_estimator_instance, 'decision_function'):
                self.prediction_method = 'decision_function'
            else:
                self.prediction_method = 'predict'
        else:
            assert prediction_method in prediction_methods, \
            "please choose one of {}, {}, {}".format(*prediction_methods)
            self.prediction_method = prediction_method
        # hard-code prediction_method for now.
        self.prediction_method = 'predict'
        del base_estimator_instance
        self.dag = nx.DiGraph()
        self.build_dag()

    def build_dag(self) -> None:
        df = self.data.drop(columns=[self.target_col])

        test_powerset = []
        
        for i in range(1, len(self.tests) + 1):
            test_powerset += combinations(self.tests, i)
        
        levels = [
            [x for x in test_powerset if len(x) == L + 1] 
            for L in range(len(self.tests))
        ]
        
        # node attribute schema: {(level, Tuple[tests]): {'features': feature_list}}
        base_indices = tuple(
            sorted([
                df.columns.get_loc(c) 
                for c in self.base_features
                if c != self.target_col
            ])
        )
        self.dag.add_node(
            node_for_adding=(0,base_indices),
            tests={},
            features=self.base_features, 
            predictions = {}, 
            errors=[], 
            ground_truth = {}
        )
        
        for i, level in enumerate(levels):
        
            for j, test_indices in enumerate(level):
        
                features_ = []
                features_ += self.base_features
                
                test_names = {}
                for t in test_indices:
                    features_ += self.tests[t].get_test_features()
                    test_names[t] = self.tests[t]

                col_indices = tuple(
                    sorted(
                        [df.columns.get_loc(c) for c in features_]
                    )
                )
                current_node = (i + 1, tuple(col_indices))
                self.dag.add_node(
                    node_for_adding=current_node, 
                    tests=test_names, 
                    features=features_, 
                    predictions = {}, 
                    errors=[], 
                    ground_truth = {}
                )
                nodes_one_level_up = [
                    node for node in self.dag.nodes if node[0] == i
                ]
                
                for node in nodes_one_level_up:
                    if all(idx in col_indices for idx in node[1]):
                        self.dag.add_edge(node, current_node)

    def fit_node_estimators(self):
        print('getting features')
        feature_set = nx.get_node_attributes(self.dag, 'features')

        with tqdm(total=len(self.dag.nodes)) as pbar:
            
            for node in self.dag.nodes:
                print('starting global fitting')
                features = list(set(feature_set[node]))
                df = self.data.copy()
                targets = np.array(df[self.target_col])
                df = df[[c for c in df.columns if c != self.target_col]]

                for i, f in enumerate(df.columns):
                    if f == self.target_col:
                        continue
                    elif f not in features:
                        df[f] = [np.NaN] * len(df)

                model = self.base_estimator(**self.base_estimator_params)

                model.fit(df, targets)
                model = InheritanceCompatibleEstimator(model, node)
                self.prediction_method = model.prediction_method
                self.dag.nodes[node]['model'] = model
                self.dag.nodes[node]['predictions'] = {}
                self.dag.nodes[node]['passthrough_predictions'] = {}
                self.dag.nodes[node]['ground_truth'] = {}
                self.dag.nodes[node]['errors'] = []
                self.dag.nodes[node]['n_samples'] = len(df)
                
                pbar.update(1)


class EstimatorWithImputation(BaseEstimator):
    def __init__(
        self, 
        estimator: BaseEstimator, 
        imputer: _BaseImputer,
        prediction_method: str = 'auto'
    ) -> None:
        if inspect.isclass(estimator):
            self.estimator = estimator()
        else: # instantiated estimator
            self.estimator = estimator
        if inspect.isclass(imputer):
            self.imputer = imputer()
        else:
            self.imputer = imputer
        self.prediction_method = prediction_method
        self.imputer_is_fitted = False
        self.estimator_is_fitted = False
        estimator_name = type(self.estimator).__name__
        imputer_name = type(imputer).__name__
        self.name = estimator_name + '_imputation_' + imputer_name

    def fit(self, X, y):
        # impute
        X_imputed = self.imputer.fit_transform(X=X)
        self.imputer_is_fitted = True
        self.estimator.fit(X_imputed, y)
        self.estimator_is_fitted = True

    def predict(self, X):
        assert self.estimator_is_fitted
        X_imputed = self.imputer.transform(X)
        if self.prediction_method == "auto":
            if hasattr(self.estimator, "predict_proba"):
                predictions = self.estimator.predict_proba(X_imputed)
            elif hasattr(self.estimator, "decision_function"):
                predictions = self.estimator.decision_function(X_imputed)
            else:
                predictions = self.estimator.predict(X_imputed)
        else:
            if not hasattr(self.estimator, self.prediction_method):
                raise ValueError(
                    "Underlying estimator does not implement {}.".format(
                        self.prediction_method
                    )
                )
            predictions = getattr(self.estimator, self.prediction_method)(
                X_imputed
            )

        return predictions


class InheritanceCompatibleEstimator(ClassifierMixin):
    def __init__(self, estimator, node, prediction_method='auto'):
        self.estimator = estimator
        self.node = node
        self.level = node[0]
        self.indices = node[1]
        prediction_methods = ['predict_proba', 'predict', 'decision_function']
        if prediction_method == 'auto':
            if hasattr(estimator, 'predict_proba'):
                self.prediction_method = 'predict_proba'
            elif hasattr(estimator, 'decision_function'):
                self.prediction_method = 'decision_function'
            else:
                self.prediction_method = 'predict'
        else:
            assert prediction_method in prediction_methods, \
            "please choose one of {}, {}, {}".format(*prediction_methods)
            self.prediction_method = prediction_method

    def fit(self, X, y):
        self.estimator.fit(X, y)

    def predict(self, X):
        predictions = getattr(self.estimator, self.prediction_method)(X)
        return predictions

    def __sklearn_is_fitted__(self):
        """Necessary for Stacking"""
        return True
