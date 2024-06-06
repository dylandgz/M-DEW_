import inspect
from itertools import combinations
import json
import sys
from typing import Iterable, Union
import warnings

warnings.filterwarnings("ignore")

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.impute._base import _BaseImputer
from tqdm import tqdm

# sys.path.append('.')
import sys
sys.path.append('/')

from data_loaders import Dataset, MedicalDataset
# from data_loaders import Dataset, MedicalDataset
from utils import find_optimal_threshold, Test

__all__ = [
    'BaseInheritance', 
    'BaseInheritanceImpute', 
    'EstimatorWithImputation',
    'InheritanceCompatibleEstimator'
]


class NullClassifier(ClassifierMixin):
    def __init__(self, c=0.5) -> None:
        self.c = c

    def fit(self, X=None, y=None):
        pass

    def predict_proba(self, X=None):
        if X is None:
            return np.array([[self.c, self.c]])
        else:
            n_samples = len(X)
            return np.array([[self.c, self.c] * n_samples])

    def predict(self, X):
        return 1

    def __sklearn_is_fitted__(self):
        """Necessary for Stacking"""
        return True


class BaseInheritanceClassifier(ClassifierMixin):
    def __init__(
        self,
        data: Union[Dataset, MedicalDataset],
        base_estimator: BaseEstimator,
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
        self.target_col = data.target_col
        self.tests = data.tests
        self.base_features = data.base_features
        self.feature_to_test_map = data.feature_to_test_map
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
        self.prediction_method = 'predict_proba'
        del base_estimator_instance
        self.dag = nx.DiGraph()
        self.build_dag()

    def build_dag(self) -> None:
        # df = self.data.drop(columns=[self.target_col])
        df = self.data[0][0].copy() # get train dataset for first cv fold
        df = df.drop(self.data.target_col, axis=1)

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
        
        for i, level in tqdm(enumerate(levels)):
        
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

    def fit_node_estimators(self, cv_fold_index=0):
        feature_set = nx.get_node_attributes(self.dag, 'features')

        with tqdm(total=len(self.dag.nodes)) as pbar:
            
            for node in self.dag.nodes:
                
                features = list(set(feature_set[node]))
                print('--------\n---------\n')
                print(features)
                if len(features) == 0:
                    print('zero features')
                    model = NullClassifier(c=0.5)
                    model = InheritanceCompatibleClassifier(model, node)

                    self.prediction_method = 'predict_proba'
                    self.dag.nodes[node]['model'] = model
                    self.dag.nodes[node]['predictions'] = {}
                    self.dag.nodes[node]['passthrough_predictions'] = {}
                    self.dag.nodes[node]['ground_truth'] = {}
                    self.dag.nodes[node]['errors'] = []
                    self.dag.nodes[node]['n_samples'] = len(df)
                    self.dag.nodes[node]['optim_threshold'] = 0.5
                    
                    continue


                df = self.data[cv_fold_index][0].copy()
                df = df.loc[:, features + [self.target_col]]
                df = df.dropna(axis=0)
                targets = df[self.data.target_col]

                for i, f in enumerate(df.columns):
                    if f == self.target_col:
                        continue
                    elif f not in features:
                        df[f] = [np.NaN] * len(df)

                model = self.base_estimator(**self.base_estimator_params)

                model.fit(df, targets)
                model = InheritanceCompatibleClassifier(model, node)
                train_proba_predictions = model.predict_proba(df)
                t = find_optimal_threshold(train_proba_predictions, targets)

                self.prediction_method = model.prediction_method
                self.dag.nodes[node]['model'] = model
                self.dag.nodes[node]['predictions'] = {}
                self.dag.nodes[node]['passthrough_predictions'] = {}
                self.dag.nodes[node]['ground_truth'] = {}
                self.dag.nodes[node]['errors'] = []
                self.dag.nodes[node]['n_samples'] = len(df)
                self.dag.nodes[node]['optim_threshold'] = t
                
                pbar.update(1)


class BaseNonmissingSubspaceClassifier(ClassifierMixin):
    def __init__(
        self,
        data: MedicalDataset,
        base_estimator: BaseEstimator,
        base_estimator_params: dict = {},
        prediction_method: str = 'auto',
        use_optimal_threshold=True,
        threshold=0.5
    ) -> None:
        if not inspect.isclass(base_estimator):
            # if base_estimator is instance, get its class...
            # ... since we instantiate later in `build_dag()`
            self.base_estimator = type(base_estimator)
        else:
            self.base_estimator = base_estimator
        self.data = data
        self.target_col = data.target_col
        self.tests = data.tests
        self.base_features = data.base_features
        self.feature_to_test_map = data.feature_to_test_map
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
        self.prediction_method = 'predict_proba'
        del base_estimator_instance
        self.use_optimal_threshold = use_optimal_threshold
        self.threshold = threshold
        self.dag = nx.DiGraph()
        self.build_dag()

    def build_dag(self) -> None:
        # df = self.data.drop(columns=[self.target_col])
        df = self.data[0][0].copy() # get train
        df = df.drop(self.data.target_col, axis=1)

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

    def fit_node_estimators(self, X=None, y=None, cv_fold_index=None):
        feature_set = nx.get_node_attributes(self.dag, 'features')

        with tqdm(total=len(self.dag.nodes)) as pbar:
            
            for node in self.dag.nodes:
                
                features = feature_set[node]
                if len(features) == 0:
                    model = NullClassifier(c=0.5)
                    model = InheritanceCompatibleClassifier(model, node)

                    self.prediction_method = 'predict_proba'
                    self.dag.nodes[node]['model'] = model
                    self.dag.nodes[node]['predictions'] = {}
                    self.dag.nodes[node]['passthrough_predictions'] = {}
                    self.dag.nodes[node]['ground_truth'] = {}
                    self.dag.nodes[node]['errors'] = []
                    self.dag.nodes[node]['n_samples'] = 0
                    self.dag.nodes[node]['optim_threshold'] = 0.5
                    
                    continue
                if cv_fold_index is not None:
                    df = self.data[cv_fold_index][0].copy()
                elif X is not None and y is not None:
                    df = X.copy()
                    df[self.data.target_col] = y
                df = df.loc[:, features + [self.target_col]]
                df = df.dropna(axis=0)
                targets = df[self.target_col]
                df = df.drop(columns=[self.target_col])
                # print('df cols: ' + str(list(df.columns)))

                model = self.base_estimator(**self.base_estimator_params)

                model = InheritanceCompatibleClassifier(model, node)
                model.fit(df, targets, use_optimal_threshold=self.use_optimal_threshold)
                # train_proba_predictions = model.predict_proba(df)[:, 1]
                # t = find_optimal_threshold(train_proba_predictions, targets)

                self.prediction_method = model.prediction_method
                self.dag.nodes[node]['model'] = model
                self.dag.nodes[node]['predictions'] = {}
                self.dag.nodes[node]['passthrough_predictions'] = {}
                self.dag.nodes[node]['ground_truth'] = {}
                self.dag.nodes[node]['errors'] = []
                self.dag.nodes[node]['n_samples'] = len(df)
                self.dag.nodes[node]['optim_threshold'] = model.threshold
                
                pbar.update(1)

    def write_dag(self, outfile, write_mode='w'):
        dag_json = json.dump(dict(self.dag))
        with open(outfile, write_mode) as f:
            f.write(dag_json)

    def __sklearn_is_fitted__(self):
        """Necessary for Stacking"""
        return True


class ClassifierWithImputation(BaseEstimator, ClassifierMixin):
    def __init__(
        self, 
        estimator: BaseEstimator, 
        imputer: _BaseImputer,
        threshold=None,

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
        self.estimator_name = str(type(self.estimator).__name__)
        self.imputer_name = str(type(self.imputer).__name__)
        if hasattr(self.imputer, 'estimator'):
            self.imputer_name += '(' + str(type(self.imputer.estimator)) + ')'
        self.name = self.estimator_name + '_imputation_' + self.imputer_name
        if threshold is None:
            self.threshold = 0.5
        else:
            self.threshold = threshold
        self.X_train_imputed = None  # Initialize storage for imputed data Dylan    
        self.last_imputed_X_test = None  # Initialize storage for imputed data Dylan
        # self.last_imputed_X_val = None  # Initialize storage for imputed data Dylan

    def fit(self, X, y, use_optimal_threshold=False):
        # impute
        X_imputed = self.imputer.fit_transform(X=X, y=y)
        self.X_train_imputed= X_imputed  # Store the last imputed data Dylan
        # self.last_imputed_X_train = X_imputed  # Store the last imputed data Dylan
        self.imputer_is_fitted = True
        self.estimator.fit(X_imputed, y)
        self.estimator_is_fitted = True
        if use_optimal_threshold:
            self.set_optimal_threshold(X, y)
        if isinstance(y, list):
            self.classes = list(set(y))
        elif isinstance(y, np.ndarray):
            if len(y.shape) > 1:
                self.classes = np.eye(y.shape[1])
            else:
                self.classes = np.unique(y)
        

    @property
    def classes_(self):
        try: 
            return self.classes
        except:
            raise Exception('Model has not been fit.')




    def predict_proba(self, X):
        assert self.estimator_is_fitted
        X_imputed = self.imputer.transform(X)
        self.X_test_imputed=X_imputed
        # self.last_imputed_X_test = X_imputed  # Store the last imputed data X test Dylan
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
        
        if isinstance(predictions, list): # random forest
            predictions = predictions[-1]
        
        

        return predictions  #, X_imputed

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        predictions = np.array([0 if x < self.threshold else 1 for x in probs])
        return predictions

    def set_optimal_threshold(self, X, y) -> None:
        proba_predictions = self.predict_proba(X)
        threshold = find_optimal_threshold(proba_predictions, y)
        self.threshold = threshold

    def __sklearn_is_fitted__(self):
        """Necessary for Stacking"""
        return True


class RegressorWithImputation(BaseEstimator, RegressorMixin):
    def __init__(
        self, 
        estimator: BaseEstimator, 
        imputer: _BaseImputer,
    ) -> None:
        if inspect.isclass(estimator):
            self.estimator = estimator()
        else: # instantiated estimator
            self.estimator = estimator
        if inspect.isclass(imputer):
            self.imputer = imputer()
        else:
            self.imputer = imputer
        self.imputer_is_fitted = False
        self.estimator_is_fitted = False
        self.estimator_name = str(type(self.estimator).__name__)
        self.imputer_name = str(type(self.imputer).__name__)
        if hasattr(self.imputer, 'estimator'):
            self.imputer_name += '(' + str(type(self.imputer.estimator)) + ')'
        self.name = self.estimator_name + '_imputation_' + self.imputer_name

    def fit(self, X, y, use_optimal_threshold=False):
        # impute
        X_imputed = self.imputer.fit_transform(X=X, y=y)
        self.imputer_is_fitted = True
        self.estimator.fit(X_imputed, y)
        self.estimator_is_fitted = True
        if use_optimal_threshold:
            self.set_optimal_threshold(X, y)
        if isinstance(y, list):
            self.classes = list(set(y))
        elif isinstance(y, np.ndarray):
            if len(y.shape) > 1:
                self.classes = np.eye(y.shape[1])
            else:
                self.classes = np.unique(y)

    def predict(self, X):
        assert self.estimator_is_fitted
        X_imputed = self.imputer.transform(X)
        predictions = self.estimator.predict(X_imputed)
        
        if isinstance(predictions, list): # random forest
            predictions = predictions[-1]
            if not isinstance(predictions, np.ndarray):
                raise RuntimeError('Predictions are of type {str(type(predictions))}, must be np.ndarray')

        return predictions

    def set_optimal_threshold(self, X, y) -> None:
        raise RuntimeWarning('set_optimal_threshold is being called on a Regressor; this is not well-defined.')

    def __sklearn_is_fitted__(self):
        """Necessary for Stacking"""
        return True


class InheritanceCompatibleClassifier(ClassifierMixin):
    def __init__(self, estimator, node, prediction_method='auto', threshold=0.5):
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
        self.threshold = threshold

    def fit(self, X, y, use_optimal_threshold=True):
        self.estimator.fit(X, y)
        if use_optimal_threshold:
            self.set_optimal_threshold(X, y)

    def predict_proba(self, X):
        predictions = self.estimator.predict_proba(X)
        return predictions

    def predict(self, X):
        probabilities = self.predict_proba(X)
        predictions = [0 if p < self.threshold else 1 for p in probabilities]
        return predictions

    def set_optimal_threshold(self, X, y) -> None:
        proba_predictions = self.predict_proba(X)
        try:
            proba_predictions = proba_predictions[:, 1]
            threshold = find_optimal_threshold(proba_predictions, y, step=1)
        except: # this means there were no samples; just use 0.5 here
            threshold = 0.5
        self.threshold = threshold

    def __sklearn_is_fitted__(self):
        """Necessary for Stacking"""
        return True


class IdentityImputer:
    def fit(self, X, y):
        pass

    def transform(self, X):
        return X

    def fit_transform(self, X, y):
        return X