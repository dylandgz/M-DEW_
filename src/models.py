from collections import Counter
from distutils.log import Log, warn
import inspect
from itertools import combinations
from logging import warning
from queue import Full
from typing import Iterable, Mapping, Union
import warnings
warnings.filterwarnings("ignore")

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import sklearn
# from sklearn.experimental import enable_iterative_imputer, enable_hist_gradient_boosting
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaseEnsemble, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.impute._base import _BaseImputer
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import xgboost as xgb

from .base import BaseInheritance, BaseInheritanceImpute,\
    EstimatorWithImputation, InheritanceCompatibleEstimator, Test
from .data_loaders import load_numom2b_analytes, PlacentalAnalytesTests
from .utils import get_n_tests_missing, label_encoded_data,\
    find_optimal_threshold


__all__ = [
    'FullInheritance', 'StackedHeterogenousClassifier',
    'StackedHeterogenousRegressor', 'StackedInheritanceEstimator',
    'StackedParametricClassifier'
]


class FullInheritance(BaseInheritance):
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
    ):
        super().__init__(
            base_estimator=base_estimator,
            data=data,
            target_col=target_col,
            tests=tests,
            base_features=base_features,
            feature_to_test_map=feature_to_test_map,
            base_estimator_params=base_estimator_params,
            prediction_method=prediction_method
        )
        self.name = type(base_estimator).__name__ + '_FullInheritance'
        # just hard-code for now to be compatible with 
        # InheritanceCompatibleEstimator. Can refactor later.
        self.prediction_method = 'predict'

    def fit(self):
        print('attempting fit')
        super().fit_node_estimators()
        self.is_fit = True

    def predict(self, X_test: pd.DataFrame):
    
        if isinstance(X_test, pd.DataFrame):
            cols = [c for c in X_test.columns if c != self.target_col]
            X_test = X_test.to_numpy()

        assert len(X_test.shape) == 2

        predictions = []
        nodes_seen = []
        
        for i in tqdm(range(X_test.shape[0])):

            sample = X_test[i,:].astype(np.float32)
            raw_sample = sample.copy()
            indices = np.ravel(np.argwhere(np.isnan(sample)))
            level = len(set(list(self.feature_to_test_map.values()))) \
                - get_n_tests_missing(cols, self.feature_to_test_map, indices)
            sample_1d = np.array([
                sample[i] if i not in indices else np.nan
                for i in range(len(sample))
            ])
            sample = sample_1d.reshape(1,-1).astype(np.float32)
            # create node: pair[level, tuple[relevant indices]]
            node = (
                level, 
                tuple(
                    [x for x in range(X_test.shape[1]) if x not in indices]
                )
            )
            nodes_seen.append(node)
            model = self.dag.nodes[node]['model']
            
            # ancestor_predictions includes prediction at current node as well
            # so initialize ancestor predictions with current node prediction
            # first_prediction = getattr(model, self.prediction_method)(sample)[0]
            # first_prediction = getattr(model, 'predict')(raw_sample)[0]
            first_prediction = getattr(model, 'predict')(sample)[0]
            self.dag.nodes[node]['passthrough_predictions'][i] = first_prediction
            ancestor_predictions = [first_prediction]

            df = self.data.drop(columns=[self.target_col])

            if level != 0:
                for a in nx.ancestors(self.dag, node):
                    
                    ancestor_model = self.dag.nodes[a]['model']
                    # current_prediction = ancestor_model.predict(raw_sample)[0]
                    current_prediction = ancestor_model.predict(sample)[0]
                    self.dag.nodes[a]['passthrough_predictions'][i] = current_prediction
                    ancestor_predictions.append(current_prediction)
            prediction = sum(ancestor_predictions) / len(ancestor_predictions)

            predictions.append(prediction)
            self.dag.nodes[node]['predictions'][i] = prediction
            
        return np.array(predictions)

    def find_each_model_val_threshold(self):
        """
        for each model in the DAG, find its optimal threshold for discerning
        positive vs. negative class.
        """
        for node in self.dag.nodes:
            model = self.dag.nodes[node]['model']
            #TODO{construct X_val, y_val}
            proba_predictions = model.predict(self.X_val)
            threshold = find_optimal_threshold(
                proba_predictions, self.y_test, step=0.1
            )
            

class StratificationOnly(BaseInheritance):

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
    ):
        super().__init__(
            base_estimator=base_estimator,
            data=data,
            target_col=target_col,
            tests=tests,
            base_features=base_features,
            feature_to_test_map=feature_to_test_map,
            base_estimator_params=base_estimator_params,
            prediction_method=prediction_method
        )
        self.name = type(base_estimator).__name__ + '_StratificationOnly'
           

    def fit(self):
        super().fit_node_estimators()

    def predict(self, X_test: pd.DataFrame):

        if isinstance(X_test, pd.DataFrame):
            cols = [c for c in X_test.columns if c != self.target_col]
            X_test = X_test.to_numpy()

        assert len(X_test.shape) == 2

        predictions = []
        nodes_seen = []
        
        for i in tqdm(range(X_test.shape[0])):

            sample = X_test[i,:].astype(np.float32)
            # raw_sample = sample.copy()
            indices = np.ravel(np.argwhere(np.isnan(sample)))
            level = len(set(list(self.feature_to_test_map.values()))) \
                - get_n_tests_missing(cols, self.feature_to_test_map, indices)
            # sample = np.array([
                # sample[i] 
                # if i not in indices else np.nan
                # for i in range(len(sample))
            # ])
            # sample = sample.reshape(1,-1).astype(np.float32)
            # create node: pair[level, tuple[relevant indices]]
            node = (
                level, 
                tuple(
                    [x for x in range(X_test.shape[1]) if x not in indices]
                )
            )
            nodes_seen.append(node)
            model = self.dag.nodes[node]['model']
            
            # prediction = getattr(model, 'predict')(raw_sample)[0]
            prediction = getattr(model, 'predict')(sample)[0]
            predictions.append(prediction)
            self.dag.nodes[node]['predictions'][i] = prediction

        return np.vstack(predictions)


class StackedHeterogenousClassifier(StackingClassifier):
    def __init__(
        self,
        estimators,
        final_estimator=None,
        *,
        cv=None,
        stack_method="auto",
        n_jobs=None,
        passthrough=False,
        verbose=0
    ):
        super().__init__(
            estimators,
            final_estimator,
            cv,
            stack_method,
            n_jobs,
            passthrough,
            verbose
        )
        self.name = type(final_estimator).__name__ + \
            '_StackedHeterogeneousClassifier'


class StackedHeterogenousRegressor(StackingRegressor):
    def __init__(
        self,
        estimators,
        final_estimator=None,
        *,
        cv=None,
        stack_method="auto",
        n_jobs=None,
        passthrough=False,
        verbose=0
    ):
        super().__init__(
            estimators,
            final_estimator,
            cv,
            stack_method,
            n_jobs,
            passthrough,
            verbose
        )
        self.name = type(final_estimator).__name__ + \
            '_StackedHeterogeneousRegressor'


class StackedInheritanceEstimator(BaseInheritance):
    def __init__(
        self,
        base_estimator: BaseEstimator,
        meta_estimator: BaseEstimator,
        data: pd.DataFrame,
        target_col: str,
        tests: Iterable[Test],
        base_features: Iterable[str],
        feature_to_test_map: dict,
        base_estimator_params: dict = {},
        meta_estimator_params: dict = {},
        prediction_method: str = 'auto',
        prediction_type: str = 'auto'
    ):
        super().__init__(
            base_estimator=base_estimator,
            data=data,
            target_col=target_col,
            tests=tests,
            base_features=base_features,
            feature_to_test_map=feature_to_test_map,
            base_estimator_params=base_estimator_params,
            prediction_method=prediction_method
        )
        self.meta_estimator_params = meta_estimator_params
        if not inspect.isclass(meta_estimator):
            meta_estimator = type(meta_estimator)
        else:
            self.meta_estimator = meta_estimator
        
        if prediction_type == 'auto':
            # infer whether classification or regression
            if len(Counter(self.data[target_col])) < np.sqrt(len(self.data)):
                self.prediction_type = 'classification'
            else:
                self.prediction_type = 'regression'
        else:
            assert prediction_type in ['classification', 'regression']
            self.prediction_type = prediction_type
        self.name = type(base_estimator).__name__ + \
            '_StackedInheritance_Meta(' + self.meta_estimator.__name__ + ')'

    def fit(self):
        print('fitting base models...')
        super().fit_node_estimators()
        print('fitting stacking models...')
        # fit stacking classifier at each node using ancestor models
        with tqdm(total=len(self.dag.nodes)) as pbar:
            for node in self.dag.nodes:
                estimators = [(str(node), self.dag.nodes[node]['model'])]
                if node[0] != 0:    
                    for ancestor in nx.ancestors(self.dag, node):
                        ancestor_model = self.dag.nodes[ancestor]['model']
                        estimators.append((str(ancestor), (ancestor_model)))
                    
                if self.prediction_type == 'classification':
                    stacking_model = StackingClassifier(
                        estimators=estimators,
                        final_estimator=self.meta_estimator(**self.meta_estimator_params),
                        cv='prefit'
                    )
                else:
                    stacking_model = StackingRegressor(
                        estimators=estimators,
                        final_estimator=self.meta_estimator(**self.meta_estimator_params),
                        cv='prefit'
                    )
                indices = node[1]
                X = self.data[[c for c in self.data.columns if c != self.target_col]]
                y = self.data[self.target_col]
                # current_node_rows = self.data.iloc[:, list(indices)].dropna(
                #     axis=0
                # ).index
                # current_data = self.data.loc[current_node_rows, :]
                # X = current_data[
                #     [c for c in current_data.columns if c != self.target_col]
                # ]
                # y = current_data[self.target_col]
                stacking_model.fit(X, y)
                self.dag.nodes[node]['stacking_model'] = stacking_model
                pbar.update(1)

    def predict(self, X_test: pd.DataFrame):

        if isinstance(X_test, pd.DataFrame):
            cols = X_test.columns
            X_test = X_test.to_numpy()

        assert len(X_test.shape) == 2

        predictions = []
        nodes_seen = []
        
        for i in tqdm(range(X_test.shape[0])):

            sample = X_test[i,:].astype(np.float32)
            raw_sample = sample.copy()
            indices = np.ravel(np.argwhere(np.isnan(sample)))
            level = len(set(list(self.feature_to_test_map.values()))) \
                - get_n_tests_missing(cols, self.feature_to_test_map, indices)
            sample = np.array([
                sample[i] 
                if i not in indices else np.nan
                for i in range(len(sample))
            ])
            sample = sample.reshape(1,-1).astype(np.float32)
            # create node: pair[level, tuple[relevant indices]]
            node = (
                level, 
                tuple(
                    [x for x in range(X_test.shape[1]) if x not in indices]
                )
            )
            nodes_seen.append(node)
            stacking_model = self.dag.nodes[node]['stacking_model']
            
            prediction = getattr(
                stacking_model, self.prediction_method
            )(sample)
            predictions.append(prediction)
            self.dag.nodes[node]['predictions'][i] = prediction

        return np.vstack(predictions)


class StackedParametricClassifier(nn.Module):
    def __init__(
        self,
        estimators: Union[
            Iterable[BaseEstimator], Mapping[str, BaseEstimator]
        ],
        input_dim: int
    ) -> None:
        super().__init__()

        if isinstance(estimators, Mapping):
            self.estimators = estimators
        else:
            estimators_dict = {}
            for model in estimators:
                estimators_dict[type(model).__name__] = model
            self.estimators = estimators_dict

        self.nn_layer1 = nn.Linear(input_dim, input_dim)
        self.nn_layer2 = nn.Linear(input_dim, len(self.estimators))

    def forward(self, X):
        estimators_predictions = []
        for model in self.estimators.values():
            estimators_predictions.append(model.predict_proba(X))
        estimators_predictions = torch.Tensor(
            np.array(estimators_predictions)
        )

        X = F.relu(self.nn_layer1(X))
        X = self.nn_layer2(X)
        weights = nn.Softmax(X)
        weighted_sum = torch.dot(weights, estimators_predictions)
        return weighted_sum


# ======================================================== #


class FullInheritanceImpute(BaseInheritanceImpute):
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
    ):
        super().__init__(
            base_estimator=base_estimator,
            data=data,
            imputer=imputer,
            target_col=target_col,
            tests=tests,
            base_features=base_features,
            feature_to_test_map=feature_to_test_map,
            base_estimator_params=base_estimator_params,
            imputer_params=imputer_params,
            prediction_method=prediction_method
        )
        self.name = type(base_estimator).__name__ + '_FullInheritanceImpute'
        # just hard-code for now to be compatible with 
        # InheritanceCompatibleEstimator. Can refactor later.
        self.prediction_method = 'predict'

    def fit(self):
        super().fit_node_estimators()

    def predict(self, X_test: pd.DataFrame):
        cols = [c for c in X_test.columns if c != self.target_col]
        X_test = self.imputer.transform(X_test)
    
        if isinstance(X_test, pd.DataFrame):
            cols = [c for c in X_test.columns if c != self.target_col]
            X_test = X_test.to_numpy()

        assert len(X_test.shape) == 2

        predictions = []
        nodes_seen = []
        
        for i in tqdm(range(X_test.shape[0])):

            sample = X_test[i,:].astype(np.float32)
            raw_sample = sample.copy()
            indices = np.ravel(np.argwhere(np.isnan(sample)))
            level = len(set(list(self.feature_to_test_map.values()))) \
                - get_n_tests_missing(cols, self.feature_to_test_map, indices)
            sample_1d = np.array([
                sample[i] for i in range(len(sample))
                if i not in indices
            ])
            sample = sample_1d.reshape(1,-1).astype(np.float32)
            # create node: pair[level, tuple[relevant indices]]
            node = (
                level, 
                tuple(
                    [x for x in range(X_test.shape[1]) if x not in indices]
                )
            )
            nodes_seen.append(node)
            model = self.dag.nodes[node]['model']
            
            # ancestor_predictions includes prediction at current node as well
            # so initialize ancestor predictions with current node prediction
            # first_prediction = getattr(model, self.prediction_method)(sample)[0]
            first_prediction = getattr(model, 'predict')(raw_sample)[0]
            self.dag.nodes[node]['passthrough_predictions'][i] = first_prediction
            ancestor_predictions = [first_prediction]

            df = self.data.drop(columns=[self.target_col])

            if level != 0:
                for a in nx.ancestors(self.dag, node):
                    
                    ancestor_model = self.dag.nodes[a]['model']
                    current_prediction = ancestor_model.predict(raw_sample)[0]
                    self.dag.nodes[a]['passthrough_predictions'][i] = current_prediction
                    ancestor_predictions.append(current_prediction)
            prediction = sum(ancestor_predictions) / len(ancestor_predictions)

            predictions.append(prediction)
            self.dag.nodes[node]['predictions'][i] = prediction

        return np.array(predictions)


if __name__ == '__main__':
    import pickle
    import seaborn as sns


    pa = ['ADAM12','ENDOGLIN','SFLT1','VEGF','AFP','fbHCG',
        'INHIBINA','PAPPA','PLGF']
    train, test = load_numom2b_analytes()
    tests = PlacentalAnalytesTests().tests
    lowercase_cols = [c for c in train.columns if c == c.lower()]
    exclude_cols = [
        'Unnamed: 0', 'Unnamed: 0.1', 'STUDYID', 'StudyID', 'GAWKSEND', 'BIRTH_TYPE', 
        'AGE_AT_V1', 'PEGHTN', 'CHRONHTN', 'OUTCOME'
    ] + lowercase_cols

    train = train[[c for c in train.columns if c not in exclude_cols]]
    test = test[[c for c in test.columns if c not in exclude_cols]]

    X_train = train[[c for c in train.columns if c != 'PEgHTN']]
    y_train = np.array(train['PEgHTN'])
    X_test = test[[c for c in test.columns if c != 'PEgHTN']]
    y_test = np.array(test['PEgHTN'])
    base_features = [c for c in train.columns 
    if c not in exclude_cols + ['PEgHTN'] + pa]

    model = StackedInheritanceEstimator(
        base_estimator=xgb.XGBClassifier,
        meta_estimator=LogisticRegression,
        base_estimator_params={'n_jobs': 1},
        data=train,
        target_col='PEgHTN',
        tests=tests,
        base_features=base_features,
        feature_to_test_map=tests
    )


    # model = FullInheritance(
    #     base_estimator=xgb.XGBClassifier, #HistGradientBoostingClassifier,
    #     data=train,
    #     target_col='PEgHTN',
    #     tests=tests,
    #     base_features=base_features,
    #     base_estimator_params={'n_jobs': 1},
    #     feature_to_test_map=tests
    # )
    model.fit()


    # clf = EstimatorWithImputation(
    #     estimator=RandomForestClassifier(n_estimators=100),
    #     imputer=KNNImputer,
    # )
    proba_predictions = model.predict(X_test)[:,1]
    # clf.fit(X_train, y_train)
    # proba_predictions = clf.predict_proba(X_test)[:,1]
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(y_test, proba_predictions)
    print(proba_predictions)
    predictions = np.where(proba_predictions > 0.22, 1, 0)
    from sklearn.metrics import classification_report
    print(classification_report(y_test, predictions))
    # del model


    sensitivities = []
    specificities = []
    ppvs = []
    npvs = []
    gmeans_sens_spec = []
    gmeans_all_metrics = []

    for i in tqdm(range(1, 1000)):
        predictions = np.array([1 if x >= i / 1000 else 0 for x in proba_predictions])
        # compute sensitivity, specificity, and gmean
        correct_pos = 0
        correct_neg = 0
        false_pos = 0
        false_neg = 0

        for y, y_hat in zip(y_test, predictions):
            if y == 0:
                if y_hat == 0:
                    correct_neg += 1
                else:
                    false_pos += 1
            else: # y = 1
                if y_hat == 1:
                    correct_pos += 1
                else:
                    false_neg += 1

        try:
            sensitivity = correct_pos / (correct_pos + false_neg)
        except:
            sensitivity = 0
        sensitivities.append(sensitivity)
        try:
            specificity = correct_neg / (correct_neg + false_pos)
        except:
            specificity = 0
        specificities.append(specificity)
        try:
            ppv = correct_pos / (correct_pos + false_pos)
        except:
            ppv = 0
        ppvs.append(ppv)
        try:
            npv = correct_neg / (correct_neg + false_neg)
        except:
            npv = 0
        npvs.append(npv)
        try:
            gmean_sens_spec = np.sqrt(sensitivity * specificity)
        except:
            gmean_sens_spec = 0
        gmeans_sens_spec.append(gmean_sens_spec)
        try:
            gmean_all_metrics = np.prod(
                [sensitivity, specificity, ppv, npv]
            )** (1 / 4)
        except:
            gmean_all_metrics = 0
        gmeans_all_metrics.append(gmean_all_metrics)
        
    print('roc auc: ' + str(roc_auc))
    print('max gmean sensitivity vs. specificity: ' + str(round(max(gmeans_sens_spec), 3)))
    print('max gmean all metrics: ' + str(round(max(gmeans_all_metrics), 3)))
    print('AUC, gmean sensitivity vs specificity: ' + str(round(sum(gmeans_sens_spec) / len(gmeans_sens_spec), 3)))
    print('AUC, gmean all metrics: ' + str(round(sum(gmeans_all_metrics) / len(gmeans_all_metrics), 3)))

    p1 = sns.lineplot(data=sensitivities, color='blue')
    p2 = sns.lineplot(data=specificities, color='orange')
    p3 = sns.lineplot(data=ppvs, color='green')
    p4 = sns.lineplot(data=npvs, color='red')
    p5 = sns.lineplot(data=gmeans_sens_spec, color='black')
    p6 = sns.lineplot(data=gmeans_all_metrics, color='purple')

    from matplotlib import pyplot as plt
    import matplotlib.patches as mpatches

    blue_patch = mpatches.Patch(color='blue', label='Sensitivities')
    orange_patch = mpatches.Patch(color='orange', label='Specificities')
    green_patch = mpatches.Patch(color='green', label='PPVs')
    red_patch = mpatches.Patch(color='red', label='NPVs')
    black_patch = mpatches.Patch(color='black', label='G-Means: Sens vs. Spec')
    purple_patch = mpatches.Patch(color='purple', label='G-Means: All metrics')


    plt.title('Metrics vs. Class Probability Cutoffs')
    plt.xlabel('Class Probability Cutoffs')
    plt.ylabel('Metric Values')
    plt.legend(handles=[blue_patch, orange_patch, green_patch, red_patch, black_patch, purple_patch])

    plt.savefig('outfile.png')