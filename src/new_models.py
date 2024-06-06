from collections import Counter
import copy
import inspect
from itertools import combinations, product
import json
from concurrent.futures import ProcessPoolExecutor
import os
import pickle
import sys
from turtle import distance
from typing import Iterable, Mapping, Tuple, Union
import warnings

# from src.new_base import BaseNonmissingSubspaceClassifier
# from src.utils import get_classification_metrics
warnings.filterwarnings("ignore")

from imblearn.under_sampling import RandomUnderSampler
import networkx as nx
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import scipy
from scipy.spatial.distance import cdist
from sklearn.experimental import enable_iterative_imputer, enable_hist_gradient_boosting
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import BaseEnsemble, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.impute._base import _BaseImputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss, max_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import torch
from torch import nn, softmax
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from tqdm import tqdm
import xgboost as xgb

sys.path.append('.')

from new_base import BaseInheritanceClassifier, BaseNonmissingSubspaceClassifier,\
    ClassifierWithImputation, InheritanceCompatibleClassifier
from data_loaders import Dataset, MedicalDataset, PlacentalAnalytesTests
from data_loaders import load_numom2b_analytes_dataset
from data_loaders import load_wisconsin_diagnosis_dataset, load_wisconsin_prognosis_dataset
import diversity
from utils import Test, get_n_tests_missing, label_encoded_data,\
    find_optimal_threshold, get_device, get_prediction_method, get_classification_metrics
from utils import get_sample_indices_with_optional_tests, plot_prediction_errors
from utils import get_prediction_method


__all__ = [
    'FullInheritance', 'StackedHeterogenousClassifier',
    'StackedHeterogenousRegressor', 'StackedInheritanceEstimator',
    'StackedParametricClassifier'
]


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

#BaseEstimator is for building our own estimator as well as the ClassifierMixin. 
#All estimators and classifier in scikit-learn derived from this class. 
class DEWClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, 
        classifier_pool:    Union[Mapping[str, BaseEstimator],\
                            Iterable[Tuple[str, BaseEstimator]]],
        n_neighbors=5,
        n_top_to_choose=[1,3,5,None],
        competence_threshold = 0.5,
        baseline_test_predictions_dict = None
    ) -> None:
        super().__init__() #calling parent class 
        self.classifier_pool = dict(classifier_pool)
        self.n_top_to_choose=n_top_to_choose
        self.n_neighbors = n_neighbors
        self.competence_threshold = competence_threshold
        self.samplewise_clf_errors = pd.DataFrame({})
        self.weight_assignments = pd.DataFrame({})
        self._temp_weights = []
        self.baseline_test_predictions_dict = baseline_test_predictions_dict
        if self.baseline_test_predictions_dict is not None:
            self.baseline_test_predictions = np.dstack([
                self.baseline_test_predictions_dict[pipeline]
                for pipeline in self.baseline_test_predictions_dict
            ])
    
    #INTERESTING FIT BEFOREIMPUTATION 
    #fitting samles to model
    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        #separate train samples from origianl data by .deepcopy(X)
        self.train_samples = copy.deepcopy(X)
        self.y_train = y
        
        if len(self.y_train.shape) > 1:
            self.y_train = pd.DataFrame(self.y_train).idxmax(axis=1)
        val_set_hard_preds_df = pd.DataFrame()
        self.diversity_df = pd.DataFrame()

        for clf_name, model in self.classifier_pool.items():
            probas = model.predict_proba(X)
            if len(probas.shape) > 1:
                val_set_hard_preds_df[clf_name] = pd.DataFrame(probas).idxmax(axis=1) # hard labels for diversity score
            else:
                val_set_hard_preds_df[clf_name] = np.round(probas).astype(int)
            label_encoder = OneHotEncoder()
            y_2d = np.array(label_encoder.fit_transform(y.reshape(-1, 1)).todense())
            errors = np.max(np.clip(y_2d - probas, 0, 1), axis=1)
            # errors = [log_loss(y[i, :], probas[i, :]) for i in range(len(y))]
            self.samplewise_clf_errors[clf_name] = errors

        self.val_set_hard_preds_matrix = val_set_hard_preds_df.to_numpy()

        self.imputed_samples_per_pipeline = {}
        for _, p in self.classifier_pool.items():
        	self.imputed_samples_per_pipeline[p] = p.imputer.transform(self.train_samples)


    def predict_proba_one_sample(
        self, sample, sample_idx
    ) -> Tuple[Mapping[str, Mapping[Union[int, None], Iterable[float]]], int]:
        predictions = {}
        diversities_per_pipeline = {}
        weights_dict = {}
        query = np.array(sample).reshape(1,-1)
        pipeline_specific_neighbor_indices, distances = self.get_nearest_neighbors(query, self.train_samples)
        pipeline_competences: dict = self.estimate_competences(
            pipeline_specific_neighbor_indices=pipeline_specific_neighbor_indices, distances=distances
        )
        full_competences = np.array(list(pipeline_competences.values())).astype(np.float32)
        full_weights = scipy.special.softmax(full_competences)
        
        sample = sample.to_frame().T
        
        for top_n in self.n_top_to_choose:
            
            competences = copy.deepcopy(full_competences)
            diversity_scores = []
            y_true_per_pipeline = [
                self.y_train[[i for i in pipeline_specific_neighbor_indices[pipeline_idx]]]
                for pipeline_idx in pipeline_specific_neighbor_indices.keys()
            ]

            competences = np.nan_to_num(competences)
            if top_n not in [None, -1, 0]:
                # rank from highest competence --> lowest_competence
                ranked = np.argsort(competences)[::-1][0: top_n]
                top_n_clf = [
                    list(self.classifier_pool.items())[i] 
                    for i in ranked
                ]
                top_n_clf_competences = np.array([
                    competences[i]
                    for i in ranked
                ]).astype(np.float32)
            else:
                ranked = list(range(len(competences)))
                top_n_clf = list(self.classifier_pool.items())
                top_n_clf_competences = competences

            top_n_clf_competences = [
                competence 
                if competence > self.competence_threshold 
                else 0 
                for competence in top_n_clf_competences
            ]

            weights = scipy.special.softmax(top_n_clf_competences)
            weights_to_report = [0] * len(full_weights)
            for i, clf_idx in enumerate(ranked):
                weights_to_report[clf_idx] = weights[i]
            weights_dict[top_n] = list(weights_to_report)
            if self.baseline_test_predictions_dict is None:
                probas = np.array([
                    model.predict_proba(sample)[0]
                    for clf_name, model in top_n_clf
                ])
                prediction = np.dot(weights, probas)
            else:
                probas = self.baseline_test_predictions[sample_idx].T
                prediction = np.dot(weights_to_report, probas)

            predictions[top_n] = prediction

        return (
            {
                'predictions': predictions, 
                'weights': weights_dict, 
                'nn_distances': list(distances)
            }, 
            sample_idx
        )
        

    def predict_proba(self, X) -> Tuple[Mapping[Union[int, None], np.ndarray]]:
        top_n_prediction_sets = {}

        predictions = {top_n: {} for top_n in self.n_top_to_choose}
        weights = {top_n: {} for top_n in self.n_top_to_choose}
        all_samples = [X.iloc[i, :].astype(np.float32) for i in range(X.shape[0])]
        sample_incides = list(range(len(all_samples)))
        nn_distances_per_sample = []

        with tqdm(total=len(all_samples)) as pbar:
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
                for result, idx in pool.map(self.predict_proba_one_sample, all_samples, sample_incides):
                    nn_distances_per_sample.append(result['nn_distances'])
                    for top_n in self.n_top_to_choose:
                        predictions[top_n][idx] = result['predictions'][top_n]
                        weights[top_n][idx] = result['weights'][top_n]
                    pbar.update(1)

        for top_n in predictions.keys():
            predictions[top_n] = dict(sorted(predictions[top_n].items(), key = lambda x: x[0]))
            predictions[top_n] = [x[1] for x in predictions[top_n].items()]
            predictions[top_n] = np.vstack(predictions[top_n])
            weights[top_n] = dict(sorted(weights[top_n].items(), key = lambda x: x[0]))
            weights[top_n] = [x[1] for x in weights[top_n].items()]
            weights[top_n] = np.vstack(weights[top_n])

            nn_distances_per_sample = np.vstack(nn_distances_per_sample)

        return predictions, weights, nn_distances_per_sample

    def predict(self, X) -> dict:
        predictions = {}
        probas_sets = self.predict_proba(X)
        for top_n, probas in probas_sets.items():
            a = (probas == probas.max(axis=1, keepdims=True)).astype(int)
            predictions[top_n] = a

        return predictions

    def set_baseline_test_predictions(self, baseline_test_predictions_dict: dict):
        self.baseline_test_predictions_dict = baseline_test_predictions_dict
        self.baseline_test_predictions = np.dstack([
            self.baseline_test_predictions_dict[pipeline]
            for pipeline in self.baseline_test_predictions_dict
        ])

    def get_nearest_neighbors(self, q, samples_df):
        pipeline_specific_neighbor_indices = {}
        for idx, p in self.classifier_pool.items():
            imputed_q = p.imputer.transform(q).reshape(1,-1)
            imputed_samples = self.imputed_samples_per_pipeline[p]
            distances = cdist(imputed_q, imputed_samples)[0]
            # we will use numerical indices simply for facilitating y_true indexing in competence esitmation
            dist_df = pd.DataFrame(data=distances, columns=['distance'], index=range(len(samples_df)))
            sorted_distances_df = dist_df.sort_values('distance')
            distances = sorted_distances_df['distance'][0: self.n_neighbors]
            pipeline_specific_neighbor_indices[idx] = sorted_distances_df.index[0: self.n_neighbors]

        return pipeline_specific_neighbor_indices, distances


    def estimate_competences(
        self, q=None, samples_df=None, pipeline_specific_neighbor_indices=None,
        distances=None
    ) -> dict:
        
        assert not np.logical_xor(pipeline_specific_neighbor_indices is None, distances is None)

        pipeline_competences = {}
        if pipeline_specific_neighbor_indices is None:
            assert q is not None and samples_df is not None
            pipeline_specific_neighbor_indices, distances = self.get_nearest_neighbors(
                q, samples_df
            )

        distance_weights = scipy.special.softmax(distances)
        
        for pipeline_idx, indices in pipeline_specific_neighbor_indices.items():
            
            errors = self.samplewise_clf_errors[pipeline_idx][indices]
            competences = 1 - errors
            competence = np.mean(competences)
            pipeline_competences[pipeline_idx] = competence

        return pipeline_competences


    def get_pipeline_weights(self, q, samples_df) -> dict:
        pipeline_weights = {}
        pipeline_competences = self.estimate_competences(q, samples_df)
        all_competence_scores = list(pipeline_competences.values())
        weights = scipy.special.softmax(all_competence_scores)
        for idx, pipeline_type in enumerate(list(pipeline_competences.keys())):
            pipeline_weights[pipeline_type] = weights[idx]

        return pipeline_weights



class DEWClassifierClustering(BaseEstimator, ClassifierMixin):
    def __init__(
        self, 
        classifier_pool:    Union[Mapping[str, BaseEstimator],\
                            Iterable[Tuple[str, BaseEstimator]]],
        n_clusters=5,
        n_top_to_choose=[1,3,5,None],
        competence_threshold = 0.5
    ) -> None:
        super().__init__()
        self.classifier_pool = dict(classifier_pool)
        self.n_top_to_choose=n_top_to_choose
        self.n_neighbors = n_clusters
        self.competence_threshold = competence_threshold
        self.samplewise_clf_errors = pd.DataFrame({})
        self.weight_assignments = pd.DataFrame({})
        self._temp_weights = []

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.train_samples = copy.deepcopy(X)

        for clf_name, model in self.classifier_pool.items():
            probas = model.predict_proba(X)
            errors = np.max(np.clip(y - probas, 0, 1), axis=1)
            self.samplewise_clf_errors[clf_name] = errors

    def predict_proba_one_sample(self, sample):
        predictions = {}
        query = np.array(sample).reshape(1,-1)
        pipeline_competences: dict = self.estimate_competences(
            q=query, samples_df=self.train_samples
        )
        competences = list(pipeline_competences.values())
        full_weights = scipy.special.softmax(competences)
        
        sample = sample.to_frame().T
        
        for top_n in self.n_top_to_choose:
            if top_n not in [None, -1, 0]:
                # rank from highest competence --> lowest_competence
                ranked = np.argsort(competences)[::-1][0: top_n]
                top_n_clf = [
                    list(self.classifier_pool.items())[i] 
                    for i in ranked
                ]
                top_n_clf_competences = np.array([
                    competences[i]
                    for i in ranked
                ])
            else:
                top_n_clf = list(self.classifier_pool.items())
                top_n_clf_competences = competences

            weights = scipy.special.softmax(top_n_clf_competences)
            probas = np.array([
                model.predict_proba(sample)[0]
                for clf_name, model in top_n_clf
            ])
            prediction = np.dot(weights, probas)
            predictions[top_n] = prediction

        return predictions
            

    def predict_proba(self, X):
        top_n_prediction_sets = {}

        predictions = {top_n: [] for top_n in self.n_top_to_choose}
        all_samples = [X.iloc[i, :].astype(np.float32) for i in range(X.shape[0])]

        with tqdm(total=len(all_samples)) as pbar:
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
                for result in pool.map(self.predict_proba_one_sample, all_samples):
                    for top_n in result.keys():
                        predictions[top_n].append(result[top_n])
                    pbar.update(1)

        for top_n in predictions.keys():
            predictions[top_n] = np.vstack(predictions[top_n])

        return predictions

    def predict(self, X) -> dict:
        predictions = {}
        probas_sets = self.predict_proba(X)
        for top_n, probas in probas_sets.items():
            a = (probas == probas.max(axis=1, keepdims=True)).astype(int)
            predictions[top_n] = a

        return predictions

    def get_nearest_neighbors(self, q, samples_df):
        pipeline_specific_neighbor_indices = {}
        for idx, p in self.classifier_pool.items():
            imputed_q = p.imputer.transform(q).reshape(1,-1)
            imputed_samples = p.imputer.transform(samples_df)
            distances = cdist(imputed_q, imputed_samples)[0]
            # we will use numerical indices simply for facilitating y_true indexing in competence esitmation
            dist_df = pd.DataFrame(data=distances, columns=['distance'], index=range(len(samples_df)))
            sorted_distances_df = dist_df.sort_values('distance')
            pipeline_specific_neighbor_indices[idx] = sorted_distances_df.index[0: self.n_neighbors]

        return pipeline_specific_neighbor_indices


    def estimate_competences(self, q, samples_df) -> dict:
        
        pipeline_competences = {}
        pipeline_specific_neighbor_indices = self.get_nearest_neighbors(
            q, samples_df
        )
        for pipeline_idx, indices in pipeline_specific_neighbor_indices.items():
            
            errors = self.samplewise_clf_errors[pipeline_idx]
            competences = 1 - errors
            competence = np.mean(competences)
            competence = competence / (np.std(competences) + 0.01) if competence > self.competence_threshold else 0
            pipeline_competences[pipeline_idx] = competence

        return pipeline_competences


    def get_pipeline_weights(self, q, samples_df) -> dict:
        pipeline_weights = {}
        pipeline_competences = self.estimate_competences(q, samples_df)
        all_competence_scores = list(pipeline_competences.values())
        weights = scipy.special.softmax(all_competence_scores)
        for idx, pipeline_type in enumerate(list(pipeline_competences.keys())):
            pipeline_weights[pipeline_type] = weights[idx]

        return pipeline_weights


class DEWRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self, 
        regressor_pool:    Union[Mapping[str, BaseEstimator],\
                            Iterable[Tuple[str, BaseEstimator]]],
        n_neighbors=5,
        n_top_to_choose=[1,3,5,None],
        competence_threshold = 0.5,
        baseline_test_predictions_dict = None
    ) -> None:
        super().__init__()
        self.regressor_pool = dict(regressor_pool)
        self.n_top_to_choose=n_top_to_choose
        self.n_neighbors = n_neighbors
        self.competence_threshold = competence_threshold # not used
        self.samplewise_errors = pd.DataFrame({})
        self.weight_assignments = pd.DataFrame({})
        self._temp_weights = []
        self.baseline_test_predictions_dict = baseline_test_predictions_dict
        if self.baseline_test_predictions_dict is not None:
            self.baseline_test_predictions = np.dstack([
                self.baseline_test_predictions_dict[pipeline]
                for pipeline in self.baseline_test_predictions_dict
            ]).squeeze()
            print(self.baseline_test_predictions.shape)

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.train_samples = copy.deepcopy(X)
        self.y_train = y
        if len(self.y_train.shape) > 1:
            self.y_train = pd.DataFrame(self.y_train).idxmax(axis=1)
        val_set_hard_preds_df = pd.DataFrame()
        self.diversity_df = pd.DataFrame()

        for clf_name, model in self.regressor_pool.items():
            preds = model.predict(X).squeeze()
            val_set_hard_preds_df[clf_name] = preds
            errors = np.abs(y.squeeze() - preds)
            self.samplewise_errors[clf_name] = errors

        self.val_set_hard_preds_matrix = val_set_hard_preds_df.to_numpy()

    def predict_one_sample(
        self, sample, sample_idx
    ) -> Tuple[Mapping[str, Mapping[Union[int, None], Iterable[float]]], int]:
        predictions = {}
        diversities_per_pipeline = {}
        weights_dict = {}
        query = np.array(sample).reshape(1,-1)
        pipeline_specific_neighbor_indices, distances = self.get_nearest_neighbors(query, self.train_samples)
        pipeline_competences: dict = self.estimate_competences(
            pipeline_specific_neighbor_indices=pipeline_specific_neighbor_indices, distances=distances
        )
        full_competences = np.array(list(pipeline_competences.values())).astype(np.float32)
        full_weights = scipy.special.softmax(full_competences)
        
        sample = sample.to_frame().T
        
        for top_n in self.n_top_to_choose:
            
            competences = copy.deepcopy(full_competences)
            competences = np.nan_to_num(competences)

            if top_n not in [None, -1, 0]:
                # rank from highest competence --> lowest_competence
                ranked = np.argsort(competences)[::-1][0: top_n]
                top_n_clf = [
                    list(self.regressor_pool.items())[i] 
                    for i in ranked
                ]
                top_n_clf_competences = np.array([
                    competences[i]
                    for i in ranked
                ]).astype(np.float32)
            else:
                ranked = list(range(len(competences)))
                top_n_clf = list(self.regressor_pool.items())
                top_n_clf_competences = competences

            top_n_clf_competences = [
                competence 
                for competence in top_n_clf_competences
            ]

            weights = scipy.special.softmax(top_n_clf_competences)
            weights_to_report = [0] * len(full_weights)
            for i, clf_idx in enumerate(ranked):
                weights_to_report[clf_idx] = weights[i]
            weights_dict[top_n] = list(weights_to_report)
            if self.baseline_test_predictions_dict is None:
                preds = np.array([
                    model.predict(sample)
                    for clf_name, model in top_n_clf
                ])
                prediction = np.dot(weights, preds)
            else:
                # probas = self.baseline_test_predictions[sample_idx, :, :].T
                #print(self.baseline_test_predictions.shape)
                preds = self.baseline_test_predictions[..., sample_idx, :].squeeze()
                prediction = np.dot(weights_to_report, preds)

            predictions[top_n] = prediction

        return (
            {
                'predictions': predictions, 
                'weights': weights_dict, 
                'nn_distances': list(distances)
            }, 
            sample_idx
        )
        

    def predict(self, X) -> Tuple[Mapping[Union[int, None], np.ndarray]]:
        top_n_prediction_sets = {}

        predictions = {top_n: {} for top_n in self.n_top_to_choose}
        weights = {top_n: {} for top_n in self.n_top_to_choose}
        all_samples = [X.iloc[i, :].astype(np.float32) for i in range(X.shape[0])]
        sample_incides = list(range(len(all_samples)))
        nn_distances_per_sample = []

        with tqdm(total=len(all_samples)) as pbar:
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
                for result, idx in pool.map(self.predict_one_sample, all_samples, sample_incides):
                    nn_distances_per_sample.append(result['nn_distances'])
                    for top_n in self.n_top_to_choose:
                        predictions[top_n][idx] = result['predictions'][top_n]
                        weights[top_n][idx] = result['weights'][top_n]
                    pbar.update(1)

        for top_n in predictions.keys():
            predictions[top_n] = dict(sorted(predictions[top_n].items(), key = lambda x: x[0]))
            predictions[top_n] = [x[1] for x in predictions[top_n].items()]
            predictions[top_n] = np.vstack(predictions[top_n])
            weights[top_n] = dict(sorted(weights[top_n].items(), key = lambda x: x[0]))
            weights[top_n] = [x[1] for x in weights[top_n].items()]
            weights[top_n] = np.vstack(weights[top_n])

            nn_distances_per_sample = np.vstack(nn_distances_per_sample)

        return predictions, weights, nn_distances_per_sample


    def set_baseline_test_predictions(self, baseline_test_predictions_dict: dict):
        self.baseline_test_predictions_dict = baseline_test_predictions_dict
        self.baseline_test_predictions = np.dstack([
            self.baseline_test_predictions_dict[pipeline]
            for pipeline in self.baseline_test_predictions_dict
        ])

    def get_nearest_neighbors(self, q, samples_df):
        pipeline_specific_neighbor_indices = {}
        for idx, p in self.regressor_pool.items():
            imputed_q = p.imputer.transform(q).reshape(1,-1)
            imputed_samples = p.imputer.transform(samples_df)
            distances = cdist(imputed_q, imputed_samples)[0]
            # we will use numerical indices simply for facilitating y_true indexing in competence esitmation
            dist_df = pd.DataFrame(data=distances, columns=['distance'], index=range(len(samples_df)))
            sorted_distances_df = dist_df.sort_values('distance')
            distances = sorted_distances_df['distance'][0: self.n_neighbors]
            pipeline_specific_neighbor_indices[idx] = sorted_distances_df.index[0: self.n_neighbors]

        return pipeline_specific_neighbor_indices, distances


    def estimate_competences(
        self, q=None, samples_df=None, pipeline_specific_neighbor_indices=None,
        distances=None
    ) -> dict:
        
        assert not np.logical_xor(pipeline_specific_neighbor_indices is None, distances is None)

        pipeline_competences = {}
        if pipeline_specific_neighbor_indices is None:
            assert q is not None and samples_df is not None
            pipeline_specific_neighbor_indices, distances = self.get_nearest_neighbors(
                q, samples_df
            )

        distance_weights = scipy.special.softmax(distances)
        
        for pipeline_idx, indices in pipeline_specific_neighbor_indices.items():
            
            errors = self.samplewise_errors[pipeline_idx][indices]
            competences = -errors
            competence = np.mean(competences)
            pipeline_competences[pipeline_idx] = competence

        return pipeline_competences


    def get_pipeline_weights(self, q, samples_df) -> dict:
        pipeline_weights = {}
        pipeline_competences = self.estimate_competences(q, samples_df)
        all_competence_scores = list(pipeline_competences.values())
        weights = scipy.special.softmax(all_competence_scores)
        for idx, pipeline_type in enumerate(list(pipeline_competences.keys())):
            pipeline_weights[pipeline_type] = weights[idx]

        return pipeline_weights
