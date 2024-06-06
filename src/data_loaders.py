from collections import namedtuple
from collections.abc import Sequence
from enum import Enum
from itertools import product
import sys
from typing import Callable, Iterable, Tuple, Union


from imblearn.base import BaseSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from scipy import stats
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder


sys.path.append('.')
from utils import create_missing_values, get_cols_without_missing_values, produce_NA, Test

def label_encoded_data(data, ignore_columns): #for categorical data
    # save data type of the columns a object datatype would indicated 
    # a categorical feature
    data_dict = dict(data.dtypes)

    features = list(data.columns)

    for feature in ignore_columns:
         features.remove(feature)

    le = LabelEncoder()

    for labels in features:
        # check if the column is categorical 
        if data_dict[labels] == np.object:
            try:
                data.loc[:, labels] = le.fit_transform(data.loc[:, labels])
            except:
                print(labels)

    return data


CustomExperimentDataObject = namedtuple(
    'CustomExperimentDataObject',
    ['data', 'dataset_name', 'target_col']
)


class MedicalDataset(Sequence):
    def __init__(
        self,
        data,
        n_folds=None,
        test_size=None,
        target_col: str = '',
        tests={},
        feature_to_test_map: dict = {},
        sampling_strategy: Union[None, Callable, BaseSampler] = None,
        sampling_strategy_kwargs: Iterable = {},
        cv_random_state: int = 42
    ):
        """
        if a `callable` sampling strategy is chosen, it must take a DataFrame
        as its first argument.
        """
        self.raw_data = data.copy()
        self.all_columns = data.columns
        self.target_col = target_col
        self.n_folds = n_folds
        
        if sampling_strategy is not None:
            y = data[target_col]
            X = data.drop(target_col, axis=1)

            if isinstance(sampling_strategy, Callable):
                X, y = sampling_strategy(
                    **sampling_strategy_kwargs
                ).fit_resample(X, y)
            elif isinstance(sampling_strategy, BaseSampler):
                X, y = sampling_strategy.fit_resample(X, y)
            data = X
            data[target_col] = y

        self.data = data
        self.targets = data[self.target_col]
        
        folder_ = StratifiedKFold(
            n_splits=n_folds,
            random_state=cv_random_state,
            shuffle=True
        )

        train_test_pairs = list(folder_.split(self.data, self.targets))
        train_sets = [train for (train, test) in train_test_pairs]
        for t in train_sets:
            np.random.shuffle(t)

        train_val_pairs = [
            (
                train_set[0: int(len(train_set) * 0.8)], 
                train_set[int(len(train_set) * 0.8):]
            )
            for train_set in train_sets
        ]
        final_train_sets = [train for (train, val) in train_val_pairs]
        final_val_sets = [val for (train, val) in train_val_pairs]
        final_test_sets = [test_set for (_, test_set) in train_test_pairs]

        self.train_val_test_triples = list(zip(
            final_train_sets, final_val_sets, final_test_sets
        ))


        # self.train_test_pairs = train_test_pairs

        self.feature_to_test_map = feature_to_test_map
        self.tests = tests
        
        test_features = []
        for x in list(self.tests.values()):
            test_features += x.features
        
        self.test_features = test_features
        
        self.base_features = [
            c 
            for c in self.data.columns 
            if c not in test_features + [self.target_col]
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        train_idx, val_idx, test_idx = self.train_val_test_triples[i]
        train, val, test = (
            self.data.iloc[train_idx, :], 
            self.data.iloc[val_idx, :], 
            self.data.iloc[test_idx, :]
        )

        train, val, test = (
            pd.DataFrame(train, columns=self.all_columns), 
            pd.DataFrame(val, columns=self.all_columns),
            pd.DataFrame(test, columns=self.all_columns)
        )
        return train, val, test
            

class Dataset(Sequence):
    def __init__(
        self,
        data,
        n_folds=None,
        test_size=None,
        target_col: str = '',
        sampling_strategy: Union[None, Callable, BaseSampler] = None,
        sampling_strategy_kwargs: Iterable = {},
        cv_random_state: int = 42
    ):
        """
        if a `callable` sampling strategy is chosen, it must take a DataFrame
        as its first argument.
        """
        self.raw_data = data.copy()
        self.all_columns = data.columns
        self.target_col = target_col
        self.n_folds = n_folds
        
        if sampling_strategy is not None:
            y = data[target_col]
            X = data.drop(target_col, axis=1)

            if isinstance(sampling_strategy, Callable):
                X, y = sampling_strategy(
                    **sampling_strategy_kwargs
                ).fit_resample(X, y)
            elif isinstance(sampling_strategy, BaseSampler):
                X, y = sampling_strategy.fit_resample(X, y)
            data = X
            data[target_col] = y

        self.data = data
        self.targets = data[self.target_col]
        
        folder_ = StratifiedKFold(
            n_splits=n_folds,
            random_state=cv_random_state,
            shuffle=True
        )
        train_test_pairs = list(folder_.split(self.data, self.targets))
        train_sets = [train for (train, test) in train_test_pairs]
        for t in train_sets:
            np.random.shuffle(t)

        train_val_pairs = [
            (
                train_set[0: int(len(train_set) * 0.8)], 
                train_set[int(len(train_set) * 0.8):]
            )
            for train_set in train_sets
        ]
        final_train_sets = [train for (train, val) in train_val_pairs]
        final_val_sets = [val for (train, val) in train_val_pairs]
        final_test_sets = [test_set for (_, test_set) in train_test_pairs]

        self.train_val_test_triples = list(zip(
            final_train_sets, final_val_sets, final_test_sets
        ))

        self.base_features = get_cols_without_missing_values(data[[c for c in data.columns if c != self.target_col]])
        self.tests = {
            f: Test(name=f, features=[f])
            for f in data.columns
            if f not in self.base_features + [self.target_col]
        }
        self.feature_to_test_map = self.tests

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        train_idx, val_idx, test_idx = self.train_val_test_triples[i]
        train, val, test = (
            self.data.iloc[train_idx, :], 
            self.data.iloc[val_idx, :], 
            self.data.iloc[test_idx, :]
        )

        train, val, test = (
            pd.DataFrame(train, columns=self.all_columns), 
            pd.DataFrame(val, columns=self.all_columns),
            pd.DataFrame(test, columns=self.all_columns)
        )
        return train, val, test


TrainValTestTriples = Iterable[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]

class MissDataset(Sequence):
    def __init__(
        self,
        data,
        n_folds=None,
        test_size=None,
        target_col: str = '',
        p_miss=0.1, 
        missing_mechanism = "MCAR", 
        opt = None, 
        p_obs = None, 
        q = None,
        sampling_strategy: Union[None, Callable, BaseSampler] = None,
        sampling_strategy_kwargs: Iterable = {},
        cv_random_state: int = 42
    ):
        """
        if a `callable` sampling strategy is chosen, it must take a DataFrame
        as its first argument.
        """
        self.p_miss = p_miss
        self.raw_data = data.copy()
        self.all_columns = data.columns
        self.target_col = target_col
        self.n_folds = n_folds
        self.cv_random_state = cv_random_state
        
        if sampling_strategy is not None:
            y = data[target_col]
            X = data.drop(target_col, axis=1)

            if isinstance(sampling_strategy, Callable):
                X, y = sampling_strategy(
                    **sampling_strategy_kwargs
                ).fit_resample(X, y)
            elif isinstance(sampling_strategy, BaseSampler):
                X, y = sampling_strategy.fit_resample(X, y)
            data = X
            data[target_col] = y

        self.targets = data[self.target_col]
        X = data.drop(columns=[self.target_col])
        X_cols = X.columns
        X_index = X.index
        scaler = MinMaxScaler(feature_range=(0, 1))
        X = scaler.fit_transform(X)
        miss_dict = produce_NA(
            X=X,
            p_miss=p_miss, 
            mecha=missing_mechanism, 
            opt=opt, p_obs=p_obs, q=q
        )
        X = miss_dict['X_incomp']

        self.data = pd.DataFrame(X, columns=X_cols, index=X_index)
        self.data[self.target_col] = self.targets
        
        folder_ = StratifiedKFold(
            n_splits=self.n_folds,
            random_state=cv_random_state,
            shuffle=True
        )
        train_test_pairs = list(folder_.split(self.data, self.targets))
        train_sets = [train for (train, test) in train_test_pairs]
        for t in train_sets:
            np.random.shuffle(t)
        
        train_val_pairs = [
            (
                train_set[0: int(len(train_set) * 0.5)], 
                train_set[int(len(train_set) * 0.5):]
            )
            for train_set in train_sets
        ]
        final_train_sets = [train for (train, val) in train_val_pairs]
        final_val_sets = [val for (train, val) in train_val_pairs]
        final_test_sets = [test_set for (_, test_set) in train_test_pairs]

        self.train_val_test_triples = list(zip(
            final_train_sets, final_val_sets, final_test_sets
        ))

        # self.base_features = get_cols_without_missing_values(data[[c for c in data.columns if c != self.target_col]])
        # self.tests = {
        #     f: Test(name=f, features=[f])
        #     for f in data.columns
        #     if f not in self.base_features + [self.target_col]
        # }
        # self.feature_to_test_map = self.tests

    def split_dataset(self):
        folder_ = StratifiedKFold(
            n_splits=self.n_folds,
            random_state=self.cv_random_state,
            shuffle=True
        )
        train_test_pairs = list(folder_.split(self.data, self.targets))
        train_sets = [train for (train, test) in train_test_pairs]
        for t in train_sets:
            np.random.shuffle(t)

        train_val_pairs = [
            (
                train_set[0: int(len(train_set) * 0.5)], 
                train_set[int(len(train_set) * 0.5):]
            )
            for train_set in train_sets
        ]
        final_train_sets = [train for (train, val) in train_val_pairs]
        final_val_sets = [val for (train, val) in train_val_pairs]
        final_test_sets = [test_set for (_, test_set) in train_test_pairs]

        self.train_val_test_triples = list(zip(
            final_train_sets, final_val_sets, final_test_sets
        ))

    def split_dataset_hook(self, splitting_function: Callable, *args, **kwargs) -> TrainValTestTriples:
        self.train_val_test_triples = splitting_function(*args, **kwargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        try:
            train_idx, val_idx, test_idx = self.train_val_test_triples[i]
            train, val, test = (
                self.data.iloc[train_idx, :], 
                self.data.iloc[val_idx, :], 
                self.data.iloc[test_idx, :]
            )

            train, val, test = (
                pd.DataFrame(train, columns=self.all_columns), 
                pd.DataFrame(val, columns=self.all_columns),
                pd.DataFrame(test, columns=self.all_columns)
            )
            return train, val, test
        except:
            train, val, test = self.train_val_test_triples[i]
            print(test.head())
            return train, val, test




class PlacentalAnalytesTests:
    def __init__(self) -> None:
        self.filename = 'placental_analytes'
        self.tests = {
            'ADAM12': Test(name='ADAM12', filename=self.filename, features=['ADAM12']),
            'ENDOGLIN': Test(name='ENDOGLIN', filename=self.filename, features=['ENDOGLIN']),
            'SFLT1': Test(name='SFLT1', filename=self.filename, features=['SFLT1']),
            'VEGF': Test(name='VEGF', filename=self.filename, features=['VEGF']),
            'AFP': Test(name='AFP', filename=self.filename, features=['AFP']),
            'fbHCG': Test(name='fbHCG', filename=self.filename, features=['fbHCG']),
            'INHIBINA': Test(name='INHIBINA', filename=self.filename, features=['INHIBINA']),
            'PAPPA': Test(name='PAPPA', filename=self.filename, features=['PAPPA']),
            'PLGF': Test(name='PLGF', filename=self.filename, features=['PLGF'])
        }

    def get_test(self, test_name):
        return self.tests[test_name]


def load_numom2b_analytes_dataset(target_col='PEgHTN') -> MedicalDataset:
    
    pa = [
        'ADAM12',
        'ENDOGLIN',
        'SFLT1',
        'VEGF',
        'AFP',
        'fbHCG',
        'INHIBINA',
        'PAPPA',
        'PLGF'
    ]

    df_pa = pd.read_csv('/volumes/identify/placental_analytes.csv', na_values=np.nan)
    df_pa = df_pa.drop(columns=[x for x in df_pa.columns if x[-1] == 'c'])
    df_pa = df_pa[df_pa['Visit'] == 2]
    df_pa = label_encoded_data(df_pa, ignore_columns=[])

    df_outcomes = pd.read_csv('/volumes/identify/pregnancy_outcomes.csv')[['StudyID', target_col]]
    df_outcomes = label_encoded_data(df_outcomes, ignore_columns=[])

    df_extra_base = pd.read_csv('/volumes/no name/new/l1_visits/Visit1_l1.csv')
    df_extra_base = df_extra_base.rename(columns={'STUDYID': 'StudyID'})
    lowercase_cols = [c for c in df_extra_base.columns if c == c.lower()]
    df_extra_base = df_extra_base.drop(columns=['AGE_AT_V1', 'GAWKSEND', 'BIRTH_TYPE', 'Unnamed: 0', 'PEGHTN', 'CHRONHTN'] + lowercase_cols)
    df_extra_base = label_encoded_data(df_extra_base, ignore_columns=[])

    df = pd.merge(left=df_pa, left_on='StudyID', right=df_outcomes,
                right_on='StudyID', how='outer'
        )

    df = pd.merge(left=df, left_on='StudyID', right=df_extra_base,
                right_on='StudyID', how='outer')

    df['StudyID'] = list(range(len(df)))

    df = df.drop(columns=['PublicID', 'VisitDate', 'ref_date', 'VisitDate_INT'])
    df = df.drop(columns=['Visit', 'StudyID'])
    if target_col == 'PEgHTN':
        df['PEgHTN'] = [0 if x == 7 else 1 for x in df['PEgHTN']]

    tests_dict = PlacentalAnalytesTests().tests
    RANDOM_STATE = 42
    lowercase_cols = [c for c in df.columns if c == c.lower()]
    exclude_cols = [
        'Unnamed: 0', 'Unnamed: 0.1', 'STUDYID', 'StudyID', 'GAWKSEND', 'BIRTH_TYPE', 
        'AGE_AT_V1', 'PEGHTN', 'CHRONHTN', 'OUTCOME'
    ] + lowercase_cols
    df = df[[c for c in df.columns if c not in exclude_cols]]
    y = df[target_col]
    X = df.drop(target_col, axis=1)
    X_cols = X.columns
    rus = RandomUnderSampler()
    X, y = rus.fit_resample(X, y)
    X = pd.DataFrame(data=X, columns=X_cols)
    df = X
    df[target_col] = y

    return MedicalDataset(
        data=df, target_col=target_col, 
        feature_to_test_map=tests_dict, tests=tests_dict, 
        n_folds=5, test_size=0.2
    )


def load_wisconsin_diagnosis_dataset(
        MASKED_FEATURE_TYPES = [
            'smoothness', 'compactness', 'concavity',
            'symmetry', 'fractaldimension', 'area'
    ],
        missingness_amounts = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    ) -> MedicalDataset:

    assert len(MASKED_FEATURE_TYPES) == len(missingness_amounts)

    colnames = [
        'radius', 
        'texture',
        'perimeter',
        'area',
        'smoothness',
        'compactness',
        'concavity',
        'concave points',
        'symmetry',
        'fractaldimension'
    ]
    feature_types = ['mean', 'stderr', 'worst']
    cols = [x[0] + '_' + x[1] for x in product(feature_types, colnames)]
    cols = ['id', 'diagnosis'] + cols
    df = pd.read_csv('../data/wdbc.data', header=None, 
                    names=cols, na_values='?')
    df = df.dropna(how='any')
    df = df.drop(columns='id')

    feature_groups = {
        c: []
        for c in colnames
    }

    for col in cols:
        splitcol = col.split('_')
        if len(splitcol) < 2:
            continue
        if splitcol[1] in colnames:
            feature_groups[splitcol[1]].append(col)

    target_col = 'diagnosis'

    df[target_col] = LabelEncoder().fit_transform(df[target_col])

    copy_df = df.__copy__()

    tests = {}
    num_tests = len(MASKED_FEATURE_TYPES)

    for i, f, amount in zip(list(range(num_tests)), MASKED_FEATURE_TYPES, missingness_amounts):
        tests[f] = Test(name=f, filename='', features=feature_groups[f])
        copy_df = create_missing_values(copy_df, feature_groups[f], num_samples=amount)
        
    base_features = []
    for f in feature_groups:
        if f not in MASKED_FEATURE_TYPES:
            base_features += feature_groups[f]

    tests_dict = {}
    for t in copy_df.columns:
        if '_' in t:
            test_type = t.split('_')[1]
            if test_type in MASKED_FEATURE_TYPES:
                tests_dict[t] = tests[test_type]

    return MedicalDataset(
        data=copy_df, target_col=target_col, 
        feature_to_test_map=tests_dict, tests=tests, 
        n_folds=5, test_size=0.2, cv_random_state=42,
        sampling_strategy=RandomUnderSampler()
    )


def load_wisconsin_prognosis_dataset(
        MASKED_FEATURE_TYPES = [
            'smoothness', 'compactness', 'concavity',
            'symmetry', 'fractaldimension', 'area'
    ],
        missingness_amounts = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    ) -> MedicalDataset:

    assert len(MASKED_FEATURE_TYPES) == len(missingness_amounts)

    colnames = [
        'radius', 
        'texture',
        'perimeter',
        'area',
        'smoothness',
        'compactness',
        'concavity',
        'concave points',
        'symmetry',
        'fractaldimension'
    ]
    feature_types = ['mean', 'stderr', 'worst']
    cols = [x[0] + '_' + x[1] for x in product(feature_types, colnames)]
    cols = ['id', 'prognosis', 'time'] + cols
    cols += ['tumorsize', 'lymphstatus']

    df = pd.read_csv('/Users/adamcatto/Dropbox/src/nuMoM2b/visit_test_sched/data/wpbc.data', header=None, 
                    names=cols, na_values='?')
    df = df.dropna(how='any')
    df = df.drop(columns='id')

    feature_groups = {
        c: []
        for c in colnames
    }

    for col in cols:
        splitcol = col.split('_')
        if len(splitcol) < 2:
            continue
        if splitcol[1] in colnames:
            feature_groups[splitcol[1]].append(col)
            
    feature_groups['time'] = ['time']
    feature_groups['tumorsize'] = ['tumorsize']
    feature_groups['lymphstatus'] = ['lymphstatus']

    target_col = 'prognosis'

    df[target_col] = LabelEncoder().fit_transform(df[target_col])

    copy_df = df.__copy__()

    tests = {}
    num_tests = len(MASKED_FEATURE_TYPES)

    for i, f, amount in zip(list(range(num_tests)), MASKED_FEATURE_TYPES, missingness_amounts):
        tests[f] = Test(name=f, filename='', features=feature_groups[f])
        copy_df = create_missing_values(copy_df, feature_groups[f], num_samples=amount)
        
    base_features = []
    for f in feature_groups:
        if f not in MASKED_FEATURE_TYPES:
            base_features += feature_groups[f]

    tests_dict = {}
    for t in copy_df.columns:
        if '_' in t:
            test_type = t.split('_')[1]
            if test_type in MASKED_FEATURE_TYPES:
                tests_dict[t] = tests[test_type]
        elif t in ['tumorsize', 'lymphstatus']:
            if t in MASKED_FEATURE_TYPES:
                tests_dict[t] = tests[t]

    return MedicalDataset(
        data=copy_df, target_col=target_col, 
        feature_to_test_map=tests_dict, tests=tests, 
        n_folds=5, test_size=0.2, cv_random_state=42,
        sampling_strategy=RandomUnderSampler()
    )


def normality_test_wisconsin():
    colnames = [
        'radius', 
        'texture',
        'perimeter',
        'area',
        'smoothness',
        'compactness',
        'concavity',
        'concave points',
        'symmetry',
        'fractaldimension'
    ]
    feature_types = ['mean', 'stderr', 'worst']
    cols = [x[0] + '_' + x[1] for x in product(feature_types, colnames)]
    cols = ['id', 'diagnosis'] + cols

    df = pd.read_csv('../data/wdbc.data', header=None, 
                    names=cols, na_values='?')
    print(df.head())
    df = df.dropna(how='any', axis=0)
    df = df.drop(columns=['id','diagnosis'])

    for i in df.columns:
        t = stats.normaltest(df[i])
        print(t)


def split_parkinsons_data(df, n_folds=5):
    ids = list(set(df.index))
    
    pos_ids = list(range(1, 21))
    num_pos_ids = len(pos_ids)
    min_pos_id, max_pos_id = min(pos_ids), max(pos_ids)
    neg_ids = list(range(21, 41))
    num_neg_ids = len(neg_ids)
    min_neg_id, max_neg_id = min(neg_ids), max(neg_ids)

    np.random.shuffle(pos_ids)
    np.random.shuffle(neg_ids)

    pos_slices = [
            (x, min(x + int(num_pos_ids/n_folds), num_pos_ids)) 
        for x in range(0, num_pos_ids, int(num_pos_ids/n_folds))
    ]
    neg_slices = [
        (x, min(x + int(num_neg_ids/n_folds), num_neg_ids)) 
        for x in range(0, num_neg_ids, int(num_neg_ids/n_folds))
    ]
    pos_set_ids = [pos_ids[x:y] for x, y in pos_slices]
    neg_set_ids = [neg_ids[x:y] for x, y in neg_slices]
    test_set_ids = [pos_ + neg_ for pos_, neg_ in zip(pos_set_ids, neg_set_ids)]
    print(test_set_ids)
    test_sets = [df.loc[x,:] for x in test_set_ids]
    train_sets = [df.drop(x.index) for x in test_sets]
    # splits = [df[1][1] for df in [groups[x: y] for x, y in slices]]
    # splits = [df for df in [groups[x: y] for x, y in slices]]
    # splits = [[x[1] for x in split] for split in splits]
    # flatten = lambda l: [item for sublist in l for item in sublist]
    
    # train_sets = [pd.concat(flatten([splits[idx] for idx in range(5) if idx != i ])) for i in range(5)]
    # test_sets = [pd.concat(x) for x in splits]

    train_val_test_triples = []
    for i, t in enumerate(train_sets):
        pos_train_df = t[t.index.isin(pos_ids)]
        pos_train_ids = list(set(pos_train_df.index))
        np.random.shuffle(pos_train_ids)
        pos_train_ids, pos_val_ids = (
            pos_train_ids[0 : int(len(pos_train_ids) / 2)], 
            pos_train_ids[int(len(pos_train_ids) / 2) : ]
        )
        
        neg_train_df = t[t.index.isin(neg_ids)]
        neg_train_ids = list(set(neg_train_df.index))
        np.random.shuffle(neg_train_ids)
        neg_train_ids, neg_val_ids = (
            neg_train_ids[0 : int(len(neg_train_ids) / 2)], 
            neg_train_ids[int(len(neg_train_ids) / 2) : ]
        )
        train_set = t[t.index.isin(pos_train_ids + neg_train_ids)]
        val_set = t[t.index.isin(pos_val_ids + neg_val_ids)]

        # train_ids = list(set(t.index))[0:int(len(set(t.index))/2)]
        # val_ids = [id_ for id_ in set(t.index) if id_ not in train_ids]
        # train_set = t.loc[train_ids, :]
        # val_set = t.loc[val_ids, :]
        train_val_test_triples.append((train_set, val_set, test_sets[i]))
        print(train_set.shape, val_set.shape, test_sets[i].shape)
        print(list(set(train_set.index)), list(set(val_set.index)), list(set(test_sets[i].index)))
        print(dict(train_set.targets.value_counts()), dict(val_set.targets.value_counts()), dict(test_sets[i].targets.value_counts()))

    return train_val_test_triples





class DataLoadersEnum(Enum):

    def prepare_eeg_eye_data(
        path_to_data: str = '../data/eeg_eye_state.csv'
    ) -> CustomExperimentDataObject:
        df = pd.read_csv(path_to_data, header=None)
        df.columns = [str(c) for c in df.columns]
        #df = df.iloc[0:600,:]
        target_col = df.columns[-1]
        return CustomExperimentDataObject(
            data=df, 
            dataset_name='eeg_eye_state', 
            target_col=target_col
        )

    def prepare_cleveland_heart_data(
        path_to_data: str = '../data/heart_cleveland_upload.csv'
    ) -> CustomExperimentDataObject:
        df = pd.read_csv(path_to_data)
        target_col = 'condition'
        dataset_name = 'cleveland_heart_disease'
        return CustomExperimentDataObject(
            data=df,
            dataset_name=dataset_name,
            target_col=target_col
        )

    def prepare_diabetic_retinopathy_dataset(
        path_to_data: str = '../data/diabetic_retinopathy_dataset.csv'
    ) -> CustomExperimentDataObject:
        df = pd.read_csv(path_to_data, header=None)
        df.columns = [str(c) for c in df.columns]
        target_col = df.columns[-1]
        
        return CustomExperimentDataObject(
            data=df, 
            dataset_name='diabetic_retinopathy', 
            target_col=target_col
        )

    def prepare_diabetes_vcu_dataset(
        path_to_data: str = '../data/diabetes_vcu.csv'
    ) -> CustomExperimentDataObject:
        df = pd.read_csv(path_to_data, na_values='?')
#        df = df.loc[:99,:]
        missing_mask_df = df.isna()#masking missing values
        target_col = 'readmitted'
        df['age'] = df['age'].apply(lambda x: int(x[1]))
        df['readmitted'] = np.where(df['readmitted'] == 'NO', 0, 1)
        df = df.drop(columns=['patient_nbr', 'weight', 'payer_code', 'medical_specialty'])
        categ_cols = [
            'race', 'gender', 'admission_type_id', 'discharge_disposition_id',
            'admission_source_id', 'diag_1', 'diag_2', 'diag_3', 'max_glu_serum','A1Cresult','metformin','repaglinide', 'nateglinide','chlorpropamide','glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide','pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin','glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed'
        ]
        #transform each cell from above categories
        for c in categ_cols:
            df[c] = LabelEncoder().fit_transform(df[c]).astype(int)
        #max_glu_serum_map = {'None': np.nan, 'Norm': 1, '>200': 2, '300': 3}
        #print(df['max_glu_serum'].unique())
        df = df.where(~missing_mask_df)
        return CustomExperimentDataObject(df,'diabetes_vcu', target_col)
    # wisconsin breast cancer ___prognosis___ dataset
    def prepare_wpbc_data(
        path_to_data: str = '../data/wpbc.data'
    ) -> CustomExperimentDataObject:
        colnames = [
            'radius', 
            'texture',
            'perimeter',
            'area',
            'smoothness',
            'compactness',
            'concavity',
            'concave points',
            'symmetry',
            'fractaldimension'
        ]
        feature_types = ['mean', 'stderr', 'worst']
        cols = [x[0] + '_' + x[1] for x in product(feature_types, colnames)]
        cols = ['id', 'prognosis', 'time'] + cols # this way col order is correct
        cols += ['tumorsize', 'lymphstatus']
        
        df = pd.read_csv(path_to_data, header=None, 
                    names=cols, na_values='?')
        df = df.drop(columns='id')
        target_col = 'prognosis'
        df[target_col] = LabelEncoder().fit_transform(df[target_col])

        return CustomExperimentDataObject(
            data=df,
            dataset_name='wisconsin_bc_prognosis',
            target_col=target_col
        )
        
    # wisconsin breast cancer ___diagnosis___ dataset
    def prepare_wdbc_data(
        path_to_data: str = '../data/wdbc.data'
    ) -> CustomExperimentDataObject:
        colnames = [
            'radius', 
            'texture',
            'perimeter',
            'area',
            'smoothness',
            'compactness',
            'concavity',
            'concave points',
            'symmetry',
            'fractaldimension'
        ]
        feature_types = ['mean', 'stderr', 'worst']
        cols = [x[0] + '_' + x[1] for x in product(feature_types, colnames)]
        cols = ['id', 'diagnosis'] + cols
        
        df = pd.read_csv(path_to_data, header=None, 
                    names=cols, na_values='?')
        df = df.drop(columns='id')
        target_col = 'diagnosis'
        df[target_col] = LabelEncoder().fit_transform(df[target_col])
        
        return CustomExperimentDataObject(
            data=df,
            dataset_name='wisconsin_bc_diagnosis',
            target_col=target_col
        )
        
    def prepare_parkinsons_data(
        path_to_data: str = '../data/parkinsons/train_data.txt'
    ) -> CustomExperimentDataObject:
        
        df = pd.read_csv(path_to_data, header=None, index_col=0)
        train_ids = set(df.index)
        pos_ids = list(range(1,21))
        neg_ids = list(range(21,41))
        colnames = [
            'Jitter (local)', 'Jitter (local, absolute)', 'Jitter (rap)', 
            'Jitter (ppq5)', 'Jitter (ddp)',  'Shimmer (local)', 
            'Shimmer (local, dB)', 'Shimmer (apq3)', 'Shimmer (apq5)', 
            'Shimmer (apq11)', 'Shimmer (dda)', 'AC', 'NTH', 'HTN', 
            'Median pitch', 'Mean pitch', 'Standard deviation', 'Minimum pitch',
            'Maximum pitch',  'Number of pulses', 'Number of periods', 
            'Mean period', 'Standard deviation of period',  
            'Fraction of locally unvoiced frames', 'Number of voice breaks', 
            'Degree of voice breaks', 'UPDRS', 'targets'
        ]
        
        df.columns = colnames

        df = df.drop(columns='UPDRS')
        target_col = 'targets'
        dataset_name = 'parkinsons'
        df[target_col] = LabelEncoder().fit_transform(df[target_col])
        return CustomExperimentDataObject(data=df, dataset_name=dataset_name, target_col=target_col)

    def prepare_cervical_cancer_data(
        path_to_data: str = '../data/risk_factors_cervical_cancer.csv',
        target_col: str = 'Hinselmann'
    ) -> CustomExperimentDataObject:
        df = pd.read_csv(path_to_data, na_values='?')
        cols_to_drop = [
            'STDs: Time since first diagnosis',
            'STDs: Time since last diagnosis',
            'Dx:Cancer', 'Dx'
        ]
        target_cols = ['Hinselmann', 'Schiller', 'Citology', 'Biopsy']
        df = df.drop(columns=cols_to_drop)
        df = df.drop(columns=[c for c in target_cols if c != target_col])
        dataset_name = 'cervical_cancer_' + target_col
        df[target_col] = LabelEncoder().fit_transform(df[target_col])
        y = df[target_col]
        X = df.drop(columns=target_col)
        X_cols = df.columns
        rus = RandomUnderSampler(random_state=0)
        X_sample, y_sample = rus.fit_resample(X, y)
        df = pd.DataFrame(data=X_sample, columns=X_cols)
        df[target_col] = y_sample
        return CustomExperimentDataObject(data=df, dataset_name=dataset_name, target_col=target_col)

    def prepare_myocardial_infarction_data(
        path_to_data: str = '../data/MI.data'
    ) -> CustomExperimentDataObject:
        df = pd.read_csv(path_to_data, header=None, index_col=0, na_values='?')
        # target_idx = -1
        target_col = 'targets'
        df.columns = [str(c) for c in df.columns[0: df.shape[1] - 1]] + [target_col]
        df = df.drop(columns=[str(x) for x in range(112, 123)]) # drop target-related cols
        df[target_col] = df[target_col].map(lambda x: 0 if x == 0 else 1)
        df[target_col] = LabelEncoder().fit_transform(df[target_col])
        y = df[target_col]
        X = df.drop(columns=target_col)
        X_cols = df.columns
        rus = RandomUnderSampler(random_state=0)
        X_sample, y_sample = rus.fit_resample(X, y)
        df = pd.DataFrame(data=X_sample, columns=X_cols)
        df[target_col] = y_sample
        return CustomExperimentDataObject(data=df, dataset_name='myocardial_infarction', target_col=target_col)
        
    def prepare_student_data(
        path_to_data: str = '../data/student/student-mat.csv',
        target_col='G3'
    ) -> CustomExperimentDataObject:
        if 'mat' in path_to_data.split('/')[-1]:
            subtype = 'mat'
        elif 'por' in path_to_data.split('/')[-1]:
            subtype = 'por'
        potential_targets = ['G1', 'G2', 'G3']
        df = pd.read_csv(path_to_data, sep=';')
        targets = np.array(df[target_col]).astype(int).squeeze()
        # print(targets)
        df = df.drop(columns=potential_targets)
        categorical_cols = [
            'school', 'sex', 'famsize', 'address', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 
            'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 
            'activities', 'nursery', 'higher', 'internet', 'romantic'
        ]
        col_transformer = ColumnTransformer(
            [('enc_' + str(i), LabelEncoder(), [c]) for i, c in enumerate(categorical_cols)], 
            remainder='passthrough'
        )
        for c in categorical_cols:
            df[c] = LabelEncoder().fit_transform(df[c]).astype(int)
        #df = col_transformer.fit_transform(df)
        #df = pd.DataFrame(df)
        df[target_col] = targets
        return CustomExperimentDataObject(
            data=df, 
            dataset_name='Student_' + subtype + '_' + target_col,
            target_col=target_col
        )


