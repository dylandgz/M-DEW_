from collections import Counter
import os
from turtle import shape
from typing import Iterable

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from pyparsing import col
import scipy
from scipy import optimize
from scipy.special import kl_div
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
import torch
from tqdm import tqdm


__all__ = [
    'get_n_tests_missing', 'label_encoded_data', 
    'find_optimal_threshold', 'Test'
]

COLORS = [
        'tab:blue',
        'tab:green',
        'tab:brown',
        'tab:red',
        'tab:cyan',
        'tab:olive',
        'tab:pink',
        'tab:orange',
        'tab:purple',
        'tab:gray'
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


def get_cols_without_missing_values(df: pd.DataFrame) -> list:
    return [x for x, y in df.isna().astype(int).max().items() if y == 0]


def get_n_tests_missing(cols, feature_to_test_map, missing_indices):
    seen_tests = []
    if len(missing_indices) == 0:
        return 0
    
    for i in missing_indices:
        feature = cols[i]
        test = feature_to_test_map[feature]
        if test not in seen_tests:
            seen_tests.append(test)

    return len(seen_tests)


def label_encoded_data(data, ignore_columns):
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


def find_optimal_threshold(proba_predictions, y_test, step=0.1):
    end_range = int(100 / step)
    sensitivities = []
    specificities = []
    ppvs = []
    npvs = []
    gmeans_sens_spec = []
    gmeans_all_metrics = []

    for i in range(1, end_range):
        predictions = np.array([1 if x >= i / end_range else 0 for x in proba_predictions])
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

    max_index = gmeans_all_metrics.index(max(gmeans_all_metrics))
    probability_threshold = max_index / end_range
    return probability_threshold


def get_classification_metrics(predictions, ground_truth):
    correct_pos = 0
    correct_neg = 0
    false_pos = 0
    false_neg = 0

    for y, y_hat in zip(ground_truth, predictions):
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
    try:
        specificity = correct_neg / (correct_neg + false_pos)
    except:
        specificity = 0
    try:
        ppv = correct_pos / (correct_pos + false_pos)
    except:
        ppv = 0
    try:
        npv = correct_neg / (correct_neg + false_neg)
    except:
        npv = 0
    try:
        gmean_sens_spec = np.sqrt(sensitivity * specificity)
    except:
        gmean_sens_spec = 0
    try:
        gmean_all_metrics = np.prod(
            [sensitivity, specificity, ppv, npv]
        )** (1 / 4)
    except:
        gmean_all_metrics = 0

    return {
        'sensitivity': round(sensitivity, 4), 
        'specificity': round(specificity, 4), 
        'ppv': round(ppv, 4), 
        'npv': round(npv, 4), 
        'gmean_sens_spec': round(gmean_sens_spec, 4), 
        'gmean_all_metrics': round(gmean_all_metrics, 4)
    }


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU 
    return device


def get_prediction_method(clf: BaseEstimator):
    if hasattr(clf, 'predict_proba'):
        prediction_method = 'predict_proba'
    elif hasattr(clf, 'decision_function'):
        prediction_method = 'decision_function'
    else:
        prediction_method = 'predict'
    return prediction_method


def create_missing_values(df, feature_combination, num_samples):
    assert num_samples > 0
    if num_samples > 1:
        assert isinstance(num_samples, int)
    else:
        num_samples = int(len(df) * num_samples)
    
    copy_df = df.copy()
    sub_table = copy_df[feature_combination]
    rows = sub_table.sample(num_samples)
    sub_table[sub_table.index.isin(rows.index)] = np.nan
    
    for f in feature_combination:
        copy_df[f] = sub_table[f]
    
    return copy_df


def get_sample_indices_with_optional_tests(df, test_features):
    df = df.loc[:, test_features]
    df = df.dropna(how='all', axis=0)
    return np.array(df.index)


def plot_prediction_errors(
    y_true: np.ndarray, proba_predictions_df: pd.DataFrame,
    title, xlabel, ylabel, outfile=None
):
    plt.figure(figsize=(24,16), dpi=300)
    colors = [
        'tab:blue',
        'tab:green',
        'tab:orange',
        'tab:cyan',
        'tab:olive',
        'tab:purple',
        'tab:red',
        'tab:brown',
        'tab:pink',
        'tab:gray'
    ]
    column_colors = {}
    lines = []
    seen_cols = []
    error_df = proba_predictions_df

    for i, c in enumerate(error_df.columns):
        error_df[c] = np.abs(error_df[c] - y_true)
        column_colors[c] = colors[i]
    
    for i, row in error_df.iterrows():
        # we iterate over rows instead of just plotting columns because
        # the values at each row must be sorted high-to-low in order for
        # all the stems to show properly.
        row = row.sort_values(ascending=False)
        for col, err in zip(row.index, row):
            markerline, stemline, baseline = plt.stem(
                [i],
                [err],
                linefmt=column_colors[col],
                label=col
            )
            plt.setp(markerline, markersize=1)

    plt.axhline(y=0.5, color='black', linestyle='--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), prop={'size': 16})
    if outfile is not None:
        plt.savefig(outfile)
    plt.clf()


def parse_synthetic_experiment_name(s):
    """
    template: `amount(X)__features(list(Y))`
    """
    amount_str, features_str = tuple(s.split('__'))
    # now we have ('amount(X)', 'features(list(Y))')
    missingness_fraction = eval(amount_str[7: -1])
    features = eval(features_str[9: -1])
    n_features = len(features)
    return missingness_fraction, n_features


def make_classifier_performance_matrix(df, col):
    if isinstance(df.index[0], int):
        df.index = df.iloc[:, 0]
    metrics = df[col]
    n_rows_cols = np.sqrt(len(metrics))
    assert n_rows_cols % 1 == 0
    performance_matrix = np.zeros(shape=(int(n_rows_cols), int(n_rows_cols)))
    metrics.index = [parse_synthetic_experiment_name(s) for s in metrics.index]
    for (missing_fraction, n_features), metric in zip(metrics.index, metrics):
        i, j = int(missing_fraction * 10 - 1), int(n_features - 1)
        performance_matrix[i, j] = metric

    return performance_matrix


def make_clf_performance_heatmap(df, col_to_plot, outfile, cols=None):
    
    perf_matrices = []
    if cols is None:
        cols = df.columns
    # final_cols = []
    # for c in cols:
    #     if 'oracle' in c:
    #         continue
    #     perf_matrices.append(make_classifier_performance_matrix(df, c))
    #     final_cols.append(c)

    # performance_col_values = np.array(df[col_to_plot])
    shape_ = tuple([np.sqrt(len(df))] * 2)
    performance_matrix = make_classifier_performance_matrix(df, col_to_plot)
    
    
    heatmap = sns.heatmap(
        data=performance_matrix, 
        annot=True,
        # cmap=COLORS[0:len(final_cols)],
        cmap='Blues',
        xticklabels=[x * 3 for x in range(1, 10)],
        yticklabels=[x / 10 for x in range(1, 10)]
    )
    plt.title('Classifier Performance Per Missingness Pattern')
    plt.ylabel('Fraction of Values Missing')
    plt.xlabel('Number of Features With Missing Values')
    # patches = [
    #     mpatches.Patch(color=COLORS[i], label=final_cols[i])
    #     for i in range(len(final_cols))
    # ]
    # plt.legend(handles=patches)
    plt.savefig(outfile, dpi=100)
    return performance_matrix


def make_best_classifier_per_missing_pattern_heatmap(df, outfile, cols=None):
    
    perf_matrices = []
    if cols is None:
        cols = df.columns
    final_cols = []
    for c in cols:
        if 'oracle' in c:
            continue
        perf_matrices.append(make_classifier_performance_matrix(df, c))
        final_cols.append(c)
    
    all_performances_tensor = np.dstack(perf_matrices)
    best_performance_matrix = np.argmax(all_performances_tensor, axis=-1)
    # best_performance_matrix = np.where(
    #     all_performances_tensor == np.amax(all_performances_tensor, axis=-1),
    # )
    print(best_performance_matrix)
    # print(Counter(best_performance_matrix.flatten().tolist()))
    print(best_performance_matrix)
    heatmap = sns.heatmap(
        data=best_performance_matrix, 
        cmap=COLORS[0:len(final_cols)],
        xticklabels=[x * 3 for x in range(1, 10)],
        yticklabels=[x / 10 for x in range(1, 10)]
    )
    plt.title('Best Performing Classifiers Per Missingness Pattern')
    plt.ylabel('Fraction of Values Missing')
    plt.xlabel('Number of Features With Missing Values')
    patches = [
        mpatches.Patch(color=COLORS[i], label=final_cols[i])
        for i in range(len(final_cols))
    ]
    plt.legend(handles=patches)
    plt.savefig(outfile, dpi=100)
    return {
        'all_performances_tensor': all_performances_tensor,
        'best_performance_matrix': best_performance_matrix
    }


def all_max_indices_along_axis(arr, axis):
    pass


def sort_model_type_strings(colnames):
    """
    column names are not exactly the model type names; we want to sort 
    columns in a specific way, namely by the `sizes` dictionary in the 
    `make_metrics_comparison_plot` function below.
    """
    sorted_cols = {c: 0 for c in colnames}
    for c in colnames:
        if 'knn' in c:
            sorted_cols[c] = 5
        elif 'mice' in c:
            sorted_cols[c] = 4
        elif 'vanilla' in c:
            sorted_cols[c] = 3
        elif 'stacked' in c:
            sorted_cols[c] = 2
        elif 'inheritance' in c:
            sorted_cols[c] = 1
        else: # 'ds'
            sorted_cols[c] = 0
        
    return [x[0] for x in sorted(sorted_cols.items(), key=lambda x: x[1])]


def make_metrics_comparison_plot(df, metric: str, experiment_type: str, experiment_subtype: str=''):
    if experiment_subtype != '':
        experiment_type += '/' + experiment_subtype
    
    cols = []
    num_yticks = len(df)

    sizes = {
        'knn': 16,
        'mice': 32,
        'vanilla': 48,
        'stacked': 72,
        'inheritance': 104,
        'ds': 120,
        # 'oracle': 6
    }

    for col in df.columns:
        if 'oracle' in col:
            continue
        elif metric in col:
            cols.append(col)

    # sort cols by markersize; now there is no circle-overlap problem.
    cols = sort_model_type_strings(cols)
    print(cols)

    fig, ax = plt.subplots(figsize=(18, 24))
    ax.set_yticklabels([parse_synthetic_experiment_name(str(x)) for x in df.index])
    ax.set_yticks(list(range(num_yticks)))
    ax.grid(axis='y')

    patches = [
        mpatches.Patch(color=COLORS[i], label=cols[i])
        for i in range(len(cols))
    ]

    for i, col in enumerate(cols):
        model_type = col.split('_')[-2]
        plt.scatter(
            x=df[col], 
            y=list(range(num_yticks)), 
            c=COLORS[i],
            s=sizes[model_type]
        )

    exp_str_fmt = {
        'wisconsin_bc_prognosis': 'Wisconsin Breast Cancer Prognosis',
        'wisconsin_bc_diagnosis': 'Wisconsin Breast Cancer Diagnosis',
        'synthetic_classification/mcar': 'MCAR Synthetic Experiment',
        'synthetic_classification/mar': 'MAR Synthetic Experiment',
        'synthetic_classification/mnar': 'MNAR Synthetic Experiment'
    }

    plt.title('Comparison of Classifiers: ' + metric + ', ' + exp_str_fmt[experiment_type])
    plt.ylabel('Experiment (missingness fraction, \# of masked tests)')
    plt.xlabel(metric)
    plt.legend(handles=patches)

    out_dir = '../results/' + experiment_type
    filename = 'comparison_' + metric + '_' + experiment_type.replace('/', '_') + '.png'

    outfile = os.path.join(out_dir, filename)

    plt.savefig(outfile, dpi=100)
    

def rank_order_df(df: pd.DataFrame, metric, experiment_type, experiment_subtype=''):
    if experiment_subtype != '':
        experiment_type += '/' + experiment_subtype
    
    cols = []

    for col in df.columns:
        if 'oracle' in col:
            continue
        elif metric in col:
            cols.append(col)

    metric_df = df[cols]
    ranked_df = metric_df.rank(axis=1, ascending=False)
    median_ranks = ranked_df.median(axis=0).to_frame().T
    mean_ranks = ranked_df.mean(axis=0).to_frame().T
    sd_ranks = ranked_df.std(axis=0).to_frame().T
    aggregate_ranks_df = pd.concat([median_ranks, mean_ranks, sd_ranks])
    outfile = os.path.join(
        '../results/', 
        experiment_type,
        metric + '_aggregate_rankings.csv'
    )
    aggregate_ranks_df.to_csv(outfile)
    return aggregate_ranks_df


def get_summary_statistics(df: pd.DataFrame, metric, experiment_type, experiment_subtype=''):
    if experiment_subtype != '':
        experiment_type += '/' + experiment_subtype
    
    cols = []

    for col in df.columns:
        if 'oracle' in col:
            continue
        elif metric in col:
            cols.append(col)

    metric_df = df[cols]
    median_df = metric_df.median(axis=0).to_frame().T
    mean_df = metric_df.mean(axis=0).to_frame().T
    sd_df = metric_df.std(axis=0).to_frame().T
    aggregate_ranks_df = pd.concat([median_df, mean_df, sd_df])
    aggregate_ranks_df.index = ['Median', 'Average', 'SD']
    outfile = os.path.join(
        '../results/', 
        experiment_type,
        metric + '_aggregate_scores.csv'
    )
    aggregate_ranks_df.to_csv(outfile)
    return aggregate_ranks_df


def plot_metric_distributions(df: pd.DataFrame, metric: str, experiment_type: str, experiment_subtype: str=''):
    exp_str_fmt = {
        'wisconsin_bc_prognosis': 'Wisconsin Breast Cancer Prognosis',
        'wisconsin_bc_diagnosis': 'Wisconsin Breast Cancer Diagnosis',
        'synthetic_classification/mcar': 'MCAR Synthetic Experiment',
        'synthetic_classification/mar': 'MAR Synthetic Experiment',
        'synthetic_classification/mnar': 'MNAR Synthetic Experiment'
    }

    if experiment_subtype != '':
        experiment_type += '/' + experiment_subtype

    cols = []

    for col in df.columns:
        if 'oracle' in col:
            continue
        elif metric in col:
            cols.append(col)

    metric_df = df[cols]
    fig, ax = plt.subplots(figsize=(16, 12))
    boxplot_ = sns.violinplot(data=metric_df)
    plt.xticks(rotation=30)

    plt.title('Distribution of ' + metric + ', ' + exp_str_fmt[experiment_type])
    plt.xlabel('Model Type')
    plt.ylabel(metric + ' value')

    outfile = os.path.join(
        '../results/', 
        experiment_type,
        'viz',
        metric + '_violinplot.png'
    )
    plt.savefig(outfile, dpi=100)


def plot_rankings_distributions(df: pd.DataFrame, metric: str, experiment_type: str, experiment_subtype: str=''):
    exp_str_fmt = {
        'wisconsin_bc_prognosis': 'Wisconsin Breast Cancer Prognosis',
        'wisconsin_bc_diagnosis': 'Wisconsin Breast Cancer Diagnosis',
        'synthetic_classification/mcar': 'MCAR Synthetic Experiment',
        'synthetic_classification/mar': 'MAR Synthetic Experiment',
        'synthetic_classification/mnar': 'MNAR Synthetic Experiment'
    }
    if experiment_subtype != '':
        experiment_type += '/' + experiment_subtype
    cols = []

    for col in df.columns:
        if 'oracle' in col:
            continue
        elif metric in col:
            cols.append(col)

    metric_df = df[cols]
    ranked_df = metric_df.rank(axis=1, ascending=False)
    fig, ax = plt.subplots(figsize=(16, 12))
    boxplot_ = sns.violinplot(data=ranked_df)
    plt.xticks(rotation=30)

    plt.title('Rankings Distribution of ' + metric + ', ' + exp_str_fmt[experiment_type])
    plt.xlabel('Model Type')
    plt.ylabel(metric + ' ranking')

    outfile = os.path.join(
        '../results/', 
        experiment_type,
        'viz',
        metric + '_rankings_dist_violinplot.png'
    )
    plt.savefig(outfile, dpi=100)


def make_3d_plot_comparison(df, col_to_plot):
    """
    make different plots: <60% auc, 60-70% auc, >70% auc.
    
    ...to show which missingness patterns are associated with different 
    performance levels
    """
    indices = df.index.astype(str)
    if indices[0] == '0':
        df = df.set_index(df.iloc[:, 0])
        indices = df.index.astype(str)
    indices = [parse_synthetic_experiment_name(idx) for idx in indices]
    
    df_60 = df[df[col_to_plot] < 0.6]
    df60_70 = df[(df[col_to_plot] >= 0.6) & (df[col_to_plot] < 0.7)]
    df70_ = df[df[col_to_plot] >= 0.7]

    for performance, df_ in zip(
        ['under_60', '60_70', 'over_70'],
        [df_60, df60_70, df70_]
    ):
        miss_rate = [idx[0] for idx in indices]
        miss_rate = sorted(list(set(miss_rate)))
        n_features = [idx[1] for idx in indices]
        n_features = sorted(list(set(n_features)))
        shape1d = int(np.sqrt(len(df_)))
        shape_ = (shape1d, shape1d)
        z = np.array(df_[col_to_plot])
        z = z[:, np.newaxis]
        z = np.reshape(z, shape_)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        plot_ = ax.contourf(X=miss_rate, Y=n_features, Z=z)
        # plot_ = ax.scatter(xs=miss_rate, ys=n_features, zs=z)
        # plot_ = ax.plot_trisurf(X=miss_rate, Y=n_features, Z=z)
        plt.savefig('test_3d_' + performance + '.png', dpi=100)
        plt.clf()


def KL(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def get_weighted_error(weights, errors):
    return np.sum(weights * errors)


def compare_dew_to_uniform(weights_dir, err_dir, out_dir):
    weights_files = sorted([os.path.join(weights_dir, x) for x in os.listdir(weights_dir)])
    err_files = sorted([os.path.join(err_dir, x) for x in os.listdir(err_dir)])

    ratio_dew_to_uniform = []
    top_performers = []
    mean_dew_errors = []
    mean_uniform_errors = []

    for w, e in tqdm(zip(weights_files, err_files), total=len(weights_files)):
        weights_df = pd.read_csv(w, index_col=0)
        err_df = pd.read_csv(e, index_col=0)
        dew_errors = err_df['DEW']
        min_num_cols = min(weights_df.shape[1], err_df.shape[1])
        weights_df = weights_df.iloc[:, 0:min_num_cols]
        err_df = err_df.iloc[:, 0:min_num_cols]

        mean_dew_err = np.mean(dew_errors)
        mean_dew_errors.append(mean_dew_err)

        uniform_errors = np.mean(err_df.to_numpy(), axis=1)
        mean_uniform_err = np.mean(uniform_errors)
        mean_uniform_errors.append(mean_uniform_err)

        fraction_dew_outperforms_uniform = (
            dew_errors < uniform_errors
        ).astype(int).sum() / len(dew_errors)
        ratio_dew_to_uniform.append(fraction_dew_outperforms_uniform)

        top_performer = 'dew' if fraction_dew_outperforms_uniform else 'uniform'
        top_performers.append(top_performer)

    agg_df = pd.DataFrame({})
    agg_df['percent of samples DEW outperforms model averaging'] = ratio_dew_to_uniform
    agg_df['top performer in experiment'] = top_performers
    agg_df.index = sorted(os.listdir(weights_dir))
    agg_df.to_csv(os.path.join(out_dir, 'dew_vs_uniform_top_performer.csv'))

    plt.axhline(y=0.5, color='black', linestyle='--')

    sns.scatterplot(
        x=range(len(ratio_dew_to_uniform)), y=ratio_dew_to_uniform,
        size=0.5
    )
    plt.legend([],[], frameon=False)
    plt.title('\% of Samples where DEW Outperforms Uniform Averaging, per Experiment')
    plt.xticks([])
    plt.xlabel('Experiment Number')
    plt.ylabel('\% of Samples')
    plt.savefig(os.path.join(out_dir, 'viz', 'dew_vs_uniform_top_performer.png'), dpi=150)
    
    plt.clf()

    dew_uniform_errors_df = pd.DataFrame({})
    dew_uniform_errors_df['dew'] = mean_dew_errors
    dew_uniform_errors_df['uniform'] = mean_uniform_errors
    dew_uniform_errors_df.index = sorted(os.listdir(weights_dir))

    dew_uniform_errors_df.to_csv(os.path.join(out_dir, 'weight_competence_error.csv'))

    ratios = np.array(mean_dew_errors) / np.array(mean_uniform_errors)
    plt.axhline(y=1.0, color='black', linestyle='--')
    sns.scatterplot(x=range(len(ratios)), y=ratios, size=0.1)
    plt.legend([],[], frameon=False)
    plt.title('Ratio of DEW Error to Uniform Averaging Error')
    plt.xticks([])
    plt.xlabel('Experiment Number')
    plt.ylabel('Ratio')
    plt.savefig(os.path.join(out_dir, 'viz', 'dew_vs_uniform_errors.png'), dpi=150)

    plt.clf()



# == the following is adapted from 
# https://github.com/R-miss-tastic/website/blob/master/static/how-to/python/

##################### START R-MISSTASTIC UTILS ############################


def nanmean(v, *args, **kwargs):
    """
    A Pytorch version on Numpy's nanmean
    """
    v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


#### Quantile ######
def quantile(X, q, dim=None):
    """
    Returns the q-th quantile.
    Parameters
    ----------
    X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
        Input data.
    q : float
        Quantile level (starting from lower values).
    dim : int or None, default = None
        Dimension allong which to compute quantiles. If None, the tensor is flattened and one value is returned.
    Returns
    -------
        quantiles : torch.DoubleTensor
    """
    return X.kthvalue(int(q * len(X)), dim=dim)[0]


#### Automatic selection of the regularization parameter ####
def pick_epsilon(X, quant=0.5, mult=0.05, max_points=2000):
    """
        Returns a quantile (times a multiplier) of the halved pairwise squared distances in X.
        Used to select a regularization parameter for Sinkhorn distances.
    Parameters
    ----------
    X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
        Input data on which distances will be computed.
    quant : float, default = 0.5
        Quantile to return (default is median).
    mult : float, default = 0.05
        Mutiplier to apply to the quantiles.
    max_points : int, default = 2000
        If the length of X is larger than max_points, estimate the quantile on a random subset of size max_points to
        avoid memory overloads.
    Returns
    -------
        epsilon: float
    """
    means = nanmean(X, 0)
    X_ = X.clone()
    mask = torch.isnan(X_)
    X_[mask] = (mask * means)[mask]

    idx = np.random.choice(len(X_), min(max_points, len(X_)), replace=False)
    X = X_[idx]
    dists = ((X[:, None] - X) ** 2).sum(2).flatten() / 2.
    dists = dists[dists > 0]

    return quantile(dists, quant, 0).item() * mult


#### Accuracy Metrics ####
def MAE(X, X_true, mask):
    """
    Mean Absolute Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.
    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.
    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)
    Returns
    -------
        MAE : float
    """
    if torch.is_tensor(mask):
        mask_ = mask.bool()
        return torch.abs(X[mask_] - X_true[mask_]).sum() / mask_.sum()
    else: # should be an ndarray
        mask_ = mask.astype(bool)
        return np.absolute(X[mask_] - X_true[mask_]).sum() / mask_.sum()



def RMSE(X, X_true, mask):
    """
    Root Mean Squared Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.
    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.
    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)
    Returns
    -------
        RMSE : float
    """
    if torch.is_tensor(mask):
        mask_ = mask.bool()
        return (((X[mask_] - X_true[mask_]) ** 2).sum() / mask_.sum()).sqrt()
    else: # should be an ndarray
        mask_ = mask.astype(bool)
        return np.sqrt(((X[mask_] - X_true[mask_])**2).sum() / mask_.sum())


##################### MISSING DATA MECHANISMS #############################

##### Missing At Random ######

def MAR_mask(X, p, p_obs):
    """
    Missing at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated. If a numpy array is provided,
        it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    p_obs : float
        Proportion of variables with *no* missing values that will be used for the logistic masking model.
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_obs = max(int(p_obs * d), 1) ## number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs ## number of variables that will have missing values

    ### Sample variables that will all be observed, and those with missing values:
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    ### Other variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask

##### Missing not at random ######

def MNAR_mask_logistic(X, p, p_params =.3, exclude_inputs=True):
    """
    Missing not at random mechanism with a logistic masking model. It implements two mechanisms:
    (i) Missing probabilities are selected with a logistic model, taking all variables as inputs. Hence, values that are
    inputs can also be missing.
    (ii) Variables are split into a set of intputs for a logistic model, and a set whose missing probabilities are
    determined by the logistic model. Then inputs are then masked MCAR (hence, missing values from the second set will
    depend on masked values.
    In either case, weights are random and the intercept is selected to attain the desired proportion of missing values.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    p_params : float
        Proportion of variables that will be used for the logistic masking model (only if exclude_inputs).
    exclude_inputs : boolean, default=True
        True: mechanism (ii) is used, False: (i)
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_params = max(int(p_params * d), 1) if exclude_inputs else d ## number of variables used as inputs (at least 1)
    d_na = d - d_params if exclude_inputs else d ## number of variables masked with the logistic model

    ### Sample variables that will be parameters for the logistic regression:
    idxs_params = np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_params]) if exclude_inputs else np.arange(d)

    ### Other variables will have NA proportions selected by a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_params, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_params], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_params].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    ## If the inputs of the logistic model are excluded from MNAR missingness,
    ## mask some values used in the logistic model at random.
    ## This makes the missingness of other variables potentially dependent on masked values

    if exclude_inputs:
        mask[:, idxs_params] = torch.rand(n, d_params) < p

    return mask

def MNAR_self_mask_logistic(X, p):
    """
    Missing not at random mechanism with a logistic self-masking model. Variables have missing values probabilities
    given by a logistic model, taking the same variable as input (hence, missingness is independent from one variable
    to another). The intercepts are selected to attain the desired missing rate.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    ### Variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, self_mask=True)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X, coeffs, p, self_mask=True)

    ps = torch.sigmoid(X * coeffs + intercepts)

    ber = torch.rand(n, d) if to_torch else np.random.rand(n, d)
    mask = ber < ps if to_torch else ber < ps.numpy()

    return mask


def MNAR_mask_quantiles(X, p, q, p_params, cut='both', MCAR=False):
    """
    Missing not at random mechanism with quantile censorship. First, a subset of variables which will have missing
    variables is randomly selected. Then, missing values are generated on the q-quantiles at random. Since
    missingness depends on quantile information, it depends on masked values, hence this is a MNAR mechanism.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    q : float
        Quantile level at which the cuts should occur
    p_params : float
        Proportion of variables that will have missing values
    cut : 'both', 'upper' or 'lower', default = 'both'
        Where the cut should be applied. For instance, if q=0.25 and cut='upper', then missing values will be generated
        in the upper quartiles of selected variables.
        
    MCAR : bool, default = True
        If true, masks variables that were not selected for quantile censorship with a MCAR mechanism.
        
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).
    """
    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_na = max(int(p_params * d), 1) ## number of variables that will have NMAR values

    ### Sample variables that will have imps at the extremes
    idxs_na = np.random.choice(d, d_na, replace=False) ### select at least one variable with missing values

    ### check if values are greater/smaller that corresponding quantiles
    if cut == 'upper':
        quants = quantile(X[:, idxs_na], 1-q, dim=0)
        m = X[:, idxs_na] >= quants
    elif cut == 'lower':
        quants = quantile(X[:, idxs_na], q, dim=0)
        m = X[:, idxs_na] <= quants
    elif cut == 'both':
        u_quants = quantile(X[:, idxs_na], 1-q, dim=0)
        l_quants = quantile(X[:, idxs_na], q, dim=0)
        m = (X[:, idxs_na] <= l_quants) | (X[:, idxs_na] >= u_quants)

    ### Hide some values exceeding quantiles
    ber = torch.rand(n, d_na)
    mask[:, idxs_na] = (ber < p) & m

    if MCAR:
    ## Add a mcar mecanism on top
        mask = mask | (torch.rand(n, d) < p)

    return mask



def produce_NA(X, p_miss, mecha = "MCAR", opt = None, p_obs = None, q = None):
    """
    Generate missing values for specifics missing-data mechanism and proportion of missing values. 
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d) or pd.DataFrame, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
        If pandas dataframe is provided, it will be converted to numpy array ==> pytorch tensor
    p_miss : float
        Proportion of missing values to generate for variables which will have missing values.
    mecha : str, 
            Indicates the missing-data mechanism to be used. "MCAR" by default, "MAR", "MNAR" or "MNARsmask"
    opt: str, 
         For mecha = "MNAR", it indicates how the missing-data mechanism is generated: using a logistic regression ("logistic"), quantile censorship ("quantile") or logistic regression for generating a self-masked MNAR mechanism ("selfmasked").
    p_obs : float
            If mecha = "MAR", or mecha = "MNAR" with opt = "logistic" or "quanti", proportion of variables with *no* missing values that will be used for the logistic masking model.
    q : float
        If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.
    
    Returns
    ----------
    A dictionnary containing:
    'X_init': the initial data matrix.
    'X_incomp': the data with the generated missing values.
    'mask': a matrix indexing the generated missing values.s
    """
    
    to_torch = not torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if to_torch:
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = X.astype(np.float32)
        X = torch.from_numpy(X)
    
    if mecha == "MAR":
        mask = MAR_mask(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "logistic":
        mask = MNAR_mask_logistic(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "quantile":
        mask = MNAR_mask_quantiles(X, p_miss, q, 1-p_obs).double()
    elif mecha == "MNAR" and opt == "selfmasked":
        mask = MNAR_self_mask_logistic(X, p_miss)
    else:
        # n_features_to_mask = X.shape[1] - int(X.shape[1] * p_obs)
        mask = (torch.rand(X.shape) < p_miss).double()
    
    X_nas = X.clone()
    X_nas[mask.bool()] = np.nan
    
    return {'X_init': X.double(), 'X_incomp': X_nas.double(), 'mask': mask}



def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
    n, d = X.shape
    if self_mask:
        coeffs = torch.randn(d)
        Wx = X * coeffs
        coeffs /= torch.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = torch.randn(d_obs, d_na)
        Wx = X[:, idxs_obs].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
    return coeffs


def fit_intercepts(X, coeffs, p, self_mask=False):
    if self_mask:
        d = len(coeffs)
        intercepts = torch.zeros(d)
        for j in range(d):
            def f(x):
                return torch.sigmoid(X * coeffs[j] + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -100, 100)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = torch.zeros(d_na)
        for j in range(d_na):
            def f(x):
                return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -100, 100)
    return intercepts


###################### END R-MISSTASTIC #########################



if __name__ == '__main__':
    
    def wbc_experiments():
        df = pd.read_csv(
            # '../results/wisconsin_bc_prognosis_accuracy.csv',
            '../results/final_results_wisconsin_bc_prognosis_FROZEN.csv',
            # '../results/numom2b_hypertension/numom2b_final_results.csv',
            index_col=0, header=0, nrows=81
        )
        # outfile = '../results/wisconsin_bc_prognosis/accuracy_clf_comparison_heatmap.png'
        # matrices_dict = make_best_classifier_per_missing_pattern_heatmap(df, outfile=outfile)

        # print(Counter(list(np.ravel(matrices_dict['best_performance_matrix']))))
        METRIC_TYPES = [
            'sensitivity', 'specificity', 'ppv', 'npv', 'gmean_sens_spec',
            'gmean_all', 'roc_auc', 'accuracy'
        ]
        for metric_type in METRIC_TYPES:
            make_metrics_comparison_plot(df, metric_type, 'wisconsin_bc_prognosis')
            ranks = rank_order_df(df, metric=metric_type, experiment_type='wisconsin_bc_prognosis')
            plot_metric_distributions(df, metric_type, 'wisconsin_bc_prognosis')
            plot_rankings_distributions(df, metric_type, 'wisconsin_bc_prognosis')
            get_summary_statistics(df, metric_type, 'wisconsin_bc_prognosis')


    def synth_experiments(miss_type='mnar'):
        METRIC_TYPES = [
            'sensitivity', 'specificity', 'ppv', 'npv', 'gmean_sens_spec',
            'gmean_all', 'roc_auc', 'accuracy'
        ]
        
        results_folder = os.path.join(
            '../results/synthetic_classification', 
            miss_type,
            'results'
        )

        trials = os.listdir(results_folder)
        trial_filepaths = [os.path.join(results_folder, t) for t in trials]

        results_dfs = []

        for f in trial_filepaths:
            results_dfs.append(pd.read_csv(f, index_col=0, header=0))

        results_df = pd.concat(results_dfs)
        results_df.columns = results_dfs[0].columns

        for metric in METRIC_TYPES:
            # make_metrics_comparison_plot(
            #     results_df, metric, 'synthetic_classification', miss_type
            # )
            ranks = rank_order_df(
                results_df, metric, 'synthetic_classification', miss_type
            )
            plot_metric_distributions(
                results_df, metric, 'synthetic_classification', miss_type
            )
            plot_rankings_distributions(
                results_df, metric, 'synthetic_classification', miss_type
            )
            get_summary_statistics(
                results_df, metric, 'synthetic_classification', miss_type
            )




    def test_3d_plot():
        df = pd.read_csv('../results/final_results_wisconsin_bc_prognosis_FROZEN.csv', nrows=81)
        make_3d_plot_comparison(df, 'roc_auc_mean_ds_metrics')

    # test_3d_plot()
    # experiments()
    # make_clf_performance_heatmap(
    #     pd.read_csv('../results/final_results_wisconsin_bc_prognosis_FROZEN.csv', nrows=81),
    #     col_to_plot='roc_auc_mean_ds_metrics',
    #     outfile='heatmap_roc_auc_mean_ds.png'
    # )
    # synth_experiments()
    for synth_exp_type in ['mar', 'mcar', 'mnar']:
        compare_dew_to_uniform(
            os.path.join('../results/synthetic_classification', synth_exp_type, 'dew_weights'), 
            os.path.join('../results/synthetic_classification', synth_exp_type, 'prediction_errors'),
            os.path.join('../results/synthetic_classification', synth_exp_type)
        )
