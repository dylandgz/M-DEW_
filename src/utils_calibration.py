#libraries
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# import sys
# sys.path.append('.')
from src.data_loaders import MissDataset

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import textwrap

def train_test_val(data_object, num):
    print(num)
    dataset = MissDataset(data_object.data, target_col=data_object.target_col, n_folds=5)
       
    # #creating train, validation, and test sets
    train, val, test = dataset.__getitem__(0)
    X_train = train.drop(columns=[dataset.target_col])
    y_train = train[dataset.target_col]

    X_val = val.drop(columns=[dataset.target_col])
    y_val = val[dataset.target_col]

    X_test = test.drop(columns=[dataset.target_col])
    y_test = test[dataset.target_col]

    return X_train, X_val, X_test, y_train, y_val, y_test

def calibration_pipeline(X_train, y_train, X_val,clf_imputer_pairs, calibrated_pipelines):
    for (base_clf, clf_name), (base_imp, imp_name) in clf_imputer_pairs:
        pipeline_name = f"Estim({clf_name})_Imputer({imp_name})"
        pipeline = Pipeline([
            ('imputer', base_imp),
            ('classifier', CalibratedClassifierCV(base_clf, method='sigmoid'))  # You can change the calibration method here
        ])
        calibrated_pipelines[pipeline_name] = pipeline

        for pipeline_name, pipeline in calibrated_pipelines.items():
            pipeline.fit(X_train, y_train)
    
        for pipeline_name, pipeline in calibrated_pipelines.items():
            y_pred_proba = pipeline.predict_proba(X_val)

def plot_calibration_curve(data_obj):
    # Plot calibration curve for each pipeline
    for pipeline_name, pipeline in calibrated_pipelines.items():  # Iterate over pipeline_name and pipeline
        prob_pos = pipeline.predict_proba(X_val)[:, 1]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_val, prob_pos, n_bins=10, strategy='uniform')

        # Wrap the long label for better visualization
        wrapped_label = "\n".join(textwrap.wrap(pipeline_name, width=20))
        plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label=wrapped_label)  # Use pipeline_name as the label

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(data_obj.dataset_name)
    # plt.legend(loc='upper left')

    plt.legend(loc='upper center', bbox_to_anchor=(1.2, 1.15), ncol=1)

    output_directory = 'figure/calibration(train_val)/'
    output_filename = f'calibration_{data_obj.dataset_name}.png'
    plt.savefig(output_directory + output_filename)

    plt.show()

