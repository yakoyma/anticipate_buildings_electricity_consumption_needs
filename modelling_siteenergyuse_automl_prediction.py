"""
===============================================================================
Regression Project: Anticipate Electricity Consumption Needs of
Buildings - Modelling - Prediction of SiteEnergyUse
===============================================================================

This file is organised as follows:
1. Load the dataset
2. Feature Engineering
3. Machine Learning
   3.1 Prediction with the feature ENERGYSTARScore
       3.1.1 PyCaret
       3.1.2 FLAML
       3.1.3 H2O
       3.1.4 EvalML
   3.2 Prediction without the feature ENERGYSTARScore
       3.2.1 PyCaret
       3.2.2 FLAML
       3.2.3 H2O
       3.2.4 EvalML
"""
# Standard libraries
import random
import platform

# Other libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import pycaret
import flaml
import shap
import h2o
import evalml


from sklearn.model_selection import train_test_split
from pycaret.regression import *
from flaml import AutoML
from shap import Explainer, maskers, plots
from h2o import init, H2OFrame, shutdown
from h2o.automl import H2OAutoML
from evalml import AutoMLSearch
from evalml.model_understanding import readable_explanation
from functions import *


# Display versions of platforms and packages
print('\n\nPython: {}'.format(platform.python_version()))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('NumPy: {}'.format(np.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))
print('PyCaret: {}'.format(pycaret.__version__))
print('FLAML: {}'.format(flaml.__version__))
print('SHAP: {}'.format(shap.__version__))
print('H2O: {}'.format(h2o.__version__))
print('EvalML: {}'.format(evalml.__version__))



# Constants
SEED = 0
MAX_ROWS_DISPLAY = 300
MAX_COLUMNS_DISPLAY = 150
FOLDS = 10

# Set the random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)

# Set the maximum number of rows and columns to display by Pandas
pd.set_option('display.max_rows', MAX_ROWS_DISPLAY)
pd.set_option('display.max_columns', MAX_COLUMNS_DISPLAY)



"""
===============================================================================
1. Load the dataset
===============================================================================
"""
print(f'\n\n\n1. Load the dataset')


# Load the dataset
INPUT_CSV = 'datasets/dataset.csv'
dataset = load_dataset(file_path=INPUT_CSV, encoding='utf-8')



"""
===============================================================================
2. Feature Engineering
===============================================================================
"""
print(f'\n\n\n2. Feature Engineering')


# Feature selection
X = dataset.dropna(subset=['SiteEnergyUse(kBtu)']).reset_index(drop=True)
y = X['SiteEnergyUse(kBtu)'].values
X = X.drop(['SiteEnergyUse(kBtu)', 'TotalGHGEmissions'], axis=1)

# Display X dataset information and description
dataset_info_description(dataset=X, max_rows=15)


# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, shuffle=True)


# Add labels to the training and test datasets
train_dataset = X_train.assign(Target=y_train)
test_dataset = X_test.assign(Target=y_test)

# Display train dataset information and description
dataset_info_description(dataset=train_dataset, max_rows=15)

# Display test dataset information and description
dataset_info_description(dataset=test_dataset, max_rows=15)



"""
===============================================================================
3. Machine Learning
===============================================================================
"""
print(f'\n\n\n3. Machine Learning')


# 3.1 Prediction with the feature ENERGYSTARScore
print(f'\n\n3.1 Prediction with the feature ENERGYSTARScore')

# 3.1.1 PyCaret
print(f'\n\n3.1.1 PyCaret')

# Instantiate the setup
s = setup(
    data=train_dataset,
    target='Target',
    index=False,
    train_size=0.8,
    keep_features=['ENERGYSTARScore'],
    remove_multicollinearity=True,
    fold=FOLDS,
    fold_shuffle=True,
    normalize=True,
    n_jobs=-1,
    session_id=SEED,
    verbose=True
)

# Selection of the best model by cross-validation
best = compare_models(
    fold=FOLDS,
    round=3,
    cross_validation=True,
    n_select=1,
    verbose=True
)
print(f'\nClassification of models:\n{best}')

# Make predictions
pred = predict_model(estimator=best, data=test_dataset)
print(f'\nPredictions:\n{pred}')

# Plot error
try:
    plot_model(best, plot='error')
except Exception as error:
    print(f'The following error occurred: {error}')

# Plot residuals
try:
    plot_model(best, plot='residuals')
except Exception as error:
    print(f'The following error occurred: {error}')

# Plot the global interpretability of the model (Feature importance)
try:
    plot_model(estimator=best, plot='feature')
except Exception as error:
    print(f'The following error occurred: {error}')

# Plot the global interpretability of the model (Summary)
try:
    plot_model(estimator=best, plot='summary')
except Exception as error:
    print(f'The following error occurred: {error}')

# Make predictions
y_pred = pred['prediction_label'].to_numpy()

# Evaluation
evaluate_regression(y_test=y_test, y_pred=y_pred, SEED=SEED)


# 3.1.2 FLAML
print(f'\n\n3.1.2 FLAML')

# Instantiate AutoML instance
flaml_automl = AutoML()
flaml_automl.fit(
    dataframe=train_dataset,
    label='Target',
    metric='mae',
    task='regression',
    n_jobs=-1,
    eval_method='auto',
    n_splits=FOLDS,
    split_type='auto',
    seed=SEED,
    early_stop=True
)

# Display information about the best model
print('\nBest estimator: {}'.format(flaml_automl.best_estimator))
print('Best hyperparameters:\n{}'.format(flaml_automl.best_config))
print('Best loss: {}'.format(flaml_automl.best_loss))
print('Training time: {}s'.format(flaml_automl.best_config_train_time))

# Make predictions
y_pred = flaml_automl.predict(test_dataset.drop(['Target'], axis=1))

# Evaluation
evaluate_regression(y_test=y_test, y_pred=y_pred, SEED=SEED)

# Plot the global interpretability of the model
try:
    feature_importance_viz = pd.DataFrame(
        data={'Importance': flaml_automl.model.estimator.feature_importances_},
        index=flaml_automl.model.estimator.feature_names_in_
    )
    feature_importance_viz = feature_importance_viz.sort_values(
        by=['Importance'], ascending=True)
    feature_importance_viz = feature_importance_viz[-50:]
    ax = feature_importance_viz.plot.barh()
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.legend(loc='best')
    ax.grid(True)
    plt.show()
except Exception as error:
    try:
        explainer = Explainer(
            model=flaml_automl.model.estimator,
            masker=maskers.Independent(
                data=train_dataset.drop(['Target'], axis=1),
                max_samples=1000
            )
        )
        shap_values = explainer(test_dataset.drop(['Target'], axis=1))
        plots.beeswarm(shap_values=shap_values, max_display=50)
    except Exception as error:
        print(f'The following error occurred: {error}')


# 3.1.3 H2O
print(f'\n\n3.1.3 H2O')

# Initialisation (start the cluster)
init()

# Instantiate the model
h2o_automl = H2OAutoML(nfolds=FOLDS, seed=SEED)

# Train the model
h2o_automl.train(
    x=list(train_dataset.drop(['Target'], axis=1).columns),
    y='Target',
    training_frame=H2OFrame(train_dataset)
)

# Display the leaderboard
leaderboard = h2o_automl.leaderboard
print(f'\nThe leaderboard:\n{leaderboard.head(rows=leaderboard.nrows)}')

# Display the best model
print(f'\nThe best model:\n{h2o_automl.leader}')

# Display the best model performance
performance = h2o_automl.leader.model_performance(
    test_data=H2OFrame(test_dataset))
print(f'\nPerformance:\n{performance}')

# Make predictions
predictions = h2o_automl.leader.predict(H2OFrame(test_dataset))
print(f'\nPredictions:\n{predictions}')

# Convert H2O frame into Pandas DataFrame
y_pred = predictions.as_data_frame()['predict'].values

# Evaluation
evaluate_regression(y_test=y_test, y_pred=y_pred, SEED=SEED)

# Interpretability of the model
try:
    h2o_automl.leader.varimp_plot()
except Exception as error:
    print(f'The following error occurred: {error}')

try:
    h2o_automl.leader.explain(H2OFrame(test_dataset))
except Exception as error:
    print(f'\nThe following error occurred: {error}')


# 3.1.4 EvalML
print(f'\n\n3.1.4 EvalML')

# Instantiate the model
evalml_automl = AutoMLSearch(
    X_train=X_train,
    y_train=y_train,
    problem_type='regression',
    objective='R2',
    max_iterations=FOLDS,
    random_seed=SEED,
    n_jobs=-1,
    verbose=True
)

# Train the model
evalml_automl.search()

# Display information about the best model
print(f'\nClassification of models:\n{evalml_automl.rankings}')
print(f'\nBest model:\n{evalml_automl.best_pipeline}')

# Make predictions
y_pred = np.array(evalml_automl.best_pipeline.predict(X_test))

# Evaluation
evaluate_regression(y_test=y_test, y_pred=y_pred, SEED=SEED)

# Plot the global interpretability of the model
try:
    readable_explanation(
        pipeline=evalml_automl.best_pipeline, importance_method='feature')
    fig = evalml_automl.best_pipeline.graph_feature_importance()
    fig.write_html('feature_importance_with_energystarscore_siteenergyuse.html')
    fig.show()
except Exception as error:
    print(f'The following error occurred: {error}')


# 3.2 Prediction without the feature ENERGYSTARScore
print(f'\n\n3.2 Prediction without the feature ENERGYSTARScore')

# Remove the feature ENERGYSTARScore
train_dataset = train_dataset.drop(['ENERGYSTARScore'], axis=1)
test_dataset = test_dataset.drop(['ENERGYSTARScore'], axis=1)
X_train = X_train.drop(['ENERGYSTARScore'], axis=1)
X_test = X_test.drop(['ENERGYSTARScore'], axis=1)


# 3.2.1 PyCaret
print(f'\n\n3.2.1 PyCaret')

# Instantiate the setup
s = setup(
    data=train_dataset,
    target='Target',
    index=False,
    train_size=0.8,
    remove_multicollinearity=True,
    fold=FOLDS,
    fold_shuffle=True,
    normalize=True,
    n_jobs=-1,
    session_id=SEED,
    verbose=True
)

# Selection of the best model by cross-validation
best = compare_models(
    fold=FOLDS,
    round=3,
    cross_validation=True,
    n_select=1,
    verbose=True
)
print(f'\nClassification of models:\n{best}')

# Make predictions
pred = predict_model(estimator=best, data=test_dataset)
print(f'\nPredictions:\n{pred}')

# Plot error
try:
    plot_model(best, plot='error')
except Exception as error:
    print(f'The following error occurred: {error}')

# Plot residuals
try:
    plot_model(best, plot='residuals')
except Exception as error:
    print(f'The following error occurred: {error}')

# Plot the global interpretability of the model (Feature importance)
try:
    plot_model(estimator=best, plot='feature')
except Exception as error:
    print(f'The following error occurred: {error}')

# Plot the global interpretability of the model (Summary)
try:
    plot_model(estimator=best, plot='summary')
except Exception as error:
    print(f'The following error occurred: {error}')

# Make predictions
y_pred = pred['prediction_label'].to_numpy()

# Evaluation
evaluate_regression(y_test=y_test, y_pred=y_pred, SEED=SEED)


# 3.2.2 FLAML
print(f'\n\n3.2.2 FLAML')

# Instantiate AutoML instance
flaml_automl = AutoML()
flaml_automl.fit(
    dataframe=train_dataset,
    label='Target',
    metric='mae',
    task='regression',
    n_jobs=-1,
    eval_method='auto',
    n_splits=FOLDS,
    split_type='auto',
    seed=SEED,
    early_stop=True
)

# Display information about the best model
print('\nBest estimator: {}'.format(flaml_automl.best_estimator))
print('Best hyperparameters:\n{}'.format(flaml_automl.best_config))
print('Best loss: {}'.format(flaml_automl.best_loss))
print('Training time: {}s'.format(flaml_automl.best_config_train_time))

# Make predictions
y_pred = flaml_automl.predict(test_dataset.drop(['Target'], axis=1))

# Evaluation
evaluate_regression(y_test=y_test, y_pred=y_pred, SEED=SEED)

# Plot the global interpretability of the model
try:
    feature_importance_viz = pd.DataFrame(
        data={'Importance': flaml_automl.model.estimator.feature_importances_},
        index=flaml_automl.model.estimator.feature_names_in_
    )
    feature_importance_viz = feature_importance_viz.sort_values(
        by=['Importance'], ascending=True)
    feature_importance_viz = feature_importance_viz[-50:]
    ax = feature_importance_viz.plot.barh()
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.legend(loc='best')
    ax.grid(True)
    plt.show()
except Exception as error:
    try:
        explainer = Explainer(
            model=flaml_automl.model.estimator,
            masker=maskers.Independent(
                data=train_dataset.drop(['Target'], axis=1),
                max_samples=1000
            )
        )
        shap_values = explainer(test_dataset.drop(['Target'], axis=1))
        plots.beeswarm(shap_values=shap_values, max_display=50)
    except Exception as error:
        print(f'The following error occurred: {error}')


# 3.2.3 H2O
print(f'\n\n3.2.3 H2O')

# Initialisation (start the cluster)
init()

# Instantiate the model
h2o_automl = H2OAutoML(nfolds=FOLDS, seed=SEED)

# Train the model
h2o_automl.train(
    x=list(train_dataset.drop(['Target'], axis=1).columns),
    y='Target',
    training_frame=H2OFrame(train_dataset)
)

# Display the leaderboard
leaderboard = h2o_automl.leaderboard
print(f'\nThe leaderboard:\n{leaderboard.head(rows=leaderboard.nrows)}')

# Display the best model
print(f'\nThe best model:\n{h2o_automl.leader}')

# Display the performance of the best model
performance = h2o_automl.leader.model_performance(
    test_data=H2OFrame(test_dataset))
print(f'\nPerformance:\n{performance}')

# Make predictions
predictions = h2o_automl.leader.predict(H2OFrame(test_dataset))
print(f'\nPredictions:\n{predictions}')

# Convert H2O frame into Pandas DataFrame
y_pred = predictions.as_data_frame()['predict'].values

# Evaluation
evaluate_regression(y_test=y_test, y_pred=y_pred, SEED=SEED)

# Interpretability of the model
try:
    h2o_automl.leader.varimp_plot()
except Exception as error:
    print(f'The following error occurred: {error}')

try:
    h2o_automl.leader.explain(H2OFrame(test_dataset))
except Exception as error:
    print(f'\nThe following error occurred: {error}')

shutdown()


# 3.2.4 EvalML
print(f'\n\n3.2.4 EvalML')

# Instantiate the model
evalml_automl = AutoMLSearch(
    X_train=X_train,
    y_train=y_train,
    problem_type='regression',
    objective='R2',
    max_iterations=FOLDS,
    random_seed=SEED,
    n_jobs=-1,
    verbose=True
)

# Train the model
evalml_automl.search()

# Display information about the best model
print(f'\nClassification of models:\n{evalml_automl.rankings}')
print(f'\nBest model:\n{evalml_automl.best_pipeline}')

# Make predictions
y_pred = np.array(evalml_automl.best_pipeline.predict(X_test))

# Evaluation
evaluate_regression(y_test=y_test, y_pred=y_pred, SEED=SEED)

# Plot the global interpretability of the model
try:
    readable_explanation(
        pipeline=evalml_automl.best_pipeline, importance_method='feature')
    fig = evalml_automl.best_pipeline.graph_feature_importance()
    fig.write_html(
        'feature_importance_without_energystarscore_siteenergyuse.html')
    fig.show()
except Exception as error:
    print(f'The following error occurred: {error}')
