"""
===============================================================================
Regression Project: Anticipate Electricity Consumption Needs of
Buildings - Modelling - Prediction of TotalGHGEmissions
===============================================================================

This file is organised as follows:
1. Load the dataset
2. Feature Engineering
3. Machine Learning
   3.1 Functions
   3.2 Prediction with the feature ENERGYSTARScore
       3.2.1 AutoKeras
       3.2.2 LSTM
   3.3 Prediction without the feature ENERGYSTARScore
       3.3.1 AutoKeras
       3.3.2 LSTM
"""
# Standard libraries
import random
import platform

# Other libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import joblib
import autokeras as ak
import tensorflow as tf
import keras
import keras_tuner


from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import (PowerTransformer, TargetEncoder,
                                   StandardScaler)
from sklearn.impute import KNNImputer
from collinearity import SelectNonCollinear
from joblib import dump
from autokeras import StructuredDataRegressor, CUSTOM_OBJECTS
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras_tuner.tuners import BayesianOptimization
from functions import *


# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('NumPy: {}'.format(np.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))
print('Joblib: {}'.format(joblib.__version__))
print('AutoKeras: {}'.format(ak.__version__))
print('TensorFlow: {}'.format(tf.__version__))
print('Keras: {}'.format(keras.__version__))
print('KerasTuner: {}'.format(keras_tuner.__version__))



# Constants
SEED = 0
MAX_ROWS_DISPLAY = 300
MAX_COLUMNS_DISPLAY = 150
CORR_THRESHOLD = 0.9
FOLDS = 10
BATCH_SIZE = 32
EPOCHS = 1000

# Set the random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Set the maximum number of rows to display by Pandas
pd.set_option('display.max_rows', MAX_ROWS_DISPLAY)
pd.set_option('display.max_columns', MAX_COLUMNS_DISPLAY)



"""
===============================================================================
1. Load the dataset
===============================================================================
"""
print('\n\n\nLoad the dataset')


# Load the dataset
INPUT_CSV = 'datasets/dataset.csv'
dataset = load_dataset(file_path=INPUT_CSV, encoding='utf-8')



"""
===============================================================================
2. Feature Engineering
===============================================================================
"""
print('\n\n\n2. Feature Engineering')


# Feature selection
X = dataset.dropna(subset=['TotalGHGEmissions']).reset_index(drop=True)
y = X['TotalGHGEmissions'].values
X = X.drop(['TotalGHGEmissions', 'SiteEnergyUse(kBtu)'], axis=1)

# Display X dataset information and description
dataset_info_description(dataset=X, max_rows=15)

numeric_features = X.select_dtypes(include=['number']).columns.tolist()
object_features = X.select_dtypes(include=['object']).columns.tolist()
print('\n\nLength of numeric features: {}'.format(len(numeric_features)))
print('Numeric features:\n{}'.format(numeric_features))
print('\nLength of object features: {}'.format(len(object_features)))
print('Object features:\n{}'.format(object_features))


# Split the dataset into train and test sets
train, test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, shuffle=True)


# Target transformation
transformer = PowerTransformer()
y_train_log = transformer.fit_transform(X=y_train.reshape(-1, 1)).ravel()
y_test_log = transformer.transform(X=y_test.reshape(-1, 1)).ravel()

# Transformer persistence
dump(transformer, 'models/transformer/totalghgemissions/transformer.joblib')


# Display the train and test targets
print(f'\n\nActual train target shape: {y_train.shape}')
print(f'Actual test target shape: {y_test.shape}')
print(f'\nTransformed train target shape: {y_train_log.shape}')
print(f'Transformed test target shape: {y_test_log.shape}')


# Encode object features
encoder = TargetEncoder(cv=FOLDS, random_state=SEED)
object_train_encoded = encoder.fit_transform(
    X=train[object_features], y=y_train_log)
object_test_encoded = encoder.transform(X=test[object_features])

# Encoder persistence
dump(encoder, 'models/encoder/totalghgemissions/encoder.joblib')


# Concatenate numerical arrays with object encoded arrays
train_set = np.column_stack(
    [np.array(train[numeric_features]), object_train_encoded])
test_set = np.column_stack(
    [np.array(test[numeric_features]), object_test_encoded])


# Normalisation
scaler = StandardScaler()
train_scaled = scaler.fit_transform(X=train_set)
test_scaled = scaler.transform(X=test_set)

# Scaler persistence
dump(scaler, 'models/scaler/totalghgemissions/scaler.joblib')


# Imputation
imputer = KNNImputer(n_neighbors=1)
train_imput = imputer.fit_transform(X=train_scaled)
test_imput = imputer.transform(X=test_scaled)

# Imputer persistence
dump(imputer, 'models/imputer/totalghgemissions/imputer.joblib')


# Convert arrays into DataFrames
X_train = pd.DataFrame(data=train_imput, columns=train.columns)
X_test = pd.DataFrame(data=test_imput, columns=test.columns)


# Get a set of correlated features based on the correlation threshold
selector = SelectNonCollinear(
    correlation_threshold=CORR_THRESHOLD, scoring=f_regression)
selector.fit(X=np.array(X_train), y=y_train_log)
selected_features = list(np.array(X_train.columns)[selector.get_support()])
correlated_features = list(set(X_train.columns) - set(selected_features))
print('\n\nSelected features:\n{}'.format(selected_features))
print('Length of selected features: {}'.format(len(selected_features)))
print('\nCorrelated features:\n{}'.format(correlated_features))
print('Length of correlated features: {}'.format(len(correlated_features)))

# Scaler persistence
dump(selector, 'models/selector/totalghgemissions/selector.joblib')


# Create the final train and test datasets
keep_feature = 'ENERGYSTARScore'
if keep_feature not in selected_features:
    selected_features.append(keep_feature)
    corr_dataset = X_train[selected_features].corr()
    selected_features = list(
        corr_dataset[(corr_dataset[keep_feature] < CORR_THRESHOLD) |
        (corr_dataset[keep_feature] == 1)].index
    )
    other_correlated_features = list(
        set(corr_dataset.columns) - set(selected_features))
    print('\nfeatures:\n{}'.format(selected_features))
    print('Length of features: {}'.format(len(selected_features)))
    print('Other correlated features:\n{}'.format(other_correlated_features))
    print('\nLength of other correlated features: {}'.format(
        len(other_correlated_features)))

train_dataset = X_train[selected_features]
test_dataset = X_test[selected_features]

# Display the train dataset information and description
dataset_info_description(dataset=train_dataset, max_rows=15)

# Display the test dataset information and description
dataset_info_description(dataset=test_dataset, max_rows=15)



"""
===============================================================================
3. Machine Learning
===============================================================================
"""
print('\n\n\n3. Machine Learning')


# 3.1 Functions
print(f'\n\n3.1 Functions')

def build_lstm_model(hp):
    """This function builds a TensorFlow LSTM (Long Short-Term Memory network)
    model with hyperparameters to be optimised in order to find the best
    architecture.

    Args:
        hp (keras_tuner.HyperParameters): hyperparameters to optimise

    Returns:
        model (tf.keras.Model): the model with the best architecture
    """

    # Instantiate the LSTM model
    model = Sequential()
    model.add(LSTM(
        hp.Int('units', 128, 1024),
        hp.Choice('activation', values=['elu', 'relu', 'tanh']),
        return_sequences=True)
    )
    model.add(LSTM(
        hp.Int('units', 128, 1024),
        hp.Choice('activation', values=['elu', 'relu', 'tanh']),
        return_sequences=True)
    )
    model.add(LSTM(
        hp.Int('units', 128, 1024),
        hp.Choice('activation', values=['elu', 'relu', 'tanh']),
        return_sequences=False)
    )
    model.add(Dense(1))

    # Compile the model
    model.compile(
        loss=hp.Choice(
            'loss', values=['mean_squared_error', 'mean_absolute_error']),
        optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
        metrics=['mean_squared_error']
    )
    return model


def display_model_history(model_history, metric: str):
    """This function plots training loss and training scores, training and
    validation scores, and training and validation losses based on the history
    of a TensorFlow model.

    Args:
        model_history (tf.keras.callbacks.History): the model history
        metric (str): the name of evaluation metric
    """

    loss = next(
        key for key in model_history.history
        if not key.startswith('val_') and key not in
        ['mean_squared_error', 'mean_absolute_error']
    )
    score = model_history.history[metric]
    val_score = model_history.history['val_' + metric]
    loss_score = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    # Plot the training loss and training score
    plt.figure(figsize=(12, 6))
    plt.plot(loss_score, color='tab:blue')
    plt.plot(score, color='tab:red')
    plt.title('History of the model training')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend([loss, metric], loc='best')
    plt.grid(True)
    plt.show()

    # Plot the training and validation scores
    plt.figure(figsize=(12, 6))
    plt.plot(score, label='Training ' + metric, color='tab:blue')
    plt.plot(val_score, label='Validation ' + metric, color='tab:red')
    plt.xlabel(xlabel='Epoch')
    plt.ylabel(ylabel=metric)
    plt.title(label=f'Training and Validation {metric.capitalize()}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    # Plot the training and validation losses
    plt.figure(figsize=(12, 6))
    plt.plot(loss_score, label='Training ' + loss, color='tab:blue')
    plt.plot(val_loss, label='Validation ' + loss, color='tab:red')
    plt.xlabel(xlabel='Epoch')
    plt.ylabel(ylabel=loss)
    plt.title(label=f'Training and Validation {loss.capitalize()}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


# 3.2 Prediction with the feature ENERGYSTARScore
print(f'\n\n3.2 Prediction with the feature ENERGYSTARScore')

# Convert the train and test datasets into arrays
X_train = np.array(train_dataset)
X_test = np.array(test_dataset)

# Display the train and test sets
print(f'\nTrain shape: {X_train.shape}')
print(f'Test shape: {X_test.shape}')


# 3.2.1 AutoKeras
print(f'\n\n3.2.1 AutoKeras')

# Instantiate the model
model = StructuredDataRegressor(
    tuner='hyperband',
    overwrite=True,
    seed=SEED
)

# Train the model
model.fit(
    x=X_train,
    y=y_train_log,
    epochs=EPOCHS
)

# Make predictions
y_pred_log = model.predict(x=X_test).flatten()

# Evaluate the model
scores = model.evaluate(x=X_test, y=y_test_log)
for metric, score in zip(['Loss (MSE)', 'Score (MSE)'], scores):
    print(f'{metric}: {score:.3f}')

# Model persistence
model_path = 'models/autokeras/totalghgemissions/energystarscore/model.keras'
model = model.export_model()
model.save(filepath=model_path)
print(f'\nSummary of the model with optimised hyperparameters:'
      f'\n{model.summary()}')

# Load the pre-trained model
model = load_model(filepath=model_path, custom_objects=CUSTOM_OBJECTS)

# Evaluation for transformed targets
print('\nPredicted transformed target shape: ', y_pred_log.shape)
evaluate_regression(y_test=y_test_log, y_pred=y_pred_log, SEED=SEED)

# Reverse target transformation
y_pred = transformer.inverse_transform(y_pred_log.reshape(-1, 1)).ravel()

# Evaluation for actual targets
print('\nPredicted target shape: ', y_pred.shape)
evaluate_regression(y_test=y_test, y_pred=y_pred, SEED=SEED)


# 3.2.2 LSTM
print(f'\n\n3.2.2 LSTM')

# Reshape features three-dimensional for the LSTM model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Display the train and test sets
print(f'\nReshaped X_train shape: {X_train.shape}')
print(f'Reshaped X_test shape: {X_test.shape}')

# Instantiate the model
tuner = BayesianOptimization(
    hypermodel=build_lstm_model,
    objective='val_loss',
    seed=SEED
)

# Create the callback that stops training the model
# when the loss stops improving
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=15
)

# Search the best architecture
tuner.search(
    x=X_train,
    y=y_train_log,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[early_stopping_callback]
)

# Build the model with the best architecture
model = tuner.hypermodel.build(tuner.get_best_hyperparameters()[0])

# Train the model
model_history = model.fit(
    X_train,
    y_train_log,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2
)

# Plot the model history
display_model_history(
    model_history=model_history, metric='mean_squared_error')

# Make predictions
y_pred_log = model.predict(x=X_test).flatten()

# Evaluate the model
scores = model.evaluate(x=X_test, y=y_test_log)
for metric, score in zip(['Loss', 'Score (MSE)'], scores):
    print(f'{metric}: {score:.3f}')

# Model persistence
model_path = 'models/keras/totalghgemissions/energystarscore/model.keras'
model.save(filepath=model_path)
print(f'\nSummary of the model with optimised hyperparameters:'
      f'\n{model.summary()}')

# Load the pre-trained model
model = load_model(filepath=model_path)

# Evaluation for transformed targets
print('\nPredicted transformed target shape: ', y_pred_log.shape)
evaluate_regression(y_test=y_test_log, y_pred=y_pred_log, SEED=SEED)

# Reverse target transformation
y_pred = transformer.inverse_transform(y_pred_log.reshape(-1, 1)).ravel()

# Evaluation for actual targets
print('\nPredicted actual target shape: ', y_pred.shape)
evaluate_regression(y_test=y_test, y_pred=y_pred, SEED=SEED)


# 3.3 Prediction without the feature ENERGYSTARScore
print(f'\n\n3.3 Prediction without the feature ENERGYSTARScore')

# Convert the train and test datasets into arrays
X_train = np.array(train_dataset.drop(['ENERGYSTARScore'], axis=1))
X_test = np.array(test_dataset.drop(['ENERGYSTARScore'], axis=1))

# Display the train and test sets
print(f'\nTrain shape: {X_train.shape}')
print(f'Test shape: {X_test.shape}')


# 3.3.1 AutoKeras
print(f'\n\n3.3.1 AutoKeras')

# Instantiate the model
model = StructuredDataRegressor(
    tuner='hyperband',
    overwrite=True,
    seed=SEED
)

# Train the model
model.fit(
    x=X_train,
    y=y_train_log,
    epochs=EPOCHS
)

# Make predictions
y_pred_log = model.predict(x=X_test).flatten()

# Evaluate the model
scores = model.evaluate(x=X_test, y=y_test_log)
for metric, score in zip(['Loss (MSE)', 'Score (MSE)'], scores):
    print(f'{metric}: {score:.3f}')

# Model persistence
model_path = 'models/autokeras/totalghgemissions/model.keras'
model = model.export_model()
model.save(filepath=model_path)
print(f'\nSummary of the model with optimised hyperparameters:'
      f'\n{model.summary()}')

# Load the pre-trained model
model = load_model(filepath=model_path, custom_objects=CUSTOM_OBJECTS)

# Evaluation for transformed targets
print('\nPredicted transformed target shape: ', y_pred_log.shape)
evaluate_regression(y_test=y_test_log, y_pred=y_pred_log, SEED=SEED)

# Reverse target transformation
y_pred = transformer.inverse_transform(y_pred_log.reshape(-1, 1)).ravel()

# Evaluation for actual targets
print('\nPredicted target shape: ', y_pred.shape)
evaluate_regression(y_test=y_test, y_pred=y_pred, SEED=SEED)


# 3.3.2 LSTM
print(f'\n\n3.3.2 LSTM')

# Reshape features three-dimensional for the LSTM model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Display the train and test sets
print(f'\nReshaped X_train shape: {X_train.shape}')
print(f'Reshaped X_test shape: {X_test.shape}')

# Instantiate the model
tuner = BayesianOptimization(
    hypermodel=build_lstm_model,
    objective='val_loss',
    seed=SEED
)

# Create the callback that stops training the model
# when the loss stops improving
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=15
)

# Search the best architecture
tuner.search(
    x=X_train,
    y=y_train_log,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[early_stopping_callback]
)

# Build the model with the best architecture
model = tuner.hypermodel.build(tuner.get_best_hyperparameters()[0])

# Train the model
model_history = model.fit(
    X_train,
    y_train_log,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2
)

# Plot the model history
display_model_history(
    model_history=model_history, metric='mean_squared_error')

# Make predictions
y_pred = model.predict(x=X_test).flatten()

# Evaluate the model
scores = model.evaluate(x=X_test, y=y_test)
for metric, score in zip(['Loss', 'Score (MSE)'], scores):
    print(f'{metric}: {score:.3f}')

# Model persistence
model_path = 'models/keras/totalghgemissions/model.keras'
model.save(filepath=model_path)
print(f'\nSummary of the model with optimised hyperparameters:'
      f'\n{model.summary()}')

# Load the pre-trained model
model = load_model(filepath=model_path)

# Evaluation for transformed targets
print('\nPredicted transformed target shape: ', y_pred_log.shape)
evaluate_regression(y_test=y_test_log, y_pred=y_pred_log, SEED=SEED)

# Reverse target transformation
y_pred = transformer.inverse_transform(y_pred_log.reshape(-1, 1)).ravel()

# Evaluation for actual targets
print('\nPredicted actual target shape: ', y_pred.shape)
evaluate_regression(y_test=y_test, y_pred=y_pred, SEED=SEED)
