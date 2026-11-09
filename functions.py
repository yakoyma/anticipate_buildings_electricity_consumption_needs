"""
===============================================================================
This file contains all the functions for the project
===============================================================================
"""
# Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from csv import Sniffer
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             mean_absolute_percentage_error,
                             median_absolute_error,
                             mean_squared_log_error,
                             max_error,
                             explained_variance_score,
                             r2_score,
                             PredictionErrorDisplay)



def load_dataset(file_path, encoding):
    """This function loads a csv file and finds the type of separators by
     sniffing the file.

    Args:
        file_path (str): the csv file path
        encoding (str): encoding to use for reading and writing

    Returns:
        dataset (pd.DataFrame): the loaded Pandas dataset
    """

    with open(file_path, 'r') as csvfile:
        separator = Sniffer().sniff(csvfile.readline()).delimiter
    dataset = pd.read_csv(
        filepath_or_buffer=file_path,
        sep=separator,
        encoding=encoding,
        encoding_errors='ignore',
        on_bad_lines='skip'
    )
    return dataset


def dataset_info_description(dataset, max_rows):
    """This function displays the information and description of a Pandas
    DataFrame.

    Args:
        dataset (pd.DataFrame): the Pandas DataFrame
        max_rows (int): the maximum number of rows in the dataset to be
                        displayed
    """

    # Display dimensions of the dataset
    print('\nDimensions of the dataset: {}'.format(dataset.shape))

    # Display information about the dataset
    print('\nInformation about the dataset:')
    print(dataset.info())

    # Display the description of the dataset
    print('\nDescription of the dataset:')
    print(dataset.describe(include='all'))

    # Display the head and the tail of the dataset
    print('\nDisplay the head and the tail of the dataset: ')
    print(pd.concat([dataset.head(max_rows), dataset.tail(max_rows)]))


def evaluate_regression(y_test, y_pred, SEED):
    """This function evaluates the result of a Regression.

    Args:
        y_test (array-like): the test labels
        y_pred (array-like): the predicted labels
        SEED (int): the random state value
    """

    print('\nMSE: {:.3f}'.format(mean_squared_error(
        y_true=y_test, y_pred=y_pred)))
    print('MAE: {:.3f}'.format(mean_absolute_error(
        y_true=y_test, y_pred=y_pred)))
    print('MAPE: {:.3f}'.format(mean_absolute_percentage_error(
        y_true=y_test, y_pred=y_pred)))
    print('MdAE: {:.3f}'.format(median_absolute_error(
        y_true=y_test, y_pred=y_pred)))
    if np.where(y_test < 0)[0].size == 0 and np.where(y_pred < 0)[0].size == 0:
        print('MSLE: {:.3f}'.format(mean_squared_log_error(
            y_true=y_test, y_pred=y_pred)))
    elif np.where(y_test < 0)[0].size > 0:
        print('Impossible to compute MSLE because the test set contains '
              'negative values.')
    elif np.where(y_pred < 0)[0].size > 0:
        print('Impossible to compute MSLE because forecasts or predictions '
              'contain negative values.')
    print('Maximum residual error: {:.3f}'.format(max_error(
        y_true=y_test, y_pred=y_pred)))
    print('Explained variance score: {:.3f}'.format(explained_variance_score(
        y_true=y_test, y_pred=y_pred)))
    print('R²: {:.3f}'.format(r2_score(
        y_test, y_pred)))

    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    PredictionErrorDisplay.from_predictions(
        y_true=y_test,
        y_pred=y_pred,
        kind='actual_vs_predicted',
        ax=axs[0],
        random_state=SEED
    )
    axs[0].set_title('Actual vs Predicted values')
    axs[0].grid(True)
    PredictionErrorDisplay.from_predictions(
        y_true=y_test,
        y_pred=y_pred,
        kind='residual_vs_predicted',
        ax=axs[1],
        random_state=SEED
    )
    axs[1].set_title('Residuals vs Predicted Values')
    axs[1].grid(True)
    fig.suptitle('Plot the results of predictions')
    plt.show()
