import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    true_positives = np.sum((y_pred == y_true) & (y_true == 0))
    true_negatives = np.sum((y_pred == y_true) & (y_true == 1))
    false_negatives = np.sum((y_pred != y_true) & (y_true == 0))
    false_positives = np.sum((y_pred != y_true) & (y_true == 1))
    try: 
        precision = true_positives/(true_positives + false_positives)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = true_positives/(true_positives + false_negatives)
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = 2*precision*recall/(precision + recall)
    except ZeroDivisionError:
        f1 = 0
    accuracy = (true_positives + true_negatives)/len(y_pred)
    return precision, recall, f1, accuracy
    
    
def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    accuracy = np.sum(y_pred == y_true)/len(y_pred)
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    r2 = 1 - ((y_pred - y_true)**2).sum()/((y_true - np.mean(y_true))**2).sum()
    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mse = (((y_pred - y_true)**2).sum())/len(y_true)
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """
    
    mae = (abs(y_true - y_pred).sum())/len(y_true)
    return mae