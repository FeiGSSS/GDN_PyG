from typing import List

import numpy as np
from scipy.stats import iqr, rankdata
from typing import List

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_and_smooth_error_scores(predictions: list or np.ndarray,
                                      ground_truth: list or np.ndarray,
                                      smoothing_window: int = 3,
                                      epsilon:float = 1e-2) -> np.ndarray:
    """
    Calculate and smooth the error scores between test predictions and ground truths.

    Args:
        predictions (list or np.ndarray): The predicted values on the test set.
        ground_truth (list or np.ndarray): The actual ground truth values of the test set.
        smoothing_window (int): The number of elements to consider for smoothing the error scores.
        epsilon (float): A small constant added for numerical stability.

    Returns:
        numpy.ndarray: Smoothed error scores.
    """
    test_delta = np.abs(np.array(predictions) - np.array(ground_truth))
    err_median = np.median(test_delta)
    err_iqr = iqr(test_delta)
    normalized_err_scores = (test_delta - err_median) / (np.abs(err_iqr) + epsilon)
    # smoothe the error scores by a moving average
    smoothed_err_scores = np.zeros_like(normalized_err_scores)
    for idx in range(smoothing_window, len(normalized_err_scores)):
        smoothed_err_scores[idx] = np.mean(normalized_err_scores[idx - smoothing_window: idx])
    return smoothed_err_scores

def calculate_nodewise_error_scores(predictions:np.ndarray,
                                    ground_truth:np.ndarray,
                                    smoothing_window:int=3,
                                    epsilon:float=1e-2) -> np.ndarray:
    # predictions: [total_time_len, num_nodes]
    # ground_truth: [total_time_len, num_nodes]
    # return: [num_nodes, total_time_len - smoothing_window + 1]
    nodewise_error_scores = []
    number_nodes = predictions.shape[1]
    for i in range(number_nodes):
        pred = predictions[:, i]
        gt = ground_truth[:, i]
        scores = calculate_and_smooth_error_scores(pred, gt,
                                                   smoothing_window,
                                                   epsilon)
        nodewise_error_scores.append(scores)
    
    # [num_nodes, total_time_len - smoothing_window + 1]
    return np.stack(nodewise_error_scores, axis=0) 
    

def test_performence(test_result:List[np.ndarray],
                     val_result:List[np.ndarray],
                     smoothing_window:int=3,
                     epsilon:float=1e-2) -> tuple:
    """ Get the precision, recall and f1 score of the testset based on the validation set.

    Args:
        test_result (List): list of [test_predictions, test_ground_truth, test_anomaly_label]
        val_result (List): list of [val_predictions, val_ground_truth, val_anomaly_label]
    Returns:
        tuple: (precision, recall, f1)
    """
    
    test_predictions, test_ground_truth, test_anomaly_label = test_result
    val_predictions, val_ground_truth, _ = val_result
    # _predictions: [total_time_len, num_nodes]
    # _ground_truth: [total_time_len, num_nodes]
    # _anomaly_label: [total_time_len]
    val_scores = calculate_nodewise_error_scores(val_predictions,
                                                 val_ground_truth,
                                                 smoothing_window,
                                                 epsilon)
    # test_scores: [num_nodes, total_time_len - smoothing_window + 1]
    test_scores = calculate_nodewise_error_scores(test_predictions,
                                                  test_ground_truth,
                                                  smoothing_window,
                                                  epsilon)
    all_results = {}
    
    precision, recall, f1 = test_perf_based_on_best(test_scores, test_anomaly_label)
    all_results['best'] = {'precision': precision, 'recall': recall, 'f1': f1}
    print("Test (best) Precision: {:.2f} Recall: {:.2f} F1: {:.4f}".format(precision*100, recall*100, f1))
    
    precision, recall, f1 = test_perf_based_on_val(val_scores, test_scores, test_anomaly_label)
    all_results['val'] = {'precision': precision, 'recall': recall, 'f1': f1}
    print("Test (val) Precision: {:.2f} Recall: {:.2f} F1: {:.4f}".format(precision*100, recall*100, f1))
    
    roc_auc, prc_auc = test_roc_prc_perf(test_scores, test_anomaly_label)
    all_results['roc_prc'] = {'roc_auc': roc_auc, 'prc_auc': prc_auc}
    print("Test ROC : {:.4f} PRC: {:.4f}".format(roc_auc, prc_auc))
    
    return all_results
    
    
    
def test_roc_prc_perf(test_scores, anomaly_labels):
    # test_scores: [num_nodes, total_time_len]
    # anomaly_labels: [total_time_len]
    test_scores = np.max(test_scores, axis=0) # [time_len]
    fpr, tpr, _ = roc_curve(anomaly_labels, test_scores)
    roc_auc = auc(fpr, tpr)
    prc_auc = average_precision_score(anomaly_labels, test_scores)
    return roc_auc, prc_auc


def test_perf_based_on_val(val_scores, test_scores, test_anomaly_label):
    # val_scores: [num_nodes, total_time_len]
    # test_scores: [num_nodes, total_time_len]
    # test_anomaly_label: [total_time_len]
    threshold = np.max(val_scores)
    test_scores_peaks = np.max(test_scores, axis=0)
    predicted_label = (test_scores_peaks > threshold).astype(int)
    test_anomaly_label = test_anomaly_label.astype(int)
    
    assert predicted_label.shape == test_anomaly_label.shape
    
    precision = precision_score(test_anomaly_label, predicted_label)
    recall = recall_score(test_anomaly_label, predicted_label)
    f1 = f1_score(test_anomaly_label, predicted_label)
    
    return (precision, recall, f1)

def test_perf_based_on_best(test_scores, anomaly_labels, threshold_steps= 400) -> tuple:
    # find the best threshold based on the f1 score of the test set
    test_scores = np.max(test_scores, axis=0)
    min_score = np.min(test_scores)
    max_score = np.max(test_scores)
    best_f1 = 0
    best_threshold = 0

    for step in range(threshold_steps):
        threshold = min_score + (max_score - min_score) * step / threshold_steps
        predicted_labels = (test_scores > threshold).astype(int)
        f1 = f1_score(anomaly_labels, predicted_labels)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    final_predicted_labels = (test_scores > best_threshold).astype(int)
    precision = precision_score(anomaly_labels, final_predicted_labels)
    recall = recall_score(anomaly_labels, final_predicted_labels)
    f1 = f1_score(anomaly_labels, final_predicted_labels)

    return (precision, recall, f1)