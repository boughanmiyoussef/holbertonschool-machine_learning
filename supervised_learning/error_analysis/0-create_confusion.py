#!/usr/bin/env python3
import numpy as np

def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix from one-hot label and prediction arrays.

    Args:
        labels (np.ndarray): One-hot array of shape (m, classes) with true labels.
        logits (np.ndarray): One-hot array of shape (m, classes) with predicted labels.

    Returns:
        np.ndarray: Confusion matrix of shape (classes, classes)
    """
    true_classes = np.argmax(labels, axis=1)     # shape (m,)
    pred_classes = np.argmax(logits, axis=1)     # shape (m,)

    num_classes = labels.shape[1]
    confusion = np.zeros((num_classes, num_classes))

    for true, pred in zip(true_classes, pred_classes):
        confusion[true][pred] += 1

    return confusion
