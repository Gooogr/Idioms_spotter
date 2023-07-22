"""
Helper functions for classification metric prints
"""

from typing import List

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def print_classification_report(
    y_true: List, y_pred: List, labels: List, normalize: str = "true"
):
    """
    Print evaluation report based on confusion matrix for a classification task.

    Args:
    - y_true : array-like of shape (n_samples,). Ground truth (correct) target values.
    - y_pred : array-like of shape (n_samples,). Estimated targets as returned by a classifier.
    - labels : array-like of shape (n_classes), default=None. List of labels to index the matrix.
        This may be used to reorder or select a subset of labels. If ``None`` is given, those that
        appear at least once in ``y_true`` or ``y_pred`` are used in sorted order.
    - normalize : {'true', 'pred', 'all'}, default=None. Normalizes confusion matrix over
        the true (rows), predicted (columns) conditions or all the population.
        If None, confusion matrix will not be normalized.

    Returns:
    - None: Function print result report.
    """
    print("Confusion Matrix:")

    # Print the column headers
    print(f"{'':<12} |", end="")
    for label in labels:
        print(f" {label:<12}", end="")
    print()
    print("-" * (13 + len(labels) * 12))

    conf_matrix = confusion_matrix(y_true, y_pred, normalize=normalize)

    # Print the rows and data
    for i, label in enumerate(labels):
        print(f"{label:<12} |", end="")
        for j in range(len(labels)):
            print(f" {conf_matrix[i, j]:<12.3f}", end="")
        print()
    # Calculate precision, recall, and F1-score for each class and get micro and macro averages
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    micro_avg = precision_recall_fscore_support(y_true, y_pred, average="micro")
    macro_avg = precision_recall_fscore_support(y_true, y_pred, average="macro")

    print("\nClass-wise Metrics:")
    for i, label in enumerate(labels):
        print(
            f"{label:<10} Precision: {precision[i]:.3f} | Recall: {recall[i]:.3f} | F1-score: {fscore[i]:.3f}"  # pylint: disable=line-too-long
        )

    print("\nMicro-average Metrics:")
    print(
        f"Precision: {micro_avg[0]:.3f} | Recall: {micro_avg[1]:.3f} | F1-score: {micro_avg[2]:.3f}"
    )

    print("\nMacro-average Metrics:")
    print(
        f"Precision: {macro_avg[0]:.3f} | Recall: {macro_avg[1]:.3f} | F1-score: {macro_avg[2]:.3f}"
    )
    print()
