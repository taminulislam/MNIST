"""
Analysis utilities for model evaluation and confusion matrix
"""

import numpy as np
import pandas as pd
from . import config


def analyze_confusion_matrix(cm, class_names=config.CLASS_NAMES):
    """
    Analyze confusion matrix and print detailed insights.

    Args:
        cm (np.ndarray): Confusion matrix
        class_names (list): List of class names

    Returns:
        dict: Analysis results
    """
    print("\n" + "="*70)
    print("Confusion Matrix Analysis")
    print("="*70)

    # Calculate per-class accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)

    print("\nPer-Class Accuracy:")
    print("-"*50)
    for name, acc in zip(class_names, class_accuracy):
        print(f"{name:15s}: {acc:.4f} ({acc*100:.2f}%)")

    # Find most common confusions
    print("\nMost Common Misclassifications:")
    print("-"*50)
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)

    confusion_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm_no_diag[i, j] > 0:
                confusion_pairs.append((cm_no_diag[i, j], i, j))

    confusion_pairs.sort(reverse=True)
    for count, true_idx, pred_idx in confusion_pairs[:10]:
        print(f"{class_names[true_idx]:15s} → {class_names[pred_idx]:15s}: {count} times")

    return {
        'class_accuracy': class_accuracy,
        'confusion_pairs': confusion_pairs,
        'overall_accuracy': cm.diagonal().sum() / cm.sum()
    }


def print_training_summary(history):
    """
    Print summary statistics from training history.

    Args:
        history: Training history object
    """
    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)

    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = np.argmax(history.history['val_accuracy']) + 1

    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"Overfitting Gap: {final_train_acc - final_val_acc:.4f}")


def create_data_volume_summary(data_sizes, train_accuracies, test_accuracies):
    """
    Create summary table for data volume experiment.

    Args:
        data_sizes (list): List of data sizes
        train_accuracies (list): Training accuracies
        test_accuracies (list): Test accuracies

    Returns:
        pd.DataFrame: Results table
    """
    results_df = pd.DataFrame({
        'Training Samples': data_sizes,
        'Training Accuracy': [f'{acc:.4f}' for acc in train_accuracies],
        'Test Accuracy': [f'{acc:.4f}' for acc in test_accuracies],
        'Overfitting Gap': [f'{train_accuracies[i] - test_accuracies[i]:.4f}'
                           for i in range(len(data_sizes))]
    })

    print("\n" + "="*70)
    print("Data Volume Analysis Results")
    print("="*70)
    print(results_df.to_string(index=False))

    # Additional analysis
    print("\n" + "-"*70)
    print(f"Accuracy with {data_sizes[0]:,} samples:  Train={train_accuracies[0]:.4f}, Test={test_accuracies[0]:.4f}")
    print(f"Accuracy with {data_sizes[-1]:,} samples: Train={train_accuracies[-1]:.4f}, Test={test_accuracies[-1]:.4f}")
    print(f"\nImprovement in test accuracy: {test_accuracies[-1] - test_accuracies[0]:.4f} ({(test_accuracies[-1] - test_accuracies[0])*100:.2f}%)")

    # Calculate diminishing returns
    accuracy_gains = np.diff(test_accuracies)
    half_point = len(accuracy_gains) // 2
    print(f"\nAverage accuracy gain per 5k samples (first half): {np.mean(accuracy_gains[:half_point]):.4f}")
    print(f"Average accuracy gain per 5k samples (second half): {np.mean(accuracy_gains[half_point:]):.4f}")

    return results_df


def evaluate_model_performance(model, test_images, test_labels):
    """
    Evaluate model performance on test set.

    Args:
        model: Trained model
        test_images (np.ndarray): Test images
        test_labels (np.ndarray): Test labels

    Returns:
        tuple: (test_loss, test_accuracy)
    """
    print("\n" + "="*70)
    print("Model Evaluation")
    print("="*70)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    return test_loss, test_acc
