"""
Visualization utilities for Fashion MNIST analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from . import config


def plot_sample_images(images, labels, class_names=config.CLASS_NAMES,
                      num_samples=25, save_path=None):
    """
    Plot sample images from the dataset.

    Args:
        images (np.ndarray): Image data
        labels (np.ndarray): Labels
        class_names (list): List of class names
        num_samples (int): Number of samples to plot
        save_path (str): Path to save the figure
    """
    plt.figure(figsize=config.VIZ_FIGSIZE_SMALL)
    grid_size = int(np.ceil(np.sqrt(num_samples)))

    for i in range(num_samples):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])

    plt.suptitle('Sample Images from Fashion MNIST', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.VIZ_DPI, bbox_inches='tight')

    plt.show()


def plot_confusion_matrix(true_labels, predicted_labels,
                         class_names=config.CLASS_NAMES, save_path=None):
    """
    Plot confusion matrix.

    Args:
        true_labels (np.ndarray): True labels
        predicted_labels (np.ndarray): Predicted labels
        class_names (list): List of class names
        save_path (str): Path to save the figure

    Returns:
        np.ndarray: Confusion matrix
    """
    cm = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=config.VIZ_FIGSIZE_LARGE)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Number of Predictions'})
    plt.title('Confusion Matrix for Fashion MNIST Classification',
             fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.VIZ_DPI, bbox_inches='tight')

    plt.show()

    return cm


def plot_training_history(history, save_path=None):
    """
    Plot training and validation accuracy over epochs.

    Args:
        history: Training history object
        save_path (str): Path to save the figure
    """
    plt.figure(figsize=config.VIZ_FIGSIZE_MEDIUM)

    epochs_range = range(1, len(history.history['accuracy']) + 1)

    plt.plot(epochs_range, history.history['accuracy'], 'b-',
            linewidth=2, label='Training Accuracy', marker='o')
    plt.plot(epochs_range, history.history['val_accuracy'], 'r-',
            linewidth=2, label='Validation Accuracy', marker='s')

    plt.title('Model Accuracy Over Training Epochs', fontsize=16, pad=20)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs_range)
    plt.ylim([0.7, 1.0])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.VIZ_DPI, bbox_inches='tight')

    plt.show()


def plot_data_volume_curve(data_sizes, train_accuracies, test_accuracies,
                          epochs=6, save_path=None):
    """
    Plot learning curve over different data volumes.

    Args:
        data_sizes (list): List of data sizes
        train_accuracies (list): Training accuracies
        test_accuracies (list): Test accuracies
        epochs (int): Number of epochs used
        save_path (str): Path to save the figure
    """
    plt.figure(figsize=config.VIZ_FIGSIZE_MEDIUM)

    plt.plot(data_sizes, train_accuracies, 'b-', linewidth=2,
            label='Training Accuracy', marker='o', markersize=8)
    plt.plot(data_sizes, test_accuracies, 'r-', linewidth=2,
            label='Test Accuracy', marker='s', markersize=8)

    plt.title(f'Model Accuracy vs. Training Data Size ({epochs} Epochs)',
             fontsize=16, pad=20)
    plt.xlabel('Number of Training Samples', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(data_sizes, [f'{x//1000}k' for x in data_sizes], rotation=45)
    plt.ylim([0.7, 1.0])

    plt.fill_between(data_sizes, train_accuracies, test_accuracies,
                    alpha=0.2, color='gray')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.VIZ_DPI, bbox_inches='tight')

    plt.show()


def plot_prediction_samples(images, true_labels, predictions,
                           class_names=config.CLASS_NAMES,
                           num_samples=15, save_path=None):
    """
    Plot sample predictions with confidence scores.

    Args:
        images (np.ndarray): Test images
        true_labels (np.ndarray): True labels
        predictions (np.ndarray): Prediction probabilities
        class_names (list): List of class names
        num_samples (int): Number of samples to plot
        save_path (str): Path to save the figure
    """
    num_rows = 5
    num_cols = 3
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))

    for i in range(min(num_samples, num_rows * num_cols)):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions, true_labels, images, class_names)

        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions, true_labels)

    plt.suptitle('Sample Predictions (Blue=Correct, Red=Incorrect)',
                fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.VIZ_DPI, bbox_inches='tight')

    plt.show()


def plot_image(i, predictions_array, true_label, img, class_names):
    """Helper function to plot a single image with prediction."""
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'

    plt.xlabel(f"{class_names[predicted_label]} {100*np.max(predictions_array):.0f}% ({class_names[true_label]})",
              color=color)


def plot_value_array(i, predictions_array, true_label):
    """Helper function to plot prediction confidence."""
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
