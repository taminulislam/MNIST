"""
Data loading and preprocessing utilities
"""

import numpy as np
from tensorflow import keras


def load_fashion_mnist():
    """
    Load and preprocess the Fashion MNIST dataset.

    Returns:
        tuple: (train_images, train_labels, test_images, test_labels)
    """
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Normalize pixel values to [0, 1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return train_images, train_labels, test_images, test_labels


def create_data_subset(images, labels, size, seed=42):
    """
    Create a random subset of the data.

    Args:
        images (np.ndarray): Image data
        labels (np.ndarray): Labels
        size (int): Number of samples to select
        seed (int): Random seed

    Returns:
        tuple: (subset_images, subset_labels)
    """
    np.random.seed(seed)
    indices = np.random.choice(len(images), size, replace=False)
    return images[indices], labels[indices]


def get_dataset_info(train_images, train_labels, test_images, test_labels):
    """
    Print dataset information.

    Args:
        train_images: Training images
        train_labels: Training labels
        test_images: Test images
        test_labels: Test labels
    """
    print("="*70)
    print("Dataset Information")
    print("="*70)
    print(f"Training images shape: {train_images.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Test images shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    print(f"Pixel value range: [{train_images.min():.2f}, {train_images.max():.2f}]")
    print(f"Number of classes: {len(np.unique(train_labels))}")
