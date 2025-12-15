"""
Model definitions for Fashion MNIST classification
"""

from tensorflow import keras
from . import config


def create_baseline_model(input_shape=config.INPUT_SHAPE,
                          hidden_units=config.HIDDEN_UNITS,
                          num_classes=config.NUM_CLASSES):
    """
    Create a baseline neural network model for Fashion MNIST classification.

    Args:
        input_shape (tuple): Shape of input images (height, width)
        hidden_units (int): Number of units in hidden layer
        num_classes (int): Number of output classes

    Returns:
        keras.Sequential: Compiled model
    """
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(hidden_units, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_improved_model(input_shape=config.INPUT_SHAPE,
                         num_classes=config.NUM_CLASSES):
    """
    Create an improved model with dropout and batch normalization.

    Args:
        input_shape (tuple): Shape of input images
        num_classes (int): Number of output classes

    Returns:
        keras.Sequential: Compiled model
    """
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_cnn_model(input_shape=config.INPUT_SHAPE,
                    num_classes=config.NUM_CLASSES):
    """
    Create a CNN model for improved performance.

    Args:
        input_shape (tuple): Shape of input images
        num_classes (int): Number of output classes

    Returns:
        keras.Sequential: Compiled model
    """
    model = keras.Sequential([
        keras.layers.Reshape((*input_shape, 1), input_shape=input_shape),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
