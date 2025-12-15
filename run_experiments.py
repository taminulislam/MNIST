"""
Run all experiments from the homework assignment

This script runs:
1. Task 1: Baseline model training (10 epochs)
2. Task 2: Confusion matrix analysis
3. Task 3: Learning curve over epochs
4. Task 4: Learning curve over data volume
"""

import os
import numpy as np
import tensorflow as tf
from src import config
from src.data_loader import load_fashion_mnist, create_data_subset
from src.models import create_baseline_model
from src.visualization import (
    plot_sample_images, plot_confusion_matrix,
    plot_training_history, plot_data_volume_curve
)
from src.analysis import (
    analyze_confusion_matrix, print_training_summary,
    evaluate_model_performance, create_data_volume_summary
)


def task1_baseline_model():
    """Task 1: Implement and train baseline model."""
    print("\n" + "="*70)
    print("TASK 1: Baseline Model Training")
    print("="*70)

    # Load data
    train_images, train_labels, test_images, test_labels = load_fashion_mnist()

    # Visualize samples
    plot_sample_images(
        train_images, train_labels,
        save_path=os.path.join(config.OUTPUT_DIR, 'task1_samples.png')
    )

    # Create and train model
    model = create_baseline_model()
    model.summary()

    history = model.fit(
        train_images, train_labels,
        epochs=config.EPOCHS_BASELINE,
        validation_data=(test_images, test_labels),
        verbose=1
    )

    # Evaluate
    print_training_summary(history)
    evaluate_model_performance(model, test_images, test_labels)

    return model, history, test_images, test_labels


def task2_confusion_matrix(model, test_images, test_labels):
    """Task 2: Confusion matrix analysis."""
    print("\n" + "="*70)
    print("TASK 2: Confusion Matrix Analysis")
    print("="*70)

    # Make predictions
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)

    # Plot and analyze confusion matrix
    cm = plot_confusion_matrix(
        test_labels, predicted_labels,
        save_path=os.path.join(config.OUTPUT_DIR, 'task2_confusion_matrix.png')
    )

    analyze_confusion_matrix(cm)

    return predictions, predicted_labels


def task3_learning_curve_epochs(history):
    """Task 3: Learning curve over epochs."""
    print("\n" + "="*70)
    print("TASK 3: Learning Curve Over Epochs")
    print("="*70)

    plot_training_history(
        history,
        save_path=os.path.join(config.OUTPUT_DIR, 'task3_learning_curve.png')
    )


def task4_data_volume_experiment():
    """Task 4: Learning curve over data volume."""
    print("\n" + "="*70)
    print("TASK 4: Learning Curve Over Data Volume")
    print("="*70)

    # Load data
    train_images, train_labels, test_images, test_labels = load_fashion_mnist()

    train_accuracies = []
    test_accuracies = []

    # Set random seed
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)

    print(f"\nTraining {len(config.DATA_SIZES)} models...")

    for size in config.DATA_SIZES:
        print(f"\n  Training with {size:,} samples...")

        # Create subset
        train_subset, labels_subset = create_data_subset(
            train_images, train_labels, size, config.RANDOM_SEED
        )

        # Create and train model
        model = create_baseline_model()
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            train_subset, labels_subset,
            epochs=config.EPOCHS_DATA_VOLUME,
            validation_data=(test_images, test_labels),
            verbose=0
        )

        train_acc = history.history['accuracy'][-1]
        test_acc = history.history['val_accuracy'][-1]

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"    Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    # Plot results
    plot_data_volume_curve(
        config.DATA_SIZES, train_accuracies, test_accuracies,
        epochs=config.EPOCHS_DATA_VOLUME,
        save_path=os.path.join(config.OUTPUT_DIR, 'task4_data_volume.png')
    )

    # Print summary
    create_data_volume_summary(config.DATA_SIZES, train_accuracies, test_accuracies)


def main():
    """Run all experiments."""
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("="*70)
    print("CS535 HW4: Fashion MNIST Classification - All Tasks")
    print("="*70)

    # Run all tasks
    model, history, test_images, test_labels = task1_baseline_model()
    predictions, predicted_labels = task2_confusion_matrix(model, test_images, test_labels)
    task3_learning_curve_epochs(history)
    task4_data_volume_experiment()

    print("\n" + "="*70)
    print("All Experiments Complete!")
    print(f"Results saved to: {config.OUTPUT_DIR}/")
    print("="*70)


if __name__ == "__main__":
    main()
