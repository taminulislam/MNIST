"""
CS535 Advanced Machine Learning: Homework #4
MNIST Fashion Classification

Student Name: Taminul Islam (856569517)

This script implements a deep learning solution for Fashion MNIST classification
including baseline model training, confusion matrix analysis, and learning curve experiments.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")


class FashionMNISTClassifier:
    """A classifier for Fashion MNIST dataset with various analysis capabilities."""

    def __init__(self):
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.model = None
        self.history = None
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None

    def load_data(self):
        """Load and preprocess the Fashion MNIST dataset."""
        print("\n" + "="*70)
        print("Loading Fashion MNIST Dataset")
        print("="*70)

        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        print(f"Training images shape: {train_images.shape}")
        print(f"Training labels shape: {train_labels.shape}")
        print(f"Test images shape: {test_images.shape}")
        print(f"Test labels shape: {test_labels.shape}")
        print(f"Number of classes: {len(self.class_names)}")

        # Normalize pixel values to be between 0 and 1
        self.train_images = train_images / 255.0
        self.test_images = test_images / 255.0
        self.train_labels = train_labels
        self.test_labels = test_labels

        print(f"\nAfter normalization:")
        print(f"Min pixel value: {self.train_images.min()}")
        print(f"Max pixel value: {self.train_images.max()}")

    def visualize_samples(self, num_samples=25):
        """Visualize sample images from the training set."""
        plt.figure(figsize=(10, 10))
        for i in range(num_samples):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.train_images[i], cmap=plt.cm.binary)
            plt.xlabel(self.class_names[self.train_labels[i]])
        plt.suptitle('Sample Images from Fashion MNIST Training Set', fontsize=16)
        plt.tight_layout()
        plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
        plt.show()

    def build_model(self):
        """Build the neural network model."""
        print("\n" + "="*70)
        print("Building Model")
        print("="*70)

        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

        self.model.summary()

    def train_model(self, epochs=10):
        """Train the model."""
        print("\n" + "="*70)
        print(f"Training Model for {epochs} Epochs")
        print("="*70)

        self.history = self.model.fit(
            self.train_images,
            self.train_labels,
            epochs=epochs,
            validation_data=(self.test_images, self.test_labels),
            verbose=1
        )

    def evaluate_model(self):
        """Evaluate the model on test data."""
        print("\n" + "="*70)
        print("Evaluating Model")
        print("="*70)

        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels, verbose=2)
        print(f"\nTest accuracy: {test_acc:.4f}")
        print(f"Test loss: {test_loss:.4f}")

        return test_loss, test_acc

    def make_predictions(self):
        """Make predictions on the test set."""
        predictions = self.model.predict(self.test_images)
        predicted_labels = np.argmax(predictions, axis=1)

        print(f"\nPredictions shape: {predictions.shape}")
        print(f"Predicted labels shape: {predicted_labels.shape}")

        return predictions, predicted_labels

    def plot_confusion_matrix(self, predicted_labels):
        """Generate and plot confusion matrix."""
        print("\n" + "="*70)
        print("Generating Confusion Matrix")
        print("="*70)

        cm = confusion_matrix(self.test_labels, predicted_labels)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Number of Predictions'})
        plt.title('Confusion Matrix for Fashion MNIST Classification', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()

        return cm

    def analyze_confusion_matrix(self, cm):
        """Analyze confusion matrix and print insights."""
        print("\n" + "="*70)
        print("Confusion Matrix Analysis")
        print("="*70)

        # Calculate per-class accuracy
        class_accuracy = cm.diagonal() / cm.sum(axis=1)

        print("\nPer-Class Accuracy:")
        print("-"*50)
        for i, (name, acc) in enumerate(zip(self.class_names, class_accuracy)):
            print(f"{name:15s}: {acc:.4f} ({acc*100:.2f}%)")

        # Find most common confusions
        print("\nMost Common Misclassifications:")
        print("-"*50)
        cm_no_diag = cm.copy()
        np.fill_diagonal(cm_no_diag, 0)

        confusion_pairs = []
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and cm_no_diag[i, j] > 0:
                    confusion_pairs.append((cm_no_diag[i, j], i, j))

        confusion_pairs.sort(reverse=True)
        for count, true_idx, pred_idx in confusion_pairs[:10]:
            print(f"{self.class_names[true_idx]:15s} → {self.class_names[pred_idx]:15s}: {count} times")

    def plot_learning_curve_epochs(self):
        """Plot learning curve over training epochs."""
        print("\n" + "="*70)
        print("Plotting Learning Curve (Over Epochs)")
        print("="*70)

        plt.figure(figsize=(12, 6))

        epochs_range = range(1, len(self.history.history['accuracy']) + 1)
        plt.plot(epochs_range, self.history.history['accuracy'], 'b-',
                linewidth=2, label='Training Accuracy', marker='o')
        plt.plot(epochs_range, self.history.history['val_accuracy'], 'r-',
                linewidth=2, label='Test Accuracy', marker='s')

        plt.title('Model Accuracy Over Training Epochs', fontsize=16, pad=20)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(epochs_range)
        plt.ylim([0.7, 1.0])

        final_train_acc = self.history.history['accuracy'][-1]
        final_test_acc = self.history.history['val_accuracy'][-1]

        print(f"Final Training Accuracy: {final_train_acc:.4f}")
        print(f"Final Test Accuracy: {final_test_acc:.4f}")
        print(f"Overfitting Gap: {final_train_acc - final_test_acc:.4f}")

        plt.tight_layout()
        plt.savefig('learning_curve_epochs.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_learning_curve_data_volume(self, data_sizes=None, epochs=6):
        """Plot learning curve over different training data sizes."""
        print("\n" + "="*70)
        print("Analyzing Learning Curve Over Data Volume")
        print("="*70)

        if data_sizes is None:
            data_sizes = list(range(5000, 65000, 5000))

        train_accuracies = []
        test_accuracies = []

        # Set random seed for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

        print(f"\nTraining {len(data_sizes)} models with different data sizes...")

        for size in data_sizes:
            print(f"\nTraining with {size:,} samples...")

            # Randomly sample training data
            indices = np.random.choice(len(self.train_images), size, replace=False)
            train_images_subset = self.train_images[indices]
            train_labels_subset = self.train_labels[indices]

            # Build new model
            model_subset = keras.Sequential([
                keras.layers.Flatten(input_shape=(28, 28)),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(10, activation='softmax')
            ])

            model_subset.compile(optimizer='adam',
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'])

            # Train model
            history_subset = model_subset.fit(
                train_images_subset, train_labels_subset,
                epochs=epochs,
                validation_data=(self.test_images, self.test_labels),
                verbose=0
            )

            final_train_acc = history_subset.history['accuracy'][-1]
            final_test_acc = history_subset.history['val_accuracy'][-1]

            train_accuracies.append(final_train_acc)
            test_accuracies.append(final_test_acc)

            print(f"  Training Accuracy: {final_train_acc:.4f}")
            print(f"  Test Accuracy: {final_test_acc:.4f}")

        # Plot results
        plt.figure(figsize=(12, 6))

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
                        alpha=0.2, color='gray', label='Overfitting Gap')

        plt.tight_layout()
        plt.savefig('learning_curve_data_volume.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Print analysis
        print("\n" + "="*70)
        print("Data Volume Analysis Summary")
        print("="*70)
        print(f"\nAccuracy with {data_sizes[0]:,} samples:  Train={train_accuracies[0]:.4f}, Test={test_accuracies[0]:.4f}")
        print(f"Accuracy with {data_sizes[-1]:,} samples: Train={train_accuracies[-1]:.4f}, Test={test_accuracies[-1]:.4f}")
        print(f"\nImprovement in test accuracy: {test_accuracies[-1] - test_accuracies[0]:.4f}")

        # Create results table
        results_df = pd.DataFrame({
            'Training Samples': data_sizes,
            'Training Accuracy': [f'{acc:.4f}' for acc in train_accuracies],
            'Test Accuracy': [f'{acc:.4f}' for acc in test_accuracies],
            'Overfitting Gap': [f'{train_accuracies[i] - test_accuracies[i]:.4f}'
                               for i in range(len(data_sizes))]
        })

        print("\n" + results_df.to_string(index=False))


def main():
    """Main execution function."""
    print("="*70)
    print("CS535 HW4: MNIST Fashion Classification")
    print("="*70)

    # Initialize classifier
    classifier = FashionMNISTClassifier()

    # Task 1: Implement Baseline Model
    print("\n### TASK 1: Implement a Baseline Model ###")
    classifier.load_data()
    classifier.visualize_samples()
    classifier.build_model()
    classifier.train_model(epochs=10)
    classifier.evaluate_model()

    # Task 2: Confusion Matrix Analysis
    print("\n### TASK 2: Analyzing Classification Errors ###")
    predictions, predicted_labels = classifier.make_predictions()
    cm = classifier.plot_confusion_matrix(predicted_labels)
    classifier.analyze_confusion_matrix(cm)

    # Task 3: Learning Curve Over Time
    print("\n### TASK 3: Learning Curve Over Epochs ###")
    classifier.plot_learning_curve_epochs()

    # Task 4: Learning Curve Over Data Volume
    print("\n### TASK 4: Learning Curve Over Data Volume ###")
    classifier.plot_learning_curve_data_volume(epochs=6)

    print("\n" + "="*70)
    print("All Tasks Completed Successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
