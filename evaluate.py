"""
Evaluation script for Fashion MNIST Classification

Usage:
    python evaluate.py --model-path models/baseline_model.keras
"""

import argparse
import os
import numpy as np
from tensorflow import keras
from src import config
from src.data_loader import load_fashion_mnist
from src.visualization import plot_confusion_matrix, plot_prediction_samples
from src.analysis import analyze_confusion_matrix, evaluate_model_performance


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Fashion MNIST Classifier')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to saved model')
    parser.add_argument('--output-dir', type=str, default=config.OUTPUT_DIR,
                       help='Output directory for plots')
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("Evaluating Fashion MNIST Classifier")
    print("="*70)

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model = keras.models.load_model(args.model_path)
    model.summary()

    # Load test data
    print("\nLoading test data...")
    _, _, test_images, test_labels = load_fashion_mnist()

    # Evaluate model
    evaluate_model_performance(model, test_images, test_labels)

    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)

    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    cm = plot_confusion_matrix(
        test_labels, predicted_labels,
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )

    # Analyze confusion matrix
    analyze_confusion_matrix(cm)

    # Plot prediction samples
    print("\nPlotting sample predictions...")
    plot_prediction_samples(
        test_images, test_labels, predictions,
        save_path=os.path.join(args.output_dir, 'prediction_samples.png')
    )

    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
