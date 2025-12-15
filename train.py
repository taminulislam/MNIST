"""
Training script for Fashion MNIST Classification

Usage:
    python train.py --model baseline --epochs 10
    python train.py --model improved --epochs 20
    python train.py --model cnn --epochs 15
"""

import argparse
import os
import tensorflow as tf
from src import config
from src.data_loader import load_fashion_mnist, get_dataset_info
from src.models import create_baseline_model, create_improved_model, create_cnn_model
from src.visualization import plot_training_history, plot_sample_images
from src.analysis import print_training_summary, evaluate_model_performance


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Fashion MNIST Classifier')
    parser.add_argument('--model', type=str, default='baseline',
                       choices=['baseline', 'improved', 'cnn'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS_BASELINE,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--save-model', action='store_true',
                       help='Save the trained model')
    parser.add_argument('--output-dir', type=str, default=config.OUTPUT_DIR,
                       help='Output directory for plots and models')
    parser.add_argument('--seed', type=int, default=config.RANDOM_SEED,
                       help='Random seed for reproducibility')
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Set random seeds
    tf.random.set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print(f"Training Fashion MNIST Classifier - {args.model.upper()} Model")
    print("="*70)

    # Load data
    print("\nLoading data...")
    train_images, train_labels, test_images, test_labels = load_fashion_mnist()
    get_dataset_info(train_images, train_labels, test_images, test_labels)

    # Visualize samples
    print("\nVisualizing sample images...")
    plot_sample_images(
        train_images, train_labels,
        save_path=os.path.join(args.output_dir, 'sample_images.png')
    )

    # Create model
    print(f"\nCreating {args.model} model...")
    if args.model == 'baseline':
        model = create_baseline_model()
    elif args.model == 'improved':
        model = create_improved_model()
    else:
        model = create_cnn_model()

    model.summary()

    # Train model
    print(f"\nTraining model for {args.epochs} epochs...")
    history = model.fit(
        train_images, train_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(test_images, test_labels),
        verbose=1
    )

    # Print summary
    print_training_summary(history)

    # Evaluate model
    evaluate_model_performance(model, test_images, test_labels)

    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(
        history,
        save_path=os.path.join(args.output_dir, f'{args.model}_training_history.png')
    )

    # Save model if requested
    if args.save_model:
        model_dir = os.path.join(config.MODELS_DIR)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'{args.model}_model.keras')
        model.save(model_path)
        print(f"\nModel saved to: {model_path}")

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
