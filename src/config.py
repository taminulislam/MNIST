"""
Configuration settings for Fashion MNIST classification
"""

# Model architecture parameters
INPUT_SHAPE = (28, 28)
HIDDEN_UNITS = 128
NUM_CLASSES = 10

# Training parameters
EPOCHS_BASELINE = 10
EPOCHS_DATA_VOLUME = 6
BATCH_SIZE = 32

# Data volume experiment parameters
DATA_SIZES = list(range(5000, 65000, 5000))

# Random seed for reproducibility
RANDOM_SEED = 42

# Class names
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Visualization parameters
VIZ_FIGSIZE_LARGE = (12, 10)
VIZ_FIGSIZE_MEDIUM = (12, 6)
VIZ_FIGSIZE_SMALL = (10, 10)
VIZ_DPI = 150

# Output directories
OUTPUT_DIR = "outputs"
MODELS_DIR = "models"
