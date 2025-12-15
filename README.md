# Fashion MNIST Classification Project

**CS535 Advanced Machine Learning: Homework #4**

A comprehensive deep learning solution for Fashion MNIST classification with modular architecture, multiple model implementations, and extensive analysis tools.

## 🎯 Project Overview

This project implements a complete pipeline for Fashion MNIST classification including:
- Baseline neural network model
- Confusion matrix analysis with detailed error interpretation
- Learning curve visualization (epochs and data volume)
- Modular, production-ready code structure
- Multiple model architectures (baseline, improved, CNN)
- Command-line training and evaluation tools

## 📁 Project Structure

```
MNIST/
├── src/                                    # Source package
│   ├── __init__.py                        # Package initialization
│   ├── config.py                          # Configuration parameters
│   ├── data_loader.py                     # Data loading utilities
│   ├── models.py                          # Model architectures
│   ├── visualization.py                   # Plotting functions
│   └── analysis.py                        # Analysis utilities
├── mnist_fashion_classification.py        # Single-file implementation
├── train.py                               # Training script
├── evaluate.py                            # Evaluation script
├── run_experiments.py                     # Run all homework tasks
├── CS535_HW4_MNIST_Fashion_Classification.ipynb  # Original notebook
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
└── outputs/                               # Generated plots and results
```

## 🚀 Quick Start

### Option 1: Run All Experiments (Homework Tasks)

Run all 4 tasks from the homework assignment:

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments
python run_experiments.py
```

This will execute:
- **Task 1**: Baseline model training (10 epochs)
- **Task 2**: Confusion matrix analysis
- **Task 3**: Learning curve over epochs
- **Task 4**: Learning curve over data volume

Results are saved to `outputs/` directory.

### Option 2: Train Custom Model

Train a specific model with custom parameters:

```bash
# Train baseline model
python train.py --model baseline --epochs 10

# Train improved model with dropout
python train.py --model improved --epochs 20 --save-model

# Train CNN model
python train.py --model cnn --epochs 15
```

### Option 3: Evaluate Saved Model

Evaluate a previously trained model:

```bash
python evaluate.py --model-path models/baseline_model.keras
```

### Option 4: Use Single-File Script

Run the complete implementation in a single file:

```bash
python mnist_fashion_classification.py
```

### Option 5: Use Jupyter Notebook

```bash
jupyter notebook CS535_HW4_MNIST_Fashion_Classification.ipynb
```

## 📊 Available Models

### 1. Baseline Model
Simple feedforward neural network:
- Flatten layer (28x28 → 784)
- Dense layer (128 units, ReLU)
- Output layer (10 units, Softmax)
- **Params**: ~101K
- **Expected Accuracy**: ~88%

### 2. Improved Model
Enhanced architecture with regularization:
- Dense layer (256 units, ReLU)
- Batch Normalization
- Dropout (0.3)
- Dense layer (128 units, ReLU)
- Batch Normalization
- Dropout (0.2)
- Output layer (10 units, Softmax)
- **Params**: ~235K
- **Expected Accuracy**: ~90-92%

### 3. CNN Model
Convolutional neural network:
- Conv2D (32 filters, 3x3)
- MaxPooling2D
- Conv2D (64 filters, 3x3)
- MaxPooling2D
- Conv2D (64 filters, 3x3)
- Dense (64 units)
- Dropout (0.5)
- Output layer (10 units, Softmax)
- **Params**: ~123K
- **Expected Accuracy**: ~91-93%

## 🛠️ Command-Line Options

### Training Script (`train.py`)

```bash
python train.py [OPTIONS]

Options:
  --model {baseline,improved,cnn}  Model architecture (default: baseline)
  --epochs INT                     Number of epochs (default: 10)
  --batch-size INT                 Batch size (default: 32)
  --save-model                     Save trained model
  --output-dir PATH                Output directory (default: outputs)
  --seed INT                       Random seed (default: 42)
```

### Evaluation Script (`evaluate.py`)

```bash
python evaluate.py --model-path PATH [OPTIONS]

Options:
  --model-path PATH     Path to saved model (required)
  --output-dir PATH     Output directory (default: outputs)
```

## 📦 Dependencies

- Python 3.8+
- TensorFlow 2.13+
- NumPy 1.24+
- Matplotlib 3.7+
- Scikit-learn 1.3+
- Seaborn 0.12+
- Pandas 2.0+

Install all dependencies:

```bash
pip install -r requirements.txt
```

## 📈 Results

### Task 1: Baseline Model
- **Test Accuracy**: 88.42%
- **Training Time**: ~2-3 minutes
- Model converges smoothly over 10 epochs

### Task 2: Confusion Matrix Analysis

**Most Common Misclassifications**:
1. Shirt ↔ T-shirt/top (129 & 109 errors)
2. Pullover ↔ Coat (110 & 90 errors)
3. Sneaker ↔ Ankle boot (71 errors)

**Per-Class Performance**:
- Best: Ankle boot (98.00%), Bag (97.30%), Trouser (97.20%)
- Worst: Shirt (67.40%), Pullover (79.40%)

### Task 3: Learning Curve Analysis
- Model shows steady improvement over epochs
- Minimal overfitting (gap ~2-3%)
- Converges around epoch 7-8

### Task 4: Data Volume Impact
- Accuracy improves from 83.26% (5K samples) to 87.60% (60K samples)
- Diminishing returns observed after ~40K samples
- Each additional 5K samples adds ~0.35% accuracy initially, ~0.20% later

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# Model parameters
HIDDEN_UNITS = 128
NUM_CLASSES = 10

# Training parameters
EPOCHS_BASELINE = 10
EPOCHS_DATA_VOLUME = 6
BATCH_SIZE = 32

# Data volume experiment
DATA_SIZES = list(range(5000, 65000, 5000))

# Random seed
RANDOM_SEED = 42
```

## 📝 Usage Examples

### Example 1: Train and evaluate baseline model

```bash
# Train
python train.py --model baseline --epochs 10 --save-model

# Evaluate
python evaluate.py --model-path models/baseline_model.keras
```

### Example 2: Compare different architectures

```bash
# Train all models
python train.py --model baseline --epochs 15 --save-model
python train.py --model improved --epochs 15 --save-model
python train.py --model cnn --epochs 15 --save-model

# Evaluate each
python evaluate.py --model-path models/baseline_model.keras
python evaluate.py --model-path models/improved_model.keras
python evaluate.py --model-path models/cnn_model.keras
```

### Example 3: Custom training configuration

```bash
python train.py \
    --model improved \
    --epochs 25 \
    --batch-size 64 \
    --save-model \
    --output-dir my_results \
    --seed 123
```

## 🧪 Testing Environment

Tested with:
- TensorFlow 2.19.1
- Keras 3.12.0
- Python 3.10
- NVIDIA RTX 6000 Ada (optional GPU)
- Ubuntu/Windows

## 📚 Key Features

✅ **Modular Design**: Clean separation of concerns
✅ **Multiple Models**: Baseline, improved, and CNN architectures
✅ **CLI Tools**: Easy training and evaluation from command line
✅ **Comprehensive Analysis**: Confusion matrix, learning curves, error analysis
✅ **Reproducible**: Fixed random seeds for consistent results
✅ **Well-Documented**: Extensive comments and docstrings
✅ **Production-Ready**: Proper package structure and error handling
✅ **Visualization**: High-quality plots saved automatically

## 📊 Output Files

Running the experiments generates:

```
outputs/
├── task1_samples.png              # Sample images from dataset
├── task2_confusion_matrix.png     # Confusion matrix heatmap
├── task3_learning_curve.png       # Accuracy over epochs
└── task4_data_volume.png          # Accuracy vs data size

models/
├── baseline_model.keras           # Saved baseline model
├── improved_model.keras           # Saved improved model
└── cnn_model.keras               # Saved CNN model
```

## 🐛 Troubleshooting

### GPU Memory Issues
If you encounter GPU memory errors:
```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

### Import Errors
Make sure you're running from the project root:
```bash
cd /path/to/MNIST
python run_experiments.py
```

### Slow Training
Use GPU acceleration or reduce epochs/data size:
```bash
python train.py --model baseline --epochs 5
```

## 📖 References

- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API](https://keras.io/)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/metrics.html)

## 👨‍💻 Author

**Taminul Islam** (856569517)
CS535 Advanced Machine Learning

## 📄 License

This is an academic project for CS535 Advanced Machine Learning.

## 🎓 Homework Submission

For homework submission:
1. Run all experiments: `python run_experiments.py`
2. Include the Jupyter notebook with outputs
3. Submit both `.ipynb` and exported `.pdf`/`.html`

---

**Note**: This project demonstrates best practices in deep learning project organization, including modular code structure, comprehensive documentation, and reproducible experiments.
