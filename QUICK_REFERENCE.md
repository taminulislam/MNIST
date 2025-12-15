# Fashion MNIST Classification - Quick Reference

## 📁 Project Structure

```
MNIST/
│
├── 📦 src/                           # Main package
│   ├── __init__.py                  # Package init
│   ├── config.py                    # Configuration settings
│   ├── data_loader.py              # Data loading utilities
│   ├── models.py                   # Model architectures (3 models)
│   ├── visualization.py            # Plotting functions
│   └── analysis.py                 # Analysis utilities
│
├── 🚀 Executable Scripts
│   ├── mnist_fashion_classification.py  # Single-file implementation
│   ├── train.py                    # Model training CLI
│   ├── evaluate.py                 # Model evaluation CLI
│   └── run_experiments.py          # Run all homework tasks
│
├── 📚 Documentation
│   ├── README_NEW.md               # Main documentation
│   ├── USAGE_GUIDE.md              # Detailed usage guide
│   ├── PROJECT_SUMMARY.md          # Refactoring summary
│   ├── QUICK_REFERENCE.md          # This file
│   └── README.md                   # Original README
│
├── 📓 Original Files
│   ├── CS535_HW4_MNIST_Fashion_Classification.ipynb
│   ├── CS535_HW4_MNIST_Fashion_Classification.html
│   └── REQUIREMENTS_CHECKLIST.md
│
├── 🛠️ Setup & Config
│   ├── requirements.txt            # Python dependencies
│   ├── setup.py                    # Package setup
│   ├── Makefile                    # Build automation
│   ├── .gitignore                  # Git ignore rules
│   └── setup_environment.sh        # Environment setup
│
└── 📂 Generated (not in repo)
    ├── outputs/                    # Plots and results
    └── models/                     # Saved models
```

## 🎯 Common Commands

### Run Everything (Homework Submission)
```bash
python run_experiments.py
```
**Output**: All 4 tasks complete, results in `outputs/`

### Train Models

```bash
# Baseline (simple, fast)
python train.py --model baseline --epochs 10

# Improved (better accuracy)
python train.py --model improved --epochs 20 --save-model

# CNN (best accuracy)
python train.py --model cnn --epochs 15 --save-model
```

### Evaluate Models
```bash
python evaluate.py --model-path models/baseline_model.keras
```

### Using Makefile
```bash
make install      # Install dependencies
make train        # Train baseline model
make experiments  # Run all experiments
make clean        # Remove generated files
```

## 📊 Model Comparison

| Model    | Params | Accuracy | Training Time | Use Case                |
|----------|--------|----------|---------------|-------------------------|
| Baseline | 101K   | ~88%     | 2-3 min      | Quick experiments       |
| Improved | 235K   | ~90-92%  | 5-7 min      | Better performance      |
| CNN      | 123K   | ~91-93%  | 7-10 min     | Best accuracy           |

## 🔧 Configuration Quick Edit

Edit `src/config.py`:

```python
# Change epochs
EPOCHS_BASELINE = 10        # Default: 10

# Change model size
HIDDEN_UNITS = 128          # Default: 128, Try: 256

# Change data volume experiment
DATA_SIZES = list(range(5000, 65000, 5000))  # 5K to 60K

# Change random seed
RANDOM_SEED = 42            # Default: 42
```

## 🐍 Python API Quick Reference

### Load Data
```python
from src.data_loader import load_fashion_mnist
train_images, train_labels, test_images, test_labels = load_fashion_mnist()
```

### Create Model
```python
from src.models import create_baseline_model, create_improved_model, create_cnn_model

model = create_baseline_model()  # or create_improved_model() or create_cnn_model()
```

### Train
```python
history = model.fit(train_images, train_labels, epochs=10,
                   validation_data=(test_images, test_labels))
```

### Visualize
```python
from src.visualization import plot_training_history, plot_confusion_matrix

plot_training_history(history)
plot_confusion_matrix(test_labels, predicted_labels)
```

### Analyze
```python
from src.analysis import analyze_confusion_matrix, evaluate_model_performance

loss, acc = evaluate_model_performance(model, test_images, test_labels)
analyze_confusion_matrix(confusion_matrix)
```

## 📝 Homework Task Mapping

### Task 1: Baseline Model
```bash
python run_experiments.py
# Or specifically:
python train.py --model baseline --epochs 10
```
**Output**: `outputs/task1_samples.png`

### Task 2: Confusion Matrix
```bash
python run_experiments.py
# Or:
python evaluate.py --model-path models/baseline_model.keras
```
**Output**: `outputs/task2_confusion_matrix.png`

### Task 3: Learning Curve (Epochs)
```bash
python run_experiments.py
```
**Output**: `outputs/task3_learning_curve.png`

### Task 4: Learning Curve (Data Volume)
```bash
python run_experiments.py
```
**Output**: `outputs/task4_data_volume.png`
**Time**: ~10-15 minutes (trains 12 models)

## 🚨 Common Issues & Quick Fixes

### Issue: Import Error
```bash
# Fix: Run from project root
cd /path/to/MNIST
python train.py
```

### Issue: Out of Memory
```bash
# Fix: Reduce batch size
python train.py --batch-size 16
```

### Issue: Slow Training
```bash
# Fix 1: Reduce epochs
python train.py --epochs 5

# Fix 2: Use baseline model
python train.py --model baseline
```

### Issue: No GPU Detected
```bash
# Check GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 📥 Installation

### Quick Install
```bash
pip install -r requirements.txt
```

### Full Setup
```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install
pip install -r requirements.txt

# 4. Verify
python -c "import tensorflow; print(tensorflow.__version__)"
```

## 📤 Homework Submission Checklist

- [ ] Run all experiments: `python run_experiments.py`
- [ ] Check outputs in `outputs/` directory:
  - [ ] task1_samples.png
  - [ ] task2_confusion_matrix.png
  - [ ] task3_learning_curve.png
  - [ ] task4_data_volume.png
- [ ] Update student name in notebook (first cell)
- [ ] Run notebook: Kernel → Restart & Run All
- [ ] Export notebook: File → Download as → PDF
- [ ] Submit files:
  - [ ] `.ipynb` file
  - [ ] `.pdf` or `.html` file

## 🎓 Assignment Requirements Met

✅ **Task 1**: Baseline model implemented (10 epochs)
✅ **Task 2**: Confusion matrix with textual labels
✅ **Task 3**: Learning curve over epochs (1-10)
✅ **Task 4**: Learning curve over data volume (5K-60K, 6 epochs)
✅ **Documentation**: Clear, easy to follow
✅ **Submission**: Both notebook and PDF ready

## 📚 Documentation Files

- **README_NEW.md** → Full project documentation
- **USAGE_GUIDE.md** → Detailed usage instructions
- **PROJECT_SUMMARY.md** → What was changed and why
- **QUICK_REFERENCE.md** → This file (quick lookup)

## 🔗 Quick Links

| What You Want to Do | File to Read |
|---------------------|--------------|
| Understand the project | README_NEW.md |
| Learn how to use it | USAGE_GUIDE.md |
| See what was improved | PROJECT_SUMMARY.md |
| Quick command lookup | QUICK_REFERENCE.md (this file) |
| Run homework tasks | Run `python run_experiments.py` |

## ⚡ One-Liner Solutions

```bash
# Complete homework in one command
python run_experiments.py

# Train best model
python train.py --model cnn --epochs 15 --save-model

# Quick test (5 minutes)
python train.py --model baseline --epochs 5

# Full evaluation
python evaluate.py --model-path models/baseline_model.keras

# Clean everything
make clean
```

## 💡 Tips

1. **Save time**: Use `--save-model` only when needed
2. **Debug mode**: Use `--epochs 3` for quick testing
3. **GPU users**: Batch size up to 128 for faster training
4. **CPU users**: Keep batch size at 32
5. **First time**: Run `python run_experiments.py` once to see everything

---

**Last Updated**: December 15, 2025
**Student**: Taminul Islam (856569517)
**Course**: CS535 Advanced Machine Learning
