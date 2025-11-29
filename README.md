# CS535 Advanced Machine Learning: Homework #4
## MNIST Fashion Classification

This project implements a deep learning solution for the Fashion MNIST classification task, including baseline model training, confusion matrix analysis, and learning curve experiments.

## Files Included

- `CS535_HW4_MNIST_Fashion_Classification.ipynb` - Main Jupyter notebook with all tasks
- `requirements.txt` - Python package dependencies
- `setup_environment.sh` - Automated environment setup script
- `README.md` - This file

## Quick Start

### Option 1: Automated Setup (Linux/Mac)

```bash
# Run the setup script
./setup_environment.sh

# Activate the environment
source cs535_hw4_env/bin/activate

# Launch Jupyter Notebook
jupyter notebook
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv cs535_hw4_env

# Activate environment
# On Linux/Mac:
source cs535_hw4_env/bin/activate
# On Windows:
# cs535_hw4_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

## Assignment Tasks

The notebook implements all four required tasks:

### Task 1: Implement a Baseline Model
- Load and preprocess Fashion MNIST dataset
- Build a neural network with Flatten and Dense layers
- Train the model for 10 epochs
- Evaluate performance on test data
- Visualize predictions

### Task 2: Analyzing Classification Errors through Confusion Matrix
- Generate confusion matrix using sklearn
- Visualize with heatmap using class names
- Analyze most common misclassifications
- Provide interpretation of error patterns

### Task 3: Observing Model's Learning Curve Over Time
- Plot training and test accuracy over 10 epochs
- Visualize convergence and detect overfitting
- Annotate key performance metrics

### Task 4: Examining Learning Curve Over Data Volume
- Train models with varying data sizes (5k to 60k samples)
- Fix training to 6 epochs per model
- Plot accuracy vs. training data size
- Analyze diminishing returns and data efficiency

## Dependencies

- Python 3.8+
- TensorFlow 2.13+
- NumPy 1.24+
- Matplotlib 3.7+
- Scikit-learn 1.3+
- Jupyter 1.0+
- Pandas 2.0+
- Seaborn 0.12+

## Usage Instructions

1. Open the Jupyter notebook in your browser
2. **IMPORTANT**: Update the student name(s) in the first cell
3. Run all cells sequentially (Cell → Run All) or execute cells individually
4. Review outputs, visualizations, and interpretations
5. Export to PDF or HTML: File → Download as → PDF/HTML

## Expected Outputs

The notebook will generate:
- Sample image visualizations
- Model architecture summary
- Training progress logs
- Confusion matrix heatmap
- Learning curve plots (epochs and data volume)
- Performance analysis tables
- Detailed interpretations

## Notes

- Training Task 4 (data volume experiment) may take 10-15 minutes as it trains 12 separate models
- All random seeds are set for reproducibility
- The notebook includes comprehensive markdown explanations for each section
- Figures are properly sized and labeled as per assignment requirements

## Troubleshooting

### GPU Support (Optional)
If you have a CUDA-capable GPU and want to use it:
```bash
pip install tensorflow[and-cuda]
```

### Memory Issues
If you encounter memory errors during Task 4:
- Close other applications
- Reduce the number of data sizes tested
- Use Google Colab instead

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

## Submission Checklist

Before submitting to D2L:

- [ ] Student name(s) added to first cell
- [ ] All cells executed successfully
- [ ] All visualizations display correctly
- [ ] Confusion matrix interpretation completed
- [ ] Learning curves properly labeled
- [ ] Notebook exported to PDF or HTML
- [ ] Both .ipynb and PDF/HTML files ready for submission

## References

- [TensorFlow Fashion MNIST Tutorial](https://www.tensorflow.org/tutorials/keras/classification)
- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [Scikit-learn Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

## License

This is an academic assignment for CS535 Advanced Machine Learning.
