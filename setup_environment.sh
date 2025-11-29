#!/bin/bash

# CS535 HW4 Environment Setup Script
# This script creates a virtual environment and installs all dependencies

echo "================================================"
echo "CS535 HW4 - Fashion MNIST Classification"
echo "Environment Setup Script"
echo "================================================"
echo ""

# Create virtual environment
echo "Step 1: Creating virtual environment..."
python3 -m venv cs535_hw4_env

if [ $? -eq 0 ]; then
    echo "✓ Virtual environment created successfully"
else
    echo "✗ Failed to create virtual environment"
    exit 1
fi

echo ""

# Activate virtual environment
echo "Step 2: Activating virtual environment..."
source cs535_hw4_env/bin/activate

if [ $? -eq 0 ]; then
    echo "✓ Virtual environment activated"
else
    echo "✗ Failed to activate virtual environment"
    exit 1
fi

echo ""

# Upgrade pip
echo "Step 3: Upgrading pip..."
pip install --upgrade pip

echo ""

# Install requirements
echo "Step 4: Installing required packages..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ All packages installed successfully"
else
    echo "✗ Failed to install packages"
    exit 1
fi

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "To get started:"
echo "1. Activate the environment: source cs535_hw4_env/bin/activate"
echo "2. Launch Jupyter: jupyter notebook"
echo "3. Open: CS535_HW4_MNIST_Fashion_Classification.ipynb"
echo ""
echo "To deactivate the environment later, run: deactivate"
echo "================================================"
