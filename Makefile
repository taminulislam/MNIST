.PHONY: help install train evaluate experiments clean test lint format

help:
	@echo "Fashion MNIST Classification - Available Commands:"
	@echo ""
	@echo "  make install        - Install package and dependencies"
	@echo "  make train          - Train baseline model"
	@echo "  make evaluate       - Evaluate saved model"
	@echo "  make experiments    - Run all homework experiments"
	@echo "  make clean          - Remove generated files"
	@echo "  make test           - Run tests"
	@echo "  make lint           - Check code style"
	@echo "  make format         - Format code"
	@echo ""

install:
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

train:
	python train.py --model baseline --epochs 10 --save-model
	@echo "✓ Training complete"

train-improved:
	python train.py --model improved --epochs 20 --save-model
	@echo "✓ Improved model training complete"

train-cnn:
	python train.py --model cnn --epochs 15 --save-model
	@echo "✓ CNN model training complete"

evaluate:
	@if [ ! -f models/baseline_model.keras ]; then \
		echo "Error: No model found. Run 'make train' first."; \
		exit 1; \
	fi
	python evaluate.py --model-path models/baseline_model.keras
	@echo "✓ Evaluation complete"

experiments:
	python run_experiments.py
	@echo "✓ All experiments complete"

clean:
	rm -rf outputs/*
	rm -rf models/*
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf *.pyc
	rm -rf .pytest_cache
	@echo "✓ Cleaned generated files"

test:
	@echo "Running tests..."
	python test_environment.py
	@echo "✓ Tests passed"

lint:
	@echo "Checking code style..."
	@which flake8 > /dev/null || pip install flake8
	flake8 src/ --max-line-length=100 --ignore=E501,W503
	@echo "✓ Linting complete"

format:
	@echo "Formatting code..."
	@which black > /dev/null || pip install black
	black src/ --line-length=100
	@echo "✓ Formatting complete"

notebook:
	jupyter notebook CS535_HW4_MNIST_Fashion_Classification.ipynb

all: install experiments
	@echo "✓ Complete pipeline executed"
