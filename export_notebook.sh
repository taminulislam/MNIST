#!/bin/bash

# CS535 HW4 - Notebook Export Script
# Exports notebook to both HTML and PDF formats

echo "=========================================="
echo "CS535 HW4 - Notebook Export Script"
echo "=========================================="
echo ""

NOTEBOOK="CS535_HW4_MNIST_Fashion_Classification.ipynb"

# Check if notebook exists
if [ ! -f "$NOTEBOOK" ]; then
    echo "ERROR: Notebook file not found: $NOTEBOOK"
    exit 1
fi

echo "Found notebook: $NOTEBOOK"
echo ""

# Export to HTML
echo "Step 1: Exporting to HTML..."
jupyter nbconvert --to html "$NOTEBOOK"

if [ $? -eq 0 ]; then
    echo "✓ HTML export successful!"
    echo "  Created: CS535_HW4_MNIST_Fashion_Classification.html"
else
    echo "✗ HTML export failed"
fi

echo ""

# Try to export to PDF
echo "Step 2: Attempting PDF export..."

# Try webpdf first
echo "  Trying webpdf method..."
jupyter nbconvert --to webpdf "$NOTEBOOK" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✓ PDF export successful (webpdf)!"
    echo "  Created: CS535_HW4_MNIST_Fashion_Classification.pdf"
else
    echo "  webpdf failed, trying LaTeX method..."
    jupyter nbconvert --to pdf "$NOTEBOOK" 2>/dev/null

    if [ $? -eq 0 ]; then
        echo "✓ PDF export successful (LaTeX)!"
        echo "  Created: CS535_HW4_MNIST_Fashion_Classification.pdf"
    else
        echo "⚠ PDF export failed"
        echo ""
        echo "ALTERNATIVE: You can convert HTML to PDF by:"
        echo "1. Open CS535_HW4_MNIST_Fashion_Classification.html in browser"
        echo "2. Press Ctrl+P (or Cmd+P on Mac)"
        echo "3. Select 'Save as PDF'"
        echo "4. Click 'Save'"
    fi
fi

echo ""
echo "=========================================="
echo "Export complete! Files ready:"
echo "=========================================="
ls -lh CS535_HW4_MNIST_Fashion_Classification.{ipynb,html,pdf} 2>/dev/null | awk '{print $9, "(" $5 ")"}'
echo ""
echo "Submit these files to D2L:"
echo "  1. CS535_HW4_MNIST_Fashion_Classification.ipynb"
echo "  2. CS535_HW4_MNIST_Fashion_Classification.html (or .pdf)"
echo "=========================================="
