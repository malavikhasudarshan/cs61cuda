#!/bin/bash
# Script to render Quarto document to HTML

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Quarto is installed
if ! command -v quarto &> /dev/null; then
    echo "Error: Quarto is not installed."
    echo ""
    echo "To install Quarto on macOS:"
    echo "  1. Visit: https://quarto.org/docs/get-started/"
    echo "  2. Or use Homebrew: brew install --cask quarto"
    echo ""
    echo "After installing, run this script again."
    exit 1
fi

echo "Converting README.md to Quarto format..."
python3 convert_to_quarto.py ../README.md webpage.qmd

echo ""
echo "Rendering Quarto document to HTML..."
quarto render webpage.qmd

echo ""
echo "✓ HTML file generated: webpage.html"
echo "✓ Open webpage.html in your browser to view"

