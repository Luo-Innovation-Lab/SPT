# Contributing to Sequential Pattern Transformer

Thank you for your interest in contributing to the Sequential Pattern Transformer project!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/sequential-pattern-transformer.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`

## Development Setup

### Project Structure

- `main.py` - Main training script
- `example.py` - Example usage
- `src/` - Source code
  - `models/` - Model implementations
  - `data/` - Data loading and tokenization
  - `analysis/` - Analysis tools
  - `visualization/` - Visualization utilities
  - `evaluation/` - Model evaluation scripts

### Running Tests

```bash
# Train a small model for testing
python main.py --epochs 5 --batch-size 8

# Test predictions
python example.py
```

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker
- Include:
  - Python version
  - PyTorch version
  - Error messages
  - Steps to reproduce

### Submitting Changes

1. Create a feature branch: `git checkout -b feature-name`
2. Make your changes
3. Test your changes
4. Commit: `git commit -m "Description of changes"`
5. Push: `git push origin feature-name`
6. Submit a pull request

### Code Style

- Follow PEP 8
- Use type hints where possible
- Add docstrings to functions and classes
- Keep functions focused and modular

### Areas for Contribution

- **New tokenization strategies**
- **Additional model architectures**
- **Evaluation metrics**
- **Visualization improvements**
- **Documentation and examples**
- **Performance optimizations**

## Questions?

Feel free to open an issue for questions or discussion!
