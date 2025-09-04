# Sequential Pattern Transformer (SPT)

# Interpretable Framework for Predicting Disease Trajectories

A PyTorch-based transformer model for learning patterns in medical disease sequences.

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/sequential-pattern-transformer.git
cd sequential-pattern-transformer
pip install -r requirements.txt
```



### Pattern Mining

First, run the PrefixSpan-based pattern mining code in the src directory. This step extracts the common patterns from disease sequences and generates the input data required for the transformer model.

### Training a Model

```bash
python main.py --data-path src/data/modified_sid_patt.txt --epochs 50 --batch-size 32
```

### Making Predictions

```bash
python example.py
```

## 📁 Project Structure

```
├── main.py                 # Main training script
├── example.py              # Example usage and predictions
├── requirements.txt        # Python dependencies
├── src/
│   ├── config.py          # Configuration settings
│   ├── pattern_mining.py  # prefixspan
│   ├── models/
│   │   ├── model.py       # Transformer model implementation
│   │   └── trainer.py     # Training pipeline
│   ├── data/
│   │   ├── tokenizer.py   # CCSR sequence tokenizer
│   │   ├── data_loading.py # Data loading utilities
│   │   └── modified_sid_patt.txt # Sample training data
│   ├── analysis/          # Analysis tools and notebooks
│   ├── evaluation/        # Model evaluation scripts
│   └── visualization/     # Visualization utilities
└── visualizations/        # Generated plots and analysis results
```

## Model Architecture

- **Type**: Transformer Decoder (GPT-like architecture)
- **Vocabulary**: CCSR disease codes (variable based on training data)

### Advanced Training Strategies

- **Progressive dynamic**: Growing context windows
- **Sliding windows**: Overlapping contexts

## Data Format

### Input Data Requirements

The training data should be a text file containing a Python list of comma-separated disease sequences. Each sequence represents a patient's medical history.

**Format Example:**

```python
['END010, END003, GEN003, CIR011, FAC025', 
'END010, END003, CIR019, CIR011, FAC025', 
'END011, END010, END003, GEN003, FAC025',
'END003, CIR019, GEN003, CIR011, FAC025']
```

### Data Preparation Guidelines

1. **Sequence Length**: Variable length sequences are supported
2. **Code Format**: Use standard CCSR codes
3. **Separator**: Comma and space (`, `) between codes
4. **File Format**: Plain text file with Python list syntax
5. **Encoding**: UTF-8 text encoding

### Creating Your Dataset

```python
# Example: Convert your medical data to the required format
sequences = [
    "END002, CIR011, FAC025",      # Diabetes → Heart disease → Screening
    "GEN002, INF002, FAC025",      # Renal failure → Sepsis → Screening  
    "END003, GEN002, CIR019",      # Diabetes complications → Renal failure → Arrhythmia
]

# Save to file
with open('my_medical_data.txt', 'w') as f:
    f.write(str(sequences))
```

## Clinical Applications

- **Comorbidity Prediction**: Identify likely co-occurring conditions
- **Disease Progression**: Model temporal relationships
- **Clinical Decision Support**: Suggest related diagnoses

## Key Files

- **`main.py`**: Train new models with custom parameters
- **`example.py`**: Load trained models and make predictions
- **`src/config.py`**: Modify model architecture and training settings
- **`src/data/tokenizer.py`**: CCSR tokenization strategies
- **`src/models/model.py`**: Transformer implementation with attention caching

## Usage Examples

### Train with Custom Settings

```bash
python main.py \
  --data-path your_data.txt \
  --epochs 100 \
  --batch-size 64 \
  --lr 1e-4 \
  --device cuda \
  --output-dir my_model
```

### Predict Next Disease Code

you can find the Example code of using the code in "example.py"

```python
from src.models.model import TransformerDecoder
import pickle

# Load trained model and tokenizer
model, tokenizer = load_model_and_tokenizer("outputs/best_model.pth", "outputs/tokenizer.pkl")

# Predict next token for a sequence
sequence = "END002 CIR011"  # Diabetes + Heart disease
predictions = predict_next_tokens(model, tokenizer, sequence, top_k=5)

for token, prob in predictions:
    print(f"{token}: {prob:.4f}")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
