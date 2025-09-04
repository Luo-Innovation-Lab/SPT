#!/usr/bin/env python3
"""
Sequential Pattern Transformer (SPT) - Main Training Script

A PyTorch-based framework for sequence pattern analysis using transformer models.
This script trains a transformer model to learn patterns in medical disease sequences.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.config import Config
from src.data.tokenizer import CCSRSequenceTokenizer
from src.data.data_loading import create_dataloaders
from src.models.model import TransformerDecoder
from src.models.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train Sequential Pattern Transformer")
    parser.add_argument("--data-path", default="src/data/modified_sid_patt.txt", 
                       help="Path to training data file")
    parser.add_argument("--epochs", type=int, default=50, 
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, 
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, 
                       help="Learning rate")
    parser.add_argument("--device", default="auto", 
                       help="Device to use: cuda, mps, cpu, or auto")
    parser.add_argument("--output-dir", default="outputs", 
                       help="Directory to save model and results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Update config with command line arguments
    config = Config()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
    
    # Set device
    if args.device == "auto":
        if torch.cuda.is_available():
            config.device = "cuda"
        elif torch.backends.mps.is_available():
            config.device = "mps"
        else:
            config.device = "cpu"
    else:
        config.device = args.device
    
    print(f"Using device: {config.device}")
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        print("Please ensure you have the training data file.")
        sys.exit(1)
    
    print("Loading and preprocessing data...")
    
    # Update config with data path
    config.data_path = args.data_path
    
    # Create data loaders (this will also create and configure the tokenizer)
    train_loader, val_loader, test_loader, tokenizer = create_dataloaders(config)
    
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Initialize model
    print("Initializing model...")
    model = TransformerDecoder(
        vocab_size=config.vocab_size,
        max_len=config.max_length,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        tie_weights=config.tie_weights
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = Trainer(model, config, tokenizer)
    
    # Train model
    print("Starting training...")
    best_acc = trainer.fit(train_loader, val_loader)
    
    print(f"Training completed! Best validation accuracy: {best_acc:.4f}")
    
    # Save tokenizer
    import pickle
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.pkl")
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to: {tokenizer_path}")
    
    # Test model
    print("Evaluating on test set...")
    test_metrics = trainer._validate(test_loader)
    print(f"Test Accuracy: {test_metrics['top_k'][1]:.4f}")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    print(f"Test Perplexity: {test_metrics['perplexity']:.2f}")
    
    print(f"All outputs saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
