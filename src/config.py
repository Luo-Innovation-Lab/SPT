# config.py
class Config:
    # Data configuration
    data_path = "SID_filtered.txt"
    max_length = 10
    train_ratio = 0.8
    val_ratio = 0.1  # new: portion of data used for validation (test = 1 - train - val)
    batch_size = 256

    # Tokenizer configuration
    tokenizer_mode = "whole_word"  # "whole_word" or "subword"
    windowing_strategy = "progressive_dynamic"  
    window_size = 8     # Only used for "sliding" (or you could use it with "fixed" if desired)
    overlap = 0         # Only used for sliding


    target_mode = "next_token"  

    # Model configuration 
    vocab_size = 499  # This will be updated after tokenizer initialization
    d_model = 256
    n_layers = 4
    n_heads = 8
    d_ff = 2048
    dropout = 0.1
    tie_weights = True

    # Training configuration
    device = "mps"  # "cuda" or "cpu"
    lr = 1e-5
    num_epochs = 300
    top_k = [1, 3, 5]
    patience = 20
    checkpoint_path = "best_model.pth"

    # Experiment tracking
    wandb_project = "SPT-Transformer-full"  # set your Weights & Biases project name


