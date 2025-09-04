import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pickle
import os
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef
from config import Config
from model import TransformerDecoder
from data_loading import CCSRDataset
from torch.utils.data import DataLoader, random_split
from tokenizer import CCSRSequenceTokenizer

class ModelEvaluator:
    def __init__(self, model_path="model_full.pth", tokenizer_path="tokenizer.pkl"):
        """
        Initialize the evaluator with a saved model and tokenizer
        """
        self.config = Config()
        self.device = torch.device(self.config.device)
        
        # Load the tokenizer from file
        print(f"Loading tokenizer from {tokenizer_path}")
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        # Update vocab size in config to match tokenizer
        self.config.vocab_size = len(self.tokenizer.idx2token)
        print(f"Vocabulary size: {self.config.vocab_size}")
        
        # Create validation data loader with the loaded tokenizer
        self._create_dataloader()
        
        # Load the saved model
        print(f"Loading model from {model_path}")
        try:
            # First try to load it as a state dict
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Initialize model
            self.model = TransformerDecoder(
                vocab_size=self.config.vocab_size,
                max_len=self.config.max_length,
                d_model=self.config.d_model,
                n_layers=self.config.n_layers,
                n_heads=self.config.n_heads,
                d_ff=self.config.d_ff,
                dropout=self.config.dropout,
                tie_weights=self.config.tie_weights
            )
            
            # Check if it's a state dict or a model object
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("Loaded model from state dictionary")
            elif isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint)
                print("Loaded model state dictionary")
            else:
                # If checkpoint is the model itself
                self.model = checkpoint
                print("Loaded model object directly")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        self.model.to(self.device)
        self.model.eval()
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.token2idx["[PAD]"], reduction='none')
    
    def _create_dataloader(self):
        """Create a data loader using the loaded tokenizer"""
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"Data file not found: {self.config.data_path}")

        # Read the data
        with open(self.config.data_path, "r") as f:
            raw_text = f.read()
        
        # Preprocess data and prepare inputs
        sequences = self.tokenizer.preprocess_data(raw_text)
        inputs = self.tokenizer.prepare_transformer_inputs(
            sequences,
            windowing_strategy=self.config.windowing_strategy,
            window_size=self.config.window_size,
            overlap=self.config.overlap,
            target_mode=self.config.target_mode
        )
        
        # Create dataset
        dataset = CCSRDataset(inputs["input_ids"], inputs["attention_mask"], inputs["targets"])
        
        # Split the dataset (we'll only use the validation set for evaluation)
        train_size = int(self.config.train_ratio * len(dataset))
        test_size = len(dataset) - train_size
        _, test_dataset = random_split(dataset, [train_size, test_size])
        
        # Create data loader with a dynamic_collate_fn
        from data_loading import dynamic_collate_fn
        self.val_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            collate_fn=dynamic_collate_fn
        )
        
        print(f"Created validation data loader with {len(test_dataset)} samples")
    
    def calculate_metrics(self, all_preds, all_targets):
        """
        Calculate evaluation metrics
        """
        metrics = {
            'precision': precision_score(all_targets, all_preds, average='macro', zero_division=0),
            'recall': recall_score(all_targets, all_preds, average='macro', zero_division=0),
            'f1': f1_score(all_targets, all_preds, average='macro', zero_division=0),
            'kappa': cohen_kappa_score(all_targets, all_preds),
            'mcc': matthews_corrcoef(all_targets, all_preds)
        }
        return metrics
    
    def evaluate_next_token_prediction(self):
        """
        Task 1: For all sequences, predict the second token and evaluate
        """
        print("\n--- Task 1: Predict second token for all sequences ---")
        all_preds = []
        all_targets = []
        total_loss = 0
        correct = {1: 0, 3: 0, 5: 0}
        total = 0
        reciprocal_ranks = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                inputs = batch['input_ids'].to(self.device)
                
                # Get sequences with at least 2 tokens
                mask = batch['attention_mask'].to(self.device)
                seq_lengths = mask.sum(dim=1)
                valid_samples = seq_lengths >= 2
                
                if not valid_samples.any():
                    continue
                    
                valid_inputs = inputs[valid_samples]
                valid_mask = mask[valid_samples]
                
                # Use only the first token to predict the second
                first_token = valid_inputs[:, 0:1]
                target_token = valid_inputs[:, 1]
                
                # Predict
                # Handle key_padding_mask instead of attention_mask
                key_padding_mask = (first_token == self.tokenizer.token2idx["[PAD]"])
                
                # Forward pass through the model
                outputs = self.model(first_token, key_padding_mask=key_padding_mask)
                
                # Handle the case where outputs is a tuple (logits, caches)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                logits = outputs[:, 0, :]  # Get predictions for the first position
                
                # Calculate loss for perplexity
                losses = self.criterion(logits, target_token)
                total_loss += losses.mean().item()
                
                # Get top-k predictions
                probs = torch.softmax(logits, dim=-1)
                
                # Top-1 predictions for standard metrics
                top1_preds = torch.argmax(logits, dim=-1)
                
                # Top-k accuracies
                for k in [1, 3, 5]:
                    topk_preds = torch.topk(logits, k=k, dim=-1).indices
                    # Check if target is in top-k predictions
                    is_correct = torch.any(topk_preds == target_token.unsqueeze(-1), dim=-1)
                    correct[k] += is_correct.sum().item()
                
                # Calculate MRR
                target_expanded = target_token.unsqueeze(-1)
                ranked_indices = torch.argsort(logits, dim=-1, descending=True)
                
                for target, ranks in zip(target_token, ranked_indices):
                    # Find rank of target (position in sorted predictions)
                    rank = (ranks == target).nonzero(as_tuple=True)[0].item() + 1
                    reciprocal_ranks.append(1.0 / rank)
                
                total += len(target_token)
                
                # Save predictions and targets for metric calculation
                all_preds.extend(top1_preds.cpu().numpy())
                all_targets.extend(target_token.cpu().numpy())
        
        # Calculate metrics
        base_metrics = self.calculate_metrics(all_preds, all_targets)
        
        # Additional metrics
        mrr = np.mean(reciprocal_ranks)
        perplexity = np.exp(total_loss / len(self.val_loader))
        
        metrics = {
            **base_metrics,
            'top1_acc': correct[1] / total if total > 0 else 0,
            'top3_acc': correct[3] / total if total > 0 else 0,
            'top5_acc': correct[5] / total if total > 0 else 0,
            'mrr': mrr,
            'perplexity': perplexity
        }
        
        # Report results
        print(f"Loss: {total_loss / len(self.val_loader):.4f}")
        print(f"Top-1 Accuracy: {metrics['top1_acc']:.4f}")
        print(f"Top-3 Accuracy: {metrics['top3_acc']:.4f}")
        print(f"Top-5 Accuracy: {metrics['top5_acc']:.4f}")
        print(f"MRR: {metrics['mrr']:.4f}")
        print(f"Perplexity: {metrics['perplexity']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Kappa: {metrics['kappa']:.4f}")
        print(f"MCC: {metrics['mcc']:.4f}")
        
        return metrics
    
    def evaluate_third_token_prediction(self):
        """
        Task 2: For sequences with length 3 or more, get first and second token,
        predict the third token and evaluate
        """
        print("\n--- Task 2: Predict third token given first two tokens ---")
        all_preds = []
        all_targets = []
        total_loss = 0
        correct = {1: 0, 3: 0, 5: 0}
        total = 0
        reciprocal_ranks = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                inputs = batch['input_ids'].to(self.device)
                
                # Get sequences with at least 3 tokens
                mask = batch['attention_mask'].to(self.device)
                seq_lengths = mask.sum(dim=1)
                valid_samples = seq_lengths >= 3
                
                if not valid_samples.any():
                    continue
                    
                valid_inputs = inputs[valid_samples]
                valid_mask = mask[valid_samples]
                
                # Use only the first two tokens to predict the third
                first_two_tokens = valid_inputs[:, 0:2]
                target_token = valid_inputs[:, 2]
                
                # Handle key_padding_mask
                key_padding_mask = (first_two_tokens == self.tokenizer.token2idx["[PAD]"])
                
                # Predict
                outputs = self.model(first_two_tokens, key_padding_mask=key_padding_mask)
                
                # Handle the case where outputs is a tuple (logits, caches)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                logits = outputs[:, 1, :]  # Get predictions for the second position
                
                # Calculate loss for perplexity
                losses = self.criterion(logits, target_token)
                total_loss += losses.mean().item()
                
                # Get top-k predictions
                probs = torch.softmax(logits, dim=-1)
                
                # Top-1 predictions for standard metrics
                top1_preds = torch.argmax(logits, dim=-1)
                
                # Top-k accuracies
                for k in [1, 3, 5]:
                    topk_preds = torch.topk(logits, k=k, dim=-1).indices
                    # Check if target is in top-k predictions
                    is_correct = torch.any(topk_preds == target_token.unsqueeze(-1), dim=-1)
                    correct[k] += is_correct.sum().item()
                
                # Calculate MRR
                target_expanded = target_token.unsqueeze(-1)
                ranked_indices = torch.argsort(logits, dim=-1, descending=True)
                
                for target, ranks in zip(target_token, ranked_indices):
                    # Find rank of target (position in sorted predictions)
                    rank = (ranks == target).nonzero(as_tuple=True)[0].item() + 1
                    reciprocal_ranks.append(1.0 / rank)
                
                total += len(target_token)
                
                # Save predictions and targets for metric calculation
                all_preds.extend(top1_preds.cpu().numpy())
                all_targets.extend(target_token.cpu().numpy())
        
        # Calculate metrics
        base_metrics = self.calculate_metrics(all_preds, all_targets)
        
        # Additional metrics
        mrr = np.mean(reciprocal_ranks)
        perplexity = np.exp(total_loss / len(self.val_loader))
        
        metrics = {
            **base_metrics,
            'top1_acc': correct[1] / total if total > 0 else 0,
            'top3_acc': correct[3] / total if total > 0 else 0,
            'top5_acc': correct[5] / total if total > 0 else 0,
            'mrr': mrr,
            'perplexity': perplexity
        }
        
        # Report results
        print(f"Loss: {total_loss / len(self.val_loader):.4f}")
        print(f"Top-1 Accuracy: {metrics['top1_acc']:.4f}")
        print(f"Top-3 Accuracy: {metrics['top3_acc']:.4f}")
        print(f"Top-5 Accuracy: {metrics['top5_acc']:.4f}")
        print(f"MRR: {metrics['mrr']:.4f}")
        print(f"Perplexity: {metrics['perplexity']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Kappa: {metrics['kappa']:.4f}")
        print(f"MCC: {metrics['mcc']:.4f}")
        
        return metrics
    
    def evaluate_fourth_token_prediction(self):
        """
        Task 3: For sequences with length 4 or more, get first three tokens,
        predict the fourth token and evaluate
        """
        print("\n--- Task 3: Predict fourth token given first three tokens ---")
        all_preds = []
        all_targets = []
        total_loss = 0
        correct = {1: 0, 3: 0, 5: 0}
        total = 0
        reciprocal_ranks = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                inputs = batch['input_ids'].to(self.device)
                
                # Get sequences with at least 4 tokens
                mask = batch['attention_mask'].to(self.device)
                seq_lengths = mask.sum(dim=1)
                valid_samples = seq_lengths >= 4
                
                if not valid_samples.any():
                    continue
                    
                valid_inputs = inputs[valid_samples]
                valid_mask = mask[valid_samples]
                
                # Use only the first three tokens to predict the fourth
                first_three_tokens = valid_inputs[:, 0:3]
                target_token = valid_inputs[:, 3]
                
                # Handle key_padding_mask
                key_padding_mask = (first_three_tokens == self.tokenizer.token2idx["[PAD]"])
                
                # Predict
                outputs = self.model(first_three_tokens, key_padding_mask=key_padding_mask)
                
                # Handle the case where outputs is a tuple (logits, caches)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                logits = outputs[:, 2, :]  # Get predictions for the third position
                
                # Calculate loss for perplexity
                losses = self.criterion(logits, target_token)
                total_loss += losses.mean().item()
                
                # Get top-k predictions
                probs = torch.softmax(logits, dim=-1)
                
                # Top-1 predictions for standard metrics
                top1_preds = torch.argmax(logits, dim=-1)
                
                # Top-k accuracies
                for k in [1, 3, 5]:
                    topk_preds = torch.topk(logits, k=k, dim=-1).indices
                    # Check if target is in top-k predictions
                    is_correct = torch.any(topk_preds == target_token.unsqueeze(-1), dim=-1)
                    correct[k] += is_correct.sum().item()
                
                # Calculate MRR
                target_expanded = target_token.unsqueeze(-1)
                ranked_indices = torch.argsort(logits, dim=-1, descending=True)
                
                for target, ranks in zip(target_token, ranked_indices):
                    # Find rank of target (position in sorted predictions)
                    rank = (ranks == target).nonzero(as_tuple=True)[0].item() + 1
                    reciprocal_ranks.append(1.0 / rank)
                
                total += len(target_token)
                
                # Save predictions and targets for metric calculation
                all_preds.extend(top1_preds.cpu().numpy())
                all_targets.extend(target_token.cpu().numpy())
        
        # Calculate metrics
        base_metrics = self.calculate_metrics(all_preds, all_targets)
        
        # Additional metrics
        mrr = np.mean(reciprocal_ranks)
        perplexity = np.exp(total_loss / len(self.val_loader))
        
        metrics = {
            **base_metrics,
            'top1_acc': correct[1] / total if total > 0 else 0,
            'top3_acc': correct[3] / total if total > 0 else 0,
            'top5_acc': correct[5] / total if total > 0 else 0,
            'mrr': mrr,
            'perplexity': perplexity
        }
        
        # Report results
        print(f"Loss: {total_loss / len(self.val_loader):.4f}")
        print(f"Top-1 Accuracy: {metrics['top1_acc']:.4f}")
        print(f"Top-3 Accuracy: {metrics['top3_acc']:.4f}")
        print(f"Top-5 Accuracy: {metrics['top5_acc']:.4f}")
        print(f"MRR: {metrics['mrr']:.4f}")
        print(f"Perplexity: {metrics['perplexity']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Kappa: {metrics['kappa']:.4f}")
        print(f"MCC: {metrics['mcc']:.4f}")
        
        return metrics
    
    def evaluate_last_token_prediction(self):
        """
        Task 4: For sequences of any length, predict the last token using all previous tokens
        """
        print("\n--- Task 4: Predict last token of sequences (any length) ---")
        all_preds = []
        all_targets = []
        total_loss = 0
        correct = {1: 0, 3: 0, 5: 0}
        total = 0
        reciprocal_ranks = []
        
        # Track metrics by sequence length for analysis
        length_metrics = {}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                inputs = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                
                # Get valid sequences (at least 2 tokens)
                seq_lengths = mask.sum(dim=1)
                valid_samples = seq_lengths >= 2
                
                if not valid_samples.any():
                    continue
                    
                valid_inputs = inputs[valid_samples]
                valid_mask = mask[valid_samples]
                valid_lengths = seq_lengths[valid_samples]
                
                batch_size = valid_inputs.size(0)
                
                for i in range(batch_size):
                    # Get the sequence length for this sample
                    seq_len = valid_lengths[i].item()
                    
                    # Use all tokens except the last one as input
                    input_seq = valid_inputs[i, :seq_len-1].unsqueeze(0)
                    target_token = valid_inputs[i, seq_len-1].unsqueeze(0)
                    
                    # Predict
                    key_padding_mask = (input_seq == self.tokenizer.token2idx["[PAD]"])
                    outputs = self.model(input_seq, key_padding_mask=key_padding_mask)
                    
                    # Handle the case where outputs is a tuple (logits, caches)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    # Get predictions for the last position
                    logits = outputs[0, -1, :]
                    
                    # Calculate loss for perplexity
                    loss = self.criterion(logits.unsqueeze(0), target_token)
                    total_loss += loss.mean().item()
                    
                    # Get top-k predictions
                    top1_pred = torch.argmax(logits)
                    
                    # Top-k accuracies
                    for k in [1, 3, 5]:
                        topk_preds = torch.topk(logits, k=k).indices
                        # Check if target is in top-k predictions
                        is_correct = (topk_preds == target_token.item()).any().item()
                        correct[k] += int(is_correct)
                    
                    # Calculate MRR
                    ranked_indices = torch.argsort(logits, descending=True)
                    rank = (ranked_indices == target_token.item()).nonzero(as_tuple=True)[0].item() + 1
                    reciprocal_ranks.append(1.0 / rank)
                    
                    # Track metrics by sequence length
                    if seq_len not in length_metrics:
                        length_metrics[seq_len] = {
                            'correct': {1: 0, 3: 0, 5: 0},
                            'total': 0,
                            'reciprocal_ranks': []
                        }
                    
                    length_metrics[seq_len]['total'] += 1
                    for k in [1, 3, 5]:
                        length_metrics[seq_len]['correct'][k] += int(is_correct)
                    length_metrics[seq_len]['reciprocal_ranks'].append(1.0 / rank)
                    
                    total += 1
                    
                    # Save predictions and targets for metric calculation
                    all_preds.append(top1_pred.cpu().item())
                    all_targets.append(target_token.cpu().item())
        
        # Calculate metrics
        base_metrics = self.calculate_metrics(all_preds, all_targets)
        
        # Additional metrics
        mrr = np.mean(reciprocal_ranks)
        perplexity = np.exp(total_loss / total) if total > 0 else float('inf')
        
        metrics = {
            **base_metrics,
            'top1_acc': correct[1] / total if total > 0 else 0,
            'top3_acc': correct[3] / total if total > 0 else 0,
            'top5_acc': correct[5] / total if total > 0 else 0,
            'mrr': mrr,
            'perplexity': perplexity,
            'by_length': {
                length: {
                    'top1_acc': data['correct'][1] / data['total'] if data['total'] > 0 else 0,
                    'top3_acc': data['correct'][3] / data['total'] if data['total'] > 0 else 0,
                    'top5_acc': data['correct'][5] / data['total'] if data['total'] > 0 else 0,
                    'mrr': np.mean(data['reciprocal_ranks']),
                    'count': data['total']
                }
                for length, data in length_metrics.items()
            }
        }
        
        # Report results
        print(f"Loss: {total_loss / total:.4f}")
        print(f"Top-1 Accuracy: {metrics['top1_acc']:.4f}")
        print(f"Top-3 Accuracy: {metrics['top3_acc']:.4f}")
        print(f"Top-5 Accuracy: {metrics['top5_acc']:.4f}")
        print(f"MRR: {metrics['mrr']:.4f}")
        print(f"Perplexity: {metrics['perplexity']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Kappa: {metrics['kappa']:.4f}")
        print(f"MCC: {metrics['mcc']:.4f}")
        
        # Report accuracy by sequence length
        print("\nAccuracy by sequence length:")
        for length in sorted(metrics['by_length'].keys()):
            data = metrics['by_length'][length]
            print(f"  Length {length} (n={data['count']}): "
                  f"Top-1 Acc={data['top1_acc']:.4f}, "
                  f"Top-3 Acc={data['top3_acc']:.4f}, "
                  f"MRR={data['mrr']:.4f}")
        
        return metrics

def main():
    # Create evaluator with the specified model and tokenizer files
    evaluator = ModelEvaluator("model_full.pth", "tokenizer.pkl")
    
    # Run the evaluation tasks
    print("\n===== MODEL EVALUATION =====")
    task1_metrics = evaluator.evaluate_next_token_prediction()
    task2_metrics = evaluator.evaluate_third_token_prediction()
    task3_metrics = evaluator.evaluate_fourth_token_prediction()
    task4_metrics = evaluator.evaluate_last_token_prediction()
    
    # Print summary
    print("\n===== EVALUATION SUMMARY =====")
    print("Task 1 (Predict 2nd token):")
    print(f"  Top-1 Acc: {task1_metrics['top1_acc']:.4f}, Top-3 Acc: {task1_metrics['top3_acc']:.4f}, Top-5 Acc: {task1_metrics['top5_acc']:.4f}")
    print(f"  MRR: {task1_metrics['mrr']:.4f}, PPL: {task1_metrics['perplexity']:.4f}")
    print(f"  Kappa: {task1_metrics['kappa']:.4f}, MCC: {task1_metrics['mcc']:.4f}")
    
    print("\nTask 2 (Predict 3rd token):")
    print(f"  Top-1 Acc: {task2_metrics['top1_acc']:.4f}, Top-3 Acc: {task2_metrics['top3_acc']:.4f}, Top-5 Acc: {task2_metrics['top5_acc']:.4f}")
    print(f"  MRR: {task2_metrics['mrr']:.4f}, PPL: {task2_metrics['perplexity']:.4f}")
    print(f"  Kappa: {task2_metrics['kappa']:.4f}, MCC: {task2_metrics['mcc']:.4f}")
    
    print("\nTask 3 (Predict 4th token):")
    print(f"  Top-1 Acc: {task3_metrics['top1_acc']:.4f}, Top-3 Acc: {task3_metrics['top3_acc']:.4f}, Top-5 Acc: {task3_metrics['top5_acc']:.4f}")
    print(f"  MRR: {task3_metrics['mrr']:.4f}, PPL: {task3_metrics['perplexity']:.4f}")
    print(f"  Kappa: {task3_metrics['kappa']:.4f}, MCC: {task3_metrics['mcc']:.4f}")
    
    print("\nTask 4 (Predict last token regardless of sequence length):")
    print(f"  Top-1 Acc: {task4_metrics['top1_acc']:.4f}, Top-3 Acc: {task4_metrics['top3_acc']:.4f}, Top-5 Acc: {task4_metrics['top5_acc']:.4f}")
    print(f"  MRR: {task4_metrics['mrr']:.4f}, PPL: {task4_metrics['perplexity']:.4f}")
    print(f"  Kappa: {task4_metrics['kappa']:.4f}, MCC: {task4_metrics['mcc']:.4f}")

if __name__ == "__main__":
    main() 