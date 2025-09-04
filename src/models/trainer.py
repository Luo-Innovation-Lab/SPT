import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score, matthews_corrcoef
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Any, Dict, List

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

class Trainer:
    def __init__(self, model: nn.Module, config: Any, tokenizer: Any) -> None:
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device(config.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token2idx["[PAD]"], label_smoothing=0.1)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=2)

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.no_improve_epochs = 0

        self.use_wandb = bool(getattr(config, "use_wandb", False)) and (wandb is not None)

        self.metrics: Dict[str, List[float]] = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'val_f1': [], 'val_precision': [], 'val_recall': [],
            'val_mrr': [], 'val_perplexity': [], 'val_kappa': [], 'val_mcc': []
        }

    def fit(self, train_loader, val_loader) -> float:
        self.model.to(self.device)
        best_val_acc = 0.0

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            self._train_epoch(train_loader)
            val_metrics = self._validate(val_loader)

            self.metrics['val_loss'].append(val_metrics['loss'])
            self.metrics['val_acc'].append(val_metrics['top_k'][1])
            self.metrics['val_f1'].append(val_metrics['f1'])
            self.metrics['val_precision'].append(val_metrics['precision'])
            self.metrics['val_recall'].append(val_metrics['recall'])
            self.metrics['val_mrr'].append(val_metrics['mrr'])
            self.metrics['val_perplexity'].append(val_metrics['perplexity'])
            self.metrics['val_kappa'].append(val_metrics['kappa'])
            self.metrics['val_mcc'].append(val_metrics['mcc'])

            print(f"Train Loss: {self.metrics['train_loss'][-1]:.4f} | Acc: {self.metrics['train_acc'][-1]:.2%}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['top_k'][1]:.2%} | Top-3 Acc: {val_metrics['top_k'].get(3, 0):.2%} | Top-5 Acc: {val_metrics['top_k'].get(5, 0):.2%}")
            print(f"F1: {val_metrics['f1']:.4f} | Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | MRR: {val_metrics['mrr']:.4f} | PPL: {val_metrics['perplexity']:.2f} | Kappa: {val_metrics['kappa']:.4f} | MCC: {val_metrics['mcc']:.4f}")

            self.scheduler.step(val_metrics['loss'])

            val_acc = val_metrics['top_k'][1]
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.no_improve_epochs = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_metrics['loss'],
                }, self.config.checkpoint_path)
            else:
                self.no_improve_epochs += 1
                if self.no_improve_epochs >= self.config.patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

            if self.use_wandb:
                current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else None
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': self.metrics['train_loss'][-1],
                    'train/acc_top1': self.metrics['train_acc'][-1],
                    'val/loss': val_metrics['loss'],
                    'val/acc_top1': val_metrics['top_k'][1],
                    'val/acc_top3': val_metrics['top_k'].get(3, 0.0),
                    'val/acc_top5': val_metrics['top_k'].get(5, 0.0),
                    'val/f1': val_metrics['f1'],
                    'val/precision': val_metrics['precision'],
                    'val/recall': val_metrics['recall'],
                    'val/mrr': val_metrics['mrr'],
                    'val/perplexity': val_metrics['perplexity'],
                    'val/kappa': val_metrics['kappa'],
                    'val/mcc': val_metrics['mcc'],
                    'lr': current_lr,
                    'best/val_acc_top1': self.best_val_acc,
                })

        self._load_best_model()
        self._plot_metrics()
        if self.use_wandb:
            try:
                wandb.log({'plots/training_metrics': wandb.Image('training_metrics.png')})
            except Exception:
                pass
        return self.best_val_acc

    def _validate(self, loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        reciprocal_ranks = []
        top_k_correct = {k: 0 for k in self.config.top_k}
        total_tokens = 0

        with torch.inference_mode():
            for batch in tqdm(loader, desc="Validating", leave=False):
                inputs = batch['input_ids'].to(self.device)
                targets = inputs[:, 1:].contiguous()
                inputs = inputs[:, :-1].contiguous()
                pad_mask = (inputs == self.tokenizer.token2idx["[PAD]"])

                outputs = self.model(inputs, key_padding_mask=pad_mask)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = self._calculate_loss(outputs, targets)
                total_loss += loss.item()

                probs = torch.softmax(outputs, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                valid_mask = (targets != self.tokenizer.token2idx["[PAD]"])

                valid_preds = preds[valid_mask].cpu().numpy()
                valid_targets = targets[valid_mask].cpu().numpy()
                all_preds.extend(valid_preds)
                all_targets.extend(valid_targets)
                all_probs.extend(probs[valid_mask].cpu().numpy())

                for k in self.config.top_k:
                    top_k_preds = torch.topk(probs, k=k, dim=-1).indices
                    correct_k = top_k_preds.eq(targets.unsqueeze(-1)).any(dim=-1)
                    top_k_correct[k] += correct_k[valid_mask].sum().item()

                top_k_indices = torch.argsort(probs, dim=-1, descending=True)
                target_expanded = targets.unsqueeze(-1)
                rr_mask = (targets != self.tokenizer.token2idx["[PAD]"]).view(-1)
                flat_targets = targets[valid_mask].view(-1)
                flat_topk = top_k_indices[valid_mask].view(-1, probs.size(-1))

                for true_label, ranked_preds in zip(flat_targets, flat_topk):
                    ranks = (ranked_preds == true_label).nonzero(as_tuple=True)[0]
                    if len(ranks) > 0:
                        reciprocal_ranks.append(1.0 / (ranks.item() + 1))

                total_tokens += valid_mask.sum().item()

        perplexity = np.exp(total_loss / len(loader))
        mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

        metrics = {
            'loss': total_loss / len(loader),
            'top_k': {k: top_k_correct[k] / total_tokens for k in self.config.top_k},
            'precision': precision_score(all_targets, all_preds, average='macro', zero_division=0),
            'recall': recall_score(all_targets, all_preds, average='macro', zero_division=0),
            'f1': f1_score(all_targets, all_preds, average='macro', zero_division=0),
            'mrr': mrr,
            'perplexity': perplexity,
            'kappa': cohen_kappa_score(all_targets, all_preds),
            'mcc': matthews_corrcoef(all_targets, all_preds)
        }
        return metrics

    
    def predict_top_k(model, tokenizer, input_sequence, device, k=5, max_length=None):
        """
        Generate the top-k predictions for the next token.
        """
        # 1) Tokenize and encode the input sequence.
        tokenized_input = tokenizer.tokenize_sequence(input_sequence.split())
        encoded_input = tokenizer.encode_sequence(tokenized_input, add_special_tokens=True)
        
        # 2) Optionally remove trailing [SEP] so we predict the token after the input.
        sep_id = tokenizer.token2idx[tokenizer.special_tokens['SEP']]
        if encoded_input and encoded_input[-1] == sep_id:
            encoded_input = encoded_input[:-1]
        
        # 3) Optionally truncate if too long.
        if max_length is not None and len(encoded_input) > max_length:
            encoded_input = encoded_input[:max_length]
        
        # 4) Convert to tensor (batch of 1).
        input_ids = torch.tensor([encoded_input], dtype=torch.long, device=device)
        
        # 5) Build masks.
        pad_id = tokenizer.token2idx[tokenizer.special_tokens['PAD']]
        key_padding_mask = (input_ids == pad_id)
        seq_len = input_ids.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
        
        # 6) Forward pass.
        with torch.no_grad():
            outputs = model(x=input_ids, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        logits = outputs[0, -1, :]
        
        # 7) Compute top-k tokens and their probabilities.
        top_k_probs, top_k_indices = torch.topk(logits, k=k)
        softmax_probs = torch.softmax(logits, dim=-1)
        top_k_probs = softmax_probs[top_k_indices].tolist()
        
        # 8) Map token IDs back to strings.
        top_k_tokens = [
            tokenizer.idx2token.get(idx.item(), tokenizer.special_tokens['UNK']) 
            for idx in top_k_indices
        ]
        
        return list(zip(top_k_tokens, top_k_probs))

    def _train_epoch(self, loader: torch.utils.data.DataLoader) -> None:
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(loader, desc="Training", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            inputs = batch['input_ids'].to(self.device)          # shape: [batch, seq_len]
            targets = batch['targets'].to(self.device)             # shape: [batch] (each target is one token)
            mask = batch['attention_mask'].to(self.device)         # shape: [batch, seq_len]
            
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(inputs, key_padding_mask=(mask == 0))
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            # outputs has shape [batch, seq_len, vocab_size]
            
            # Determine the last valid token for each sample
            last_indices = mask.sum(dim=1) - 1  # shape: [batch]
            logits = outputs[torch.arange(outputs.size(0)), last_indices, :]  # shape: [batch, vocab_size]
            
            loss = self.criterion(logits, targets)
            
            # Print sample predictions every few batches
            if batch_idx % 10 == 0:
                sample_idx = 0
                sample_input = inputs[sample_idx]      # sample input sequence
                sample_target = targets[sample_idx]      # target token
                sample_last_idx = mask[sample_idx].sum() - 1
                sample_logits = outputs[sample_idx, sample_last_idx, :]
                sample_pred = torch.argmax(sample_logits, dim=-1)
                
                decoded_input = " ".join([self.tokenizer.idx2token.get(token.item(), "[UNK]") 
                                        for token in sample_input])
                decoded_target = self.tokenizer.idx2token.get(sample_target.item(), "[UNK]")
                decoded_pred = self.tokenizer.idx2token.get(sample_pred.item(), "[UNK]")
                
                print(f"\n[Batch {batch_idx}]")
                print("Input:  ", decoded_input)
                print("Target: ", decoded_target)
                print("Pred:   ", decoded_pred)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Optionally, compute accuracy here (comparing logits vs. targets)
            with torch.inference_mode():
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({
                'loss': f"{epoch_loss/(progress_bar.n+1):.4f}",
                'acc': f"{(correct/total):.2%}" if total > 0 else '0%'
            })
        
        self.metrics['train_loss'].append(epoch_loss / len(loader))
        self.metrics['train_acc'].append(correct / total if total > 0 else 0)


    def _calculate_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Ensure we're working with logits only.
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        logits = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        # Create a mask to ignore both [PAD] and [SEP]
        ignore_idx_pad = self.tokenizer.token2idx["[PAD]"]
        ignore_idx_sep = self.tokenizer.token2idx["[SEP]"]
        mask = (targets != ignore_idx_pad) & (targets != ignore_idx_sep)
        
        # Apply mask: only compute loss on positions where mask is True.
        masked_logits = logits[mask]
        masked_targets = targets[mask]
        return self.criterion(masked_logits, masked_targets)

    def _calculate_accuracy(self, outputs: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
        # Ensure we're working with logits only.
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        preds = torch.argmax(outputs, dim=-1)
        # Flatten targets and predictions.
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        # Create mask to ignore [PAD] and [SEP].
        ignore_idx_pad = self.tokenizer.token2idx["[PAD]"]
        ignore_idx_sep = self.tokenizer.token2idx["[SEP]"]
        valid_mask = (targets != ignore_idx_pad) & (targets != ignore_idx_sep)
        
        correct = (preds[valid_mask] == targets[valid_mask]).sum().item()
        total = valid_mask.sum().item()
        return correct, total

    def _load_best_model(self) -> None:
        """Load the best model from checkpoint."""
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def _plot_metrics(self) -> None:
        """Plot and save training metrics."""
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(self.metrics['train_loss'], label='Train')
        plt.plot(self.metrics['val_loss'], label='Validation')
        plt.title('Loss')
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 3, 2)
        plt.plot(self.metrics['train_acc'], label='Train')
        plt.plot(self.metrics['val_acc'], label='Validation')
        plt.title('Top-1 Accuracy')
        plt.legend()

        # Validation metrics plot
        plt.subplot(1, 3, 3)
        plt.plot(self.metrics['val_f1'], label='F1')
        plt.plot(self.metrics['val_precision'], label='Precision')
        plt.plot(self.metrics['val_recall'], label='Recall')
        plt.title('Validation Metrics')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()

