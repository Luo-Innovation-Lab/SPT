import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pickle
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import precision_score, recall_score, f1_score
import Levenshtein
from rouge_score import rouge_scorer
from config import Config
from models.model import TransformerDecoder
from data.data_loading import CCSRDataset
from torch.utils.data import DataLoader, random_split
from collections import Counter, defaultdict

class HorizontalEvaluator:
    """
    Evaluates a model's ability to predict disease trajectories horizontally:
    - Given n tokens, predict the next 1, 2, 3... tokens
    - Evaluate using various metrics for trajectory accuracy and likelihood
    """
    
    def __init__(self, model_path="model_full.pth", tokenizer_path="tokenizer.pkl", max_horizon=3):
        """
        Initialize the evaluator with a saved model and tokenizer
        
        Args:
            model_path: Path to the saved model
            tokenizer_path: Path to the saved tokenizer
            max_horizon: Maximum number of steps to predict ahead
        """
        self.config = Config()
        self.device = torch.device(self.config.device)
        self.max_horizon = max_horizon
        
        # Load the tokenizer from file
        print(f"Loading tokenizer from {tokenizer_path}")
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        # Update vocab size in config to match tokenizer
        self.config.vocab_size = len(self.tokenizer.idx2token)
        print(f"Vocabulary size: {self.config.vocab_size}")
        
        # Create validation data loader with the loaded tokenizer
        self._create_dataloader()
        
        # Initialize model
        try:
            # Load the saved model
            print(f"Loading model from {model_path}")
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
        
        # Initialize metrics
        try:
            import nltk
            nltk.download('wordnet', quiet=True)
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.smoothing_function = SmoothingFunction().method1
        except Exception as e:
            print(f"Warning: NLTK metrics initialization failed: {e}")
            print("Some metrics may not be available.")
    
    def _create_dataloader(self):
        """Create a data loader using the loaded tokenizer"""
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"Data file not found: {self.config.data_path}")

        # Read the data
        with open(self.config.data_path, "r") as f:
            raw_text = f.read()
        
        # Preprocess data and prepare inputs
        sequences = self.tokenizer.preprocess_data(raw_text)
        self.raw_sequences = sequences  # Store raw sequences for evaluation
        
        # Filter sequences that are long enough for our evaluation
        self.filtered_sequences = [seq for seq in sequences if len(seq) >= self.max_horizon + 1]
        print(f"Found {len(self.filtered_sequences)} sequences with length >= {self.max_horizon + 1}")
        
        # Prepare inputs using tokenizer
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
        
        # Create data loader
        from data.data_loading import dynamic_collate_fn
        self.val_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            collate_fn=dynamic_collate_fn
        )
        
        print(f"Created validation data loader with {len(test_dataset)} samples")
    
    def greedy_generate(self, input_sequence, num_tokens=1):
        """
        Generate tokens autoregressively using greedy decoding
        
        Args:
            input_sequence: Input token IDs
            num_tokens: Number of tokens to generate
            
        Returns:
            Generated token IDs
        """
        self.model.eval()
        
        # Convert input to tensor if needed
        if not isinstance(input_sequence, torch.Tensor):
            input_sequence = torch.tensor(input_sequence, dtype=torch.long, device=self.device)
        
        # Add batch dimension if needed
        if input_sequence.dim() == 1:
            input_sequence = input_sequence.unsqueeze(0)
        
        input_ids = input_sequence.to(self.device)
        generated_tokens = []
        log_probs = []
        
        with torch.no_grad():
            for _ in range(num_tokens):
                # Create padding mask
                key_padding_mask = (input_ids == self.tokenizer.token2idx["[PAD]"])
                
                # Forward pass through the model
                outputs = self.model(input_ids, key_padding_mask=key_padding_mask)
                
                # Handle the case where outputs is a tuple (logits, caches)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Get the last token prediction
                logits = outputs[:, -1, :]
                
                # Calculate probabilities
                probs = torch.softmax(logits, dim=-1)
                
                # Greedy decoding
                next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
                next_token_prob = torch.max(probs, dim=-1)[0]
                log_probs.append(torch.log(next_token_prob).item())
                
                # Append to generated tokens
                generated_tokens.append(next_token.item())
                
                # Update input sequence for next iteration
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return generated_tokens, log_probs
    
    def beam_search_generate(self, input_sequence, num_tokens=1, beam_width=3):
        """
        Generate tokens using beam search decoding
        
        Args:
            input_sequence: Input token IDs
            num_tokens: Number of tokens to generate
            beam_width: Beam width for search
            
        Returns:
            Best generated trajectory and its log probability
        """
        self.model.eval()
        
        # Convert input to tensor if needed
        if not isinstance(input_sequence, torch.Tensor):
            input_sequence = torch.tensor(input_sequence, dtype=torch.long, device=self.device)
        
        # Add batch dimension if needed
        if input_sequence.dim() == 1:
            input_sequence = input_sequence.unsqueeze(0)
        
        # Initialize beam with the input sequence
        beams = [(input_sequence.to(self.device), 0.0, [])]  # (input_ids, log_prob, generated_tokens)
        
        with torch.no_grad():
            for _ in range(num_tokens):
                new_beams = []
                
                for beam_input, beam_log_prob, beam_tokens in beams:
                    # Create padding mask
                    key_padding_mask = (beam_input == self.tokenizer.token2idx["[PAD]"])
                    
                    # Forward pass through the model
                    outputs = self.model(beam_input, key_padding_mask=key_padding_mask)
                    
                    # Handle the case where outputs is a tuple (logits, caches)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    # Get the last token prediction
                    logits = outputs[:, -1, :]
                    
                    # Calculate log probabilities
                    log_probs = torch.log_softmax(logits, dim=-1)
                    
                    # Get top-k next tokens and their probabilities
                    top_log_probs, top_indices = torch.topk(log_probs, beam_width)
                    
                    for i in range(beam_width):
                        next_token = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                        next_log_prob = top_log_probs[0, i].item()
                        new_log_prob = beam_log_prob + next_log_prob
                        
                        # Create new beam
                        new_input = torch.cat([beam_input, next_token], dim=1)
                        new_tokens = beam_tokens + [next_token.item()]
                        new_beams.append((new_input, new_log_prob, new_tokens))
                
                # Sort beams by log probability and keep top-k
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Return the best beam
        _, best_log_prob, best_tokens = beams[0]
        return best_tokens, best_log_prob
    
    def calculate_edit_distance(self, generated_seq, reference_seq):
        """Calculate Levenshtein edit distance between two token sequences"""
        # Convert token IDs to strings for Levenshtein
        gen_str = " ".join(str(t) for t in generated_seq)
        ref_str = " ".join(str(t) for t in reference_seq)
        
        # Calculate edit distance
        distance = Levenshtein.distance(gen_str, ref_str)
        
        # Normalize by reference length
        max_len = max(len(gen_str), len(ref_str))
        return distance, distance / max_len if max_len > 0 else 0
    
    def calculate_sequence_overlap(self, generated_seq, reference_seq):
        """Calculate percentage of correctly predicted tokens in proper order"""
        if not reference_seq:
            return 0.0
            
        correct = 0
        i, j = 0, 0
        
        while i < len(generated_seq) and j < len(reference_seq):
            if generated_seq[i] == reference_seq[j]:
                correct += 1
                i += 1
                j += 1
            else:
                j += 1
        
        return correct / len(reference_seq)
    
    def calculate_bleu(self, generated_seq, reference_seq):
        """Calculate BLEU score"""
        try:
            # Convert token IDs to strings
            gen_tokens = [str(t) for t in generated_seq]
            ref_tokens = [str(t) for t in reference_seq]
            
            # Calculate BLEU with smoothing
            bleu1 = sentence_bleu([ref_tokens], gen_tokens, 
                                 weights=(1, 0, 0, 0),
                                 smoothing_function=self.smoothing_function)
            bleu2 = sentence_bleu([ref_tokens], gen_tokens, 
                                 weights=(0.5, 0.5, 0, 0),
                                 smoothing_function=self.smoothing_function)
            
            return bleu1, bleu2
        except Exception as e:
            print(f"Error calculating BLEU: {e}")
            return 0.0, 0.0
    
    def calculate_rouge(self, generated_seq, reference_seq):
        """Calculate ROUGE scores"""
        try:
            # Convert token IDs to strings
            gen_str = " ".join(str(t) for t in generated_seq)
            ref_str = " ".join(str(t) for t in reference_seq)
            
            # Calculate ROUGE
            scores = self.rouge_scorer.score(ref_str, gen_str)
            
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def calculate_position_accuracy(self, generated_seq, reference_seq):
        """
        Calculate the accuracy of predictions at each position
        
        Returns:
            - position_correct: List of booleans indicating correctness at each position
            - position_accuracy: Accuracy at each position
        """
        min_len = min(len(generated_seq), len(reference_seq))
        position_correct = [generated_seq[i] == reference_seq[i] for i in range(min_len)]
        
        # If generated is shorter than reference, mark missing positions as incorrect
        if len(generated_seq) < len(reference_seq):
            position_correct.extend([False] * (len(reference_seq) - len(generated_seq)))
        
        position_accuracy = sum(position_correct) / len(reference_seq) if reference_seq else 0
        return position_correct, position_accuracy
    
    def calculate_ngram_metrics(self, generated_seq, reference_seq, n=2):
        """
        Calculate n-gram precision, recall, and F1 score
        
        Args:
            generated_seq: Generated token sequence
            reference_seq: Reference token sequence
            n: Size of n-grams
            
        Returns:
            Dictionary with precision, recall, and F1 score
        """
        if len(generated_seq) < n or len(reference_seq) < n:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Generate n-grams
        def get_ngrams(seq, n):
            return [tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)]
        
        gen_ngrams = get_ngrams(generated_seq, n)
        ref_ngrams = get_ngrams(reference_seq, n)
        
        # Count n-grams
        gen_counter = Counter(gen_ngrams)
        ref_counter = Counter(ref_ngrams)
        
        # Find common n-grams
        common_ngrams = sum((gen_counter & ref_counter).values())
        
        # Calculate metrics
        precision = common_ngrams / sum(gen_counter.values()) if gen_counter else 0
        recall = common_ngrams / sum(ref_counter.values()) if ref_counter else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def calculate_partial_match_metrics(self, generated_seq, reference_seq):
        """
        Calculate metrics for partially matching sequences
        
        Args:
            generated_seq: Generated token sequence
            reference_seq: Reference token sequence
            
        Returns:
            Dictionary with partial match metrics
        """
        # No tokens correct if either sequence is empty
        if not generated_seq or not reference_seq:
            return {
                'position_accuracy': 0.0,
                'longest_common_prefix': 0,
                'longest_common_prefix_percent': 0.0,
                'subsequence_match_percent': 0.0
            }
        
        # Position-wise accuracy
        _, position_accuracy = self.calculate_position_accuracy(generated_seq, reference_seq)
        
        # Longest common prefix
        common_prefix_len = 0
        for i in range(min(len(generated_seq), len(reference_seq))):
            if generated_seq[i] == reference_seq[i]:
                common_prefix_len += 1
            else:
                break
        
        # Longest common subsequence
        # Dynamic programming approach
        m, n = len(generated_seq), len(reference_seq)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if generated_seq[i-1] == reference_seq[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        longest_common_subseq = dp[m][n]
        
        return {
            'position_accuracy': position_accuracy,
            'longest_common_prefix': common_prefix_len,
            'longest_common_prefix_percent': common_prefix_len / len(reference_seq) if reference_seq else 0,
            'longest_common_subseq': longest_common_subseq,
            'subsequence_match_percent': longest_common_subseq / len(reference_seq) if reference_seq else 0
        }

    def evaluate_horizons(self, beam_width=3):
        """
        Evaluate model on horizons of different lengths
        
        For sequences of sufficient length, predict the next 1, 2, 3... tokens
        and evaluate using various metrics.
        """
        print("\n===== HORIZONTAL TRAJECTORY EVALUATION =====")
        
        # Keep track of metrics for each horizon
        horizon_metrics = {}
        
        # Extract sequences from validation set
        test_sequences = []
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['input_ids']
                mask = batch['attention_mask']
                
                # Extract sequences
                for i in range(inputs.size(0)):
                    seq_len = mask[i].sum().item()
                    if seq_len >= self.max_horizon + 1:
                        test_sequences.append(inputs[i, :seq_len].tolist())
        
        print(f"Found {len(test_sequences)} valid sequences for evaluation")
        
        # Limit the number of sequences for evaluation if too many
        max_eval_sequences = 100  # Increased from 5 to 100 for more robust evaluation
        if len(test_sequences) > max_eval_sequences:
            print(f"Limiting evaluation to {max_eval_sequences} sequences")
            import random
            random.seed(42)
            test_sequences = random.sample(test_sequences, max_eval_sequences)
        
        # Track metrics by input sequence length for analysis
        length_based_metrics = {h: {} for h in range(1, self.max_horizon + 1)}
        
        # Track token prediction correctness
        prediction_analysis = {h: {'correct': Counter(), 'incorrect': defaultdict(Counter)} 
                             for h in range(1, self.max_horizon + 1)}
        
        # Evaluate for different horizons
        for horizon in range(1, self.max_horizon + 1):
            print(f"\n--- Evaluating Horizon {horizon} ---")
            
            # Initialize metrics
            metrics = {
                'perplexity': [],
                'log_likelihood': [],
                'edit_distance': [],
                'edit_distance_norm': [],
                'sequence_overlap': [],
                'bleu1': [],
                'bleu2': [],
                'rouge1': [],
                'rouge2': [],
                'rougeL': [],
                'greedy_exact_match': 0,
                'beam_exact_match': 0,
                'position_accuracy': [],
                'partial_match': {'common_prefix': [], 'common_prefix_percent': [], 'subseq_match_percent': []},
                'ngram_metrics': {'precision': [], 'recall': [], 'f1': []},
                'examples': []
            }
            
            # Process each sequence
            for seq in tqdm(test_sequences, desc=f"Horizon {horizon}"):
                if len(seq) < horizon + 1:
                    continue
                
                # Split into input and target
                input_seq = seq[:-horizon]
                target_seq = seq[-horizon:]
                
                # Skip if either input or target contains padding
                if self.tokenizer.token2idx["[PAD]"] in input_seq or self.tokenizer.token2idx["[PAD]"] in target_seq:
                    continue
                
                # Generate tokens - greedy decoding
                generated_tokens, log_probs = self.greedy_generate(input_seq, num_tokens=horizon)
                
                # Generate tokens - beam search
                beam_tokens, beam_log_prob = self.beam_search_generate(
                    input_seq, num_tokens=horizon, beam_width=beam_width)
                
                # Check for exact matches
                greedy_exact_match = generated_tokens == target_seq
                beam_exact_match = beam_tokens == target_seq
                
                if greedy_exact_match:
                    metrics['greedy_exact_match'] += 1
                if beam_exact_match:
                    metrics['beam_exact_match'] += 1
                
                # Calculate perplexity
                perplexity = np.exp(-np.mean(log_probs))
                log_likelihood = np.sum(log_probs)
                
                # Calculate trajectory accuracy metrics
                edit_dist, edit_dist_norm = self.calculate_edit_distance(generated_tokens, target_seq)
                seq_overlap = self.calculate_sequence_overlap(generated_tokens, target_seq)
                bleu1, bleu2 = self.calculate_bleu(generated_tokens, target_seq)
                rouge_scores = self.calculate_rouge(generated_tokens, target_seq)
                
                # Calculate partial match metrics
                _, position_accuracy = self.calculate_position_accuracy(generated_tokens, target_seq)
                partial_metrics = self.calculate_partial_match_metrics(generated_tokens, target_seq)
                ngram_metrics = self.calculate_ngram_metrics(generated_tokens, target_seq, n=2)
                
                # Track token predictions for analysis
                for i, (pred, true) in enumerate(zip(generated_tokens, target_seq)):
                    if i < len(generated_tokens) and i < len(target_seq):
                        # Convert tokens to human-readable strings
                        pred_token = self.tokenizer.idx2token.get(pred, '[UNK]')
                        true_token = self.tokenizer.idx2token.get(true, '[UNK]')
                        
                        if pred == true:
                            prediction_analysis[horizon]['correct'][true_token] += 1
                        else:
                            prediction_analysis[horizon]['incorrect'][true_token][pred_token] += 1
                
                # Store metrics
                metrics['perplexity'].append(perplexity)
                metrics['log_likelihood'].append(log_likelihood)
                metrics['edit_distance'].append(edit_dist)
                metrics['edit_distance_norm'].append(edit_dist_norm)
                metrics['sequence_overlap'].append(seq_overlap)
                metrics['bleu1'].append(bleu1)
                metrics['bleu2'].append(bleu2)
                metrics['rouge1'].append(rouge_scores['rouge1'])
                metrics['rouge2'].append(rouge_scores['rouge2'])
                metrics['rougeL'].append(rouge_scores['rougeL'])
                metrics['position_accuracy'].append(position_accuracy)
                metrics['partial_match']['common_prefix'].append(partial_metrics['longest_common_prefix'])
                metrics['partial_match']['common_prefix_percent'].append(partial_metrics['longest_common_prefix_percent'])
                metrics['partial_match']['subseq_match_percent'].append(partial_metrics['subsequence_match_percent'])
                metrics['ngram_metrics']['precision'].append(ngram_metrics['precision'])
                metrics['ngram_metrics']['recall'].append(ngram_metrics['recall'])
                metrics['ngram_metrics']['f1'].append(ngram_metrics['f1'])
                
                # Track metrics by input length
                input_length = len(input_seq)
                if input_length not in length_based_metrics[horizon]:
                    length_based_metrics[horizon][input_length] = {
                        'count': 0,
                        'perplexity': [],
                        'sequence_overlap': [],
                        'bleu1': [],
                        'greedy_exact_match': 0,
                        'beam_exact_match': 0,
                        'position_accuracy': [],
                        'subseq_match_percent': []
                    }
                
                length_data = length_based_metrics[horizon][input_length]
                length_data['count'] += 1
                length_data['perplexity'].append(perplexity)
                length_data['sequence_overlap'].append(seq_overlap)
                length_data['bleu1'].append(bleu1)
                length_data['position_accuracy'].append(position_accuracy)
                length_data['subseq_match_percent'].append(partial_metrics['subsequence_match_percent'])
                if greedy_exact_match:
                    length_data['greedy_exact_match'] += 1
                if beam_exact_match:
                    length_data['beam_exact_match'] += 1
                
                # Store examples for display
                if len(metrics['examples']) < 10:  # Increased from 5 to 10
                    input_tokens = [self.tokenizer.idx2token.get(idx, '[UNK]') for idx in input_seq]
                    target_tokens = [self.tokenizer.idx2token.get(idx, '[UNK]') for idx in target_seq]
                    generated_token_strs = [self.tokenizer.idx2token.get(idx, '[UNK]') for idx in generated_tokens]
                    beam_token_strs = [self.tokenizer.idx2token.get(idx, '[UNK]') for idx in beam_tokens]
                    
                    metrics['examples'].append({
                        'input': input_tokens,
                        'target': target_tokens,
                        'greedy_generated': generated_token_strs,
                        'beam_generated': beam_token_strs,
                        'greedy_match': greedy_exact_match,
                        'beam_match': beam_exact_match,
                        'perplexity': perplexity,
                        'seq_overlap': seq_overlap,
                        'bleu1': bleu1,
                        'position_accuracy': position_accuracy,
                        'common_prefix': partial_metrics['longest_common_prefix'],
                        'subsequence_match': partial_metrics['subsequence_match_percent']
                    })
            
            # Process length-based metrics
            for input_length, data in length_based_metrics[horizon].items():
                if data['count'] > 0:
                    data['avg_perplexity'] = np.mean(data['perplexity'])
                    data['avg_sequence_overlap'] = np.mean(data['sequence_overlap'])
                    data['avg_bleu1'] = np.mean(data['bleu1'])
                    data['avg_position_accuracy'] = np.mean(data['position_accuracy'])
                    data['avg_subseq_match'] = np.mean(data['subseq_match_percent'])
                    data['greedy_match_rate'] = data['greedy_exact_match'] / data['count']
                    data['beam_match_rate'] = data['beam_exact_match'] / data['count']
            
            # Calculate average metrics
            total_sequences = len(metrics['perplexity'])
            avg_metrics = {
                'perplexity': np.mean(metrics['perplexity']) if metrics['perplexity'] else float('inf'),
                'log_likelihood': np.mean(metrics['log_likelihood']) if metrics['log_likelihood'] else float('-inf'),
                'edit_distance': np.mean(metrics['edit_distance']) if metrics['edit_distance'] else 0,
                'edit_distance_norm': np.mean(metrics['edit_distance_norm']) if metrics['edit_distance_norm'] else 0,
                'sequence_overlap': np.mean(metrics['sequence_overlap']) if metrics['sequence_overlap'] else 0,
                'bleu1': np.mean(metrics['bleu1']) if metrics['bleu1'] else 0,
                'bleu2': np.mean(metrics['bleu2']) if metrics['bleu2'] else 0,
                'rouge1': np.mean(metrics['rouge1']) if metrics['rouge1'] else 0,
                'rouge2': np.mean(metrics['rouge2']) if metrics['rouge2'] else 0,
                'rougeL': np.mean(metrics['rougeL']) if metrics['rougeL'] else 0,
                'position_accuracy': np.mean(metrics['position_accuracy']) if metrics['position_accuracy'] else 0,
                'common_prefix_percent': np.mean(metrics['partial_match']['common_prefix_percent']) if metrics['partial_match']['common_prefix_percent'] else 0,
                'subseq_match_percent': np.mean(metrics['partial_match']['subseq_match_percent']) if metrics['partial_match']['subseq_match_percent'] else 0,
                'ngram_precision': np.mean(metrics['ngram_metrics']['precision']) if metrics['ngram_metrics']['precision'] else 0,
                'ngram_recall': np.mean(metrics['ngram_metrics']['recall']) if metrics['ngram_metrics']['recall'] else 0,
                'ngram_f1': np.mean(metrics['ngram_metrics']['f1']) if metrics['ngram_metrics']['f1'] else 0,
                'greedy_exact_match_rate': metrics['greedy_exact_match'] / total_sequences if total_sequences > 0 else 0,
                'beam_exact_match_rate': metrics['beam_exact_match'] / total_sequences if total_sequences > 0 else 0,
                'total_sequences': total_sequences,
                'examples': metrics['examples']
            }
            
            # Print results
            print(f"Total sequences evaluated: {total_sequences}")
            print(f"Perplexity: {avg_metrics['perplexity']:.4f}")
            print(f"Log-likelihood: {avg_metrics['log_likelihood']:.4f}")
            print(f"Edit Distance: {avg_metrics['edit_distance']:.4f}")
            print(f"Normalized Edit Distance: {avg_metrics['edit_distance_norm']:.4f}")
            print(f"Sequence Overlap: {avg_metrics['sequence_overlap']:.4f}")
            print(f"BLEU-1: {avg_metrics['bleu1']:.4f}")
            print(f"BLEU-2: {avg_metrics['bleu2']:.4f}")
            print(f"ROUGE-1: {avg_metrics['rouge1']:.4f}")
            print(f"ROUGE-2: {avg_metrics['rouge2']:.4f}")
            print(f"ROUGE-L: {avg_metrics['rougeL']:.4f}")
            
            # Print partial match metrics
            print("\nPartial Matching Metrics:")
            print(f"Position-wise Accuracy: {avg_metrics['position_accuracy']:.4f}")
            print(f"Common Prefix %: {avg_metrics['common_prefix_percent']:.4f}")
            print(f"Subsequence Match %: {avg_metrics['subseq_match_percent']:.4f}")
            print(f"Bigram Precision: {avg_metrics['ngram_precision']:.4f}")
            print(f"Bigram Recall: {avg_metrics['ngram_recall']:.4f}")
            print(f"Bigram F1: {avg_metrics['ngram_f1']:.4f}")
            
            # Print exact match rates
            print(f"\nExact Match (Greedy): {metrics['greedy_exact_match']} / {total_sequences} = {avg_metrics['greedy_exact_match_rate']:.4f}")
            print(f"Exact Match (Beam): {metrics['beam_exact_match']} / {total_sequences} = {avg_metrics['beam_exact_match_rate']:.4f}")
            
            # Print metrics by input length
            if length_based_metrics[horizon]:
                print("\nPerformance by Input Sequence Length:")
                print(f"{'Input Length':^12} | {'Count':^6} | {'PPL':^8} | {'Pos Acc':^8} | {'Subseq %':^8} | {'Greedy Match':^12}")
                print("-" * 70)
                for length in sorted(length_based_metrics[horizon].keys()):
                    data = length_based_metrics[horizon][length]
                    if data['count'] > 0:
                        print(f"{length:^12} | {data['count']:^6} | {data['avg_perplexity']:8.4f} | {data['avg_position_accuracy']:8.4f} | {data['avg_subseq_match']:8.4f} | {data['greedy_match_rate']:12.4f}")
            
            # Analyze token predictions
            print("\nToken Prediction Analysis:")
            
            # Most frequent correct predictions
            correct_preds = prediction_analysis[horizon]['correct']
            if correct_preds:
                print("\nMost Common Correctly Predicted Tokens:")
                for token, count in correct_preds.most_common(5):
                    print(f"  {token}: {count} times")
            
            # Most frequent incorrect predictions
            incorrect_preds = prediction_analysis[horizon]['incorrect']
            if incorrect_preds:
                print("\nMost Common Incorrect Predictions:")
                for true_token, pred_counter in sorted(incorrect_preds.items(), 
                                                     key=lambda x: sum(x[1].values()), 
                                                     reverse=True)[:5]:
                    print(f"  True: {true_token} → Predicted as:")
                    for pred_token, count in pred_counter.most_common(3):
                        print(f"    {pred_token}: {count} times")
            
            # Print examples
            print("\nExamples:")
            for i, example in enumerate(avg_metrics['examples']):
                print(f"\nExample {i+1}:")
                print(f"Input: {' '.join(example['input'])}")
                print(f"Target: {' '.join(example['target'])}")
                print(f"Greedy: {' '.join(example['greedy_generated'])} (PPL: {example['perplexity']:.2f}, BLEU: {example['bleu1']:.2f})")
                print(f"Beam: {' '.join(example['beam_generated'])}")
                
                # Show match status with more details
                greedy_match = "✓" if example['greedy_match'] else "✗"
                beam_match = "✓" if example['beam_match'] else "✗"
                print(f"Greedy match: {greedy_match}, Position Accuracy: {example['position_accuracy']:.2f}, Subseq Match: {example['subsequence_match']:.2f}")
                print(f"Beam match: {beam_match}")
            
            # Store metrics for this horizon
            horizon_metrics[horizon] = avg_metrics
        
        # Print comprehensive summary of all horizons
        print("\n===== HORIZONTAL EVALUATION SUMMARY =====")
        
        # Table 1: Basic metrics
        print("\nBasic Performance Metrics:")
        print(f"{'Horizon':^7} | {'Perplexity':^10} | {'Seq Overlap':^11} | {'BLEU-1':^6} | {'ROUGE-L':^7} | {'Exact Match':^11}")
        print("-" * 65)
        for horizon in range(1, self.max_horizon + 1):
            metrics = horizon_metrics[horizon]
            print(f"{horizon:^7} | {metrics['perplexity']:10.4f} | {metrics['sequence_overlap']:11.4f} | {metrics['bleu1']:6.4f} | {metrics['rougeL']:7.4f} | {metrics['greedy_exact_match_rate']:11.4f}")
        
        # Table 2: Partial matching metrics
        print("\nPartial Match Performance:")
        print(f"{'Horizon':^7} | {'Pos Accuracy':^12} | {'Prefix %':^10} | {'Subseq %':^10} | {'Bigram F1':^10}")
        print("-" * 65)
        for horizon in range(1, self.max_horizon + 1):
            metrics = horizon_metrics[horizon]
            print(f"{horizon:^7} | {metrics['position_accuracy']:12.4f} | {metrics['common_prefix_percent']:10.4f} | {metrics['subseq_match_percent']:10.4f} | {metrics['ngram_f1']:10.4f}")
        
        # Table 3: Comparison of greedy vs beam search
        print("\nGreedy vs Beam Search Performance:")
        print(f"{'Horizon':^7} | {'Greedy Match':^12} | {'Beam Match':^10} | {'Greedy PPL':^10} | {'Edit Distance':^12}")
        print("-" * 65)
        for horizon in range(1, self.max_horizon + 1):
            metrics = horizon_metrics[horizon]
            print(f"{horizon:^7} | {metrics['greedy_exact_match_rate']:12.4f} | {metrics['beam_exact_match_rate']:10.4f} | {metrics['perplexity']:10.4f} | {metrics['edit_distance_norm']:12.4f}")
        
        # Calculate performance changes with increasing horizon
        if self.max_horizon > 1:
            print("\nPerformance Change with Increasing Horizon:")
            print(f"{'Metric':^15} | {'H1→H2':^7} | {'H2→H3':^7}")
            print("-" * 35)
            
            metrics = {
                'Perplexity': [m['perplexity'] for m in [horizon_metrics[h] for h in range(1, self.max_horizon + 1)]],
                'Pos Accuracy': [m['position_accuracy'] for m in [horizon_metrics[h] for h in range(1, self.max_horizon + 1)]],
                'Subseq Match': [m['subseq_match_percent'] for m in [horizon_metrics[h] for h in range(1, self.max_horizon + 1)]],
                'Exact Match': [m['greedy_exact_match_rate'] for m in [horizon_metrics[h] for h in range(1, self.max_horizon + 1)]]
            }
            
            for metric_name, values in metrics.items():
                changes = []
                for i in range(len(values) - 1):
                    if metric_name == 'Perplexity':
                        # For perplexity, lower is better, so we want the inverse of the ratio
                        change = (values[i] - values[i+1]) / values[i] if values[i] != 0 else 0
                    else:
                        # For other metrics, higher is better
                        change = (values[i+1] - values[i]) / values[i] if values[i] != 0 else 0
                    changes.append(change)
                
                # Format changes as percentages
                change_strs = [f"{c*100:+7.1f}%" for c in changes]
                print(f"{metric_name:^15} | {change_strs[0]:^7} | {change_strs[1] if len(change_strs) > 1 else '':^7}")
                        
        return horizon_metrics

def main():
    # Create evaluator with the specified model and tokenizer files
    evaluator = HorizontalEvaluator("model_full.pth", "tokenizer.pkl", max_horizon=3)
    
    # Run the horizontal evaluation with a beam width of 3
    horizon_metrics = evaluator.evaluate_horizons(beam_width=3)

if __name__ == "__main__":
    main() 