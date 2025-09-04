# tokenizer.py
import re
import os
import json
import torch
import regex
from typing import List, Dict, Any, Union, Tuple, Optional
import ast
from functools import lru_cache

@lru_cache()
def bytes_to_unicode():
    """
    Returns a dict mapping utf-8 bytes to unique unicode strings (GPT-2 style).
    Ensures every possible byte is mapped to a single character.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word: Tuple[str, ...]) -> set:
    """
    Return the set of adjacent symbol pairs in a word (tuple of symbols).
    Used in the BPE merge loop.
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class CCSRSequenceTokenizer:
    def __init__(self, mode: str = "whole_word", max_length: int = 512, 
                 vocab_file: Optional[str] = None, merges_file: Optional[str] = None,
                 add_prefix_space: bool = False) -> None:
        """
        Args:
            mode: Tokenization mode ("whole_word", "letter_based", "bpe", or "split").
            max_length: Maximum sequence length.
            vocab_file: Path to vocabulary file for BPE mode.
            merges_file: Path to merges file for BPE mode.
            add_prefix_space: Whether to add prefix space for BPE tokenization.
        """
        self.mode = mode
        self.max_length = max_length
        self.special_tokens: Dict[str, str] = {
            'PAD': '[PAD]',
            'UNK': '[UNK]',
            'CLS': '[CLS]',
            'SEP': '[SEP]',
            'MASK': '[MASK]',
            'CODE_START': '[CODE]',
            'CODE_END': '[/CODE]'
        }
        # New tokens for SOS/EOS mode.
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"

        self.vocab_size: int = 0
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        
        # For BPE tokenization
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.add_prefix_space = add_prefix_space
        
        # For custom BPE
        if mode == "bpe":
            self.byte_encoder = bytes_to_unicode()
            self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
            self.encoder = {}  # str -> token id
            self.decoder = {}  # token id -> str
            self.bpe_ranks = {}  # merge priority
            self.bpe_cache = {}  # token -> merged result
            
            # A regex pattern for BPE tokenization (similar to GPT-2)
            self.bpe_pat = regex.compile(
                r"(?:'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)",
                flags=regex.IGNORECASE | regex.UNICODE,
            )
            
            # Initialize BPE if vocab and merges files are provided
            if self.vocab_file and self.merges_file:
                self._init_bpe()


    def preprocess_data(self, raw_text: str) -> List[List[str]]:
        data = ast.literal_eval(raw_text)
        sequences = [
            [token.strip() for token in seq.split(',') if token.strip()]
            for seq in data
        ]
        # Filter out any sequences that are empty.
        return [seq for seq in sequences if len(seq) > 0]





    def _init_bpe(self):
        """Load BPE vocabulary and merges"""
        # Load the encoder (JSON)
        with open(self.vocab_file, "r", encoding="utf-8") as vf:
            self.encoder = json.load(vf)
        # Build the decoder
        self.decoder = {int(v): k for k, v in self.encoder.items()}

        # Load merges
        with open(self.merges_file, "r", encoding="utf-8") as mf:
            merges = mf.read().split("\n")
            # Skip the first line if it's a version line
            if merges[0].startswith('#version'):
                merges = merges[1:]
            # Remove any empty lines at the end
            merges = [m for m in merges if m.strip()]
        merges = [tuple(m.split()) for m in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
    
    def train_bpe_from_corpus(self, sequences: List[List[str]], vocab_size: int = 5000, min_frequency: int = 2):
        """Train BPE from scratch using the provided corpus"""
        try:
            from tokenizers import ByteLevelBPETokenizer
        except ImportError:
            raise ImportError("The 'tokenizers' library is required for BPE training. Install it with 'pip install tokenizers'.")
        
        # Prepare corpus file
        os.makedirs("temp_bpe", exist_ok=True)
        corpus_file = "temp_bpe/corpus.txt"
        with open(corpus_file, "w", encoding="utf-8") as f:
            for sequence in sequences:
                f.write(" ".join(sequence) + "\n")
        
        # Train tokenizer
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=[corpus_file],
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(self.special_tokens.values()) + [self.sos_token, self.eos_token]
        )
        
        # Save trained files
        tokenizer.save_model("temp_bpe", "my_bpe")
        
        # Update paths and mode
        self.vocab_file = "temp_bpe/my_bpe-vocab.json"
        self.merges_file = "temp_bpe/my_bpe-merges.txt"
        self.mode = "bpe"
        self._init_bpe()
    
    def _tokenize_bpe(self, text: str) -> List[str]:
        """GPT-2 style Byte-Pair Encoding tokenization"""
        tokens = []
        if self.add_prefix_space and not text.startswith(" "):
            text = " " + text

        for tok in self.bpe_pat.findall(text):
            token_bytes = tok.encode("utf-8")
            token_transformed = "".join(self.byte_encoder[b] for b in token_bytes)
            merged = self._bpe(token_transformed).split(" ")
            tokens.extend(merged)
        return tokens

    def _bpe(self, token: str) -> str:
        """Merge subword pairs using self.bpe_ranks"""
        if token in self.bpe_cache:
            return self.bpe_cache[token]

        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            self.bpe_cache[token] = token
            return token

        while True:
            bigram = min(pairs, key=lambda x: self.bpe_ranks.get(x, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        merged = " ".join(word)
        self.bpe_cache[token] = merged
        return merged
    
    def tokenize_sequence(self, sequence: List[str]) -> List[str]:
        if self.mode == "whole_word":
            return sequence
        elif self.mode == "letter_based":
            return [char for word in sequence for char in word]
        elif self.mode == "bpe":
            # For BPE, we need to join the sequence and retokenize
            text = " ".join(sequence)
            return self._tokenize_bpe(text)
        elif self.mode == "split":
            pattern = r'[A-Za-z]+\d+|[A-Za-z]+|\d+|\.'
            return [token for word in sequence for token in re.findall(pattern, word)]
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def build_vocab(self, sequences: List[List[str]]) -> None:
        vocab: Dict[str, int] = {token: idx for idx, token in enumerate(self.special_tokens.values())}
        # if "<SOS>" not in vocab:
        #     vocab["<SOS>"] = len(vocab)
        # if "<EOS>" not in vocab:
        #     vocab["<EOS>"] = len(vocab)
        unique_tokens = set()
        for sequence in sequences:
            unique_tokens.update(self.tokenize_sequence(sequence))
        for token in sorted(unique_tokens):
            if token not in vocab:
                vocab[token] = len(vocab)
        self.token2idx = vocab
        self.idx2token = {idx: token for token, idx in vocab.items()}
        self.vocab_size = len(vocab)

    def encode_sequence(self, sequence: List[str], add_special_tokens: bool = False,
                        special_token_mode: str = "default") -> List[int]:

        if self.mode == "bpe":
            # For BPE, we rely on the encoder dictionary from training
            text = " ".join(sequence)
            tokens = self._tokenize_bpe(text)
            # Convert tokens to IDs using the BPE encoder
            token_ids = [int(self.encoder.get(token, self.encoder.get(self.special_tokens['UNK'], 0))) for token in tokens]
            
            # Add special tokens if needed
            if add_special_tokens:
                if special_token_mode == "sos_eos":
                    sos_id = int(self.encoder.get(self.sos_token, 0))
                    eos_id = int(self.encoder.get(self.eos_token, 0))
                    token_ids = [sos_id] + token_ids + [eos_id]
                else:
                    cls_id = int(self.encoder.get(self.special_tokens['CLS'], 0))
                    sep_id = int(self.encoder.get(self.special_tokens['SEP'], 0))
                    token_ids = [cls_id] + token_ids + [sep_id]
            return token_ids
        else:
            # Original behavior for non-BPE modes
            tokenized_seq = self.tokenize_sequence(sequence)
            if add_special_tokens:
                if special_token_mode == "sos_eos":
                    tokens = [self.sos_token] + tokenized_seq + [self.eos_token]
                else:
                    tokens = [self.special_tokens['CLS']] + tokenized_seq + [self.special_tokens['SEP']]
            else:
                tokens = tokenized_seq
            return [self.token2idx.get(token, self.token2idx[self.special_tokens['UNK']]) for token in tokens]

  
    def prepare_transformer_inputs(self, sequences: List[List[str]], windowing_strategy: str = "fixed",
                                   window_size: int = None, overlap: int = 0, target_mode: str = "shifted") -> Dict[str, torch.Tensor]:
        if windowing_strategy == "fixed":
            return self._prepare_fixed_inputs(sequences, target_mode=target_mode)
        elif windowing_strategy == "fixed_with_eos_sos":
            return self._prepare_fixed_with_eos_sos_inputs(sequences, target_mode=target_mode)
        elif windowing_strategy == "progressive_dynamic":
            return self._prepare_progressive_dynamic_inputs(sequences, target_mode=target_mode)
        elif windowing_strategy == "aggressive_dynamic":
            return self._prepare_aggressive_dynamic_inputs(sequences, target_mode=target_mode)
        elif windowing_strategy == "fixed_longest":
            return self._prepare_fixed_longest_inputs(sequences, target_mode=target_mode)
        # elif windowing_strategy == "dynamic":
        #     return self._prepare_dynamic_inputs(sequences, target_mode=target_mode)
        elif windowing_strategy == "sliding":
            if window_size is None:
                raise ValueError("window_size must be specified for sliding window strategy.")
            return self._prepare_sliding_inputs(sequences, window_size, overlap, target_mode=target_mode)
        else:
            raise ValueError(f"Unsupported windowing strategy: {windowing_strategy}")

 
    def _prepare_fixed_inputs(self, sequences: List[List[str]], target_mode: str = "shifted") -> Dict[str, torch.Tensor]:
        processed_inputs = []
        processed_targets = []
        processed_attention = []
        for seq in [self.encode_sequence(s, add_special_tokens=True) for s in sequences]:
            if len(seq) < self.max_length:
                seq = seq + [self.token2idx[self.special_tokens['PAD']]] * (self.max_length - len(seq))
            else:
                seq = seq[:self.max_length]
            attn = [1 if token != self.token2idx[self.special_tokens['PAD']] else 0 for token in seq]
            if target_mode == "shifted":
                input_seq = seq[:-1]
                target_seq = seq[1:]
                attn = attn[:-1]
            elif target_mode == "next_token":
                input_seq = seq[:-1]
                target_seq = [seq[-1]]
                attn = attn[:-1]
            else:
                raise ValueError(f"Unsupported target mode: {target_mode}")
            processed_inputs.append(input_seq)
            processed_targets.append(target_seq)
            processed_attention.append(attn)
        return {
            'input_ids': torch.tensor(processed_inputs, dtype=torch.long),
            'attention_mask': torch.tensor(processed_attention, dtype=torch.long),
            'targets': torch.tensor(processed_targets, dtype=torch.long)
        }


    def _prepare_fixed_with_eos_sos_inputs(self, sequences: List[List[str]], 
                                         target_mode: str = "shifted", 
                                         window_size: int = 16) -> Dict[str, torch.Tensor]:
        PAD_idx = self.token2idx[self.special_tokens['PAD']]
        
        continuous_tokens = []
        for s in sequences:
            encoded = self.encode_sequence(s, add_special_tokens=True, special_token_mode="sos_eos")
            continuous_tokens.extend(encoded)
        
        windows = []
        for i in range(0, len(continuous_tokens), window_size):
            window = continuous_tokens[i:i+window_size]
            if len(window) < window_size:
                window = window + [PAD_idx] * (window_size - len(window))
            windows.append(window)
        
        if target_mode == "next_token":

            input_windows = []
            target_values = []
            attention_windows = []
            for window in windows:
                
                input_window = window[:-1]
                
                target_value = window[-1]
                input_windows.append(input_window)
                target_values.append(target_value)
                
                
                attn = [1 if token != PAD_idx else 0 for token in input_window]
                attention_windows.append(attn)
            
            input_ids = torch.tensor(input_windows, dtype=torch.long)  # shape: (num_windows, window_size - 1)
            attention_mask = torch.tensor(attention_windows, dtype=torch.long)  # same shape as input_ids
            targets = torch.tensor(target_values, dtype=torch.long)  # shape: (num_windows,)
        
        else: 
        
            shifted_tokens = continuous_tokens[1:]
            remainder = len(shifted_tokens) % window_size
            if remainder != 0:
                shifted_tokens = shifted_tokens + [PAD_idx] * (window_size - remainder)
            
            target_windows = []
            for i in range(0, len(shifted_tokens), window_size):
                
                target_window = shifted_tokens[i:i+window_size]
                
                target_windows.append(target_window)
            
            attention_windows = []
            for window in windows:
                attn = [1 if token != PAD_idx else 0 for token in window]
                attention_windows.append(attn)
            
            input_ids = torch.tensor(windows, dtype=torch.long)
            attention_mask = torch.tensor(attention_windows, dtype=torch.long)
            
            targets = torch.tensor(target_windows, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            
            'attention_mask': attention_mask,
            'targets': targets
        }


    def _prepare_progressive_dynamic_inputs(self, sequences: List[List[str]], target_mode: str = "next_token") -> Dict[str, Any]:
        inputs = []
        targets = []
        attention_masks = []
        for s in sequences:
            # Get the tokenized and encoded sequence (with special tokens if desired)
            encoded = self.encode_sequence(s, add_special_tokens=False)
            # Generate progressive pairs.
            # Start at index 1 so that the input (prefix) is non-empty.
            for i in range(1, len(encoded)):
                # The prefix up to i becomes the input
                input_seq = encoded[:i]
                # The token at position i is the target
                target_token = encoded[i]
                inputs.append(input_seq)
                targets.append(target_token)
                # Attention mask: all ones (length equals the length of the input sequence)
                attention_masks.append([1] * len(input_seq))
        return {
            'input_ids': inputs,         # List of lists (variable-length)
            'attention_mask': attention_masks,
            'targets': targets           # List of integers
        }

    def _prepare_aggressive_dynamic_inputs(self, sequences: List[List[str]], target_mode: str = "next_token") -> Dict[str, Any]:
        """
        Aggressive dynamic windowing strategy that generates maximum training examples:
        1. Progressive windowing (like progressive_dynamic)
        2. Overlapping sliding windows of different sizes
        3. Reverse sequence learning
        4. Multi-target predictions where possible
        
        This creates much more training data from the same sequences.
        """
        inputs = []
        targets = []
        attention_masks = []
        
        for s in sequences:
            # Get the tokenized and encoded sequence
            encoded = self.encode_sequence(s, add_special_tokens=False)
            seq_len = len(encoded)
            
            if seq_len < 2:
                continue
            
            # 1. Progressive windowing (original progressive_dynamic)
            for i in range(1, seq_len):
                input_seq = encoded[:i]
                target_token = encoded[i]
                inputs.append(input_seq)
                targets.append(target_token)
                attention_masks.append([1] * len(input_seq))
            
            # 2. Sliding windows of different sizes (creates overlapping contexts)
            for window_size in range(2, min(seq_len, self.max_length)):
                for start_idx in range(seq_len - window_size):
                    # Create window
                    window = encoded[start_idx:start_idx + window_size]
                    # Input is all but last token, target is last token
                    input_seq = window[:-1]
                    target_token = window[-1]
                    
                    # Avoid duplicates from progressive windowing
                    if start_idx > 0 or len(input_seq) > 1:
                        inputs.append(input_seq)
                        targets.append(target_token)
                        attention_masks.append([1] * len(input_seq))
            
            # 3. Reverse sequence learning (learn to predict previous tokens)
            # This helps the model understand bidirectional dependencies
            if seq_len > 2:  # Only for longer sequences
                for i in range(seq_len - 2, 0, -1):  # Reverse direction
                    # Use suffix to predict earlier token
                    input_seq = encoded[i:]  # From position i to end
                    target_token = encoded[i-1]  # Predict token before the suffix
                    
                    inputs.append(input_seq)
                    targets.append(target_token)
                    attention_masks.append([1] * len(input_seq))
            
            # 4. Multi-position predictions (predict middle tokens from context)
            if seq_len >= 4:  # Need at least 4 tokens for this
                for target_pos in range(1, seq_len - 1):  # Don't predict first or last
                    # Context: tokens before and after the target
                    context_before = encoded[:target_pos]
                    context_after = encoded[target_pos + 1:]
                    
                    # Create input by concatenating before and after (skip target)
                    if len(context_before) > 0 and len(context_after) > 0:
                        input_seq = context_before + context_after
                        target_token = encoded[target_pos]
                        
                        # Only add if input is not too long
                        if len(input_seq) <= self.max_length - 1:
                            inputs.append(input_seq)
                            targets.append(target_token)
                            attention_masks.append([1] * len(input_seq))
        
        print(f"Aggressive dynamic generated {len(inputs)} training examples from {len(sequences)} sequences")
        print(f"Expansion ratio: {len(inputs) / len(sequences):.1f}x")
        
        return {
            'input_ids': inputs,
            'attention_mask': attention_masks,
            'targets': targets
        }

    def _prepare_fixed_longest_inputs(self, sequences: List[List[str]], target_mode: str = "next_token") -> Dict[str, torch.Tensor]:
        encoded_sequences = [self.encode_sequence(s, add_special_tokens=False) for s in sequences]
        
        
        longest = max(len(seq) for seq in encoded_sequences)
        fixed_input_length = longest - 1
        
        inputs = []
        targets = []
        
        attention_masks = []
        PAD_idx = self.token2idx[self.special_tokens['PAD']]
        
        special_indices = {self.token2idx[token] for token in self.special_tokens.values()}
        
        for seq in encoded_sequences:
            input_tokens = seq[:-1]
            target_token = None
            
            for token in reversed(seq):
                
                if token not in special_indices:
                    target_token = token
                    
                    break
            if target_token is None:
                
                raise ValueError("TARGET")
            
            pad_length = fixed_input_length - len(input_tokens)
            padded_input = input_tokens + [PAD_idx] * pad_length
            
            attn = [1] * len(input_tokens) + [0] * pad_length
            
            
            inputs.append(padded_input)
            targets.append(target_token)
            attention_masks.append(attn)
        
        return {
            'input_ids': torch.tensor(inputs, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
            
            'targets': torch.tensor(targets, dtype=torch.long)
        }
    def _prepare_sliding_inputs(self, sequences: List[List[str]], window_size: int, overlap: int,
                                  target_mode: str = "shifted") -> Dict[str, torch.Tensor]:
        def split_into_windows(sequence: List[int], window_size: int, overlap: int) -> List[List[int]]:
            step = window_size - overlap
            if step <= 0:
                raise ValueError("Overlap must be less than the window size.")
            windows = [sequence[i:i + window_size] for i in range(0, len(sequence), step)]
            
            return [window + [self.token2idx[self.special_tokens['PAD']]] * (window_size - len(window))
                    if len(window) < window_size else window for window in windows]
        def generate_attention_mask(window: List[int]) -> List[int]:
            return [1 if token != self.token2idx[self.special_tokens['PAD']] else 0 for token in window]
        encoded_sequences = [self.encode_sequence(s, add_special_tokens=True) for s in sequences]
        all_windows, all_attention, all_targets = [], [], []
        for seq in encoded_sequences:
            
            windows = split_into_windows(seq, window_size, overlap)
            for window in windows:
                attn = generate_attention_mask(window)
                if target_mode == "shifted":
                    input_seq = window[:-1]
                    target_seq = window[1:]
                    attn_input = attn[:-1]
                elif target_mode == "next_token":
                    input_seq = window[:-1]
                    
                    target_seq = [window[-1]]
                    attn_input = attn[:-1]
                else:
                    raise ValueError(f"Unsupported target mode: {target_mode}")
                all_windows.append(input_seq)
                all_targets.append(target_seq)
                all_attention.append(attn_input)
        return {
            'input_ids': torch.tensor(all_windows, dtype=torch.long),
            'attention_mask': torch.tensor(all_attention, dtype=torch.long),
            'targets': torch.tensor(all_targets, dtype=torch.long)
        }

    def decode_sequence(self, indices: Union[torch.Tensor, List[int], int], filter_special: bool = False) -> List[str]:
        if isinstance(indices, int):
            indices = [indices]
            
        if torch.is_tensor(indices):
            indices = indices.cpu().tolist()
            
        if self.mode == "bpe":
            # For BPE, use the decoder dictionary
            tokens = [self.decoder.get(idx, self.special_tokens['UNK']) for idx in indices]
            if filter_special:
                special_tokens = set(self.special_tokens.values()).union({self.sos_token, self.eos_token})
                tokens = [token for token in tokens if token not in special_tokens]
            
            # For BPE, we need to decode byte-level encoding
            decoded_tokens = []
            for token in tokens:
                # Convert unicode chars back to bytes
                byte_arr = [self.byte_decoder.get(ch, ord("?")) for ch in token]
                # Decode bytes to string
                try:
                    decoded = bytearray(byte_arr).decode("utf-8", errors="replace")
                    decoded_tokens.append(decoded)
                except Exception:
                    decoded_tokens.append("[DECODE_ERROR]")
            
            return decoded_tokens
        else:
            # Original behavior for non-BPE modes
            tokens = [self.idx2token.get(idx, self.special_tokens['UNK']) for idx in indices]
            if filter_special:
                return [token for token in tokens if token not in self.special_tokens.values()]
            else:
                return tokens


