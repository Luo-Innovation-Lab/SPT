import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------- Positional Encoding with Dropout ----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

# ---------------- Custom Multihead Attention with Caching ----------------
class MultiheadAttentionWithCaching(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
    def forward(self, x, attn_mask=None, key_padding_mask=None, cache=None):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.size()
        
        Q = self.q_proj(x)  # (batch, seq_len, d_model)
        K_new = self.k_proj(x)  # (batch, seq_len, d_model)
        V_new = self.v_proj(x)  # (batch, seq_len, d_model)
        
        # Reshape for multi-head attention: (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        K_new = K_new.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        V_new = V_new.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        # If using caching, concatenate previous keys/values with the new ones.
        if cache is not None and cache.get('k') is not None and cache.get('v') is not None:
            K = torch.cat([cache['k'], K_new], dim=2)  # Concatenate along sequence length.
            V = torch.cat([cache['v'], V_new], dim=2)
        else:
            K = K_new
            V = V_new
        
        new_cache = {'k': K, 'v': V} if cache is not None else None
        
        # Compute scaled dot-product attention.
        # Q: (batch, num_heads, seq_len, head_dim)
        # K: (batch, num_heads, total_len, head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, num_heads, seq_len, total_len)
        
        total_len = K.size(2)
        if attn_mask is None:
            # Generate a causal mask to prevent attending to future tokens.
            causal_mask = torch.triu(
                torch.ones(seq_len, total_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            # If an external mask is provided, apply it.
            if attn_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
            else:
                attn_scores = attn_scores + attn_mask
        
        # Apply key padding mask if provided. Expected shape: (batch, total_len)
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        attn_output = torch.matmul(attn_probs, V)  # (batch, num_heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_probs, new_cache

# ---------------- Decoder Layer with Pre-Norm and GELU ----------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttentionWithCaching(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
        # Initialize feed-forward layers.
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        
    def forward(self, x, attn_mask=None, key_padding_mask=None, cache=None):
        # Pre-layer norm before self-attention.
        x_norm = self.norm1(x)
        attn_output, attn_weights, new_cache = self.self_attn(
            x_norm, attn_mask=attn_mask, key_padding_mask=key_padding_mask, cache=cache
        )
        x = x + self.dropout(attn_output)
        
        # Pre-layer norm before the feed-forward network.
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout(ffn_output)
        return x, attn_weights, new_cache

# ---------------- Transformer Decoder with Weight Tying and Caching ----------------
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=512, n_layers=6, n_heads=8,
                 d_ff=2048, dropout=0.1, tie_weights=True):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        if tie_weights:
            if self.embedding.weight.shape != self.fc_out.weight.shape:
                raise ValueError("Cannot tie weights due to shape mismatch.")
            self.fc_out.weight = self.embedding.weight
            
        nn.init.xavier_uniform_(self.fc_out.weight)
        
    def forward(self, x, attn_mask=None, key_padding_mask=None, caches=None, return_attention=False):
        # x: (batch, seq_len)
        # caches: Optional list of caches for each layer (used during incremental decoding).
        # return_attention: If True, return attention weights and hidden states
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        new_caches = []
        all_attn_weights = []
        for i, layer in enumerate(self.layers):
            layer_cache = caches[i] if caches is not None else None
            x, attn_weights, new_cache = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, cache=layer_cache)
            new_caches.append(new_cache)
            if return_attention:
                all_attn_weights.append(attn_weights)
        
        hidden_states = x  # Store hidden states before final projection
        logits = self.fc_out(x)
        
        if return_attention:
            return {
                'logits': logits,
                'hidden_states': hidden_states,
                'attentions': all_attn_weights,
                'caches': new_caches
            }
        else:
            return logits, new_caches

    @staticmethod
    def generate_causal_mask(seq_len, device):
        """Generate a causal mask to prevent attending to future tokens."""
        return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).triu(1)
