import torch
import torch.nn as nn
import tiktoken   # Real tokenizer from book Chapter 2/5/6

class GPTConfig:
    """Configuration from Book Chapter 4 (small size for fast CPU demo)"""
    vocab_size = 50257      # GPT-2 vocab size
    context_length = 256
    n_embd = 64
    n_head = 4
    n_layer = 4
    drop_rate = 0.1

class MultiHeadAttention(nn.Module):
    """Book Chapter 3 + 4: Core Transformer Multi-Head Attention (QKV projection)"""
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.d_out = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.W_query = nn.Linear(config.n_embd, config.n_embd)
        self.W_key = nn.Linear(config.n_embd, config.n_embd)
        self.W_value = nn.Linear(config.n_embd, config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.drop_rate)
        self.register_buffer("mask", torch.triu(torch.ones(config.context_length, config.context_length), diagonal=1))

    def forward(self, x):
        b, t, d = x.shape
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)
        Q = Q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        K = K.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        V = V.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.mask[:t, :t] == 1, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ V
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, t, d)
        return self.out_proj(context_vec)

class FeedForward(nn.Module):
    """Book Chapter 4: FeedForward layer"""
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.drop_rate)
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    """Book Chapter 4: Full Transformer Block"""
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.drop_shortcut = nn.Dropout(config.drop_rate)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

class GPTModel(nn.Module):
    """Book Chapter 4: Full GPT Model (exact from-scratch Transformer architecture)"""
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.context_length, config.n_embd)
        self.drop_emb = nn.Dropout(config.drop_rate)
        self.trf_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.out_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx):
        b, t = idx.shape
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(t, device=idx.device))
        x = tok_emb + pos_emb
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# Real tokenizer from book (Chapter 2/5/6)
class TiktokenTokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.tokenizer.n_vocab

    def encode(self, text):
        return torch.tensor([self.tokenizer.encode(text)], dtype=torch.long)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids.tolist())