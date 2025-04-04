import math
from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    """
    Config for defining model params.

    Attr:
        max_batch_size: max batch size for inference
        max_seq_len: max sequence length
        vocab_size: size of vocabulary
        dim: model dimension
        n_layers: # of transformer layers
        n_heads: # of attention heads
        head_dim: dimension of each attention head
        hidden_dim: hidden dimension for feed-forward layers
        rope_theta: Base for rotary positional encoding
        rope_scaling: whether to scale RoPE for extended context
        dropout: Dropout prob.
    """
    max_batch_size: int = 4
    max_seq_len: int = 4096
    vocab_size: int = 32000
    dim: int = 1024  # Model dimension
    n_layers: int = 12
    n_heads: int = 16
    head_dim: int = 64
    hidden_dim: int = 2048  # Hidden dimension
    rope_theta: float = 10000.0
    rope_scaling: bool = False
    dropout: float = 0.0

    @property
    def rope_dim(self) -> int:
        return self.head_dim

    @property
    def intermediate_size(self) -> int:
        return self.hidden_dim


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # normalize and scale
        x = x / rms * self.weight
        return x


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials with given dimensions.

    Args:
        dim: Dimension of the embedding
        max_seq_len: Maximum sequence length
        theta: Base for the frequencies

    Returns:
        Precomputed frequencies tensor
    """
    # Make sure dim is even for complex number handling
    if dim % 2 != 0:
        dim = dim - 1  # Adjust to even dimension

    # Use only half the dimension since we're dealing with complex numbers
    half_dim = dim // 2

    # Compute frequencies
    freqs = 1.0 / (theta ** (torch.arange(0, half_dim).float() / half_dim))

    # Create position tensor
    t = torch.arange(max_seq_len, dtype=torch.float)

    # Outer product of position and frequencies
    freqs = torch.outer(t, freqs)  # [max_seq_len, half_dim]

    # Convert to complex exponentials
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # [max_seq_len, half_dim]

    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embeddings to the input tensor.

    Args:
        x: Input tensor of shape [batch_size, seq_len, n_heads, head_dim]
        freqs_cis: Precomputed frequency tensor

    Returns:
        Tensor with rotary embeddings applied
    """
    # Ensure proper dimensions
    batch_size, seq_len, n_heads, head_dim = x.shape

    # Make sure head_dim is even
    if head_dim % 2 != 0:
        x = x[..., :head_dim - 1]  # Truncate to even dimension
        head_dim = head_dim - 1

    # Make sure we have enough positions in freqs_cis
    if freqs_cis.shape[0] < seq_len:
        raise ValueError(f"Not enough positional embeddings. Need {seq_len}, got {freqs_cis.shape[0]}")

    # Get the frequencies we need for this sequence
    freqs_cis = freqs_cis[:seq_len]

    # Reshape for complex number manipulation: [batch, seq, heads, dim/2, 2]
    x_reshaped = x.float().reshape(batch_size, seq_len, n_heads, head_dim // 2, 2)

    # View as complex numbers: [batch, seq, heads, dim/2]
    x_complex = torch.view_as_complex(x_reshaped)

    # Reshape freqs_cis for broadcasting: [1, seq, 1, dim/2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    # Apply rotation through complex multiplication
    x_rotated = x_complex * freqs_cis

    # Convert back to real and reshape
    x_out = torch.view_as_real(x_rotated).reshape(batch_size, seq_len, n_heads, head_dim)

    return x_out.type_as(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention with rotary positional embeddings"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.head_dim

        # Make sure head_dim is even for rotary embeddings
        if self.head_dim % 2 != 0:
            self.head_dim = self.head_dim - 1
            print(f"Warning: head_dim adjusted to {self.head_dim} to be even for rotary embeddings")

        # Initialize Q, K, V projections
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # Scaling factor for attention scores
        self.attn_scale = 1.0 / math.sqrt(self.head_dim)

        # Dropout
        self.dropout = nn.Dropout(args.dropout)

        # Key and value caching for inference
        self.k_cache = None
        self.v_cache = None

    def forward(
            self,
            x: torch.Tensor,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            start_pos: int = 0
    ) -> torch.Tensor:
        """
        Forward pass of self-attention.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            freqs_cis: Precomputed frequencies for rotary embeddings
            mask: Attention mask of shape [seq_len, seq_len]
            start_pos: Starting position for inference caching

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project to queries, keys, values
        q = self.wq(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.wv(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim)

        # Get the frequencies for the current sequence
        seq_freqs = freqs_cis[start_pos:start_pos + seq_len]

        # Apply rotary embeddings to queries and keys
        q = apply_rotary_emb(q, seq_freqs)
        k = apply_rotary_emb(k, seq_freqs)

        # Initialize KV cache for inference if needed
        if start_pos > 0:
            if self.k_cache is None:
                # Create cache during first inference step
                self.k_cache = k
                self.v_cache = v
            else:
                # Append to existing cache
                self.k_cache = torch.cat([self.k_cache, k], dim=1)
                self.v_cache = torch.cat([self.v_cache, v], dim=1)

            # Use the cached keys and values
            k = self.k_cache
            v = self.v_cache

        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch_size, n_heads, cached_len, head_dim]
        v = v.transpose(1, 2)  # [batch_size, n_heads, cached_len, head_dim]

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(2, 3)) * self.attn_scale

        # Apply causal mask if provided
        if mask is not None:
            scores = scores + mask.unsqueeze(0).unsqueeze(0)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(x)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)  # [batch_size, n_heads, seq_len, head_dim]

        # Reshape and project to output dimension
        output = output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)
        output = self.wo(output)

        return output


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        hidden = F.silu(self.w1(x)) * self.w3(x)

        # Output projection
        output = self.w2(hidden)
        output = self.dropout(output)

        return output


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward layers"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn_norm = RMSNorm(args.dim)
        self.attn = SelfAttention(args)

        self.ffn_norm = RMSNorm(args.dim)
        self.ffn = FeedForward(args)

    def forward(
            self,
            x: torch.Tensor,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            start_pos: int = 0
    ) -> torch.Tensor:
        # Self-attention with residual connection
        h = x + self.attn(self.attn_norm(x), freqs_cis, mask, start_pos)

        # Feed-forward with residual connection
        out = h + self.ffn(self.ffn_norm(h))

        return out


class SimpleLLM(nn.Module):
    """Simplified transformer language model for running on a single GPU"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # Token embeddings
        self.embedding = nn.Embedding(args.vocab_size, args.dim)

        # Make sure head_dim is even for rotary embeddings
        if args.head_dim % 2 != 0:
            args.head_dim = args.head_dim - 1
            print(f"Warning: head_dim adjusted to {args.head_dim} to be even for rotary embeddings")

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(args) for _ in range(args.n_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(args.dim)

        # Output projection
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Precompute rotary embeddings
        self.freqs_cis = precompute_freqs_cis(
            dim=args.head_dim,
            max_seq_len=args.max_seq_len,
            theta=args.rope_theta
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
            self,
            tokens: torch.Tensor,
            start_pos: int = 0
    ) -> torch.Tensor:
        """
        Forward pass of the language model.

        Args:
            tokens: Input token ids of shape [batch_size, seq_len]
            start_pos: Starting position for inference (used for caching)

        Returns:
            Logits of shape [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = tokens.shape

        # Get embeddings
        h = self.embedding(tokens)

        # Get the appropriate slice of precomputed freqs
        freqs_cis = self.freqs_cis.to(h.device)

        # Prepare mask for causal attention
        mask = None
        if seq_len > 1:
            # Create causal mask for training or generation
            mask = torch.full(
                (seq_len, seq_len),
                float("-inf"),
                device=tokens.device
            ).triu_(1)

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, freqs_cis, mask, start_pos)

        # Apply final normalization
        h = self.norm(h)

        # Get logits
        logits = self.output(h)

        return logits

    @torch.inference_mode()
    def generate(
            self,
            prompt_ids: torch.Tensor,
            max_new_tokens: int,
            temperature: float = 1.0,
            top_p: float = 0.9,
            stop_ids: Optional[list] = None
    ) -> torch.Tensor:
        """
        Generate text from a prompt.

        Args:
            prompt_ids: Tensor of token ids [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (1.0 = no change, <1.0 = less random, >1.0 = more random)
            top_p: Nucleus sampling probability threshold
            stop_ids: List of token ids that stop generation when generated

        Returns:
            Generated token ids [batch_size, seq_len + max_new_tokens]
        """
        # Start with the prompt
        batch_size, seq_len = prompt_ids.shape
        generated_ids = prompt_ids.clone()

        # Clear KV cache
        for layer in self.layers:
            layer.attn.k_cache = None
            layer.attn.v_cache = None

        # Process the full prompt first (to build the KV cache)
        logits = self(prompt_ids)

        # Initialize stop flags if stop_ids are provided
        stop_flags = None
        if stop_ids is not None:
            stop_flags = torch.zeros(batch_size, dtype=torch.bool, device=prompt_ids.device)

        # Generate new tokens one by one
        for pos in range(max_new_tokens):
            # Get logits for the last token
            if pos > 0:
                # Only process the last token for subsequent steps
                last_token = generated_ids[:, -1:]
                logits = self(last_token, start_pos=seq_len + pos - 1)
            else:
                # Use the logits from the full prompt processing
                logits = logits[:, -1:, :]

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Get probabilities
            probs = F.softmax(logits[:, -1], dim=-1)

            # Apply top-p sampling
            if top_p < 1.0:
                # Sort probabilities in descending order
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

                # Calculate cumulative probabilities
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Create mask for tokens to keep (below threshold)
                to_remove = cumulative_probs > top_p

                # Ensure we keep at least one token
                to_remove[..., 0] = False

                # For each sample in the batch, remove tokens above the threshold
                for i in range(batch_size):
                    # Get indices in the original distribution that should be removed
                    remove_idx = sorted_indices[i, to_remove[i]]

                    # Set their probabilities to 0
                    probs[i, remove_idx] = 0

                # Renormalize the probabilities
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

            # Sample from the filtered distribution
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to the sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Check for stop tokens
            if stop_ids is not None:
                for stop_id in stop_ids:
                    stop_flags = stop_flags | (next_token.squeeze(-1) == stop_id)
                if stop_flags.all():
                    break

        return generated_ids


def create_model(
        vocab_size: int = 32000,
        dim: int = 1024,
        n_layers: int = 12,
        n_heads: int = 16,
        hidden_dim: int = 2048,
        max_seq_len: int = 4096,
        dropout: float = 0.0
) -> SimpleLLM:
    """
    Create a SimpleLLM model with the specified parameters.

    Returns:
        Initialized SimpleLLM model
    """
    # Ensure head_dim is even for rotary embeddings
    head_dim = dim // n_heads
    if head_dim % 2 != 0:
        head_dim = head_dim - 1
        print(f"Warning: head_dim adjusted to {head_dim} to be even for rotary embeddings")

    args = ModelArgs(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        dropout=dropout
    )

    model = SimpleLLM(args)
    return model


# Example usage
if __name__ == "__main__":
    # Set precision to BF16 if available, otherwise FP16
    dtype = torch.bfloat16 if torch.cuda.is_available() and hasattr(torch, 'bfloat16') else torch.float16

    # Create a small test model
    model = create_model(
        vocab_size=32000,
        dim=768,
        n_layers=8,
        n_heads=12,
        hidden_dim=1536,
        max_seq_len=2048
    )

    # Move model to GPU and set precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).to(dtype)

    # Test with random input
    with torch.inference_mode():
        input_ids = torch.randint(0, 32000, (1, 256), device=device)
        output = model(input_ids)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {output.shape}")

        # Test generation
        generated = model.generate(input_ids, max_new_tokens=20)
        print(f"Generated shape: {generated.shape}")