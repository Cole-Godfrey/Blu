import torch
import requests
import os
from model import create_model

# config
MODEL_PATH = "shakespeare_model.pt"
DATASET_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CONTEXT_LENGTH = 64
PROMPTS = [
    "ROMEO: ",
    "HAMLET: To be, or not to be,",
    "Once upon a time",
    "The king",
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Character-level tokenizer
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_idx.get(ch, 0) for ch in text]

    def decode(self, indices):
        return ''.join([self.idx_to_char.get(idx, ' ') for idx in indices])


# Download dataset if not available (needed for vocabulary)
def download_dataset(url, save_path="shakespeare.txt"):
    if not os.path.exists(save_path):
        print(f"Downloading dataset from {url}...")
        response = requests.get(url)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Dataset saved to {save_path}")
    else:
        print(f"Dataset already exists at {save_path}")

    with open(save_path, "r", encoding="utf-8") as f:
        text = f.read()

    return text



def fix_rotary_emb_issue(model):
    """Fix the rotary embedding issue for generation since it was deciding to not gen positional embeddings correctly and
    i don't want to wait another hour for this thing to train again"""

    # function to fix this fuckup
    def safe_apply_rotary_emb(x, freqs_cis, start_pos=0):
        batch_size, seq_len, n_heads, head_dim = x.shape

        # Make sure head_dim is even
        if head_dim % 2 != 0:
            x = x[..., :head_dim - 1]
            head_dim = head_dim - 1
        x_reshaped = x.float().reshape(batch_size, seq_len, n_heads, head_dim // 2, 2)

        # Convert to complex
        x_complex = torch.view_as_complex(x_reshaped)

        # Make sure we don't go out of bounds with positions
        max_pos = freqs_cis.shape[0]
        if start_pos >= max_pos:
            # Just use the last position's embedding if we're beyond precomputed positions
            pos_emb = freqs_cis[-1:].unsqueeze(0).unsqueeze(2)
        else:
            # Use correct position slice, handling boundary cases
            end_pos = min(start_pos + seq_len, max_pos)
            seq_freqs = freqs_cis[start_pos:end_pos]

            # If we didn't get enough positions, pad with the last position
            if seq_freqs.shape[0] < seq_len:
                padding = seq_len - seq_freqs.shape[0]
                last_pos = seq_freqs[-1:].repeat(padding, 1)
                seq_freqs = torch.cat([seq_freqs, last_pos], dim=0)

            pos_emb = seq_freqs.unsqueeze(0).unsqueeze(2)

        # Apply rotary embeddings via complex multiplication
        x_rotated = x_complex * pos_emb

        #convert back to real
        x_out = torch.view_as_real(x_rotated).reshape(batch_size, seq_len, n_heads, head_dim)

        return x_out.type_as(x)

    #  patch the apply_rotary_emb function in each attention layer
    def patched_attn_forward(self, x, freqs_cis, mask=None, start_pos=0):
        batch_size, seq_len, _ = x.shape

        # Project to queries, keys, values
        q = self.wq(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.wv(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim)

        # Apply rotary embeddings with safe function
        q = safe_apply_rotary_emb(q, freqs_cis, start_pos)
        k = safe_apply_rotary_emb(k, freqs_cis, start_pos)

        # Rest of the attention logic remains the same
        if start_pos > 0:
            if self.k_cache is None:
                self.k_cache = k
                self.v_cache = v
            else:
                self.k_cache = torch.cat([self.k_cache, k], dim=1)
                self.v_cache = torch.cat([self.v_cache, v], dim=1)

            k = self.k_cache
            v = self.v_cache

        #transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # compute attention scores
        scores = torch.matmul(q, k.transpose(2, 3)) * self.attn_scale

        # Apply causal mask if provide
        if mask is not None:
            scores = scores + mask.unsqueeze(0).unsqueeze(0)

        #  softmax to get attention weights
        attn_weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).type_as(x)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)

        # Reshape and project to output dimension
        output = output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)
        output = self.wo(output)

        return output

    # Patch all attention layers
    for layer in model.layers:
        layer.attn.original_forward = layer.attn.forward
        # Replace it with our patched version
        layer.attn.forward = patched_attn_forward.__get__(layer.attn)

    return model


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Please train the model first.")
        return

    text = download_dataset(DATASET_URL)
    tokenizer = CharTokenizer(text)

    # create model with same architecture
    model = create_model(
        vocab_size=tokenizer.vocab_size,
        dim=256,
        n_layers=4,
        n_heads=8,
        hidden_dim=512,
        max_seq_len=CONTEXT_LENGTH * 4  #Increase max sequence length for safety
    )

    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device).eval()
    print(f"Model loaded from {MODEL_PATH}")

    # Apply fixes to the model
    model = fix_rotary_emb_issue(model)

    # Generate text from different prompts
    for i, prompt in enumerate(PROMPTS):
        print(f"\nPrompt {i + 1}: '{prompt}'")
        print("-" * 50)

        # Tokenize prompt
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)

        # If prompt is longer than context length, truncate
        if input_ids.size(1) > CONTEXT_LENGTH:
            input_ids = input_ids[:, -CONTEXT_LENGTH:]

        # Generate text
        with torch.no_grad():
            try:
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=200,
                    temperature=0.8,
                    top_p=0.95
                )

                # Decode and print generated text
                generated_text = tokenizer.decode(output_ids[0].cpu().numpy())
                print(generated_text)
            except Exception as e:
                print(f"Generation failed: {e}")
                print("Attempting simpler generation method...")

                # Fallback
                generated_ids = input_ids.clone()
                for _ in range(100):  # Generate 100 tokens
                    with torch.no_grad():
                        # Forward pass through the model
                        logits = model(generated_ids)

                        # Get predictions for the next token (last position)
                        next_token_logits = logits[:, -1, :]

                        # Apply temperature and sampling
                        probs = torch.nn.functional.softmax(next_token_logits / 0.8, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)

                        # Add to the sequence
                        generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # Decode and print the result
                generated_text = tokenizer.decode(generated_ids[0].cpu().numpy())
                print(generated_text)

        print("-" * 50)

    print("\nGeneration complete!")


if __name__ == "__main__":
    main()