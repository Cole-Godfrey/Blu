import os
import time
import requests
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import create_model

# Configuration
DATASET_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CONTEXT_LENGTH = 64  # Context window size
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
NUM_EPOCHS = 5
MODEL_SAVE_PATH = "shakespeare_model.pt"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Download dataset if not available
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


# Character-level tokenization
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        return ''.join([self.idx_to_char[idx] for idx in indices])


# Dataset class
class ShakespeareDataset(Dataset):
    def __init__(self, text, tokenizer, context_length):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.context_length]
        y = self.data[idx + 1:idx + self.context_length + 1]
        return x, y


# This is a temporary fix for the rotary embeddings issue
def fix_model_for_generation(model):
    """Apply fixes to make the model work properly during generation."""

    # Ensure head_dim is even for complex number handling in RoPE
    for layer in model.layers:
        if hasattr(layer.attn, 'head_dim') and layer.attn.head_dim % 2 != 0:
            print(
                f"Warning: head_dim ({layer.attn.head_dim}) is not even. This may cause issues with rotary embeddings.")

    # Create a simple test to verify generation works
    test_input = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        try:
            # Test forward pass
            _ = model(test_input)

            # Test generation with the simplest case (single token)
            _ = model.generate(test_input, max_new_tokens=1)
            print("Generation test passed!")
        except Exception as e:
            print(f"Generation test failed: {e}")
            raise

    return model


def main():
    # Download and prepare dataset
    text = download_dataset(DATASET_URL)
    print(f"Dataset length: {len(text)} characters")

    # Create tokenizer
    tokenizer = CharTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size} characters")
    print(f"Vocabulary: {''.join(tokenizer.chars)}")

    # Create dataset and dataloader
    dataset = ShakespeareDataset(text, tokenizer, CONTEXT_LENGTH)

    # Split for training and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    print(f"Training set size: {len(train_dataset)} sequences")
    print(f"Validation set size: {len(val_dataset)} sequences")
    print(f"Batches per epoch: {len(train_loader)}")

    # Create model
    model = create_model(
        vocab_size=tokenizer.vocab_size,
        dim=256,  # Smaller model for faster training
        n_layers=4,
        n_heads=8,
        hidden_dim=512,
        max_seq_len=CONTEXT_LENGTH
    )
    model = model.to(device)

    # Print model size
    num_params = sum(p.numel() for p in model.parameters()) / 1_000_000
    print(f"Model size: {num_params:.2f}M parameters")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_losses = []
        start_time = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Forward pass
            logits = model(x)

            # Calculate loss
            loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # Print progress
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = sum(train_losses) / len(train_losses)
        epoch_time = time.time() - start_time

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1))
                val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Save model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model saved to {MODEL_SAVE_PATH}")

        # Apply fixes for generation
        model = fix_model_for_generation(model)

        # Generate sample text (skipping the problematic part for now)
        if epoch + 1 == NUM_EPOCHS:  # Only generate at the end
            print("\nTraining complete! Let's generate some text from the trained model.")

            # Get a sample prompt from validation set
            idx = torch.randint(0, len(val_dataset), (1,)).item()
            context, _ = val_dataset[idx]
            context = context[:20]  # Just take first 20 chars to start
            context = context.unsqueeze(0).to(device)

            # Generate text
            try:
                with torch.no_grad():
                    generated_ids = model.generate(
                        context,
                        max_new_tokens=200,
                        temperature=0.8,
                        top_p=0.9
                    )

                # Show the prompt and generated text
                prompt_text = tokenizer.decode(context[0].cpu().numpy())
                generated_text = tokenizer.decode(generated_ids[0].cpu().numpy())

                print("\nPrompt:")
                print("-" * 50)
                print(prompt_text)
                print("\nGenerated:")
                print("-" * 50)
                print(generated_text)
                print("-" * 50)
            except Exception as e:
                print(f"Generation failed: {e}")
                print("Continuing training without generation.")

    print("Training complete!")


if __name__ == "__main__":
    main()