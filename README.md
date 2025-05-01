# Blu: A Lightweight LLM

Blu is a lightweight transformer-based language model which I built simply because I wanted to try building an LLM from scratch. It is very simple character-based text gen, and I would not recommend actually using this model as there are much better alternatives out there. But technically it is an LLM that is able to learn from training data to generate text that isn't gibberish (sometimes).

Note that in model.py the class is named SimpleLLM since I didn't have a name for it yet.

## Features

- Decoder-only transformer architecture optimized for sub-optimal GPUs
- Character-level text gen
- Rotary positional embeddings (RoPE)
- RMSNorm for normalization and SwiGLU for feed-forward activation
- key-value caching for faster text generation
- Implementation that balances capabilities and efficiency (a fancy way of saying it makes use of my single mid-level NVIDIA GPU somewhat efficiently)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible NVIDIA GPU

## Quick Start

1. Clone repo:
```bash
git clone https://github.com/yourusername/blu.git
cd blu
```

2. Install dependencies:
```bash
pip install torch numpy requests
```
There may be other dependencies but I doubt anyone will care to actually run this model. If you do and you run into problems then I will update this.

3. Train the model (trains on the basic tiny shakespeare dataset but you can customize this to any txt file):
```bash
python train.py
```

4. Generate text (it is currently set for a shakespeare model, so you may have to modify the prompts and/or .pt file to have you generate the text you want):
```bash
python blu.py
```

## Model Architecture

Blu uses a compact transformer architecture consisting of:

- Character embeddings (vocabulary size derived from Shakespeare text)
- Transformer blocks with:
  - Multi-head self-attention with rotary positional embeddings
  - Feed-forward network with SwiGLU-like activation
  - RMSNorm for layer normalization
- Output projection layer

The default configuration (256-dimensional model with 4 layers and 8 attention heads) is 2.66M parameters, but you can scale the model architecture however you like.

## Training

### Configuration

Key training parameters in `train.py`:

```python
CONTEXT_LENGTH = 64   # Sequence length for training
BATCH_SIZE = 16       # Number of sequences per batch
LEARNING_RATE = 5e-4  # Learning rate
NUM_EPOCHS = 5        # Number of training epochs
```

## Text Generation

After training (assuming that you are sticking with the shakespeare dataset), generate Shakespeare-style text with:

```bash
python blu.py
```

This script:
- Loads your trained model
- Provides sample prompts like "ROMEO: " or "HAMLET: To be, or not to be," (dont use this if you are using a different dataset)

### Example Outputs

After 5 epochs of training, this was Blu's output for the sample prompts:

```
Prompt 1: 'ROMEO: '
--------------------------------------------------
ROMEO: lling forth.

VOLUMNIA:
These are the ushers of Marcius: before him he
challenged for than dragque, d
ot,yalllivyouyst'sreberreoweres''erwonthatandthenrion!;I aLvmrrw nm ets-he,shfus,moppthvupLgms,hel
--------------------------------------------------

Prompt 2: 'HAMLET: To be, or not to be,'
--------------------------------------------------
HAMLET: To be, or not to be, the freedom of my knowledge:
we cannot with such magnificence--in so rather,
Bassaver ClauriLwume Istorourothanduaseme,;imeor,oallo ifounrsstardon;redagoacngineaitipanchiothereaeesmuesz,ruccie
e syd:
--------------------------------------------------

Prompt 3: 'Once upon a time'
--------------------------------------------------
Once upon a time pertain
Your highness said.

SEBASTIAN:
Before the times of lap, which is it
Wore now mam dilybongerrouncurer
ouneacler
arpayesthhay: upeakbit Qrtesio,rrniorththelcytadwneron saroulc sofwww
sindc?;
e
--------------------------------------------------

Prompt 4: 'The king'
--------------------------------------------------
The king dishonour foes
When is thou didst learn, had that innocence?
A most deligh! withins rrettheines wartin ybeaur perdemlerstwsef wwthet moudd; earegwo gaidewaruewirimdealdi-hvikopdagses
wiereselArrserth
--------------------------------------------------
```
I know, this rivals even the greatest reasoning models known to man.

### Generation Parameters

Adjust these parameters in `blu.py` to control text generation:

```python
max_new_tokens = 200  # Length of generated text
temperature = 0.8     # Controls randomness (lower = more focused)
top_p = 0.9           # Nucleus sampling threshold
```

## Improving Results

For better text quality:

1. **Train longer**: 10+ epochs significantly improves coherence
2. **Adjust temperature**: Try 0.6-0.7 for more coherent output
3. **Scale the model**: Increase dimensions or add layers if you have more VRAM
4. **Use learning rate scheduling**: Implement a warmup and decay schedule

## Memory Requirements

The default configuration works well on GPUs with 6GB VRAM (at least that is what I used). For larger GPUs, you can scale the model by increasing these parameters in `train.py`, although you could also just train a model with better architecture:

```python
model = create_model(
    vocab_size=tokenizer.vocab_size,
    dim=256,        # Try 384 or 512 for more capacity
    n_layers=4,     # Try 6 or 8 for deeper models
    n_heads=8,      # Try 12 or 16 for more attention heads
    hidden_dim=512  # Try 1024 for larger feed-forward layers
)
```

## Acknowledgments

- This was originally going to be a further optmized deepseek-v3, however after reading the source code, many of their optimizations and novel improvements were tailored towards a multi-gpu setup with a large amount of RAM, and I felt that modifying this code to specifically fit my very bad laptop was not worth it with their MLA and MoE designed for large scale training.
- Because of this, I decided to write a much simpler LLM designed for my single GPU and limited RAM, mainly focusing on a traditional transformer architecture.
- However, the code for deepseek-v3 was still a heavy reference when building this model, as I still used their logic for RoPE among other things.
