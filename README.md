<!-- Neural Network Glassmorphism Banner -->
<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,12,20,24&height=200&section=header&text=Custom%20LLM&fontSize=60&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Neural%20Network%20Powered%20Transformer&descSize=20&descAlignY=55" alt="Neural Network Banner" width="100%"/>
</div>

<!-- Animated Neural Network Loader -->
<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif" width="400">
  <br>
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=00D9FF&center=true&vCenter=true&multiline=true&repeat=true&width=600&height=100&lines=Spinning+Neural+Nodes+%F0%9F%A7%A0;Glass+Effect+Processing...;AI+Model+Loading..." alt="Neural Loader">
</div>

<div align="center">
  
# Custom LLM (PyTorch)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](./requirements.txt) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](./CONTRIBUTING.md)

A visually polished, GPT-style transformer model built from scratch in PyTorch with full training, inference, tokenizer, and docs. No API keys. No external LLMs. 100% local and hackable.

Join the community: [Discord invite](https://discord.gg/dzsKgWMgjJ)

</div>

---

## Highlights

- Multi-Head Self-Attention, GELU FFN, LayerNorm, Residual, Causal masking
- Character tokenizer (easy to swap for BPE), temperature + top-k sampling
- Clean training loop with AdamW, cosine LR, gradient clipping, checkpoints
- CLI chat and scriptable inference
- Fully documented with Quickstart, Setup, and rich README visuals

## Quickstart

```bash
# 1) Install deps
pip install -r requirements.txt

# 2) Train
python train_llm_advanced.py

# 3) Chat / generate
python inference.py "Who created you?"
python inference.py  # interactive
```

## Repository Structure

```
custom-llm/
├── custom_llm_complete.py   # Core transformer + tokenizer + generation
├── train_llm_advanced.py    # Training pipeline, checkpoints
├── inference.py             # CLI and chat
├── README.md                # This page (polished)
├── QUICKSTART.md            # 5‑minute guide
├── SETUP_GITHUB.md          # Repo polish + tips
├── requirements.txt         # Dependencies
├── .gitignore               # Clean repo
├── LICENSE                  # MIT
└── FILES_SUMMARY.txt        # Summary of all files
```

## Architecture (animated SVG)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/AnishVyapari/assets/main/llm-arch-dark.svg">
  <img alt="Transformer Architecture" src="https://raw.githubusercontent.com/AnishVyapari/assets/main/llm-arch.svg" />
</picture>

```text
Input → [Tokenizer] → Embedding → +Positional → [x N Transformer Blocks]
  ↓                                                    ↓
  └────────────────────────────────> Output Logits → Sample

Transformer Block:
  Input
   ↓
  [LayerNorm] → [Multi-Head Self-Attention] → [Residual Add]
   ↓
  [LayerNorm] → [FFN (GELU)] → [Residual Add]
   ↓
  Output
```

## Key Features

### 1. Transformer Architecture
- **Multi-Head Self-Attention**: Parallel attention heads for richer representations
- **Position-wise FFN**: Two-layer network with GELU activation
- **Layer Normalization**: Pre-norm architecture for stable training
- **Residual Connections**: Skip connections around each sub-layer
- **Causal Masking**: Ensures autoregressive generation

### 2. Training Pipeline
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: Cosine annealing schedule
- **Gradient Clipping**: Prevents exploding gradients
- **Checkpointing**: Automatic model saving
- **Loss Tracking**: Comprehensive training metrics

### 3. Generation
- **Temperature Sampling**: Control randomness
- **Top-k Sampling**: Limit to k most likely tokens
- **Interactive Chat**: CLI interface for conversations
- **Batch Processing**: Efficient inference

## Training

```python
from train_llm_advanced import train_model

# Train with default parameters
train_model(
    data_path="your_text_data.txt",
    num_epochs=10,
    batch_size=32,
    learning_rate=3e-4
)
```

## Inference

```python
from inference import generate_text

# Generate text
generate_text(
    prompt="Once upon a time",
    max_length=100,
    temperature=0.8,
    top_k=40
)
```

## Configuration

Model hyperparameters can be adjusted in `custom_llm_complete.py`:

```python
config = {
    'vocab_size': 65,
    'embed_dim': 512,
    'num_heads': 8,
    'num_layers': 6,
    'ff_dim': 2048,
    'max_seq_len': 512,
    'dropout': 0.1
}
```

## Performance

- **Training Speed**: ~1000 tokens/sec on GPU
- **Memory Usage**: ~2GB VRAM for base model
- **Generation Speed**: ~50 tokens/sec

## Requirements

```
torch>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](./LICENSE) file for details.

## Acknowledgments

- Inspired by the original Transformer paper "Attention Is All You Need"
- Built with PyTorch and love ❤️

## Contact

For questions or feedback, join our [Discord](https://discord.gg/dzsKgWMgjJ) or open an issue!

---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,12,20,24&height=100&section=footer" alt="Footer" width="100%"/>
</div>
