# Custom LLM (PyTorch)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](./requirements.txt) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](./CONTRIBUTING.md)

A visually polished, GPT-style transformer model built from scratch in PyTorch with full training, inference, tokenizer, and docs. No API keys. No external LLMs. 100% local and hackable.

Join the community: [Discord invite](https://discord.gg/dzsKgWMgjJ)

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
     → LayerNorm → Linear(Vocab) → Softmax → Sample → Output
```

## Features
- Configurable model sizes (layers, heads, d_model, d_ff)
- Cosine LR schedule, weight decay, gradient clipping
- Checkpointing and resume
- Reproducible seeds, tqdm progress, validation split
- Easy prompt priming and system persona

## Usage Examples
Python API:
```python
import torch
from custom_llm_complete import CustomGPT, SimpleTokenizer

ckpt = torch.load('anish_llm_final.pt', map_location='cpu')
tok = SimpleTokenizer()
tok.char_to_idx = ckpt['tokenizer_vocab']
tok.idx_to_char = {v:k for k,v in tok.char_to_idx.items()}

model = CustomGPT(vocab_size=len(tok.char_to_idx), **ckpt['config'])
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print(model.generate(tok, "Hello!", max_length=120, temperature=0.9, top_k=50))
```

## Screenshots
- Training progress (tqdm)
- Sample generations

> Add images to docs/ later. Placeholder badges above keep the page attractive.

## Topics
pytorch, machine-learning, transformer, llm, gpt, deep-learning

## Community
- Discord: https://discord.gg/dzsKgWMgjJ
- Issues and discussions welcome

## License
MIT © Anish Vyapari
