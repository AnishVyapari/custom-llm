# ⚡ Quickstart

- Requirements: Python 3.10+, CUDA 11.8+ (optional), Git
- Setup:
  1. git clone https://github.com/AnishVyapari/custom-llm && cd custom-llm
  2. python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
  3. pip install -U pip && pip install -r requirements.txt
- Data:
  - Prepare JSONL files with {"text": "..."}
  - Place at data/train.jsonl and data/eval.jsonl
- Train:
  - python train_llm_advanced.py --model_name_or_path gpt2 \
    --train_file data/train.jsonl --eval_file data/eval.jsonl \
    --output_dir outputs/llm-exp1
- Resume (optional):
  - python train_llm_advanced.py --model_name_or_path gpt2 \
    --train_file data/train.jsonl --eval_file data/eval.jsonl \
    --output_dir outputs/llm-exp1 --resume_from_checkpoint outputs/llm-exp1/checkpoint-final
- Inference (CLI chat):
  - python inference.py --model outputs/llm-exp1/checkpoint-best
- One-off prompt:
  - python inference.py --model outputs/llm-exp1/checkpoint-best --prompt "Hello!"
- Track runs: W&B enabled by default (set WANDB_PROJECT or use --no_wandb to disable)
- Discord: https://discord.gg/dzsKgWMgjJ
