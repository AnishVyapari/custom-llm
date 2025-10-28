#!/usr/bin/env python3
"""
Minimal inference CLI for chatting with a causal LM.
Usage:
  python inference.py --model ./outputs/llm-exp1/checkpoint-best --max_new_tokens 128
"""
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from rich.console import Console
from rich.prompt import Prompt

console = Console()


def load(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    return tok, model


def generate(tok, model, prompt: str, max_new_tokens: int = 128, temperature: float = 0.8, top_p: float = 0.95):
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0], skip_special_tokens=True)


def chat_loop(tok, model):
    console.print("[bold green]Custom LLM Chat[/] — type /exit to quit")
    history = []
    while True:
        user = Prompt.ask("[bold cyan]You[/]")
        if user.strip().lower() in {"/exit", "exit", "quit", ":q"}:
            console.print("Bye!")
            break
        prompt = "\n".join(history + [f"User: {user}", "Assistant:"])
        text = generate(tok, model, prompt)
        console.print(f"[bold yellow]Assistant[/]: {text.split('Assistant:')[-1].strip()}")
        history.append(f"User: {user}")
        history.append(f"Assistant: {text.split('Assistant:')[-1].strip()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path or hub id")
    ap.add_argument("--prompt", default=None, help="One-off prompt for non-interactive mode")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    tok, model = load(args.model)
    if args.prompt:
        out = generate(tok, model, args.prompt, args.max_new_tokens, args.temperature, args.top_p)
        console.print(out)
    else:
        chat_loop(tok, model)


if __name__ == "__main__":
    main()
