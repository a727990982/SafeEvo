"""Safety Circuit Alignment (SCA) training.

Replaces each MLP Linear in a base model with a MaskedLoRALinear whose
LoRA B matrix output rows are masked by a circuit support mask. Trains
only the LoRA A/B parameters. Base weights are frozen; the fixed support
mask makes the adapter update zero outside the selected output rows.

Inputs:
    --model_name_or_path: path to base model (e.g. Meta-Llama-3-8B)
    --mask_path: path to the circuit support .pt file
    --train_file: JSONL with `text` field (LLM-LAT formatted)
    --output_dir: where to save checkpoints / lora weights / config

Saves under output_dir/:
    - sca_lora_final.bin        (only lora_A/lora_B parameters)
    - mask_used.pt                 (copy of the circuit mask for provenance)
    - run_meta.json
    - checkpoint-*/ (via SFTTrainer)
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer


class MaskedLoRALinear(nn.Module):
    """Linear + masked LoRA adapter.

    Forward: y = base(x) + ((x @ A.T) @ (B * mask_out).T) * scaling
    Only rows of B where mask_out == 1 ever contribute to output; their
    gradients flow; other rows stay at init (zero).
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float,
        mask_out: torch.Tensor,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        in_f = base_linear.in_features
        out_f = base_linear.out_features
        assert mask_out.numel() == out_f, (
            f"mask size {mask_out.numel()} != out_features {out_f}"
        )

        device = base_linear.weight.device
        dtype = base_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B stays at 0 to make ΔW = 0 at init
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        mask_bool = (mask_out > 0.5).to(dtype=dtype, device=device).view(-1, 1)
        self.register_buffer("mask_out", mask_bool, persistent=False)
        self.active_count = int((mask_out > 0.5).sum().item())
        self.out_features = out_f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_drop = self.dropout(x)
        lora_out = (x_drop @ self.lora_A.T) @ ((self.lora_B * self.mask_out).T)
        return base_out + lora_out * self.scaling


def patch_mlp_with_masked_lora(
    model: nn.Module,
    masks: dict,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: tuple = ("gate_proj", "up_proj", "down_proj"),
) -> tuple[int, int]:
    """Replace each model.layers.{i}.mlp.{gate,up,down}_proj with MaskedLoRALinear."""
    num_patched = 0
    total_active = 0
    for name, module in list(model.named_modules()):
        if not hasattr(module, "gate_proj"):
            continue
        # `module` is an MLP block
        layer_prefix = name.rsplit(".mlp", 1)[0]  # e.g. "model.layers.0"
        for proj_name in target_modules:
            key = f"{layer_prefix}.mlp.{proj_name}"
            if key not in masks:
                raise KeyError(f"missing mask for {key}; available sample: {list(masks.keys())[:3]}")
            mask = masks[key]
            base = getattr(module, proj_name)
            masked = MaskedLoRALinear(base, rank=rank, alpha=alpha, mask_out=mask, dropout=dropout)
            setattr(module, proj_name, masked)
            num_patched += 1
            total_active += masked.active_count
    return num_patched, total_active


def freeze_non_lora(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False


def count_trainable(model: nn.Module) -> tuple[int, int]:
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return train, total


class LoRAOnlySFTTrainer(SFTTrainer):
    """Save trainable adapter parameters without serializing base-model weights."""

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        sd = {n: p.detach().cpu() for n, p in self.model.named_parameters() if p.requires_grad}
        torch.save(sd, os.path.join(output_dir, "sca_lora.bin"))
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--mask_path", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--num_train_epochs", type=float, default=16.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--save_steps", type=int, default=250)
    p.add_argument("--save_total_limit", type=int, default=10)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--max_steps", type=int, default=-1, help="If >0, cap training steps (for smoke tests)")
    p.add_argument("--seed", type=int, default=42,
                   help="Training seed for LoRA initialization and data shuffling (default: 42).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Seed LoRA init + any pre-Trainer RNG (Trainer also re-seeds from SFTConfig.seed below).
    set_seed(args.seed)

    # --- Tokenizer & model ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_ddp = "LOCAL_RANK" in os.environ
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.use_bf16 else "auto",
        device_map=None if use_ddp else "auto",
    )

    # --- Load circuit masks ---
    masks = torch.load(args.mask_path, map_location="cpu", weights_only=False)

    # --- Patch MLPs with masked LoRA ---
    n_patched, total_active = patch_mlp_with_masked_lora(
        model,
        masks,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    freeze_non_lora(model)
    train_p, total_p = count_trainable(model)
    print(
        f"[SCA] patched {n_patched} MLP linears, total active circuit neurons={total_active}, "
        f"trainable={train_p:,} / {total_p:,} = {train_p/total_p*100:.4f}%"
    )
    # persist the mask snapshot used
    torch.save(masks, Path(args.output_dir) / "mask_used.pt")

    # --- Data ---
    dataset = load_dataset("json", data_files=args.train_file, split="train")
    if "text" not in dataset.column_names:
        raise ValueError("train_file must contain a `text` field")

    # --- Trainer ---
    cfg = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps",
        bf16=args.use_bf16,
        report_to=[],
        save_only_model=True,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = LoRAOnlySFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=cfg,
    )
    trainer.train()

    # Final save: only LoRA weights
    sd = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(sd, Path(args.output_dir) / "sca_lora_final.bin")
    tokenizer.save_pretrained(args.output_dir)

    meta = {
        "model_name_or_path": args.model_name_or_path,
        "mask_path": args.mask_path,
        "train_file": args.train_file,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "seed": args.seed,
        "n_patched_linears": n_patched,
        "total_active_circuit_neurons": total_active,
        "trainable_params": train_p,
        "total_params": total_p,
    }
    Path(args.output_dir, "run_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
