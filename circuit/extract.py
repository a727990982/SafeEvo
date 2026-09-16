"""Optimize circuit-selection matrices with paired completion supervision.

The operators M and I-M are defined in matrix_masks.py; objective.py combines
refusal, compliance and sparsity terms. Saved files retain raw mask logits at
projection paths.
"""

import os
import json
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from peft import PeftModel
from torch.utils.data import Dataset
from typing import Dict, List, Any

from matrix_masks import (
    RowProjection, install_masks, save_masks, set_partition,
)
from objective import extraction_loss


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Dataset — contrastive completion pairs with chat template
# ============================================================================

def _has_chat_template(tokenizer):
    """Check if tokenizer has a chat template."""
    return hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None


def _format_prompt(tokenizer, prompt):
    """Format prompt with chat template if available, else use plain format."""
    if _has_chat_template(tokenizer):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"User: {prompt}\nAssistant:"


def _format_full(tokenizer, prompt, completion):
    """Format full conversation (prompt + completion)."""
    if _has_chat_template(tokenizer):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return f"User: {prompt}\nAssistant: {completion}"


class SafetyCircuitDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=256):
        tokenizer.padding_side = "right"
        self.raw_data = data_list
        self.samples = []
        has_template = _has_chat_template(tokenizer)
        if not has_template:
            print("  [INFO] No chat template found, using plain 'User:/Assistant:' format")

        for item in data_list:
            scenarios = {
                'accept': (item['prompt'], item['accept']),
                'refuse': (item['prompt'], item['refuse'])
            }
            encoded_scenarios = {}
            for key, (prompt, completion) in scenarios.items():
                prompt_full = _format_prompt(tokenizer, prompt)
                prompt_ids = tokenizer.encode(prompt_full, add_special_tokens=False)

                full_text = _format_full(tokenizer, prompt, completion)
                full_enc = tokenizer(full_text, truncation=True, max_length=max_length,
                                      padding='max_length', return_tensors="pt")

                input_ids = full_enc["input_ids"].squeeze(0)
                attention_mask = full_enc["attention_mask"].squeeze(0)
                labels = input_ids.clone()
                labels[:len(prompt_ids)] = -100
                labels[attention_mask == 0] = -100

                if key == 'accept':
                    eos_positions = (input_ids == tokenizer.eos_token_id)
                    labels[eos_positions] = -100

                encoded_scenarios[key] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                }
            self.samples.append(encoded_scenarios)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class CircuitDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
        batch = {}
        for key in ['accept', 'refuse']:
            batch[key] = {
                'input_ids': torch.stack([f[key]['input_ids'] for f in features]),
                'attention_mask': torch.stack([f[key]['attention_mask'] for f in features]),
                'labels': torch.stack([f[key]['labels'] for f in features])
            }
        return batch


class StopAfterEpochCallback(TrainerCallback):
    """Stop after a requested snapshot without shortening the LR-schedule horizon."""

    def __init__(self, stop_after_epoch):
        self.stop_after_epoch = stop_after_epoch

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.stop_after_epoch is not None and state.epoch >= self.stop_after_epoch:
            control.should_training_stop = True
        return control


# ============================================================================
# CircuitTrainer — paired completion losses and soft-mask sparsity
# ============================================================================

class CircuitTrainer(Trainer):
    def __init__(self, mask_params_mlp,
                 accept_weight=1.0, refuse_weight=1.0,
                 sparsity_weight_mlp=1.0,
                 output_dir=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_params_mlp = mask_params_mlp
        self.accept_weight = accept_weight
        self.refuse_weight = refuse_weight
        self.sparsity_weight_mlp = sparsity_weight_mlp
        self.output_dir_for_best = output_dir
        self.best_val_loss = None
        self.best_epoch = -1

    def _objective(self, model, inputs):
        result = extraction_loss(
            model, inputs, self.mask_params_mlp,
            alpha=self.accept_weight, beta=self.refuse_weight,
            lambda_mlp=self.sparsity_weight_mlp,
        )
        self._last_loss_stats = result.log_values()
        return result

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        result = self._objective(model, inputs)
        return (result.total, result.refusal_output) if return_outputs else result.total

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            result = self._objective(model, inputs)
        return result.total.detach(), None, None

    def log(self, logs, *args, **kwargs):
        return

    def _compute_mask_stats(self, model):
        stats = {"mlp": {"total": 0, "zeros": 0}}
        threshold = 0.5
        for _, module in model.named_modules():
            if isinstance(module, RowProjection):
                mask = (torch.sigmoid(module.M.q) > threshold).float()
                stats["mlp"]["total"] += mask.numel()
                stats["mlp"]["zeros"] += (mask == 0).sum().item()
        return stats

    def _format_mask_stats(self, stats):
        def _line(label, total, zeros):
            sparsity = (zeros / total * 100.0) if total > 0 else 0.0
            active = total - zeros
            return f"  {label}: active={active:7d}/{total:7d} ({100-sparsity:5.2f}%) | inactive={zeros:7d} ({sparsity:5.2f}%)"
        return [
            "[Safety Mask Sparsity]",
            _line("MLP neurons    ", stats["mlp"]["total"], stats["mlp"]["zeros"]),
        ]

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys,
                                    metric_key_prefix=metric_key_prefix)

        if hasattr(self, '_last_loss_stats') and self._last_loss_stats:
            print("\n[Eval Loss Breakdown]", flush=True)
            for k, v in self._last_loss_stats.items():
                print(f"  {k:25s}: {v:.6f}", flush=True)
            nan_components = [k for k, v in self._last_loss_stats.items() if np.isnan(v)]
            if nan_components:
                print(f"\n WARNING: NaN in: {', '.join(nan_components)}", flush=True)

        stats = self._compute_mask_stats(self.model)
        for line in self._format_mask_stats(stats):
            print(line, flush=True)

        if hasattr(self, '_last_loss_stats') and self._last_loss_stats and self.output_dir_for_best:
            val_loss = (self.accept_weight * self._last_loss_stats['loss_accept'] +
                        self.refuse_weight * self._last_loss_stats['loss_refuse'])
            if self.best_val_loss is None or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = int(self.state.epoch) if self.state and self.state.epoch else -1
                save_path = os.path.join(self.output_dir_for_best, "best_safety_masks.pt")
                save_masks(self.model, save_path)
                print(f"\n>> Best model at epoch {self.best_epoch} (val_loss={val_loss:.6f})", flush=True)

        print("", flush=True)
        return metrics


# ============================================================================
# Generation helper
# ============================================================================

@torch.no_grad()
def generate_and_save_results(model, tokenizer, val_data, batch_size=4,
                                output_file="results.json", use_inverse_mask=False):
    tokenizer.padding_side = "left"
    model.eval()
    results = []

    set_partition(model, complement=use_inverse_mask)

    for i in tqdm(range(0, len(val_data), batch_size), desc=f"Gen (inverse={use_inverse_mask})"):
        batch_items = val_data[i:i+batch_size]
        prompts_raw = [item['prompt'] for item in batch_items]
        prompts = [_format_prompt(tokenizer, p) for p in prompts_raw]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)

        for j in range(len(prompts)):
            input_len = inputs.input_ids[j].shape[0]
            resp = tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True)
            results.append({
                "prompt": batch_items[j]['prompt'],
                "accept": batch_items[j]['accept'],
                "refuse": batch_items[j]['refuse'],
                "generation": resp
            })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved generation results to {output_file}")

    set_partition(model, complement=False)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Extract a one-pass circuit")
    parser.add_argument("--base_model_path", type=str,
                        required=True,
                        help="Path to Qwen2.5-1.5B base model")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to LoRA checkpoint (None = use base model directly)")
    parser.add_argument("--checkpoint_name", type=str, default="base",
                        help="Name for output directory (e.g., 'base', 'checkpoint-250')")
    parser.add_argument("--data_path", type=str,
                        required=True,
                        help="Path to LLM-LAT contrastive data")
    parser.add_argument("--output_base", type=str,
                        default="outputs/extraction",
                        help="Base output directory")
    # Extraction parameters
    parser.add_argument("--accept_weight", type=float, default=1.0)
    parser.add_argument("--refuse_weight", type=float, default=1.0)
    parser.add_argument("--sparsity_weight_mlp", type=float, default=0.5)
    parser.add_argument("--init_value", type=float, default=0.2)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (1.5B model fits bs=8 on 48GB GPU)")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--train_samples", type=int, default=100)
    parser.add_argument("--val_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stop_after_epoch", type=int, default=None,
                        help="Stop after this epoch while retaining num_epochs as the LR horizon")
    parser.add_argument("--skip_post_generation", action="store_true",
                        help="Skip the optional 50-sample qualitative generation pass")
    return parser.parse_args()


def main():
    args = parse_args()

    OUTPUT_DIR = os.path.join(args.output_base, args.checkpoint_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print(f"Circuit extraction — {args.checkpoint_name}")
    print("=" * 70)
    print(f"Base model:         {args.base_model_path}")
    print(f"Checkpoint:         {args.checkpoint_path or '(base model, no LoRA)'}")
    print(f"Output:             {OUTPUT_DIR}")
    print(f"accept_weight:      {args.accept_weight}")
    print(f"refuse_weight:      {args.refuse_weight}")
    print(f"sparsity_mlp:       {args.sparsity_weight_mlp}")
    print(f"init_value:         {args.init_value}")
    print(f"epochs:             {args.num_epochs}")
    print(f"lr:                 {args.lr}")
    print(f"batch_size:         {args.batch_size}")
    print(f"seed:               {args.seed}")
    print(f"stop_after_epoch:   {args.stop_after_epoch or '(disabled)'}")
    print(f"Dataset:            LLM-LAT/harmful-dataset")
    print("=" * 70)

    set_random_seed(args.seed)

    # === Load data ===
    print(f"\nLoading LLM-LAT data from {args.data_path}...")
    with open(args.data_path, "r") as f:
        llm_lat = json.load(f)
    full_data = llm_lat["results"]
    print(f"  Total samples: {len(full_data)}")

    train_data = full_data[:args.train_samples]
    val_data = full_data[args.train_samples:args.train_samples + args.val_samples]
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    # === Load model ===
    print(f"\nLoading base model from {args.base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )

    # Load LoRA adapter if checkpoint specified
    if args.checkpoint_path is not None:
        print(f"Loading LoRA adapter from {args.checkpoint_path}...")
        model = PeftModel.from_pretrained(model, args.checkpoint_path)
        model = model.merge_and_unload()
        print("  LoRA merged and unloaded.")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Print model info
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, 'num_key_value_heads', num_heads)
    intermediate_size = model.config.intermediate_size
    print(f"\nModel: {num_layers} layers, {num_heads} heads ({num_kv_heads} KV), "
          f"intermediate={intermediate_size}")
    total_coordinates = sum(layer.out_features for name, layer in model.named_modules()
                            if name.endswith(("gate_proj", "up_proj", "down_proj")))
    print(f"  Candidate output coordinates: {total_coordinates}")

    # === Build datasets ===
    print("\nBuilding datasets with chat template...")
    train_dataset = SafetyCircuitDataset(train_data, tokenizer, max_length=args.max_length)
    val_dataset = SafetyCircuitDataset(val_data, tokenizer, max_length=args.max_length)

    # === Patch model ===
    mask_params_mlp = install_masks(model, init_value=args.init_value)
    print(f"MLP mask tensors: {len(mask_params_mlp)}")

    # === Print initial sparsity ===
    init_active_mlp = sum((torch.sigmoid(p) > 0.5).sum().item() for p in mask_params_mlp)
    total_mlp = sum(p.numel() for p in mask_params_mlp)
    print(f"\nInitial mask state (init_value={args.init_value}, "
          f"sigmoid={torch.sigmoid(torch.tensor(args.init_value)).item():.4f}):")
    print(f"  MLP:  {init_active_mlp}/{total_mlp} active ({init_active_mlp/total_mlp*100:.1f}%)")

    # === Training ===
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_strategy="no",
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        label_names=[],
    )

    callbacks = []
    if args.stop_after_epoch is not None:
        callbacks.append(StopAfterEpochCallback(args.stop_after_epoch))

    trainer = CircuitTrainer(
        mask_params_mlp=mask_params_mlp,
        accept_weight=args.accept_weight,
        refuse_weight=args.refuse_weight,
        sparsity_weight_mlp=args.sparsity_weight_mlp,
        output_dir=OUTPUT_DIR,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=CircuitDataCollator(),
        callbacks=callbacks,
    )

    trainer.train()

    print(f"\n{'='*70}")
    print("Training completed!")
    if trainer.best_epoch >= 0:
        print(f"Best epoch: {trainer.best_epoch} (val_loss={trainer.best_val_loss:.6f})")
    print(f"{'='*70}\n")

    # Save final masks
    save_masks(model, f"{OUTPUT_DIR}/final_safety_masks.pt")

    # Optional qualitative generation after saving the extracted masks.
    if not args.skip_post_generation:
        generate_and_save_results(model, tokenizer, val_data, args.batch_size,
                                   f"{OUTPUT_DIR}/post_mask_accept.json", use_inverse_mask=False)
        generate_and_save_results(model, tokenizer, val_data, args.batch_size,
                                   f"{OUTPUT_DIR}/post_mask_refuse.json", use_inverse_mask=True)

    # Save config for reproducibility
    config = vars(args)
    config["sparsity_fix"] = True
    config["best_epoch"] = trainer.best_epoch
    config["best_val_loss"] = trainer.best_val_loss
    with open(f"{OUTPUT_DIR}/run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
