"""Generate responses from a base model optionally patched with a SCA (masked LoRA) adapter.

Loads the base model, patches its MLPs with MaskedLoRALinear using the same
circuit mask used for training, then loads the trained `sca_lora*.bin`
parameters back into those modules. Runs greedy decoding on the eval prompts.

If neither --sca_dir nor --peft_dir is supplied, generates with the base model.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Reuse the patching machinery from train_sca.py
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_sca import patch_mlp_with_masked_lora, freeze_non_lora  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--sca_dir", default=None,
                   help="Directory with mask_used.pt and sca_lora_final.bin. Omit for base-only eval.")
    p.add_argument("--sca_weights", default=None,
                   help="Override: path to a specific sca_lora_*.bin (else uses <sca_dir>/sca_lora_final.bin)")
    p.add_argument("--peft_dir", default=None,
                   help="Path to a PEFT adapter directory. Takes precedence over --sca_dir when both are supplied.")
    p.add_argument("--eval_file", required=True, help="JSONL with `prompt` field")
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--tag", default="sca", help="label written into each output record")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    if args.peft_dir is not None:
        from peft import PeftModel
        print(f"[eval_sca] loading PEFT adapter from {args.peft_dir}")
        model = PeftModel.from_pretrained(model, args.peft_dir, is_trainable=False)
        model.eval()
    elif args.sca_dir is not None:
        mask_path = Path(args.sca_dir) / "mask_used.pt"
        weights_path = Path(args.sca_weights) if args.sca_weights else Path(args.sca_dir) / "sca_lora_final.bin"
        print(f"[eval_sca] loading mask {mask_path} and weights {weights_path}")
        masks = torch.load(mask_path, map_location="cpu", weights_only=False)
        n_patched, total_active = patch_mlp_with_masked_lora(
            model, masks, rank=args.lora_rank, alpha=args.lora_alpha, dropout=0.0
        )
        print(f"[eval_sca] patched {n_patched} linears, {total_active} active neurons")
        sd = torch.load(weights_path, map_location="cpu", weights_only=False)
        # Load full module-path keys as saved by train_sca.py.
        missing, unexpected = model.load_state_dict(sd, strict=False)
        lora_missing = [k for k in missing if ("lora_A" in k or "lora_B" in k)]
        lora_unexpected = [k for k in unexpected if ("lora_A" in k or "lora_B" in k)]
        assert not lora_missing, f"missing LoRA keys: {lora_missing[:5]}"
        assert not lora_unexpected, f"unexpected LoRA keys: {lora_unexpected[:5]}"
        print(f"[eval_sca] loaded {sum(1 for k in sd if 'lora_B' in k)} LoRA B tensors")

    # Load prompts
    prompts = []
    with open(args.eval_file) as f:
        for line in f:
            ex = json.loads(line)
            prompts.append(ex)
    print(f"[eval_sca] generating on {len(prompts)} prompts")

    # Batched greedy decoding
    out_f = open(args.output_jsonl, "w")
    model_device = next(model.parameters()).device
    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch = prompts[i : i + args.batch_size]
        texts = [ex["prompt"] for ex in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model_device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Strip input from generations
        input_len = enc["input_ids"].shape[1]
        gens = tokenizer.batch_decode(out[:, input_len:], skip_special_tokens=True)
        for ex, gen in zip(batch, gens):
            rec = dict(ex)
            rec["generation"] = gen.strip()
            rec["tag"] = args.tag
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()
    print(f"[eval_sca] wrote {args.output_jsonl}")


if __name__ == "__main__":
    main()
