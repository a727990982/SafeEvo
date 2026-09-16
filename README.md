# SafeEvo

Core code for circuit extraction, checkpoint evolution, and Safety Circuit Alignment (SCA).
This package covers these three workflows.

## Structure

```text
circuit/
  extract.py              Circuit extraction entry point
  matrix_masks.py         Diagonal selection operators M and I-M
  objective.py            Paired completion losses and sparsity regularization
evolution/
  extract_checkpoint.py   Extraction with warm initialization and optional restriction
  run_chain.sh            Checkpoint-chain launch script
  analyze.py              Support overlap and checkpoint-distance analysis
alignment/
  train_sca.py            Circuit-localized LoRA SFT
  evaluate.py             Load an adapter and generate responses
requirements.txt
README.md
.gitignore
```

## Setup and inputs

Use Python 3.10+ and CUDA-enabled PyTorch, then run `pip install -r requirements.txt`.
The requirements list dependencies without pinning the historical environment versions.
Run the commands below from the repository root.

Prepare the following inputs:

- **Starting model:** a local model directory with its matching tokenizer.
- **Extraction pairs:** JSON with `{"results": [{"prompt": "...", "accept": "safe refusal", "refuse": "harmful compliance"}]}`.
  Here, `accept` contains a safe refusal and `refuse` contains a response complying with the harmful request.
  Preserve record order. The extraction example uses the first 100 records for training and the next 50 for validation;
  the evolution workflow shuffles before splitting, as described below. The source corpus is LLM-LAT.
- **SCA training:** JSONL with a `text` field. The recorded 4,948-example export uses the template below,
  with surrounding whitespace stripped from each safe completion. Preserve the original order and empty responses.
- **Generation:** JSONL with a `prompt` field, passed to the model as supplied
  (for example `User: ...\nAssistant:`).

Use the following training template:

```text
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
{safe_completion}
```

## 1. Extract a circuit

`matrix_masks.py` implements `M = Diag(STE(sigmoid(q)))` and the branches `MWh` and `(I-M)Wh`.
The diagonal matrix is stored as a vector without allocating a dense matrix.
Projection biases, where present, are also gated. `objective.py` combines refusal,
compliance, and sparsity losses while the starting-model weights remain frozen.

```bash
export BASE_MODEL=/path/to/model
export PAIR_DATA=/path/to/llm_lat_data.json
python circuit/extract.py \
  --base_model_path "$BASE_MODEL" --data_path "$PAIR_DATA" \
  --output_base outputs/extraction --checkpoint_name circuit \
  --seed 1 --num_epochs 100 --stop_after_epoch 2 \
  --batch_size 4 --lr 0.01 --max_length 256 \
  --train_samples 100 --val_samples 50 \
  --accept_weight 1 --refuse_weight 1 \
  --sparsity_weight_mlp 0.05 \
  --init_value 0.2 --skip_post_generation
```

This is the circuit extraction configuration for Llama-3-8B. Keep the 100-epoch learning-rate
schedule horizon even when stopping after epoch 2. Outputs include `best_safety_masks.pt`
and `final_safety_masks.pt`, both dictionaries mapping module paths to **raw logits**.

Optimization gates use `sigmoid(q) > 0.5`; the downstream SCA reader uses `q > 0.5`.
The diagnostic circuit C1 instead retains the globally highest raw logits among candidate coordinates:
10,376 for Llama, 9,972 for Qwen, and 7,677 for Mistral. The command above saves logits;
it does not perform this top-k conversion. These readout rules are distinct.
Diagnostic extraction does not automatically reconstruct the separately extracted support
(the set of selected coordinates) used by each main-table SCA run.

## 2. Track checkpoint evolution

Place LoRA adapters under `SFT_DIR/checkpoint-{1..30}` and first-pass mask files under
`C1_MASK_DIR/checkpoint-N/s0.5/best_safety_masks.pt`.
Use the starting model and tokenizer that produced the adapters.

```bash
export BASE_MODEL=/path/to/evolution_starting_model
export PAIR_DATA=/path/to/llm_lat_data.json
export SFT_DIR=/path/to/checkpoints
export C1_MASK_DIR=/path/to/first_pass_masks
export OUTPUT_DIR="$PWD/outputs/evolution"
MODE=free bash evolution/run_chain.sh
python evolution/analyze.py \
  --mask-dir "$OUTPUT_DIR" --output outputs/evolution_summary.json
```

Both chains begin with a restricted cold extraction at checkpoint 1.
Afterward, `MODE=free` initializes each extraction from the preceding mask without further support restriction;
`MODE=warm` retains the corresponding support restriction at each checkpoint.
The configuration uses 100 epochs, batch size 8, learning rate 0.01, a 100/50 training/validation split,
and sparsity weight 0.05.

Unlike diagnostic extraction, evolution shuffles the input with Python random seed 1 before
selecting the 100/50 records; the Trainer uses seed 42.
To analyze `MODE=warm` outputs, pass
`--mask-pattern 'full_ckpt{step}_warm/best_safety_masks.pt'` to `analyze.py`.
The analysis thresholds mask logits at 0.5, exports an overlap matrix, and computes mean
Jaccard similarity by checkpoint distance over checkpoints 6–30.

Following the manuscript configuration, this workflow uses Llama-3-8B.
Use matching Llama-3-8B adapters and masks.

## 3. Align through the circuit

Supply the intended SCA support. `train_sca.py` implements `W' = W + MBA`:
W and M remain fixed while LoRA A/B are trained. Raw masks are read at `q > 0.5`;
binary support dictionaries can also be supplied.

```bash
export BASE_MODEL=/path/to/alignment_starting_model
export SCA_MASK=/path/to/sca_support.pt
python alignment/train_sca.py \
  --model_name_or_path "$BASE_MODEL" --mask_path "$SCA_MASK" \
  --train_file /path/to/train_sca.jsonl --output_dir outputs/sca \
  --lora_rank 16 --lora_alpha 16 --lora_dropout 0 \
  --learning_rate 1e-4 --num_train_epochs 16 --max_seq_length 512 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
  --seed 42 --use_bf16

python alignment/evaluate.py \
  --base_model "$BASE_MODEL" --sca_dir outputs/sca \
  --eval_file /path/to/prompts.jsonl --output_jsonl outputs/generations.jsonl \
  --batch_size 16 --lora_rank 16 --lora_alpha 16
```

The example uses an effective batch size of 16 for Llama; for Qwen/Mistral, use microbatch size 4
and gradient accumulation 1. Outputs include `sca_lora_final.bin`, `mask_used.pt`,
tokenizer files, and `run_meta.json`. The generation example uses the final adapter;
to evaluate an intermediate adapter, pass its `sca_lora.bin` with `--sca_weights`.

Match generation `--lora_rank` and `--lora_alpha` to training; the script does not automatically
read these parameters from `run_meta.json`. Generation reloads the saved mask and adapter;
omit `--sca_dir` to generate from the starting model.

## Implementation notes

This package provides the core implementations of circuit extraction, evolution analysis, and alignment.
