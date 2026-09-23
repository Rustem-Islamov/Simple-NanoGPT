# Simple-NanoGPT

A minimal PyTorch setup for pretraining a ~124M-parameter GPT-style model on
[FineWeb](https://huggingface.co/datasets/kjj0/fineweb100B-gpt2) with **AdamW**.
It is intended for hyperparameter studies (learning rate, batch size, weight decay,
betas, token budget). Runs are tracked with [Weights & Biases](https://wandb.ai).

**Model:** 12 layers, 6 heads, 768-dim embeddings, RoPE, QK-norm, RMSNorm without
learnable scale, squared-ReLU MLP, tied input/output embeddings, bf16 autocast, `torch.compile`.

**Optimizer:** AdamW with two parameter groups. The (tied) embedding/LM-head gets
`lr * emb_lr_ratio` and `weight_decay / emb_lr_ratio`, so the effective per-step decay is
the same for both groups. The LR schedule is linear warmup, constant, then linear warmdown.

## Repository layout

```
.
├── train_gpt_adamw.py        # training script (DDP, gradient accumulation, W&B logging)
├── configs/
│   └── adamw.yaml            # default config
├── data/
│   └── cached_fineweb100B.py # downloads pre-tokenized FineWeb shards from Hugging Face
└── utils/
    ├── gpt.py                # model
    ├── dataloader.py         # sharded .bin token loader
    └── configs.py            # YAML loading helpers
```

## Requirements

- Linux with an NVIDIA GPU and a CUDA-enabled PyTorch build (the script asserts `torch.cuda.is_available()` and uses NCCL)
- Python >= 3.10
- PyTorch >= 2.4
- Python packages: `torch`, `numpy`, `pyyaml`, `tqdm`, `wandb`, `huggingface_hub`

## Installation

From the root of the cloned repository:

```bash
conda create -n nanogpt python=3.11 -y
conda activate nanogpt

# Install PyTorch matching your CUDA version (see https://pytorch.org/get-started/locally/)
# Example for CUDA 12.4:
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

## Weights & Biases

Logging goes through W&B, so set it up before training:

1. Log in with `wandb login`.
2. In `configs/adamw.yaml`, replace the placeholder `entity: "wandb_entity"` with your W&B
   username or team. `project` and `run` can be changed as well.

To run without an account, use `export WANDB_MODE=offline`.

## Data

Download the pre-tokenized GPT-2 FineWeb shards (roughly 100M tokens / 200 MB each).
The argument is the number of training chunks; the validation shard is always fetched.

```bash
python data/cached_fineweb100B.py 15     # ~1.5B tokens
python data/cached_fineweb100B.py        # default: 305 chunks (~30B tokens)
```

Files are stored in `data/fineweb100B/`, which matches the paths in the config.
Download at least `ceil(token_budget / 1e8)` chunks; if the budget exceeds the data on disk,
the loader silently repeats data.

## Running training

Single GPU:

```bash
torchrun --standalone --nproc_per_node=1 train_gpt_adamw.py --config=configs/adamw.yaml
```

Multiple GPUs on one node (e.g. 4):

```bash
torchrun --standalone --nproc_per_node=4 train_gpt_adamw.py --config=configs/adamw.yaml
```

Constraints checked by the script:

- `batch_size % (device_batch_size * n_gpus) == 0`. Gradient accumulation steps are `batch_size / (device_batch_size * n_gpus)`.
- `val_tokens % (device_batch_size * sequence_length_eval * n_gpus) == 0`.

If you run out of GPU memory, lower `device_batch_size`; the global `batch_size` is unchanged
because gradient accumulation makes up the difference.

### Running on a Slurm cluster

Submit a batch job that requests one GPU, activates the environment above, changes into the
repository directory, and runs the same `torchrun` command as in the previous section. If your
job script writes Slurm output files to a subdirectory, create that directory before calling `sbatch`.

## Configuration

All settings live in the YAML file (`configs/adamw.yaml`).

| Key | Meaning |
|---|---|
| `project`, `entity`, `run` | W&B project, entity and run name |
| `input_bin`, `input_val_bin` | Glob patterns for train / validation shards |
| `batch_size` | Global batch size **in sequences** (tokens per step = `batch_size * sequence_length`) |
| `device_batch_size` | Sequences per GPU per forward/backward pass |
| `sequence_length`, `sequence_length_eval` | Context length for training / evaluation |
| `num_iterations` | Number of training iterations |
| `warmup_iters`, `warmdown_iters` | Length of LR warmup and linear warmdown (in iterations) |
| `val_loss_every`, `val_tokens` | Evaluation interval (iterations) and number of validation tokens |
| `n_layer`, `n_head`, `n_embd` | Model size |
| `lr` | Peak learning rate for non-embedding parameters |
| `emb_lr_ratio` | Embedding/LM-head LR is `lr * emb_lr_ratio` |
| `weight_decay` | Weight decay for non-embedding parameters (embeddings use `weight_decay / emb_lr_ratio`) |
| `beta1`, `beta2` | AdamW betas |
| `seed` | Base seed (rank is added per process) |

**Token budget:** `tokens = num_iterations * batch_size * sequence_length`.
The default config gives `5100 * 512 * 1024 ≈ 2.67B` tokens.
`save_every` is present in the config but currently unused; no checkpoints are written.

## Outputs

- **W&B:** `train_loss`, `val_loss`, `lr_hidden`, `lr_embed`, `tokens/sec`, and the final `run/finish_val_loss`.
- **`logs/<uuid>.txt`:** a copy of the training script plus PyTorch/CUDA and `nvidia-smi` info for reproducibility.

## Data source

Training data comes from [`kjj0/fineweb100B-gpt2`](https://huggingface.co/datasets/kjj0/fineweb100B-gpt2),
a GPT-2-tokenized version of FineWeb. The model structure follows the
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) / [llm.c](https://github.com/karpathy/llm.c)
lineage.
