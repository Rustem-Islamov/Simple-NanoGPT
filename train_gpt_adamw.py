import os
import sys
import uuid
import time
import wandb
import torch
import torch.distributed as dist
import torch._inductor.config as config

import random
import numpy as np

from argparse import ArgumentParser
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from utils.gpt import GPT, GPTConfig
from utils.dataloader import DistributedDataLoader
from utils.configs import load_config

def seed_everything(seed: int, rank: int):
    seed = seed + rank  # IMPORTANT: different seed per rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

os.environ.setdefault("NCCL_TIMEOUT", "30")

with open(sys.argv[0]) as f:
    code = f.read() 


parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
cli_args = parser.parse_args()
args = load_config(cli_args.config)


assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])


BASE_SEED = getattr(args, "seed", 1337)
seed_everything(BASE_SEED, ddp_rank)

device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

try:
    if master_process:
        print("======== Arguments ========")
        print(args)
        print("===========================")
        
    # convenience variables
    B, T = args.device_batch_size, args.sequence_length
    T_eval = args.sequence_length_eval
    # calculate the number of steps to take in the val loop.
    assert args.val_tokens % (B * T_eval * ddp_world_size) == 0
    val_steps = args.val_tokens // (B * T_eval * ddp_world_size)
    # calculate the steps of gradient accumulation required to attain the desired global batch size.
    print(args.batch_size, B, ddp_world_size)
    assert args.batch_size % (B * ddp_world_size) == 0
    train_accumulation_steps = args.batch_size // (B * ddp_world_size)

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = DistributedDataLoader(args.input_val_bin, B, T_eval, ddp_rank, ddp_world_size)
    if master_process:
        print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
        print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")

    num_vocab = 50304
    model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd))
    model = model.cuda()
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True # suggested by @Chillee

    import torch._dynamo
    torch._dynamo.config.optimize_ddp = False

    model = DDP(model, device_ids=[ddp_local_rank])
    model = torch.compile(model)

    raw_model = model.module # always contains the "raw" unwrapped model
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    # --- split params: embedding/lm_head get lr * emb_lr_ratio, everything else gets lr ---
    # With weight tying, wte.weight and lm_head.weight are the SAME nn.Parameter.
    # Quick sanity check that tying is via shared storage (not a copy):
    if master_process:
        tied = (raw_model.transformer.wte.weight is raw_model.lm_head.weight)
        print(f"embedding/lm_head weight tying (shared tensor): {tied}")

    # Collect embedding params, dedup by identity so a tied tensor is grouped exactly once.
    _embed_candidates = list(raw_model.transformer.wte.parameters()) \
                      + list(raw_model.lm_head.parameters())
    _seen = set()
    embed_params = [p for p in _embed_candidates
                    if id(p) not in _seen and not _seen.add(id(p))]
    embed_ids = {id(p) for p in embed_params}

    # Everything not in the embed group goes to the hidden group.
    hidden_params = [p for p in raw_model.parameters() if id(p) not in embed_ids]

    # Verify the split is exhaustive and non-overlapping.
    n_embed = sum(p.numel() for p in embed_params)
    n_hidden = sum(p.numel() for p in hidden_params)
    n_total = sum(p.numel() for p in raw_model.parameters())
    assert n_embed + n_hidden == n_total, \
        f"param split mismatch: {n_embed} + {n_hidden} != {n_total}"
    # Scale embed weight decay down by the LR ratio so the *effective* per-step
    # decay (lr * wd) is equal across groups:
    #   hidden: lr * weight_decay
    #   embed : (lr * emb_lr_ratio) * (weight_decay / emb_lr_ratio) = lr * weight_decay
    embed_weight_decay = args.weight_decay / args.emb_lr_ratio
    if master_process:
        print(f"embed params: {n_embed:,} (lr={args.lr * args.emb_lr_ratio:.2e}, "
              f"wd={embed_weight_decay:.2e}) | "
              f"hidden params: {n_hidden:,} (lr={args.lr:.2e}, wd={args.weight_decay:.2e})")

    optimizer1 = torch.optim.AdamW(
        [
            {"params": hidden_params, "lr": args.lr,
             "weight_decay": args.weight_decay},
            {"params": embed_params,  "lr": args.lr * args.emb_lr_ratio,
             "weight_decay": embed_weight_decay},
        ],
        betas=(args.beta1, args.beta2),
        fused=True,  # faster fused CUDA kernel
    )
    optimizers = [optimizer1]

    def get_lr(it):
        assert it <= args.num_iterations
        if it < args.warmup_iters:
            ratio = (it + 1) / args.warmup_iters
        elif it < args.num_iterations - args.warmdown_iters:
            ratio = 1.0
        else:
            ratio = (args.num_iterations - 1 - it) / args.warmdown_iters
        min_ratio = 1e-5 / args.lr
        return max(ratio, min_ratio)

    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

    if master_process:
        run_id = str(uuid.uuid4())
        wandb.init(
            project=args.project, 
            entity=args.entity,
            name=args.run, 
            config=vars(args)
        )
        logdir = 'logs/%s/' % run_id
        os.makedirs(logdir, exist_ok=True)
        logfile = 'logs/%s.txt' % run_id
        with open(logfile, "w") as f:
            f.write('='*100 + '\n')
            f.write(code)
            f.write('='*100 + '\n')
            f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
            import subprocess
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            f.write(f'{result.stdout}\n')
            f.write('='*100 + '\n')

    training_time_ms = 0
    # ----------------------------
    torch.cuda.synchronize()
    global_start_time = time.time()

    model.eval()
    val_loader.reset()

    val_loss = torch.zeros((), device=device)
    for _ in range(val_steps):
        x_val, y_val = val_loader.next_batch()
        with ctx:
            _, loss = model(x_val, y_val, return_logits=False)
            val_loss += loss.detach()

    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
    val_loss /= val_steps

    if master_process:
        wandb.log({
            "val_loss": val_loss.item(),
        })
        
    val_loader.reset()
    train_loader.reset()
    x, y = train_loader.next_batch()
    train_iter_start = time.time()  # timer for training iterations only

    for step in tqdm(range(0, args.num_iterations + 1)):
        last_step = (step == args.num_iterations)

        # --------------- EVALUATION -----------------
        val_loss = None
        if last_step or (args.val_loss_every > 0 and step > 0 and step % args.val_loss_every == 0):
            model.eval()
            val_loader.reset()
            val_loss = torch.zeros((), device=device)
            for _ in range(val_steps):
                x_val, y_val = val_loader.next_batch()
                with ctx:
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss.detach()
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss /= val_steps

        if last_step:
            break

        # --------------- TRAINING SECTION -----------------.
        train_iter_start = time.time()
        model.train()
        batch_tokens = B * T * ddp_world_size * train_accumulation_steps

        for i in range(1, train_accumulation_steps+1):
            with ctx:
                _, loss = model(x, y, return_logits=False)
                train_loss = loss.detach()
            x, y = train_loader.next_batch()
            if i < train_accumulation_steps:
                with model.no_sync():
                    loss.backward()
            else:
                loss.backward()
        for p in model.parameters():
            p.grad /= train_accumulation_steps
        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()
        model.zero_grad(set_to_none=True)

        # ----------------- TRAIN TIMING -----------------
        torch.cuda.synchronize()
        train_iter_end = time.time()
        iter_elapsed_sec = (train_iter_end - train_iter_start)
        tokens_per_sec = batch_tokens / iter_elapsed_sec

        # ----------------- LOGGING -----------------
        if master_process:
            last_lrs = sched.get_last_lr()  # one entry per param group
            lr = last_lrs[0]                 # hidden-layer LR (kept for final-block reuse)
            wandb.log({
                "train_loss": train_loss.item(),
                "val_loss": val_loss.item() if val_loss else None,
                "lr_hidden": last_lrs[0],
                "lr_embed": last_lrs[1],
                "tokens/sec": tokens_per_sec,
                "step": step
            })
            
    model.eval()
    val_loader.reset()

    val_loss = torch.zeros((), device=device)

    for _ in range(val_steps):
        x_val, y_val = val_loader.next_batch()
        with ctx:
            _, loss = model(x_val, y_val, return_logits=False)
            val_loss += loss.detach()

    # ALL ranks must participate
    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
    val_loss /= val_steps

    # ONLY rank 0 logs
    if master_process:
        wandb.log({
            "run/finish_val_loss": val_loss.item(),
        })

    if master_process:
        last_lrs = sched.get_last_lr()
        wandb.log({
            "train_loss": train_loss.item(),
            "val_loss": val_loss.item() if val_loss else None,
            "lr_hidden": last_lrs[0],
            "lr_embed": last_lrs[1],
            "tokens/sec": tokens_per_sec,
        })
        wandb.finish()
        print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    dist.destroy_process_group()

except Exception as e:
    print(f"[rank {ddp_rank}] Exception: {repr(e)}", flush=True)
    raise
finally:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
