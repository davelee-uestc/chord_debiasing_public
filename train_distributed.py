"""Train an audio target with data parallelism and gradient accumulation."""
import json
import os
import random
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from common import get_args, load_config
from data_utils import load_dataset, collate_audio
from models import build_target, target_forward, to_device


class AudioForward(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waves, mask):
        return target_forward(self.model, (waves, mask), 'audio')[0]


def main():
    args = get_args(__doc__)
    if len(args.datasets) != 1:
        raise ValueError('Select one dataset for distributed training.')
    dataset = args.datasets[0]
    cfg = load_config(dataset)
    params = cfg['train']
    rank = int(os.environ['LOCAL_RANK'])
    torch.set_num_threads(4)
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=5))
    world = dist.get_world_size()
    if params['batch_size'] % world:
        raise ValueError('The global batch size must be divisible by the number of GPUs.')
    target_batch = params['batch_size'] // world
    batch_size = max(n for n in range(1, min(target_batch, params['per_device_batch_size']) + 1)
                     if target_batch % n == 0)
    accumulation = target_batch // batch_size
    torch.manual_seed(cfg['target'])
    np.random.seed(cfg['target'])
    random.seed(cfg['target'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = build_target(dataset).to(rank)
    parallel = DistributedDataParallel(AudioForward(model), device_ids=[rank], static_graph=True)
    data = load_dataset(dataset, 'train')
    sampler = DistributedSampler(data, num_replicas=world, rank=rank, seed=cfg['target'], shuffle=True)
    loader = DataLoader(data, batch_size=batch_size, sampler=sampler,
                        num_workers=4, collate_fn=collate_audio, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['epochs'])
    torch.manual_seed(cfg['target'] + rank)
    np.random.seed(cfg['target'] + rank)
    random.seed(cfg['target'] + rank)
    for epoch in range(params['epochs']):
        sampler.set_epoch(epoch)
        parallel.train()
        started = time.monotonic()
        correct = seen = 0
        optimizer.zero_grad()
        for step, (x, labels) in enumerate(loader):
            x, labels = to_device(x, rank), labels.to(rank)
            logits = parallel(*x)
            loss = nn.functional.cross_entropy(logits, labels)
            if accumulation > 1:
                # Weight the final accumulation window by its actual sample count.
                window_size = min(target_batch, len(sampler) - (step // accumulation) * target_batch)
                loss = loss * (len(labels) / window_size)
            loss.backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                optimizer.step()
                optimizer.zero_grad()
            correct += int((logits.argmax(1) == labels).sum())
            seen += len(labels)
        scheduler.step()
        totals = torch.tensor([correct, seen], device=rank, dtype=torch.float64)
        dist.all_reduce(totals)
        if rank == 0:
            print(f'{dataset} epoch={epoch+1} train_accuracy={float(totals[0]/totals[1]):.6f} '
                  f'seconds={time.monotonic()-started:.1f}', flush=True)
        dist.barrier()
    if rank == 0:
        directory = args.run_dir / 'checkpoints' / dataset
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, directory / 'encoder.pt')
        (directory / 'erm_config.json').write_text(json.dumps(cfg, indent=2) + '\n')
        print(f'Saved {directory / "encoder.pt"}', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
