"""Train and save the target models."""
import random
import json
import time
import subprocess
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import ROOT, get_args, load_config
from data_utils import load_dataset, collate_audio
from models import build_target, target_forward, to_device


def train(dataset, run_dir, device):
    cfg = load_config(dataset)
    params = cfg['train']
    path = run_dir / 'checkpoints' / dataset / 'encoder.pt'
    if path.exists():
        print(f'Already trained: {path}', flush=True)
        return
    if 'per_device_batch_size' in params:
        available = torch.cuda.device_count()
        if available == 0:
            raise RuntimeError('Audio distributed training requires a CUDA GPU.')
        devices = max(n for n in range(1, min(available, params['batch_size']) + 1)
                      if params['batch_size'] % n == 0)
        subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                        f'--nproc_per_node={devices}', str(ROOT / 'train_distributed.py'),
                        '--datasets', dataset, '--run-dir', str(run_dir)], check=True)
        return
    torch.manual_seed(cfg['target'])
    np.random.seed(cfg['target'])
    random.seed(cfg['target'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = build_target(dataset).to(device)
    data = load_dataset(dataset, 'train')
    loader = DataLoader(data, batch_size=params['batch_size'], shuffle=True,
                        num_workers=4, collate_fn=collate_audio if cfg['modality'] == 'audio' else None)
    if cfg['modality'] == 'vision':
        optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=.9, weight_decay=params['weight_decay'])
    elif cfg['modality'] == 'text':
        from transformers import AdamW
        decay, no_decay = [], []
        for name, parameter in model.named_parameters():
            (no_decay if 'bias' in name or 'LayerNorm.weight' in name else decay).append(parameter)
        optimizer = AdamW([{'params': decay, 'weight_decay': params['weight_decay']},
                           {'params': no_decay, 'weight_decay': 0}], lr=params['lr'], eps=1e-8)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    if cfg['modality'] == 'text':
        from transformers import get_linear_schedule_with_warmup
        # Advance the learning-rate schedule once per epoch.
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, params['epochs'])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['epochs'])
    for epoch in range(params['epochs']):
        started = time.monotonic()
        model.train()
        seen = correct = 0
        for x, labels in tqdm(loader, desc=f'{dataset} epoch {epoch+1}/{params["epochs"]}'):
            x, labels = to_device(x, device), labels.to(device)
            logits, _ = target_forward(model, x, cfg['modality'])
            loss = torch.nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += int((logits.argmax(1) == labels).sum())
            seen += len(labels)
        scheduler.step()
        elapsed = time.monotonic() - started
        print(f'{dataset} epoch={epoch+1} train_accuracy={correct/seen:.6f} '
              f'seconds={elapsed:.1f} samples_per_second={seen/elapsed:.1f}', flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), path)
    (path.parent / 'erm_config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    print(f'Saved last-epoch model: {path}', flush=True)


if __name__ == '__main__':
    args = get_args(__doc__)
    torch.set_num_threads(4)
    for dataset in args.datasets:
        train(dataset, args.run_dir, args.device)
