"""Fit the original SmoothL1 offline Adapter recipe to adjusted labels."""
import argparse
import csv
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .prepare import CACHE, HERE
from . import train_from_cache_20260820 as original
from ..paths import DATA, unique_path

from scripts.experiments import peer_pca_replay as replay

def train(seed, splits, prefix):
    out = HERE / f'{prefix}seed{seed}'
    out.mkdir(exist_ok=True)
    if (out / 'model.pt').exists():
        return
    original.seed(seed)
    device = splits['train'][0].device
    model = original.Model(512, 10, 'C', 64, .1).to(device)
    optimizer = torch.optim.AdamW([{'params': model.head.parameters(), 'lr': 3e-4},
                                   {'params': model.adapter.parameters(), 'lr': 1e-4}],
                                  weight_decay=1e-4, fused=device.type == 'cuda')
    train_hidden, train_labels, train_dates = splits['train']
    val_hidden, val_labels, val_dates = splits['val']
    val_indices = list(original.DateSampler(val_dates, 32, 1, False))
    best_loss, best_state, stale = float('inf'), None, 0
    with (out / 'training.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch', 'training_loss', 'validation_loss', 'best_epoch'])
        for epoch in range(1, 6):
            model.train()
            losses = []
            train_indices = list(original.DateSampler(train_dates, 32, seed + epoch, True))
            # Leave the batch and its dropout draw identical to the original date-based sampler.
            for rows in train_indices:
                optimizer.zero_grad(set_to_none=True)
                loss = F.smooth_l1_loss(model(train_hidden[rows]), train_labels[rows])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                losses.append(loss.detach())
            model.eval()
            with torch.inference_mode():
                val = torch.stack([F.smooth_l1_loss(model(val_hidden[rows]), val_labels[rows])
                                   for rows in val_indices]).mean().item()
            if not np.isfinite(val):
                raise ValueError(f'invalid validation loss for {seed}')
            if val < best_loss:
                best_loss, stale, best_epoch = val, 0, epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            else:
                stale += 1
            train_loss = torch.stack(losses).mean().item()
            writer.writerow([epoch, train_loss, val, best_epoch])
            stream.flush()
            print(f'seed {seed} epoch {epoch}: train={train_loss:.7f} validation={val:.7f}', flush=True)
            if stale >= 2:
                break
    target = out / 'model.tmp'
    torch.save(best_state, target)
    target.replace(out / 'model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=replay.SEEDS)
    parser.add_argument('--raw-control', action='store_true')
    parser.add_argument('--parallel-copy', action='store_true')
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    data = {}
    supplement = unique_path(DATA, '*/csi300/*/reference/hidden_cache')
    for split in ('train', 'val'):
        hidden = np.load((CACHE if split == 'train' else supplement) / f'hidden_{split}.npy')
        labels = np.load((supplement if args.raw_control else HERE) / f'labels_{split}.npy')
        dates = pd.read_csv(CACHE / f'meta_{split}.csv', usecols=['as_of_date']).as_of_date.tolist()
        if not len(hidden) == len(labels) == len(dates):
            raise ValueError(f'{split}: misaligned data')
        data[split] = torch.as_tensor(hidden, device=device), torch.as_tensor(labels, device=device), dates
    prefix = 'raw_parallel_' if args.parallel_copy else 'raw_' if args.raw_control else ''
    for seed in args.seeds:
        train(seed, data, prefix)
