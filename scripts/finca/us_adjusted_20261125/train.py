"""Refit the offline US predictors on adjusted targets."""
import argparse
import csv
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .prepare import HERE, ROOT
from ..paths import DATA
from scripts.experiments import peer_pca_replay as replay


def fit(market, seed, arrays, device, input_hash):
    out = HERE / market / f'seed{seed}'
    out.mkdir(exist_ok=True)
    signature = out / 'input_hash.txt'
    if (out / 'model.pt').exists() and signature.exists() and signature.read_text() == input_hash:
        return
    replay.seed_all(seed)
    torch.set_num_threads(2)
    hidden, labels, groups, val_hidden, val_labels = arrays
    model = replay.new_model().to(device)
    use_graph = device.type == 'cuda'
    adam = torch.optim.AdamW([dict(params=model.adapter.parameters(), lr=1e-4),
                             dict(params=model.head.parameters(), lr=3e-4)],
                            weight_decay=1e-4, fused=use_graph, capturable=use_graph)
    indices = torch.zeros(16, dtype=torch.long, device=device)
    mask = torch.ones(16, device=device)
    model.train()

    def step():
        adam.zero_grad(set_to_none=True)
        errors = F.smooth_l1_loss(model(hidden[indices]), labels[indices], reduction='none')
        loss = (errors.mean(dim=1) * mask).sum() / mask.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        adam.step()
        return loss.detach()

    if use_graph:
        graph = torch.cuda.CUDAGraph()
        initial = {name: p.detach().clone() for name, p in model.named_parameters()}
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph):
            capture_loss = step()
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.copy_(initial[name])
            for state in adam.state.values():
                for field in state.values():
                    field.zero_()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(gpu_rng)
    best, stale = np.inf, 0
    with (out / 'training.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch', 'validation_loss', 'best_epoch'])
        for epoch in range(1, 6):
            model.train()
            rng = np.random.default_rng(seed + epoch)
            batches = []
            for indices_for_date in groups:
                rows = rng.permutation(indices_for_date)
                batches.extend(rows[start:start + 16] for start in range(0, len(rows), 16))
            order = rng.permutation(len(batches))
            data = np.zeros((len(batches), 16), dtype=np.int64)
            weights = np.zeros(data.shape, dtype=np.float32)
            for i, j in enumerate(order):
                block = batches[j]
                data[i, :len(block)] = block
                weights[i, :len(block)] = 1.
            ids_all = torch.as_tensor(data, device=device)
            masks_all = torch.as_tensor(weights, device=device)
            for i in range(len(batches)):
                indices.copy_(ids_all[i])
                mask.copy_(masks_all[i])
                if use_graph:
                    graph.replay()
                else:
                    step()
            model.eval()
            with torch.inference_mode():
                prediction = torch.cat([model(part) for part in val_hidden.split(4096)])
                loss = F.smooth_l1_loss(prediction, val_labels).item()
            if not np.isfinite(loss):
                raise FloatingPointError(f'{market} {seed}: invalid loss')
            if loss < best:
                stale, best, best_epoch = 0, loss, epoch
                state = {name: p.detach().cpu().clone() for name, p in model.state_dict().items()}
            else:
                stale += 1
            writer.writerow([epoch, loss, best_epoch])
            stream.flush()
            print(market, seed, epoch, loss, flush=True)
            if stale == 2:
                break
    torch.save(state, out / 'model.tmp')
    (out / 'model.tmp').replace(out / 'model.pt')
    signature.write_text(input_hash)


def main(market):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cache = ROOT / f'.peer_pca_inputs/{market}/hidden_cache'
    reference = 'paper_suite_20261112' if market == 'nasdaq100' else '\u5171\u4eab\u8d44\u6e90'
    validation, = [p for p in DATA.rglob('hidden_val.npy')
                   if market in p.parts and reference in p.parts]
    tensors = []
    hashes = []
    for split, file in [('train', cache / 'hidden_train.npy'), ('val', validation)]:
        x = np.load(file)
        label_file = HERE / market / f'labels_{split}.npy'
        y = np.load(label_file)
        hashes.append(replay.sha(label_file))
        assert x.shape == (len(y), 512) and np.isfinite(y).all()
        tensors.extend([torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)])
    groups = list(pd.read_csv(cache / 'meta_train.csv', usecols=['as_of_date'])
                  .groupby('as_of_date', sort=True).indices.values())
    for seed in replay.SEEDS:
        fit(market, seed, (*tensors[:2], groups, *tensors[2:]), device, '\n'.join(hashes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('market', choices=['nasdaq100','sp500'])
    main(parser.parse_args().market)
