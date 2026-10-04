"""Reload one successful seed and verify it on the exact cached training data."""
import argparse
from pathlib import Path
import numpy as np
import torch
import experiment_runner as runner
import experiment_tasks as tasks
import parity8_align_20260924 as core


def predict(path, seed, device="cpu"):
    weights = torch.load(runner.project_path(path), map_location=device, weights_only=True)
    if weights['format_version'] != runner.FORMAT or weights['core_hash'] != runner.core_hash(core):
        raise runner.RunError('權重格式或核心版本不符')
    key = str(seed)
    if key not in weights['theta_star']:
        raise runner.RunError(f'seed {seed} 沒有首次成功權重；可用 seeds：{list(weights["theta_star"])}')
    cache = tasks.resolve_data_dir(weights['data_dir']) / weights['task'] / 'dataset.npz'
    if not cache.exists():
        raise runner.RunError('缺少訓練資料快取，請複製原 data 資料夾')
    X, y, meta = tasks.load_task(weights['task'], weights['data_dir'])
    if runner.data_identity(meta) != weights['data']:
        raise runner.RunError('資料 hash 或形狀不符')
    P = {k: v.to(device=device, dtype=torch.float64).unsqueeze(0) for k, v in weights['theta_star'][key].items()}
    with torch.no_grad():
        output = core.fwd(torch.tensor(X, dtype=torch.float64, device=device), P, weights['p'], weights['model'], weights['omega0'] or 1.0)
        labels = core.decide(output, weights['model'])[0].cpu().numpy()
    accuracy = float(np.mean(labels == y))
    return labels, accuracy, weights['conv_by_seed'][key]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='project-relative weights/*_theta_star.pt')
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    try:
        labels, accuracy, epoch = predict(runner.project_path(args.weights), args.seed, args.device)
    except (runner.RunError, tasks.DataError, OSError) as exc:
        parser.exit(1, f'推論錯誤：{exc}\n')
    print(f'seed={args.seed}, first_success_epoch={epoch}, K={len(labels)}, accuracy={accuracy:.6f}')
    if accuracy != 1.0:
        parser.exit(1, '首次成功權重未重現 100% 訓練準確率\n')


if __name__ == '__main__':
    main()
