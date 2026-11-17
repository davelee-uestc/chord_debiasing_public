"""Shared paths, configuration, and command-line arguments."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = ['waterbirds', 'celeba', 'multinli', 'civilcomments', 'cremad', 'mead']


def load_config(dataset):
    return json.loads((ROOT / 'configs' / f'{dataset}.json').read_text())


def get_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs')
    args = parser.parse_args()
    if not args.run_dir.is_absolute():
        args.run_dir = ROOT / args.run_dir
    return args
