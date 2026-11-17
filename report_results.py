"""Print ERM and CHORD tables from the results of completed runs."""
import json
import numpy as np
from common import DATASETS, get_args, load_config


def write_summary(summaries, run_dir):
    lines = ['Dataset             ERM mAcc  ERM WGA  CHORD mAcc CHORD WGA CHORD std']
    for row in summaries:
        lines.append(f"{row['dataset']:18s} {100*row['erm']['mean_accuracy']:8.2f} "
              f"{100*row['erm']['wga']:8.2f} {100*row['mean_accuracy']:9.2f} "
              f"{100*row['wga']:8.2f} {100*row['wga_std']:8.2f}")
    for title, datasets in [('Main', DATASETS[:4]), ('Speech', DATASETS[4:])]:
        rows = [r for r in summaries if r['dataset'] in datasets]
        if len(rows) == len(datasets):
            lines.append(f"{title} mean: ERM mAcc={100*np.mean([r['erm']['mean_accuracy'] for r in rows]):.2f} "
                  f"WGA={100*np.mean([r['erm']['wga'] for r in rows]):.2f}; "
                  f"CHORD mAcc={100*np.mean([r['mean_accuracy'] for r in rows]):.2f} "
                  f"WGA={100*np.mean([r['wga'] for r in rows]):.2f}")

    lines.append('ERM: one fixed target seed per dataset; CHORD: classifier seeds 0-9, std ddof=0.')

    for row in summaries:
        if load_config(row['dataset'])['attributes']:
            lines.append('')
            lines.append(row['dataset'] + ' attributes:       ERM WGA  CHORD WGA  CHORD std')
            for name, metric in row['erm']['attributes'].items():
                values = [r['attributes'][name]['wga'] for r in row['seeds']]
                lines.append(f"  {name:26s} {100*metric['wga']:8.2f} "
                             f"{100*np.mean(values):9.2f} {100*np.std(values):9.2f}")

    report = '\n'.join(lines) + '\n'
    (run_dir / 'results' / 'summary.txt').write_text(report)
    print('\n' + report)


if __name__ == '__main__':
    args = get_args(__doc__)
    rows = [json.loads((args.run_dir / 'results' / f'{dataset}.json').read_text())
            for dataset in args.datasets]
    write_summary(rows, args.run_dir)
