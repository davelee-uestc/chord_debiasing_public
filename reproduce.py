"""Run CHORD and evaluate ERM, extracting missing features as needed."""
import csv
import json

import numpy as np
import torch

from chord import (neutralize_prototypes, fit_feature_transform,
                  validate_view_partition, select_geometry,
                  aggregate_reversal_weights, train_balanced_classifier)
from ssc_common import group_metrics, identify, proto, nearest_prototype, within_class_whiten

from common import ROOT, load_config, get_args


def build_view_partitions(dataset, val, setting, run_dir):
    """Build target and pretrained partitions before deciding whether geo is needed."""
    features, labels = val['f'].float(), val['y'].flatten().long()
    n_classes = setting['classes']
    directory = run_dir / 'cache' / dataset
    if dataset in ('waterbirds', 'celeba'):
        cached = torch.load(directory / 'views.pt', map_location='cpu', weights_only=True)
        assert torch.equal(cached['y'].flatten(), labels)
        views = {name: cached['views'][name] for name in ['anchor'] + setting['views']}
    else:
        views = {'anchor': identify(labels, features, n_classes)}
        for name in setting['views']:
            path = directory / 'views' / f'{name}.pt'
            cached = torch.load(path, map_location='cpu', weights_only=True)
            if 'y' in cached:
                assert torch.equal(cached['y'].flatten(), labels)
            views[path.stem] = identify(labels, cached['f'].float(), n_classes)
    return {'anchor': views['anchor'], **{k: views[k] for k in sorted(views) if k != 'anchor'}}


def evaluation_groups(dataset, test):
    attributes = load_config(dataset)['attributes']
    if not attributes:
        return {'group': test['p'].numpy().reshape(-1)}
    with (ROOT / 'configs' / 'splits' / f'{dataset}.csv').open() as stream:
        rows = [row for row in csv.DictReader(stream) if row['split'] == 'test']
    labels = np.array([int(row['label']) for row in rows])
    assert np.array_equal(labels, test['y'].numpy().reshape(-1))
    groups = {}
    for name in attributes:
        values = sorted({row[name] for row in rows})
        index = {value: i for i, value in enumerate(values)}
        groups[name] = labels * len(values) + np.array([index[row[name]] for row in rows])
    return groups


def evaluate_predictions(dataset, predictions, test):
    metrics = {name: group_metrics(predictions, test['y'].numpy(), groups)
               for name, groups in evaluation_groups(dataset, test).items()}
    return {'mean_accuracy': float((predictions == test['y'].numpy().reshape(-1)).mean()),
            'wga': float(np.mean([m['wga'] for m in metrics.values()])),
            'attributes': metrics}


def evaluate_erm(dataset, test, run_dir):
    """Evaluate the saved ERM classifier on its target model's test features."""
    checkpoint = run_dir / 'checkpoints' / dataset / 'encoder.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    prefix = 'head' if load_config(dataset)['modality'] == 'audio' else 'fc'
    if prefix == 'fc' and 'classifier.weight' in state:
        prefix = 'classifier'
    logits = torch.nn.functional.linear(
        test['f'].float(), state[f'{prefix}.weight'].float(), state[f'{prefix}.bias'].float())
    result = evaluate_predictions(dataset, logits.argmax(1).numpy(), test)
    result.update(target_seed=load_config(dataset)['target'])
    print(f"{dataset} ERM: mAcc={100*result['mean_accuracy']:.4f} "
          f"WGA={100*result['wga']:.4f}", flush=True)
    return result


def run_dataset(dataset, device, run_dir):
    from extract_features import ensure_cache
    setting = load_config(dataset)
    print(f'\n{dataset}: ERM target seed={setting["target"]}, CHORD seeds={setting["seeds"]}', flush=True)
    checkpoint = run_dir / 'checkpoints' / dataset / 'encoder.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'{checkpoint}: run bash train_models.sh --datasets {dataset} --run-dir {run_dir} first')
    ensure_cache(dataset, run_dir, device)
    directory = run_dir / 'cache' / dataset
    val = torch.load(directory / 'val.pt', map_location='cpu', weights_only=True)
    test = torch.load(directory / 'test.pt', map_location='cpu', weights_only=True)
    features, labels = val['f'].float(), val['y'].flatten().long()
    n_classes, steps = setting['classes'], setting['transform']

    # Propose within-class partitions in each view.
    view_partitions = build_view_partitions(dataset, val, setting, run_dir)

    # Fit the target transformation used by the final classifier.
    global_prototypes = proto(labels, features, None, n_classes)[0][:-1]
    target_outliers = nearest_prototype(global_prototypes, features) != labels
    target_centers = neutralize_prototypes(labels, features, target_outliers, n_classes)
    target_transform = fit_feature_transform(features, labels, target_centers, steps, device)
    transformed_val = target_transform(features.to(device)).detach()
    transformed_test = target_transform(test['f'].float().to(device)).detach()
    original_predictions = nearest_prototype(global_prototypes.to(device), features.to(device))
    stable_mask = original_predictions == labels.to(device)
    original_errors = (~stable_mask).cpu()

    # Validate each proposal by wrong-to-right prototype reversals in target space.
    view_reversals = {}
    for name, outliers in view_partitions.items():
        view_reversals[name] = validate_view_partition(
            features, labels, outliers, original_errors, n_classes, steps, device)
        print(f"  {name}: {int(view_reversals[name].sum())} reversals", flush=True)

    # Check one fixed-scale geo proposal using fitting-split evidence only.
    geometry_outliers = identify(
        labels, within_class_whiten(labels, features, n_classes, 0.1), n_classes)
    geometry_reversals = validate_view_partition(
        features, labels, geometry_outliers, original_errors, n_classes, steps, device)
    use_geometry, geometry_evidence = select_geometry(
        geometry_outliers, geometry_reversals, view_reversals['anchor'],
        [view_reversals[name] for name in setting['views']], original_errors)
    print(f'  geo={"on" if use_geometry else "off"}; {geometry_evidence}', flush=True)
    if use_geometry:
        view_reversals['tight_lda0.1'] = geometry_reversals

    # Aggregate the anchor first, followed by the other views in name order.
    view_reversals = {'anchor': view_reversals['anchor'],
                     **{k: view_reversals[k] for k in sorted(view_reversals) if k != 'anchor'}}

    # Aggregate class-normalized evidence across views into sampling weights.
    reversal_weights = aggregate_reversal_weights(view_reversals, labels, n_classes)

    # Train and evaluate one balanced classifier for each seed.
    checkpoint_dir = run_dir / 'checkpoints' / dataset
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in setting['seeds']:
        head = train_balanced_classifier(
            transformed_val, labels.to(device), reversal_weights, stable_mask,
            n_classes, setting['head'], seed, device)
        predictions = head(transformed_test).argmax(1).detach().cpu().numpy()
        # Evaluation attributes are used only after prediction.
        row = {'seed': seed, **evaluate_predictions(dataset, predictions, test)}
        results.append(row)
        torch.save({'transformation.w': target_transform.w.cpu(),
                    'transformation.b': target_transform.b.cpu(),
                    'fc.weight': head.weight.detach().cpu(),
                    'fc.bias': head.bias.detach().cpu()},
                   checkpoint_dir / f'chord_seed{seed}.pt')
        print(f"{dataset} seed={seed}: mAcc={100*row['mean_accuracy']:.4f} "
              f"WGA={100*row['wga']:.4f}", flush=True)
    summary = {'dataset': dataset, 'config': setting, 'views': list(view_reversals),
               'geometry_evidence': geometry_evidence, 'geo_enabled': use_geometry,
               'geo_scale': 0.1 if use_geometry else None,
               'corrected_count': int((reversal_weights > 0).sum()),
               'mean_accuracy': float(np.mean([r['mean_accuracy'] for r in results])),
               'wga': float(np.mean([r['wga'] for r in results])),
               'wga_std': float(np.std([r['wga'] for r in results])), 'seeds': results,
               'erm': evaluate_erm(dataset, test, run_dir)}
    output = run_dir / 'results'
    output.mkdir(exist_ok=True)
    (output / f'{dataset}.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    args = get_args(__doc__)
    torch.set_num_threads(4)
    summaries = [run_dataset(dataset, args.device, args.run_dir) for dataset in args.datasets]
    from report_results import write_summary
    write_summary(summaries, args.run_dir)


if __name__ == '__main__':
    main()
