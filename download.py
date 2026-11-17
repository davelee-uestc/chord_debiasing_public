"""Download the main-experiment data and pretrained models from public sources."""
import argparse
import csv
import concurrent.futures
import json
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile

from common import ROOT, DATASETS, load_config


SOURCES = json.loads((ROOT / 'configs' / 'sources.json').read_text())


def download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Download {url} -> {path}', flush=True)
    temporary = path.with_name(path.name + '.part')
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(path)


def unpack(dataset):
    archive = ROOT / 'downloads' / f'{dataset}.tar.gz'
    download(SOURCES['data'][dataset]['archive'], archive)
    directory = ROOT / 'downloads' / dataset
    directory.mkdir(parents=True, exist_ok=True)
    print(f'Extract {archive}', flush=True)
    with tarfile.open(archive) as stream:
        stream.extractall(directory, filter='data')
    return directory


def copy_tree(source, target):
    shutil.copytree(source, target, dirs_exist_ok=True)


def mead_audio(root):
    """Decode the fixed MEAD split from the public nested audio archive."""
    with (ROOT / 'configs/splits/mead.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    missing = [row for row in rows if not (root / row['wav_path']).exists()]
    if not missing:
        return
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('Install ffmpeg before downloading MEAD.')
    archive = ROOT / 'downloads/MEAD.zip'
    download(SOURCES['data']['mead']['archive'], archive)

    def decode(row, content):
        target = root / row['wav_path']
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.part.wav')
        with tempfile.NamedTemporaryFile(suffix='.m4a') as source:
            source.write(content)
            source.flush()
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1',
                            '-i', source.name, '-ac', '1', '-ar', '16000',
                            '-c:a', 'pcm_s16le', '-threads', '1', '-y', str(temporary)], check=True)
        temporary.replace(target)

    wanted = {(row['actor'], 'audio/' + '/'.join(row['wav_path'].split('/')[2:]).removesuffix('.wav') + '.m4a'): row
              for row in missing}
    with zipfile.ZipFile(archive) as outer, concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        names = set(outer.namelist())
        for actor in sorted({row['actor'] for row in missing}):
            name = f'{actor}/audio(1).tar' if f'{actor}/audio(1).tar' in names else f'{actor}/audio.tar'
            pending = []
            with outer.open(name) as source, tarfile.open(fileobj=source, mode='r|*') as inner:
                for member in inner:
                    row = wanted.get((actor, member.name))
                    if row is None:
                        continue
                    pending.append(pool.submit(decode, row, inner.extractfile(member).read()))
                    if len(pending) == 32:
                        for future in pending:
                            future.result()
                        pending = []
            for future in pending:
                future.result()
            print(f'MEAD audio ready: {actor}', flush=True)
    if any(not (root / row['wav_path']).exists() for row in rows):
        raise RuntimeError('MEAD archive is missing files listed in the supplied split.')


def data(dataset):
    root = ROOT / 'data' / dataset
    root.mkdir(parents=True, exist_ok=True)
    source = SOURCES['data'][dataset]
    if dataset == 'waterbirds':
        if not (root / 'metadata.csv').exists():
            directory = unpack(dataset)
            copy_tree(next(directory.rglob('metadata.csv')).parent, root)
    elif dataset == 'celeba':
        if not (root / 'img_align_celeba' / '202599.jpg').exists():
            archive = ROOT / 'downloads/celeba.zip'
            download(source['archive'], archive)
            with zipfile.ZipFile(archive) as stream:
                stream.extractall(root)
        download(source['metadata'], root / 'metadata.csv')
    elif dataset == 'multinli':
        names = ['cached_train_bert-base-uncased_128_mnli', 'cached_dev_bert-base-uncased_128_mnli',
                 'cached_dev_bert-base-uncased_128_mnli-mm']
        if not all((root / n).exists() for n in names):
            directory = unpack(dataset)
            for name in names:
                shutil.copy2(next(directory.rglob(name)), root / name)
        download(source['metadata'], root / 'metadata_random.csv')
    elif dataset == 'civilcomments':
        target = root / 'civilcomments_v1.0' / 'all_data_with_identities.csv'
        download(source['csv'], target)
    elif dataset == 'mead':
        mead_audio(root)
        shutil.copy2(ROOT / 'configs/splits/mead.csv', root / 'manifest.csv')
    else:
        for name, url in source['files'].items():
            download(url, root / name)
        shutil.copy2(ROOT / 'configs' / 'splits' / f'{dataset}.csv', root / 'manifest.csv')
    print(f'Data ready: {dataset}', flush=True)


def pretrained(datasets):
    from huggingface_hub import snapshot_download
    names = set()
    for dataset in datasets:
        cfg = load_config(dataset)
        names.add({'vision': 'resnet50', 'text': 'bert', 'audio': 'wav2vec2'}[cfg['modality']])
        names.update(n.removesuffix('_vision') for n in cfg['views'])
    for name in sorted(names):
        root = ROOT / 'pretrained'
        if name in SOURCES['models']['files']:
            download(SOURCES['models']['files'][name], root / f'{name}.pt')
        else:
            directory = root / name
            weights = 'model.safetensors' if (directory / 'model.safetensors').exists() else 'pytorch_model.bin'
            snapshot_download(SOURCES['models']['huggingface'][name], local_dir=directory,
                              allow_patterns=['*.json', '*.txt', '*.model', weights])
        print(f'Pretrained model ready: {name}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    args = parser.parse_args()
    for dataset in args.datasets:
        data(dataset)
    pretrained(args.datasets)
    print('Ready. Next: bash train_models.sh, then bash run_chord.sh')
