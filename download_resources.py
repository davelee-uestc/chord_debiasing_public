"""Download and unpack the supplied ERM models, CHORD weights, and caches."""
import argparse
import json
import tarfile

import gdown

from common import ROOT


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    source = json.loads((ROOT / 'configs/sources.json').read_text())['resources']
    archive = ROOT / 'downloads/chord_debiasing_public_resources.tgz'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        temporary = archive.with_name(archive.name + '.part')
        result = gdown.download(source, str(temporary), fuzzy=True, resume=True, use_cookies=False)
        if result is None:
            raise RuntimeError('Resource download failed. Check the Google Drive link and retry.')
        temporary.replace(archive)
    print(f'Extract {archive}', flush=True)
    with tarfile.open(archive, 'r:gz') as stream:
        stream.extractall(ROOT, filter='data')
    print('Resources ready. Next: bash reproduce.sh', flush=True)


if __name__ == '__main__':
    main()
