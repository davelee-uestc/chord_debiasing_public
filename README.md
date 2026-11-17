# Cross-view Harmonic Outlier Reversal for Debiasing (CHORD)

Code for **Cross-view Harmonic Outlier Reversal for Debiasing (CHORD)**, based on the [NSF implementation](https://github.com/davelee-uestc/nsf_debiasing).

## Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davelee-uestc/chord_debiasing_public/blob/main/reproduce_colab.ipynb)

Use [the Colab notebook](reproduce_colab.ipynb) to clone this repository, install the pinned environment, download the supplied resources, and reproduce the six-dataset results. Select a **T4 GPU** runtime and run the cells in order. The notebook displays live command output, plots ERM and CHORD worst-group accuracy, and lets you download the results, figure, and regenerated classifier weights.

The resource archive is approximately **5.27 GB**. This workflow uses the provided ERM models and feature caches; it does not retrain the encoders from raw data. No Google Drive mount is required.

## Environment

Use Linux, Python 3.12, and an NVIDIA GPU with CUDA support. Install FFmpeg for the audio download workflow.

```bash
git clone https://github.com/davelee-uestc/chord_debiasing_public.git
cd chord_debiasing_public
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
sudo apt-get install ffmpeg
```

## Reproduce with supplied resources

Download and unpack the [supplied resources](https://drive.google.com/file/d/1mDpLaRinvDXcKTFLj92iOwLIxUdjm1R0/view?usp=sharing), then reproduce the main tables:

```bash
bash download_resources.sh
bash reproduce.sh
```

The download script saves the archive in `downloads/` and extracts the ERM models, CHORD weights, and feature caches into `resources/`. If these resources are already present, skip the download command.

`reproduce.sh` evaluates ERM and reruns CHORD from the supplied features. Results are written to `resources/results/summary.txt`; regenerated CHORD weights are saved under `resources/checkpoints/`.

## Reproduce from data

Download the data and pretrained encoders, train the target models, generate their feature caches, and run CHORD:

```bash
bash download.sh
bash train_models.sh
bash extract_cache.sh
bash run_chord.sh
bash summarize.sh
```

Data and pretrained encoders are saved in `data/` and `pretrained/`. Models, caches, and results from this workflow are saved in `runs/`. Missing caches are also generated automatically by `run_chord.sh`.

To run an individual dataset, pass `--datasets waterbirds`, `celeba`, `multinli`, `civilcomments`, `cremad`, or `mead` to each command. Commands that produce models or results accept `--run-dir <directory>`. Use `--device cuda:0` to select a GPU for single-device training and evaluation, or `CUDA_VISIBLE_DEVICES` to select GPUs for MEAD training.

The supplied-cache route reproduces the provided models. Training new models and regenerating features can yield different numerical results.

## License

GPL-3.0; see `LICENSE`. Datasets and pretrained models retain their original licenses.
