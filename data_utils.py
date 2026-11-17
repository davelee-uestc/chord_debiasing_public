"""Dataset loaders with fixed splits and sample order."""
import io
import pickle

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from common import ROOT, load_config


class InputFeatures:
    # Compatibility record for MultiNLI archives created with utils_glue.
    pass


class FeatureUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'utils_glue' and name == 'InputFeatures':
            return InputFeatures
        return super().find_class(module, name)


class LegacyPickle:
    __name__ = 'pickle'
    Unpickler = FeatureUnpickler
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)


def image_transform(train=False):
    from torchvision import transforms as T
    crop = [T.RandomResizedCrop((224, 224), scale=(0.7, 1.0)), T.RandomHorizontalFlip()] if train else [T.Resize((256, 256)), T.CenterCrop((224, 224))]
    return T.Compose(crop + [T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225))])


class Images(Dataset):
    def __init__(self, dataset, split, transform=None):
        self.root = ROOT / 'data' / dataset
        rows = pd.read_csv(self.root / 'metadata.csv')
        self.rows = rows[rows.split == ['train', 'val', 'test'].index(split)].reset_index(drop=True)
        self.labels = torch.tensor(self.rows.y.to_numpy()).long()
        self.groups = torch.tensor(self.rows['place'].to_numpy()).long()
        self.transform = transform or image_transform(split == 'train')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image = Image.open(self.root / self.rows.iloc[index].img_filename).convert('RGB')
        return self.transform(image), self.labels[index]


class Text(Dataset):
    def __init__(self, dataset, split):
        from transformers import BertTokenizer, BertTokenizerFast
        tokenizer = BertTokenizerFast if dataset == 'multinli' else BertTokenizer
        self.tokenizer = tokenizer.from_pretrained(ROOT / 'pretrained' / 'bert')
        root = ROOT / 'data' / dataset
        self.dataset = dataset
        if dataset == 'multinli':
            metadata = pd.read_csv(root / 'metadata_random.csv')
            objects = []
            for name in ('cached_train_bert-base-uncased_128_mnli', 'cached_dev_bert-base-uncased_128_mnli', 'cached_dev_bert-base-uncased_128_mnli-mm'):
                objects.extend(torch.load(root / name, map_location='cpu', weights_only=False, pickle_module=LegacyPickle))
            indices = np.flatnonzero(metadata.split.to_numpy() == ['train', 'val', 'test'].index(split))
            rows = metadata.iloc[indices]
            self.tokens = torch.tensor([[objects[i].input_ids, objects[i].input_mask, objects[i].segment_ids] for i in indices]).transpose(1, 2)
            self.labels = torch.tensor(rows.gold_label.to_numpy()).long()
            self.groups = torch.tensor(rows.sentence2_has_negation.to_numpy()).long()
        else:
            rows = pd.read_csv(root / 'civilcomments_v1.0' / 'all_data_with_identities.csv')
            rows = rows[rows.split == split]
            self.text = rows.comment_text.fillna('').tolist()
            self.labels = torch.tensor((rows.toxicity >= .5).to_numpy()).long()
            cols = ['male', 'female', 'LGBTQ', 'black', 'white', 'christian', 'muslim', 'other_religions']
            self.groups = torch.tensor((rows[cols] >= .5).any(axis=1).to_numpy()).long()


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.dataset == 'multinli':
            return self.tokens[index], self.labels[index]
        encoded = self.tokenizer(self.text[index], padding='max_length', truncation=True, max_length=220, return_tensors='pt')
        x = torch.stack([encoded[k][0] for k in ('input_ids', 'attention_mask', 'token_type_ids')], -1)
        return x, self.labels[index]

    def strings(self, start, stop):
        if self.dataset != 'multinli':
            return self.text[start:stop], None
        pairs = []
        for row in self.tokens[start:stop, :, 0].tolist():
            first = row.index(102)
            second = row.index(102, first + 1)
            pairs.append((row[1:first], row[first + 1:second]))
        return (self.tokenizer.batch_decode([p[0] for p in pairs], skip_special_tokens=True),
                self.tokenizer.batch_decode([p[1] for p in pairs], skip_special_tokens=True))


class Audio(Dataset):
    def __init__(self, dataset, split):
        root = ROOT / 'data' / dataset
        self.root = root
        rows = pd.read_csv(root / 'manifest.csv')
        self.rows = rows[rows.split == split].reset_index(drop=True)
        self.labels = torch.tensor(self.rows.label.to_numpy()).long()
        self.groups = torch.tensor(self.rows.group.to_numpy()).long()
        if 'wav_path' in rows.columns:
            return
        self.key = 'source_file' if 'source_file' in rows.columns else 'source_split'
        self.audio = {}
        for source in self.rows[self.key].unique():
            name = source if str(source).endswith('.parquet') else f'{source}.parquet'
            self.audio[source] = pd.read_parquet(root / name, columns=['audio']).audio.to_numpy()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from scipy.io import wavfile
        from scipy.signal import resample_poly
        from math import gcd
        row = self.rows.iloc[index]
        if 'wav_path' in self.rows.columns:
            rate, wave = wavfile.read(self.root / row.wav_path)
        else:
            record = self.audio[row[self.key]][int(row['row'])]
            rate, wave = wavfile.read(io.BytesIO(record['bytes']))
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        wave = wave.astype(np.float32) / 32768.0
        if rate != 16000:
            factor = gcd(int(rate), 16000)
            wave = resample_poly(wave, 16000 // factor, int(rate) // factor).astype(np.float32)
        wave = (wave - wave.mean()) / (wave.std() + 1e-7)
        return torch.from_numpy(wave), self.labels[index]


def collate_audio(batch):
    waves, labels = zip(*batch)
    padded = torch.zeros(len(waves), max(len(w) for w in waves))
    mask = torch.zeros_like(padded, dtype=torch.long)
    for i, wave in enumerate(waves):
        padded[i, :len(wave)] = wave
        mask[i, :len(wave)] = 1
    return (padded, mask), torch.stack(labels)


def load_dataset(dataset, split):
    kind = load_config(dataset)['modality']
    return {'vision': Images, 'text': Text, 'audio': Audio}[kind](dataset, split)
