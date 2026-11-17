"""Extract missing target and frozen-view caches; preserve dataset row order."""
import gc
import torch
from torch.utils.data import DataLoader

from common import ROOT, load_config, get_args
from data_utils import Images, Text, Audio, load_dataset, collate_audio, image_transform
from models import build_target, target_forward, to_device, ssl_encoder, pool_audio
from ssc_common import identify


def loader(data, cfg):
    return DataLoader(data, batch_size=cfg['extract_batch_size'], shuffle=False,
                      num_workers=4, collate_fn=collate_audio if cfg['modality']=='audio' else None)


def save_features(path, features, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'f': features, 'y': data.labels, 'p': data.groups}, path)
    print(f'Saved {path}: {tuple(features.shape)}', flush=True)


@torch.no_grad()
def target_features(dataset, run_dir, device):
    cfg = load_config(dataset)
    directory = run_dir / 'cache' / dataset
    missing = [s for s in ('val', 'test') if not (directory/f'{s}.pt').exists()]
    if not missing:
        return
    checkpoint = run_dir / 'checkpoints' / dataset / 'encoder.pt'
    if not checkpoint.exists():
        raise FileNotFoundError(f'{checkpoint}: run bash train_models.sh --datasets {dataset} first')
    model = build_target(dataset, checkpoint).eval().to(device)
    for split in missing:
        data = load_dataset(dataset, split)
        print(f'Extract {dataset} target/{split}: {len(data)} samples', flush=True)
        parts = [target_forward(model, to_device(x, device), cfg['modality'])[1].cpu()
                 for x, _ in loader(data, cfg)]
        save_features(directory/f'{split}.pt', torch.cat(parts).float(), data)


@torch.no_grad()
def vision_view(dataset, name, device, cfg):
    if name in ('clip_vision', 'blip_vision'):
        from transformers import AutoImageProcessor, CLIPVisionModel, BlipForImageTextRetrieval
        source = ROOT/'pretrained'/name.split('_')[0]
        processor = AutoImageProcessor.from_pretrained(source)
        transform = lambda image: processor(images=image, return_tensors='pt')['pixel_values'][0]
        model = (CLIPVisionModel.from_pretrained(source) if name=='clip_vision'
                 else BlipForImageTextRetrieval.from_pretrained(source).vision_model)
    else:
        transform = image_transform(False)
        model = ssl_encoder(name)
    data = Images(dataset, 'val', transform)
    model = model.eval().to(device)
    parts = []
    for x, _ in loader(data, cfg):
        out = model(x.to(device))
        if name == 'clip_vision':
            out = out.pooler_output
        elif name == 'blip_vision':
            out = out.last_hidden_state[:, 0]
        parts.append(out.float().cpu())
    return torch.cat(parts), data


@torch.no_grad()
def text_view(dataset, name, device, cfg):
    from transformers import AutoTokenizer, AutoModel, CLIPTextModel, BlipForImageTextRetrieval
    source = ROOT/'pretrained'/name
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = (CLIPTextModel.from_pretrained(source) if name=='clip' else
             BlipForImageTextRetrieval.from_pretrained(source).text_encoder if name=='blip' else
             AutoModel.from_pretrained(source))
    model = model.eval().to(device)
    data = Text(dataset, 'val')
    parts = []
    for start in range(0, len(data), cfg['extract_batch_size']):
        left, right = data.strings(start, start+cfg['extract_batch_size'])
        if name=='clip' and right is not None:
            left, right = [f'{a} {b}' for a,b in zip(left,right)], None
        tokens = tokenizer(left, right, padding=True, truncation=True,
                           max_length=77 if name=='clip' else 128, return_tensors='pt')
        tokens = {k:v.to(device) for k,v in tokens.items()}
        if name=='blip':
            tokens.pop('token_type_ids',None)
        out = model(**tokens)
        if name=='clip':
            pooled = out.pooler_output
        elif name in ('blip','simcse'):
            pooled = out.last_hidden_state[:,0]
        else:
            mask = tokens['attention_mask'].unsqueeze(-1).to(out.last_hidden_state.dtype)
            pooled = (out.last_hidden_state*mask).sum(1)/mask.sum(1).clamp_min(1)
        parts.append(pooled.float().cpu())
    return torch.cat(parts), data


@torch.no_grad()
def audio_view(dataset, name, device, cfg):
    from transformers import AutoModel, WhisperModel, WhisperFeatureExtractor
    source = ROOT/'pretrained'/name
    data = Audio(dataset,'val')
    parts = []
    if name=='whisper':
        processor = WhisperFeatureExtractor.from_pretrained(source)
        model = WhisperModel.from_pretrained(source).encoder.eval().to(device)
        for start in range(0,len(data),cfg['extract_batch_size']):
            waves = [data[i][0].numpy() for i in range(start,min(start+cfg['extract_batch_size'],len(data)))]
            inputs = processor(waves,sampling_rate=16000,return_tensors='pt')
            out = model(inputs.input_features.to(device)).last_hidden_state
            keep = [max(1,int(round(len(w)/16000*50))) for w in waves]
            parts.append(torch.stack([out[i,:keep[i]].mean(0) for i in range(len(waves))]).float().cpu())
    else:
        model = AutoModel.from_pretrained(source).eval().to(device)
        for x,_ in loader(data,cfg):
            parts.append(pool_audio(model,*to_device(x,device)).float().cpu())
    return torch.cat(parts),data


def visual_partition(features, labels):
    # Mark samples closer to another class center than to their own.
    centers = torch.stack([features[labels==c].mean(0) for c in torch.unique(labels,sorted=True)])
    distances = torch.cdist(features.float(),centers.float()).square()
    own = distances[torch.arange(len(labels)),labels].clone()
    distances[torch.arange(len(labels)),labels] = torch.inf
    return own > distances.min(1).values


def ensure_cache(dataset, run_dir, device):
    cfg = load_config(dataset)
    directory = run_dir/'cache'/dataset
    needed = [directory/'val.pt',directory/'test.pt']
    needed += ([directory/'views.pt'] if cfg['modality']=='vision' else
               [directory/'views'/f'{n}.pt' for n in cfg['views']])
    if all(p.exists() for p in needed):
        return
    target_features(dataset,run_dir,device)
    val = torch.load(directory/'val.pt',map_location='cpu',weights_only=True)
    labels = val['y'].flatten().long()
    extract = {'vision':vision_view,'text':text_view,'audio':audio_view}[cfg['modality']]
    views = {'anchor':visual_partition(val['f'].float(),labels)} if cfg['modality']=='vision' else {}
    for name in cfg['views']:
        path = directory/'views'/f'{name}.pt'
        if not path.exists():
            print(f'Extract {dataset} view/{name}', flush=True)
            features,data = extract(dataset,name,device,cfg)
            assert torch.equal(data.labels,labels)
            save_features(path,features,data)
            gc.collect()
            torch.cuda.empty_cache()
        if cfg['modality']=='vision':
            cached = torch.load(path,map_location='cpu',weights_only=True)
            assert torch.equal(cached['y'].flatten(),labels)
            views[name] = (identify(labels,cached['f'].float(),cfg['classes']) if name in ('clip_vision','blip_vision')
                           else visual_partition(cached['f'].float(),labels))
    if views:
        torch.save({'y':labels,'views':views},directory/'views.pt')


if __name__ == '__main__':
    args = get_args(__doc__)
    torch.set_num_threads(4)
    for dataset in args.datasets:
        ensure_cache(dataset, args.run_dir, args.device)
