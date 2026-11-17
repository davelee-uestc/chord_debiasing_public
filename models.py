"""The ResNet-50, BERT, and wav2vec2 models used by the main experiments."""
import pickle
import torch
from torch import nn
from common import ROOT, load_config


class AudioClassifier(nn.Module):
    def __init__(self, n_classes, training):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(ROOT / 'pretrained' / 'wav2vec2')
        self.encoder.config.layerdrop = training['layerdrop']
        if training['gradient_checkpointing']:
            self.encoder.gradient_checkpointing_enable()
        self.head = nn.Linear(self.encoder.config.hidden_size, n_classes)

    def pool(self, waves, mask):
        return pool_audio(self.encoder, waves, mask)


def pool_audio(encoder, waves, mask):
    out = encoder(input_values=waves).last_hidden_state
    lengths = encoder._get_feat_extract_output_lengths(mask.sum(-1)).clamp(max=out.size(1))
    frames = (torch.arange(out.size(1), device=out.device)[None] < lengths[:, None]).unsqueeze(-1).float()
    return (out * frames).sum(1) / frames.sum(1).clamp_min(1)


def build_target(dataset, checkpoint=None):
    cfg = load_config(dataset)
    if cfg['modality'] == 'vision':
        from torchvision.models import resnet50
        model = resnet50(weights=None)
        if checkpoint is None:
            model.load_state_dict(torch.load(ROOT / 'pretrained' / 'resnet50.pt', weights_only=True))
        model.fc = nn.Linear(model.fc.in_features, cfg['classes'])
    elif cfg['modality'] == 'text':
        from transformers import BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained(ROOT / 'pretrained' / 'bert', num_labels=cfg['classes'])
    else:
        model = AudioClassifier(cfg['classes'], cfg['train'])
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if cfg['modality'] == 'text':
            state = {k.replace('fc.', 'classifier.'): v for k, v in state.items()}
        model.load_state_dict(state)
    return model


def to_device(x, device):
    return tuple(v.to(device) for v in x) if isinstance(x, (tuple, list)) else x.to(device)


def target_forward(model, x, modality):
    if modality == 'vision':
        features = x
        for layer in list(model.children())[:-1]:
            features = layer(features)
        features = features.flatten(1)
        return model.fc(features), features
    if modality == 'text':
        output = model.bert(input_ids=x[:, :, 0], attention_mask=x[:, :, 1], token_type_ids=x[:, :, 2])
        features = model.dropout(output.pooler_output)
        return model.classifier(features), features
    features = model.pool(*x)
    return model.head(features), features


class UnusedLARS:
    # The VISSL download includes optimizer metadata; only its backbone is used.
    _exclude_bias_and_norm = None


class BackboneUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'vissl.optimizers.lars' and name == '_LARS':
            return UnusedLARS
        return super().find_class(module, name)


class BackbonePickle:
    __name__ = 'pickle'
    Unpickler = BackboneUnpickler
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)


def ssl_encoder(name):
    from torchvision.models import resnet50
    state = torch.load(ROOT / 'pretrained' / f'{name}.pt', map_location='cpu', weights_only=False, pickle_module=BackbonePickle)
    if 'classy_state_dict' in state:
        state = state['classy_state_dict']['base_model']['model']['trunk']
    state = {k.removeprefix('_feature_blocks.').removeprefix('module.'): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if k.startswith(('conv1.', 'bn1.', 'layer1.', 'layer2.', 'layer3.', 'layer4.'))}
    model = resnet50(weights=None)
    model.fc = nn.Identity()
    model.load_state_dict(state)
    return model
