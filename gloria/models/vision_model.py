from numpy.lib.function_base import extract
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cnn_backbones


class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super(ImageEncoder, self).__init__()

        self.output_dim = cfg.model.text.embedding_dim
        self.norm = cfg.model.norm
        self.cfg = cfg

        model_function = getattr(cnn_backbones, cfg.model.vision.model_name)
        self.model, self.feature_dim, self.interm_feature_dim = model_function(
            pretrained=cfg.model.vision.pretrained
        )

        if cfg.model.vision.pretrained_path is not None:
            print(f'Loading image encoder from {cfg.model.vision.pretrained_path}')
            pretrained_model_state_dict = torch.load(cfg.model.vision.pretrained_path)['state_dict']
            pretrained_model_state_dict = {k.split('gloria.img_encoder.model.')[-1]:v for k, v in pretrained_model_state_dict.items() if 'gloria.img_encoder.model.' in k}
            self.model.load_state_dict(pretrained_model_state_dict)

        self.global_embedder = nn.Linear(self.feature_dim, self.output_dim)
        self.local_embedder = nn.Conv2d(
            self.interm_feature_dim,
            self.output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        if cfg.model.ckpt_path is not None:
            self.init_trainable_weights()

        if cfg.model.vision.freeze_cnn:
            print("Freezing CNN model")
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x, get_local=False):
        # --> fixed-size input: batch x 3 x 299 x 299
        if "resne" in self.cfg.model.vision.model_name:
            global_ft, local_ft = self.resnet_forward(x, extract_features=True)
        elif "densenet" in self.cfg.model.vision.model_name:
            global_ft, local_ft = self.densenet_forward(x, extract_features=True)

        if get_local:
            return global_ft, local_ft
        else:
            return global_ft

    def generate_embeddings(self, global_features, local_features):

        global_emb = self.global_embedder(global_features)
        local_emb = self.local_embedder(local_features)

        if self.norm is True:
            local_emb = local_emb / torch.norm(
                local_emb, 2, dim=1, keepdim=True
            ).expand_as(local_emb)
            global_emb = global_emb / torch.norm(
                global_emb, 2, dim=1, keepdim=True
            ).expand_as(global_emb)

        return global_emb, local_emb

    def resnet_forward(self, x, extract_features=False):

        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=True)(x)

        x = self.model.conv1(x)  # (batch_size, 64, 150, 150)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)  # (batch_size, 64, 75, 75)
        x = self.model.layer2(x)  # (batch_size, 128, 38, 38)
        x = self.model.layer3(x)  # (batch_size, 256, 19, 19)
        local_features = x
        x = self.model.layer4(x)  # (batch_size, 512, 10, 10)

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x, local_features

    def densenet_forward(self, x, extract_features=False):
        features = self.model.features.conv0(x)
        features = self.model.features.norm0(features)
        features = self.model.features.relu0(features)
        features = self.model.features.pool0(features)

        features = self.model.features.denseblock1(features)
        features = self.model.features.transition1(features)

        features = self.model.features.denseblock2(features)
        features = self.model.features.transition2(features)

        local_features = self.model.features.denseblock3(features)
        features = self.model.features.transition3(local_features)

        features = self.model.features.denseblock4(features)
        features = self.model.features.norm5(features)

        x = F.relu(features, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        return x, local_features
      

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)


class PretrainedImageClassifier(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        num_cls: int,
        feature_dim: int,
        freeze_encoder: bool = True,
    ):
        super(PretrainedImageClassifier, self).__init__()
        self.img_encoder = image_encoder
        self.classifier = nn.Linear(feature_dim, num_cls)
        if freeze_encoder:
            for param in self.img_encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.img_encoder(x)
        pred = self.classifier(x)
        return pred


class ImageClassifier(nn.Module):
    def __init__(self, cfg, image_encoder=None):
        super(ImageClassifier, self).__init__()

        model_function = getattr(cnn_backbones, cfg.model.vision.model_name)
        self.img_encoder, self.feature_dim, _ = model_function(
            pretrained=cfg.model.vision.pretrained
        )

        self.classifier = nn.Linear(self.feature_dim, cfg.model.vision.num_targets)

        if cfg.model.pretrained_ckpt_path is not None and cfg.model.pretrained_ckpt_path != '':
            pretrained_ckpt = torch.load(cfg.model.pretrained_ckpt_path)
            self.load_state_dict(pretrained_ckpt['state_dict'])

    def forward(self, x):
        x = self.img_encoder(x)
        pred = self.classifier(x)
        return pred