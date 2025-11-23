import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from iql.load_r3m import load_r3m


SUPPORTED_VISUAL_ENCODERS = ["resnet18_imagenet", "resnet18_r3m"]


class Resnet18_IMAGENET(object):
    def __init__(self, args, eval=True, share_memory=False, use_conv_feat=True):
        self.device = torch.device('cuda') if args.gpu else torch.device('cpu')
        self.model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1').to(self.device)

        if eval:
            self.model = self.model.eval()

        if share_memory:
            self.model.share_memory()

        if use_conv_feat:
            self.model = nn.Sequential(*list(self.model.children())[:-2])

    def extract_feature(self, x):
        return self.model(x)


class Resnet18_R3M(object):
    def __init__(self, args, eval=True, share_memory=False, use_conv_feat=True):
        self.device = torch.device('cuda') if args.gpu else torch.device('cpu')
        full_r3m_model = load_r3m("resnet18").module # unpack parallel model
        self.model = full_r3m_model.convnet
        self.model.to(self.device)

        if eval:
            self.model = self.model.eval()

        if share_memory:
            self.model.share_memory()

        if use_conv_feat:
            self.model = nn.Sequential(*list(self.model.children())[:-2])

    def extract_feature(self, x):
        return self.model(x)


class Resnet(object):
    def __init__(self, resnet_args, eval=True, share_memory=False, use_conv_feat=True):
        self.model_type = resnet_args.visual_encoder
        self.gpu = resnet_args.gpu

        # choose model type
        if self.model_type == "resnet18_imagenet":
            self.resnet_model = Resnet18_IMAGENET(resnet_args, eval, share_memory, use_conv_feat=use_conv_feat)
        elif self.model_type == "resnet18_r3m":
            self.resnet_model = Resnet18_R3M(resnet_args, eval, share_memory, use_conv_feat=use_conv_feat)
        else:
            raise NotImplementedError

        # normalization transform
        self.transform = self.get_default_transform()

    @staticmethod
    def get_default_transform():
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),  # Convert to tensor and divide by 255
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def featurize(self, images, batch=1):
        # This function is only called during inference, ie, when collecting data from environments, so len(images) is
        # always 1, then this for loop below is not too consume, no need to vectorize
        images_normalized = torch.stack([self.transform(i) for i in images], dim=0)
        images_normalized = images_normalized.to(self.resnet_model.device)

        out = []
        with torch.no_grad():
            for i in range(0, images_normalized.size(0), batch):
                b = images_normalized[i : i + batch]
                out.append(self.resnet_model.extract_feature(b))
        return torch.cat(out, dim=0)


