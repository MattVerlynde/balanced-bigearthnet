import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Model(nn.Module):
    def __init__(self, num_classes, weights=None, finetune=False):
        self.name = 'InceptionV3'
        super().__init__()
        self.inceptionv3 = torchvision.models.inception_v3(weights=weights)

        classif_in_features = self.inceptionv3.fc.in_features
        self.inceptionv3.fc = nn.Linear(classif_in_features, num_classes)

        if not finetune:
            for param in self.inceptionv3.parameters():
                param.requires_grad = False
            self.inceptionv3.fc.weight.requires_grad = True

        self.preproc = torchvision.transforms.Compose([
            torchvision.transforms.Resize((299, 299)),
        ])

        self.classifier = self.inceptionv3.fc

    def forward(self, x):
        x = self.preproc(x)
        x = self.inceptionv3(x)
        if isinstance(x, torchvision.models.InceptionOutputs):
            x = x.logits  # Extract the main logits
        return x