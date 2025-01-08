import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, pretrained=False, label_type='original'):
        self.name = 'ShortCNN_All'
        nb_class = 43 if label_type == 'original' else 19
        super().__init__()
        self.conv1 = nn.Conv2d(12, 32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64*30*30, nb_class)

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)

        self.features = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool1,
            self.conv2,
            nn.ReLU(),
            self.pool2,
            self.conv3,
            nn.ReLU()
        )

        self.classifier = self.fc

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, 64*30*30))
        return x
