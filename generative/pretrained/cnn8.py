import torch.nn as nn

NUM_CLASSES = 10

cfg = [64, 64, 'M', 128, 128, 'M', 196, 196, 'M']
# cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M']

class CNN8(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN8, self).__init__()
        self.features = self._make_layers(cfg)
        
#         self.classifier = nn.Linear(512, 10)
        self.classifier = nn.Sequential(
          nn.Linear(3136, 256),
          nn.BatchNorm1d(num_features=256),
          nn.ReLU(inplace=True),
          nn.Linear(256, 10),
        )
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

 