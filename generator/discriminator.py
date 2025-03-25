import torch

class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        layers.append(torch.nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1))
        layers.append(torch.nn.LeakyReLU(0.2))
        layers.append(torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1))
        layers.append(torch.nn.LeakyReLU(0.2))
        layers.append(torch.nn.Conv2d(128, 1, kernel_size=4, stride=2, padding=1))
        layers.append(torch.nn.AdaptiveAvgPool2d(1))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(2,3)