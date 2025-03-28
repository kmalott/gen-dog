import torch

class ResDownBlock(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        layers = []
        layers.append(torch.nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1))
        layers.append(torch.nn.BatchNorm2d(out_c))
        layers.append(torch.nn.LeakyReLU(0.2))
        layers.append(torch.nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1))
        layers.append(torch.nn.BatchNorm2d(out_c))
        self.model = torch.nn.Sequential(*layers)
        res_layers = []
        res_layers.append(torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
        res_layers.append(torch.nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0))
        self.res = torch.nn.Sequential(*res_layers)
        self.lrelu = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.lrelu(self.model(x) + self.res(x))

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
    
def debug_model(batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 128, 128).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = Discriminator()
    output = model(sample_batch)

    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()