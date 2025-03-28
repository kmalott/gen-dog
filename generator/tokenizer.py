import torch



def hwc_to_chw(x: torch.Tensor) -> torch.Tensor:
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-1]] + [dims[-3]] + [dims[-2]]
    return x.permute(*dims)


def chw_to_hwc(x: torch.Tensor) -> torch.Tensor:
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-2]] + [dims[-1]] + [dims[-3]]
    return x.permute(*dims)

class ResBlock(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        layers = []
        layers.append(torch.nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(torch.nn.BatchNorm2d(c))
        layers.append(torch.nn.LeakyReLU(0.1))
        layers.append(torch.nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(torch.nn.BatchNorm2d(c))
        self.model = torch.nn.Sequential(*layers)
        self.lrelu = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.lrelu(self.model(x) + x)
    
class ResDownBlock(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        layers = []
        layers.append(torch.nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(torch.nn.BatchNorm2d(out_c))
        layers.append(torch.nn.LeakyReLU(0.1))
        layers.append(torch.nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(torch.nn.BatchNorm2d(out_c))
        self.model = torch.nn.Sequential(*layers)
        res_layers = []
        res_layers.append(torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
        res_layers.append(torch.nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False))
        self.res = torch.nn.Sequential(*res_layers)
        self.lrelu = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.lrelu(self.model(x) + self.res(x))
    
class ResUpBlock(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        layers = []
        layers.append(torch.nn.ConvTranspose2d(in_c, out_c, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False))
        layers.append(torch.nn.BatchNorm2d(out_c))
        layers.append(torch.nn.LeakyReLU(0.1))
        layers.append(torch.nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(torch.nn.BatchNorm2d(out_c))
        self.model = torch.nn.Sequential(*layers)
        res_layers = []
        # option 1 (conv)
        res_layers.append(torch.nn.ConvTranspose2d(in_c, out_c, kernel_size=1, stride=2, padding=0, output_padding=1, bias=False))
        # option 2 (interpolation)
        # res_layers.append(torch.nn.Upsample(scale_factor=2, mode='bicubic'))
        # res_layers.append(torch.nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False))
        self.res = torch.nn.Sequential(*res_layers)
        self.lrelu = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.lrelu(self.model(x) + self.res(x))

class DownBlock(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        layers = []
        layers.append(torch.nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(torch.nn.BatchNorm2d(out_c))
        layers.append(torch.nn.LeakyReLU(0.1))
        layers.append(torch.nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(torch.nn.BatchNorm2d(out_c))
        layers.append(torch.nn.LeakyReLU(0.1))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class UpBlock(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        layers = []
        layers.append(torch.nn.ConvTranspose2d(in_c, out_c, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False))
        layers.append(torch.nn.BatchNorm2d(out_c))
        layers.append(torch.nn.LeakyReLU(0.1))
        layers.append(torch.nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(torch.nn.BatchNorm2d(out_c))
        layers.append(torch.nn.LeakyReLU(0.1))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class BSQTokenizer(torch.nn.Module):
    class Encoder(torch.nn.Module):
        def __init__(self, patch_size: int, latent_dim: int):
            super().__init__()
            layers = []
            # first layer
            layers.append(torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(torch.nn.BatchNorm2d(64))
            layers.append(torch.nn.LeakyReLU(0.1))
            # series of ResDownBlock
            layers.append(ResDownBlock(64, 128))
            layers.append(ResDownBlock(128, 256))
            # final layer
            layers.append(torch.nn.Conv2d(256, latent_dim, kernel_size=1, stride=1, padding=0))
            self.model = torch.nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return chw_to_hwc(self.model(x))
        
    class Decoder(torch.nn.Module):
        def __init__(self, patch_size: int, latent_dim: int):
            super().__init__()
            layers = []
            # first layer
            layers.append(torch.nn.Conv2d(latent_dim, 256, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(torch.nn.BatchNorm2d(256))
            layers.append(torch.nn.LeakyReLU(0.1))
            # series of ResUpBlock
            layers.append(ResUpBlock(256, 128))
            layers.append(ResUpBlock(128, 64))
            # final layer
            layers.append(torch.nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1))
            self.model = torch.nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(hwc_to_chw(x))

    def __init__(self, patch_size: int, latent_dim: int, codebook: int):
        super().__init__()
        self.encoder = self.Encoder(patch_size, latent_dim)
        self.down_project = torch.nn.Linear(latent_dim, codebook)
        self.up_project = torch.nn.Linear(codebook, latent_dim)
        self.decoder = self.Decoder(patch_size, latent_dim)
        self.codebook = codebook

    def forward(self, x: torch.Tensor):
        cnt = torch.bincount(self.encode_int(x).flatten(), minlength=2**14)
        return (self.decode(self.encode(x)), cnt)
        # return self.decode(self.encode(x))
    
    def diff_sign(self, x: torch.Tensor) -> torch.Tensor:
        sign = 2 * (x >= 0).float() - 1
        return x + (sign - x).detach()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        v = self.down_project(z)
        u = torch.nn.functional.normalize(v, p=2, dim=-1)
        return self.diff_sign(u)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        z_hat = self.up_project(x)
        return self.decoder(z_hat)

    def encode_int(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        x = (x >= 0).int()
        return (x * (1 << torch.arange(x.size(-1)).to(x.device))).sum(dim=-1)

    def decode_int(self, x: torch.Tensor) -> torch.Tensor:
        x = 2 * ((x[..., None] & (1 << torch.arange(self.codebook).to(x.device))) > 0).float() - 1
        return self.decode(x)
    

def debug_model(batch_size: int = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 128, 128).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = BSQTokenizer(2, 128, 20)
    output, cnt = model(sample_batch)

    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()