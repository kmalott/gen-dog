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
        layers.append(torch.nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1))
        layers.append(torch.nn.BatchNorm2d(c))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1))
        layers.append(torch.nn.BatchNorm2d(c))
        layers.append(torch.nn.ReLU())
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x) + x

class DownBlock(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        layers = []
        layers.append(torch.nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1))
        layers.append(torch.nn.BatchNorm2d(out_c))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1))
        layers.append(torch.nn.BatchNorm2d(out_c))
        layers.append(torch.nn.ReLU())
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class UpBlock(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        layers = []
        layers.append(torch.nn.ConvTranspose2d(in_c, out_c, kernel_size=3, stride=2, padding=1, output_padding=1))
        layers.append(torch.nn.BatchNorm2d(out_c))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1))
        layers.append(torch.nn.BatchNorm2d(out_c))
        layers.append(torch.nn.ReLU())
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class BSQTokenizer(torch.nn.Module):
    class Encoder(torch.nn.Module):
        def __init__(self, patch_size: int, latent_dim: int):
            super().__init__()
            layers = []
            # first layer
            layers.append(torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1))
            layers.append(torch.nn.BatchNorm2d(64))
            layers.append(torch.nn.ReLU())
            # series of resBlock -> downBlock
            layers.append(ResBlock(64))
            layers.append(DownBlock(64, 128))
            layers.append(ResBlock(128))
            layers.append(DownBlock(128, 256))
            # (repeated?) resBlock(s)
            layers.append(ResBlock(256))
            layers.append(ResBlock(256))
            # final layer
            layers.append(torch.nn.Conv2d(256, latent_dim, kernel_size=3, stride=1, padding=1))
            self.model = torch.nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return chw_to_hwc(self.model(x))
        
    class Decoder(torch.nn.Module):
        def __init__(self, patch_size: int, latent_dim: int):
            super().__init__()
            layers = []
            # first layer
            layers.append(torch.nn.Conv2d(latent_dim, 256, kernel_size=3, stride=1, padding=1))
            layers.append(torch.nn.BatchNorm2d(256))
            layers.append(torch.nn.ReLU())
            # (repeated?) resBlock(s)
            layers.append(ResBlock(256))
            layers.append(ResBlock(256))
            # series of resBlock -> upBlock
            layers.append(ResBlock(256))
            layers.append(UpBlock(256, 128))
            layers.append(ResBlock(128))
            layers.append(UpBlock(128, 64))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
    
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
    output = model(sample_batch)

    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()