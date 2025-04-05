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
        def __init__(self, latent_dim: int):
            super().__init__()
            layers = []
            # first layers
            layers.append(torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(torch.nn.BatchNorm2d(64))
            layers.append(torch.nn.LeakyReLU(0.1))
            # series of resBlock -> downBlock
            layers.append(ResBlock(64))
            layers.append(DownBlock(64, 128))
            layers.append(ResBlock(128))
            layers.append(DownBlock(128, 256))
            # resBlock
            layers.append(ResBlock(256))
            # final layers
            layers.append(torch.nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(torch.nn.BatchNorm2d(128))
            layers.append(torch.nn.LeakyReLU(0.1))
            layers.append(torch.nn.Conv2d(128, latent_dim, kernel_size=1, stride=1, padding=0))
            self.model = torch.nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return chw_to_hwc(self.model(x))
        
    class Decoder(torch.nn.Module):
        def __init__(self, latent_dim: int):
            super().__init__()
            layers = []
            # first layers
            layers.append(torch.nn.Conv2d(latent_dim, 128, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(torch.nn.BatchNorm2d(128))
            layers.append(torch.nn.LeakyReLU(0.1))
            layers.append(torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(torch.nn.BatchNorm2d(256))
            layers.append(torch.nn.LeakyReLU(0.1))
            # resBlock
            layers.append(ResBlock(256))
            # series of resBlock -> upBlock
            layers.append(ResBlock(256))
            layers.append(UpBlock(256, 128))
            layers.append(ResBlock(128))
            layers.append(UpBlock(128, 64))
            # final layer
            layers.append(torch.nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1))
            layers.append(torch.nn.Sigmoid())
            self.model = torch.nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(hwc_to_chw(x))

    def __init__(self, latent_dim: int, codebook: int, gamma0: float = 1.0, gamma: float = 1.0):
        super().__init__()
        self.encoder = self.Encoder(latent_dim)
        self.down_project = torch.nn.Linear(latent_dim, codebook)
        self.up_project = torch.nn.Linear(codebook, latent_dim)
        self.decoder = self.Decoder(latent_dim)
        self.codebook = codebook
        self.gamma0 = gamma0
        self.gamma = gamma

    def forward(self, x: torch.Tensor):
        uq, entropy_loss, used_codes = self.encode(x)
        return self.decode(uq), entropy_loss, used_codes
    
    def diff_sign(self, x: torch.Tensor) -> torch.Tensor:
        sign = 2 * (x >= 0).float() - 1
        return x + (sign - x).detach()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        v = self.down_project(z)
        u = torch.nn.functional.normalize(v, p=2, dim=-1)
        uq = self.diff_sign(u)
        used_codes = torch.unique(self.encode_int(uq.detach()), return_counts=False)
        per_sample_entropy, codebook_entropy = self.soft_entropy_loss(v)
        entropy_loss = self.gamma0 * per_sample_entropy - self.gamma * codebook_entropy
        return uq, entropy_loss, used_codes

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        z_hat = self.up_project(x)
        return self.decoder(z_hat)

    def encode_int(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * (1 << torch.arange(x.size(-1)).to(x.device))).sum(dim=-1)

    def decode_int(self, x: torch.Tensor) -> torch.Tensor:
        x = 2 * ((x[..., None] & (1 << torch.arange(self.codebook).to(x.device))) > 0).float() - 1
        return x
    
    def soft_entropy_loss(self, v):
        p = torch.sigmoid(-4 * v)
        prob = torch.stack([p, 1-p], dim=-1)
        per_sample_entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=-1)
        per_sample_entropy = per_sample_entropy.sum(dim=-1).mean()
        avg_prob = torch.mean(prob, dim=(0,1,2))
        codebook_entropy = -(avg_prob * torch.log(avg_prob + 1e-8)).sum(dim=-1)
        codebook_entropy = codebook_entropy.sum(dim=-1)
        return per_sample_entropy, codebook_entropy

def debug_model(batch_size: int = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 128, 128).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = BSQTokenizer(128, 20)
    output, _, _ = model(sample_batch)

    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()