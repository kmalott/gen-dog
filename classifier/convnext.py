import torch

class Permute(torch.nn.Module):
    """
    Permutation 'layer'. Allows for permutations inside torch.nn.Sequential(...)
    Preforms the same function as tensor.permute(...)
    """
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class ConvNextBlock(torch.nn.Module):
    """
    Standard ConvNext block. Note: does not include learnable gamma parameter in original implementation.
    Tensor Input: [B, C, H, W]
    Tensor Output: [B, C, H, W]

    Layers: 
       - 7x7 depth-wise convolution
       - Permutation layer - makes it easier to apply layerNorm over channel dim. only
       - LayerNorm over only channel dimension
       - Permutation layer - back to channel first for convolution layers
       - 1x1 convolution (pointwise). Channel dim. -> 4*Channel dim.
       - GELU non-linearity
       - 1x1 convolution (pointwise). 4 * Channel dim -> Channel dim. 
    """
    def __init__(self, in_channel: int):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=7, stride=1, padding=3, groups=in_channel),
            Permute(0, 2, 3, 1), # [B, C, H, W] -> [B, H, W, C]
            torch.nn.LayerNorm(in_channel), # ConvNext only applies layerNorm over channel dimension
            Permute(0, 3, 1, 2), # [B, H, W, C] -> [B, C, H, W]
            torch.nn.Conv2d(in_channels=in_channel, out_channels=(4*in_channel), kernel_size=1, stride=1, padding=0),
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=(4*in_channel), out_channels=in_channel, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) + x
    
    
class DownSampleBlock(torch.nn.Module):
    """
    Down-sampling ConvNext block.
    Tensor Input: [B, C, H, W]
    Tensor Output: [B, 2*C, H//2, W//2]

    Layers: 
       - Permutation layer - makes it easier to apply layerNorm over channel dim. only
       - LayerNorm over only channel dimension
       - Permutation layer - back to channel first for convolution layer
       - 2x2 convolution. Stride = 2 (H, W -> H//2, W//2).  Channel dim. -> 2*Channel dim.
    """
    def __init__(self, in_channel: int):
        super().__init__()
        self.model = torch.nn.Sequential(
            Permute(0, 2, 3, 1), # [B, C, H, W] -> [B, H, W, C]
            torch.nn.LayerNorm(in_channel), # ConvNext only applies layerNorm over channel dimension
            Permute(0, 3, 1, 2), # [B, H, W, C] -> [B, C, H, W]
            torch.nn.Conv2d(in_channels=in_channel, out_channels=(2*in_channel), kernel_size=2, stride=2, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ConvNext(torch.nn.Module):
    """
    ConvNext CNN
    ------------
    ConvNext is a CNN architecture that aimed to match the performance of transformer based image classifiers.
    For full details see the original paper: https://arxiv.org/abs/2201.03545

    This implementation mostly matches the original with a few changes:
       - The first convolution has stride=2 (original had stride=4) due to smaller input image dimensions. 
       - The ConvNext blocks have no learable gamma parameter that is present in original implementation. This can be added with torch.nn.Parameter(...) 
       - Each 'section' has fewer ConvNext blocks (2, 2, 4, 2) than even the original tiny variant (3, 3, 9, 3). This can be easily changed... (see input arguments)
       - The channel dimensions (64, 128, 256, 512) are smaller than the original (tiny variant had: 96, 192, 384, 768)

    Inputs: 
       - in_channels: number of channels in input image
       - out_classes: number of output classes
       - blocks_section_x: number of ConvNext blocks in section x of network.

    Input tensor: [B, in_channels, H, W]
    Output tensor: [B, 1, 1, num_classes]
    """
    def __init__(self, 
                in_channels: int = 3, 
                out_classes: int = 120,
                blocks_section_1: int = 2,
                blocks_section_2: int = 2,
                blocks_section_3: int = 4,
                blocks_section_4: int = 2,
                 ):
        super().__init__()
        layers = []
        # input layers
        layers.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1))
        layers.append(Permute(0, 2, 3, 1))
        layers.append(torch.nn.LayerNorm(64))
        layers.append(Permute(0, 3, 1, 2))

        # section 1 layers
        for _ in range(0, blocks_section_1):
            layers.append(ConvNextBlock(64))

        # section 2 layers
        layers.append(DownSampleBlock(64))
        for _ in range(0, blocks_section_2):
            layers.append(ConvNextBlock(128))

        # section 3 layers
        layers.append(DownSampleBlock(128))
        for _ in range(0, blocks_section_3):
            layers.append(ConvNextBlock(256))

        # section 4 layers
        layers.append(DownSampleBlock(256))
        for _ in range(0, blocks_section_4):
            layers.append(ConvNextBlock(512))

        # output layers
        layers.append(torch.nn.AdaptiveAvgPool2d(1))
        layers.append(Permute(0, 2, 3, 1))
        layers.append(torch.nn.LayerNorm(512))
        layers.append(torch.nn.Linear(in_features=512, out_features=out_classes))
        
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
def debug_model(batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 128, 128).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = ConvNext()
    output = model(sample_batch)

    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()