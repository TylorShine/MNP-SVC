import torch
import torch.nn as nn


# GRN (Global Response Normalization) class
# reference: https://github.com/facebookresearch/ConvNeXt-V2
#             - models/utils.py
class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * Nx) + self.beta + x
    
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
    
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[None, None, None, :] * x + self.bias[None, None, None, :]
        return x
    
    
class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    
    def forward(self, x):
        return x.transpose(*self.dims)
    
    
class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    
    def forward(self, x):
        return x.permute(*self.dims)
    

class ConvNeXtV2LikeBlock2d(nn.Module):
    def __init__(self, dim, kernel_size=5, dilation=1, bottoleneck_dilation=4):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.model = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, 1, padding,
                            dilation=dilation, groups=dim),
            Permute((0, 2, 3, 1)),
            LayerNorm(dim),
            nn.Linear(dim, dim * bottoleneck_dilation),
            nn.GELU(),
            GRN(dim * bottoleneck_dilation),
            nn.Linear(dim * bottoleneck_dilation, dim),
            Permute((0, 3, 1, 2)),
        )

    def forward(self, x):
        return x + self.model(x)
    
    
class ConvNeXtV2LikeEncoder2d(nn.Module):
    def __init__(self, num_layers, dim_model, kernel_size=5, dilation=1, bottoleneck_dilation=4):
        super().__init__()
        self.model = nn.Sequential(
            *[ConvNeXtV2LikeBlock2d(dim_model, kernel_size, dilation, bottoleneck_dilation) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.model(x)
    
    