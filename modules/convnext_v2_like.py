import torch
import torch.nn as nn


# GRN (Global Response Normalization) class (modified for 1D)
# reference: https://github.com/facebookresearch/ConvNeXt-V2
#             - models/utils.py
class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))
        self.eps = eps

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=-1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * Nx) + self.beta + x
    
class LayerNorm1d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
    
    def forward(self, x):
        u = x.mean((1, 2), keepdim=True)
        s = (x - u).pow(2).mean((1, 2), keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[None, None, :] * x + self.bias[None, None, :]
        return x
    
    
class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    
    def forward(self, x):
        return x.transpose(*self.dims)
    

class ConvNeXtV2LikeBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, dilation=1, bottoleneck_dilation=4):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.model = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, 1, padding,
                            dilation=dilation, groups=dim),
            Transpose((2, 1)),
            LayerNorm1d(dim),
            nn.Linear(dim, dim * bottoleneck_dilation),
            # nn.GELU(),
            nn.CELU(),
            GRN(dim * bottoleneck_dilation),
            nn.Linear(dim * bottoleneck_dilation, dim),
            Transpose((2, 1))
        )

    def forward(self, x):
        return x + self.model(x)
    
    
class ConvNeXtV2LikeEncoder(nn.Module):
    def __init__(self, num_layers, dim_model, kernel_size=5, dilation=1, bottoleneck_dilation=4):
        super().__init__()
        self.model = nn.Sequential(
            Transpose((2, 1)),
            *[ConvNeXtV2LikeBlock(dim_model, kernel_size, dilation, bottoleneck_dilation) for _ in range(num_layers)],
            Transpose((2, 1)))

    def forward(self, x):
        return self.model(x)
    
    