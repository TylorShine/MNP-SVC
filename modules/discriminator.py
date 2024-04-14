import torch

from torch import nn
from torch.nn import Conv1d, Conv2d
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm

import torchaudio


def get_keep_size_padding(kernel_size: int, dilation: int):
    return (kernel_size - 1) * dilation // 2


class DiscriminatorSpec(torch.nn.Module):
    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ) -> None:
        super(DiscriminatorSpec, self).__init__()
        self.period = period
        self.spec = torchaudio.transforms.Spectrogram(period, center=False, pad_mode="constant", power=1.)
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        16,
                        (kernel_size, 7),
                        (stride, 2),
                        padding=(get_keep_size_padding(kernel_size, 1), 2),
                    )
                ),
                norm_f(
                    Conv2d(
                        16,
                        32,
                        (kernel_size, 7),
                        (stride, 2),
                        padding=(get_keep_size_padding(kernel_size, 1), 2),
                    )
                ),
                # norm_f(
                #     Conv2d(
                #         32,
                #         32,
                #         (kernel_size, 7),
                #         (stride, 2),
                #         padding=(get_keep_size_padding(kernel_size, 1), 2),
                #     )
                # ),
                # norm_f(
                #     Conv2d(
                #         32,
                #         32,
                #         (kernel_size, 7),
                #         (stride, 2),
                #         padding=(get_keep_size_padding(kernel_size, 1), 2),
                #     )
                # ),
                norm_f(
                    Conv2d(
                        32,
                        32,
                        (kernel_size, 3),
                        (1, 1),
                        padding=(get_keep_size_padding(kernel_size, 1), 2),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(32, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []

        with torch.no_grad():
            x = self.spec(x)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm: bool = False) -> None:
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                # norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                # norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                # norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(256, 1, 3, 1, padding=1))
        

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiSpecDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm: bool = False) -> None:
        super(MultiSpecDiscriminator, self).__init__()
        # periods = [61, 89, 131, 193, 283]
        periods = [31, 89, 467]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorSpec(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs