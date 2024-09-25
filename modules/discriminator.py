import torch

from torch import nn
from torch.nn import Conv1d, Conv2d
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm

import torchaudio

from .san_modules import SANConv1d, SANConv2d


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
        # self.spec = torchaudio.transforms.Spectrogram(period, center=False, pad_mode="constant", power=1.)
        # self.spec = torchaudio.transforms.Spectrogram(period, center=False, pad_mode="constant", power=None)
        self.spec = torchaudio.transforms.Spectrogram(period, center=True, pad_mode="reflect", power=None)
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
        self.conv_post = SANConv2d(32, 1, (3, 1), 1, padding=(1, 0))
        
        # self.convs_p = nn.ModuleList(
        #     [
        #         norm_f(
        #             Conv2d(
        #                 1,
        #                 16,
        #                 (kernel_size, 7),
        #                 (stride, 2),
        #                 padding=(get_keep_size_padding(kernel_size, 1), 2),
        #             )
        #         ),
        #         norm_f(
        #             Conv2d(
        #                 16,
        #                 32,
        #                 (kernel_size, 7),
        #                 (stride, 2),
        #                 padding=(get_keep_size_padding(kernel_size, 1), 2),
        #             )
        #         ),
                
        #         norm_f(
        #             Conv2d(
        #                 32,
        #                 32,
        #                 (kernel_size, 3),
        #                 (1, 1),
        #                 padding=(get_keep_size_padding(kernel_size, 1), 2),
        #             )
        #         ),
        #     ]
        # )
        # self.conv_p_post = SANConv2d(32, 1, (3, 1), 1, padding=(1, 0))

    def forward(self,
                x: torch.Tensor,
                flg_train: bool = False) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []

        with torch.no_grad():
            # x = self.spec(x)
            # x, xp = torch.sqrt(x.real**2. + x.imag**2.), torch.atan2(x.imag, x.real)
            # x = torch.sqrt(x.real**2. + x.imag**2.)
            
            x = self.spec(x)
            x = torch.view_as_real(x)
            x = torch.norm(x, p='fro', dim=-1)

        for i, layer in enumerate(self.convs):
            x = layer(x)
            fmap.append(x)
            x = F.leaky_relu(x, 0.1)
            # fmap.append(x)
        x = self.conv_post(x, flg_train=flg_train)
        if flg_train:
            x_fun, x_dir = x
            fmap.append(x_fun)
            x_fun = torch.flatten(x_fun, 1, -1)
            x_dir = torch.flatten(x_dir, 1, -1)
            x = [x_fun, x_dir]
        else:
            fmap.append(x)
            x = torch.flatten(x, 1, -1)
        
        # for i, layer in enumerate(self.convs_p):
        #     xp = layer(xp)
        #     fmap.append(xp)
        #     xp = F.leaky_relu(xp, 0.1)
        #     # fmap.append(x)
        # xp = self.conv_p_post(xp, flg_train=flg_train)
        # if flg_train:
        #     xp_fun, xp_dir = xp
        #     fmap.append(xp_fun)
        #     xp_fun = torch.flatten(xp_fun, 1, -1)
        #     xp_dir = torch.flatten(xp_dir, 1, -1)
        #     x[0] = torch.cat((x[0], xp_fun), dim=1)
        #     x[1] = torch.cat((x[1], xp_dir), dim=1)
        # else:
        #     fmap.append(xp)
        #     xp = torch.flatten(xp, 1, -1)
        #     x = torch.cat((x, xp), dim=1)

        # return x, fmap
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
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
                # norm_f(Conv1d(256, 256, 5, 1, padding=2)),
            ]
        )
        # self.conv_post = norm_f(Conv1d(256, 1, 3, 1, padding=1))
        self.conv_post = SANConv1d(1024, 1, 3, 1, padding=1)
        # self.conv_post = SANConv1d(256, 1, 3, 1, padding=1)
        

    def forward(self,
                x: torch.Tensor,
                flg_train: bool = False) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []

        for i, layer in enumerate(self.convs):
            x = layer(x)
            # fmap.append(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x, flg_train=flg_train)
        if flg_train:
            x_fun, x_dir = x
            fmap.append(x_fun)
            x_fun = torch.flatten(x_fun, 1, -1)
            x_dir = torch.flatten(x_dir, 1, -1)
            x = (x_fun, x_dir)
        else:
            fmap.append(x)
            x = torch.flatten(x, 1, -1)

        return x, fmap
    
    
class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.d_mult = 1
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, int(32*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_keep_size_padding(5, 1), 0))),
            norm_f(Conv2d(int(32*self.d_mult), int(128*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_keep_size_padding(5, 1), 0))),
            norm_f(Conv2d(int(128*self.d_mult), int(512*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_keep_size_padding(5, 1), 0))),
            norm_f(Conv2d(int(512*self.d_mult), int(1024*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_keep_size_padding(5, 1), 0))),
            norm_f(Conv2d(int(1024*self.d_mult), int(1024*self.d_mult), (kernel_size, 1), 1, padding=(2, 0))),
            # norm_f(Conv2d(int(128*self.d_mult), int(128*self.d_mult), (kernel_size, 1), 1, padding=(2, 0))),
        ])
        # self.conv_post = norm_f(Conv2d(int(1024*self.d_mult), 1, (3, 1), 1, padding=(1, 0)))
        self.conv_post = SANConv2d(int(1024*self.d_mult), 1, (3, 1), 1, padding=(1, 0))
        # self.conv_post = SANConv2d(int(128*self.d_mult), 1, (3, 1), 1, padding=(1, 0))

    def forward(self, x, flg_train: bool = False):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            # fmap.append(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x, flg_train=flg_train)
        # fmap.append(x)
        # x = torch.flatten(x, 1, -1)
        if flg_train:
            x_fun, x_dir = x
            fmap.append(x_fun)
            x_fun = torch.flatten(x_fun, 1, -1)
            x_dir = torch.flatten(x_dir, 1, -1)
            x = (x_fun, x_dir)
        else:
            fmap.append(x)
            x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiSpecDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm: bool = False) -> None:
        super(MultiSpecDiscriminator, self).__init__()
        # periods = [61, 89, 131, 193, 283]
        # periods = [31, 89, 467]
        # periods = [173, 337, 673]
        # periods = [62, 178, 346]
        # periods = [14, 46, 178, 746]    # 7*2, (first prime of < [n-1]*2)*2...
        # periods = [10, 46, 178, 746]    # 5*2, (first prime of < [n-1]*2)*2...
        # periods = [4, 10, 46, 178, 746]    # 4, 5*2, (first prime of < [n-1]*2)*2...
        # periods = [4, 7, 13, 178, 746]    # 4, 5*2, (first prime of < [n-1]*2)*2...
        # periods = [4, 6, 22, 82, 326]    # 4, 6, (first prime of < [n-1]*2)*2...
        # periods = [4, 6, 14, 46, 746]    # 4, 6, (first prime of < [n-1]*2)*2...
        # periods = [4, 7, 46, 512]    # 4, 6, (first prime of < [n-1]*2)*2...
        # periods = [8, 14, 46, 768]    # 4, 6, (first prime of < [n-1]*2)*2...
        # periods = [8, 14, 46, 178, 768]    # 4, 6, (first prime of < [n-1]*2)*2...
        # periods = [10, 18, 768]    # 4, 6, (first prime of < [n-1]*2)*2...
        # periods = [10, 14, 46, 178, 768]    # 10, 14, (first prime of < [n-1]*2)*2...
        # periods = [10, 14, 46]    # 10, 14, (first prime of < [n-1]*2)*2...
        # periods = [10, 14, 46, 178]
        # periods = [7, 13, 23]
        # periods = [7, 13, 46, 178]
        # periods = [7, 13, 43]
        # periods = [3, 7, 17]
        periods = [128, 512, 2048]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorSpec(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        flg_train: bool = False,
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y, flg_train=flg_train)
            y_d_g, fmap_g = d(y_hat, flg_train=flg_train)
            # if flg_train:
            #     y_d_rs.append([ydr - ydg for ydr, ydg in zip(y_d_r, y_d_g)])
            #     y_d_gs.append([ydg - ydr for ydr, ydg in zip(y_d_r, y_d_g)])
            # else:
            #     y_d_rs.append(y_d_r - y_d_g)
            #     y_d_gs.append(y_d_g - y_d_r)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            # fmap_rs.append([fr - fg for fr, fg in zip(fmap_r, fmap_g)])
            # fmap_gs.append([fg - fr for fr, fg in zip(fmap_r, fmap_g)])
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
    
    
class MultiPeriodSignalDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm: bool = False) -> None:
        super(MultiPeriodSignalDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]    # 10, 14, (first prime of < [n-1]*2)*2...
        # periods = [3, 11, 37]    # 10, 14, (first prime of < [n-1]*2)*2...

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        flg_train: bool = False,
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y, flg_train=flg_train)
            y_d_g, fmap_g = d(y_hat, flg_train=flg_train)
            # if flg_train:
            #     y_d_rs.append([ydr - ydg for ydr, ydg in zip(y_d_r, y_d_g)])
            #     y_d_gs.append([ydg - ydr for ydr, ydg in zip(y_d_r, y_d_g)])
            # else:
            #     y_d_rs.append(y_d_r - y_d_g)
            #     y_d_gs.append(y_d_g - y_d_r)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            # fmap_rs.append([fr - fg for fr, fg in zip(fmap_r, fmap_g)])
            # fmap_gs.append([fg - fr for fr, fg in zip(fmap_r, fmap_g)])
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs