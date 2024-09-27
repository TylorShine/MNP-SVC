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
        self.spec = torchaudio.transforms.Spectrogram(period, center=False, pad_mode="constant", power=None)
        # self.spec = torchaudio.transforms.Spectrogram(period, center=True, pad_mode="reflect", power=None)
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
    
    
# Adapted from https://github.com/open-mmlab/Amphion/blob/main/models/vocoders/gan/discriminator/mssbcqtd.py under the MIT license.
class DiscriminatorCQT(nn.Module):
    def __init__(self, hop_length: int, n_octaves:int, bins_per_octave: int,
                 sampling_rate: float = 44100., filters: int = 128, max_filters: int = 1024, dilations: list[int] = [1, 2, 4],
                 normalize_volume: bool = False,
                 in_channels: int = 1, out_channels: int = 1):
        super().__init__()

        self.filters = filters
        self.max_filters = max_filters
        self.filters_scale = 1
        self.kernel_size = (3, 9)
        self.dilations = dilations
        self.stride = (1, 2)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fs = sampling_rate
        self.hop_length = hop_length
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave

        # Lazy-load
        from nnAudio import features

        # self.cqt_transform = features.cqt.CQT2010v2(
        self.cqt_transform = features.cqt.CQT1992v2(
            # sr=self.fs * 2,
            sr=self.fs,
            hop_length=self.hop_length,
            n_bins=self.bins_per_octave * self.n_octaves,
            bins_per_octave=self.bins_per_octave,
            output_format="Complex",
            pad_mode="constant",
        )

        self.conv_pres = nn.ModuleList()
        for _ in range(self.n_octaves):
            self.conv_pres.append(
                nn.Conv2d(
                    self.in_channels * 2,
                    self.in_channels * 2,
                    kernel_size=self.kernel_size,
                    padding=self.get_2d_padding(self.kernel_size),
                )
            )

        self.convs = nn.ModuleList()

        self.convs.append(
            nn.Conv2d(
                self.in_channels * 2,
                self.filters,
                kernel_size=self.kernel_size,
                padding=self.get_2d_padding(self.kernel_size),
            )
        )

        in_chs = min(self.filters_scale * self.filters, self.max_filters)
        for i, dilation in enumerate(self.dilations):
            out_chs = min(
                (self.filters_scale ** (i + 1)) * self.filters, self.max_filters
            )
            self.convs.append(
                weight_norm(
                    nn.Conv2d(
                        in_chs,
                        out_chs,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        dilation=(dilation, 1),
                        padding=self.get_2d_padding(self.kernel_size, (dilation, 1)),
                    )
                )
            )
            in_chs = out_chs
        out_chs = min(
            (self.filters_scale ** (len(self.dilations) + 1)) * self.filters,
            self.max_filters,
        )
        self.convs.append(
            weight_norm(
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                    padding=self.get_2d_padding(
                        (self.kernel_size[0], self.kernel_size[0])
                    ),
                )
            )
        )

        # self.conv_post = weight_norm(
        #     nn.Conv2d(
        #         out_chs,
        #         self.out_channels,
        #         kernel_size=(self.kernel_size[0], self.kernel_size[0]),
        #         padding=self.get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
        #     )
        # )
        
        self.conv_post = SANConv2d(
            out_chs,
            self.out_channels,
            kernel_size=(self.kernel_size[0], self.kernel_size[0]),
            padding=self.get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
        )

        self.activation = torch.nn.LeakyReLU(negative_slope=0.1)
        # self.resample = torchaudio.transforms.Resample(orig_freq=self.fs, new_freq=self.fs * 2)

        self.cqtd_normalize_volume = normalize_volume
        if self.cqtd_normalize_volume:
            print(
                f"[INFO] cqtd_normalize_volume set to True. Will apply DC offset removal & peak volume normalization in CQTD!"
            )

    def get_2d_padding(
        self,
        kernel_size: tuple[int, int],
        dilation: tuple[int, int] = (1, 1),
    ):
        return (
            ((kernel_size[0] - 1) * dilation[0]) // 2,
            ((kernel_size[1] - 1) * dilation[1]) // 2,
        )

    def forward(self, x: torch.tensor, flg_train: bool = False) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []

        if self.cqtd_normalize_volume:
            # Remove DC offset
            x = x - x.mean(dim=-1, keepdims=True)
            # Peak normalize the volume of input audio
            x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)

        # x = self.resample(x)

        z = self.cqt_transform(x)

        z_amplitude = z[:, :, :, 0].unsqueeze(1)
        z_phase = z[:, :, :, 1].unsqueeze(1)

        z = torch.cat([z_amplitude, z_phase], dim=1)
        z = torch.permute(z, (0, 1, 3, 2))  # [B, C, W, T] -> [B, C, T, W]

        latent_z = []
        for i in range(self.n_octaves):
            latent_z.append(
                self.conv_pres[i](
                    z[
                        :,
                        :,
                        :,
                        i * self.bins_per_octave : (i + 1) * self.bins_per_octave,
                    ]
                )
            )
        latent_z = torch.cat(latent_z, dim=-1)

        for i, l in enumerate(self.convs):
            latent_z = l(latent_z)

            latent_z = self.activation(latent_z)
            fmap.append(latent_z)

        # latent_z = self.conv_post(latent_z)
        latent_z = self.conv_post(latent_z, flg_train=flg_train)
        
        if flg_train:
            latent_z_fun, latent_z_dir = latent_z
            fmap.append(latent_z_fun)
            latent_z_fun = torch.flatten(latent_z_fun, 1, -1)
            latent_z_dir = torch.flatten(latent_z_dir, 1, -1)
            latent_z = (latent_z_fun, latent_z_dir)
        else:
            fmap.append(latent_z)
            latent_z = torch.flatten(latent_z, 1, -1)

        return latent_z, fmap


class MultiScaleSubbandCQTDiscriminator(nn.Module):
    def __init__(self, hop_lengths: list[int] = [512, 256, 256],
                 n_octaves: list[int] = [9, 9, 9], bins_per_octave: list[int] = [24, 36, 48],
                #  sampling_rate: float = 44100., filters: int = 128, max_filters: int = 1024, dilations: list[int] = [1, 2, 4],
                sampling_rate: float = 44100., filters: int = 64, max_filters: int = 256, dilations: list[int] = [1, 4],
                 normalize_volume: bool = False,
                 in_channels: int = 1, out_channels: int = 1):
        super().__init__()

        self.cqtd_filters = filters
        self.cqtd_max_filters = max_filters
        self.cqtd_filters_scale = 1
        self.cqtd_dilations = dilations
        self.cqtd_in_channels = in_channels
        self.cqtd_out_channels = out_channels
        # Multi-scale params to loop over
        self.cqtd_hop_lengths = hop_lengths
        self.cqtd_n_octaves = n_octaves
        self.cqtd_bins_per_octaves = bins_per_octave

        self.discriminators = nn.ModuleList(
            [
               DiscriminatorS(use_spectral_norm=False)
            ] +
            [
                DiscriminatorCQT(
                    hop_length=self.cqtd_hop_lengths[i],
                    n_octaves=self.cqtd_n_octaves[i],
                    bins_per_octave=self.cqtd_bins_per_octaves[i],
                    sampling_rate=sampling_rate, filters=self.cqtd_filters, max_filters=self.cqtd_max_filters,
                    dilations=self.cqtd_dilations, normalize_volume=normalize_volume,
                    in_channels=self.cqtd_in_channels, out_channels=self.cqtd_out_channels,
                )
                for i in range(len(self.cqtd_hop_lengths))
            ]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor, flg_train: bool = False) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[list[torch.Tensor]],
        list[list[torch.Tensor]],
    ]:

        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for disc in self.discriminators:
            # y_d_r, fmap_r = disc(y)
            # y_d_g, fmap_g = disc(y_hat)
            y_d_r, fmap_r = disc(y, flg_train=flg_train)
            y_d_g, fmap_g = disc(y_hat, flg_train=flg_train)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


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
        periods = [512, 2048, 128]

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