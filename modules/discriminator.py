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
        self.log_freq_scale = self.log_frequency_scale(period)[None, :, None]
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
                
                norm_f(
                    Conv2d(
                        32,
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
        
        
    def log_frequency_scale(self, n_fft, min_clip_bins=12):
        ret = torch.log2(torch.max(torch.tensor(min_clip_bins), torch.arange(0, n_fft//2+1)) + 2) / torch.log2(torch.tensor(n_fft//2+1 + 2))
        ret[0] = 1.
        return ret
    
    def to(self, device=None, dtype=None, non_blocking=False, memory_format=torch.preserve_format):
        super().to(device=device, dtype=dtype, non_blocking=non_blocking, memory_format=memory_format)
        self.log_freq_scale = self.log_freq_scale.to(device=device, dtype=dtype, non_blocking=non_blocking, memory_format=memory_format)
        return self

    def forward(self,
                x: torch.Tensor,
                flg_train: bool = False) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []

        with torch.no_grad():
            x = self.spec(x)
            x = torch.view_as_real(x)
            x[:, :, :, 0] = x[:, :, :, 0]*(self.log_freq_scale.to(x))
            x = torch.norm(x, p='fro', dim=-1)
            
        x = self.convs[0](x)
        x = F.leaky_relu(x, 0.1)
        fmap.append(x)

        for _, layer in enumerate(self.convs[1:]):
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
        
        return x, fmap
    

def variable_hann_window(window_length, width=1.0, periodic=True):
    if periodic:
        n = torch.arange(window_length) / window_length
    else:
        n = torch.linspace(0.0, 1.0, window_length)
    mask = torch.where(n*width + 0.5-width*0.5 < 0.0, 0.0, 1.0)
    mask = mask * torch.where(n*width + 0.5-width*0.5 > 1.0, 0.0, 1.0)
    window = (0.5 + 0.5*torch.cos(2.*torch.pi*(n - 0.5)*width))*mask
    return window
    
    
class DiscriminatorSpecW(torch.nn.Module):
    def __init__(
        self,
        period: int,
        hop_length: int,
        window_fn: callable = torch.hann_window,
        wkargs: dict | None = None,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ) -> None:
        super(DiscriminatorSpecW, self).__init__()
        self.period = period
        self.spec = torchaudio.transforms.Spectrogram(period, hop_length=hop_length, window_fn=window_fn, wkwargs=wkargs, center=False, pad_mode="constant", power=None)
        self.log_freq_scale = self.log_frequency_scale(period)[None, :, None]
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
                
                norm_f(
                    Conv2d(
                        32,
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
        
    def log_frequency_scale(self, n_fft, min_clip_bins=12):
        ret = torch.log2(torch.max(torch.tensor(min_clip_bins), torch.arange(0, n_fft//2+1)) + 2) / torch.log2(torch.tensor(n_fft//2+1 + 2))
        ret[0] = 1.
        return ret
    
    def to(self, device=None, dtype=None, non_blocking=False, memory_format=torch.preserve_format):
        super().to(device=device, dtype=dtype, non_blocking=non_blocking, memory_format=memory_format)
        self.log_freq_scale = self.log_freq_scale.to(device=device, dtype=dtype, non_blocking=non_blocking, memory_format=memory_format)
        return self

    def forward(self,
                x: torch.Tensor,
                flg_train: bool = False) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []

        with torch.no_grad():
            x = self.spec(x)
            x = torch.view_as_real(x)
            x[:, :, :, 0] = x[:, :, :, 0]*(self.log_freq_scale.to(x))
            x = torch.norm(x, p='fro', dim=-1)
            
        x = self.convs[0](x)
        x = F.leaky_relu(x, 0.1)
        fmap.append(x)

        for _, layer in enumerate(self.convs[1:]):
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
        
        return x, fmap
    
    
class DiscriminatorMel(torch.nn.Module):
    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        sampling_rate: int = 44100,
        n_mels: int = 128,
        use_spectral_norm: bool = False,
    ) -> None:
        super(DiscriminatorMel, self).__init__()
        self.period = period
        self.mel = torchaudio.transforms.MelSpectrogram(
            sampling_rate, n_fft=period, n_mels=n_mels, power=1, center=False, pad_mode="constant", mel_scale="slaney")
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 7),
                        (stride, 2),
                        padding=(get_keep_size_padding(kernel_size, 1), 2),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
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

    def forward(self,
                x: torch.Tensor,
                flg_train: bool = False) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []

        with torch.no_grad():
            x = self.mel(x)
            # x = torch.view_as_real(x)
            # x = torch.norm(x, p='fro', dim=-1)
        
        x = self.convs[0](x)
        x = F.leaky_relu(x, 0.1)
        fmap.append(x)

        for _, layer in enumerate(self.convs[1:]):
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
            norm_f(Conv2d(1, int(4*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_keep_size_padding(5, 1), 0))),
            norm_f(Conv2d(int(4*self.d_mult), int(16*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_keep_size_padding(5, 1), 0))),
            norm_f(Conv2d(int(16*self.d_mult), int(64*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_keep_size_padding(5, 1), 0))),
            # norm_f(Conv2d(int(128*self.d_mult), int(512*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_keep_size_padding(5, 1), 0))),
            # norm_f(Conv2d(int(512*self.d_mult), int(1024*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_keep_size_padding(5, 1), 0))),
            # norm_f(Conv2d(int(1024*self.d_mult), int(1024*self.d_mult), (kernel_size, 1), 1, padding=(2, 0))),
            # norm_f(Conv2d(int(128*self.d_mult), int(128*self.d_mult), (kernel_size, 1), 1, padding=(2, 0))),
            norm_f(Conv2d(int(64*self.d_mult), int(64*self.d_mult), (kernel_size, 1), 1, padding=(2, 0))),
        ])
        # self.conv_post = norm_f(Conv2d(int(1024*self.d_mult), 1, (3, 1), 1, padding=(1, 0)))
        # self.conv_post = SANConv2d(int(1024*self.d_mult), 1, (3, 1), 1, padding=(1, 0))
        # self.conv_post = SANConv2d(int(128*self.d_mult), 1, (3, 1), 1, padding=(1, 0))
        self.conv_post = SANConv2d(int(64*self.d_mult), 1, (3, 1), 1, padding=(1, 0))

    def forward(self, x, flg_train: bool = False):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        
        x = self.convs[0](x)
        x = F.leaky_relu(x, 0.1)
        fmap.append(x)
        
        for l in self.convs[1:]:
            x = l(x)
            fmap.append(x)
            x = F.leaky_relu(x, 0.1)
            # fmap.append(x)

        # for l in self.convs:
        #     x = l(x)
        #     # fmap.append(x)
        #     x = F.leaky_relu(x, 0.1)
        #     fmap.append(x)
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
            
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
    
    
class MultiSpecMelDiscriminator(torch.nn.Module):
    def __init__(self, sampling_rate: float = 44100., use_spectral_norm: bool = False) -> None:
        super(MultiSpecMelDiscriminator, self).__init__()
        spec_periods = [7, 13, 43]
        
        mel_periods = [64, 128, 256, 512, 1024]
        mel_nmels = [10, 20, 40, 80, 160]

        discs = [
            DiscriminatorSpec(period, use_spectral_norm=use_spectral_norm) for period in spec_periods
        ]
        discs = discs + [
            DiscriminatorMel(period, sampling_rate=sampling_rate, n_mels=n_mel, use_spectral_norm=use_spectral_norm)
            for period, n_mel in zip(mel_periods, mel_nmels)
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
            
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
    
    
class MultiMelDiscriminator(torch.nn.Module):
    def __init__(self, sampling_rate: float = 44100., use_spectral_norm: bool = False) -> None:
        super(MultiMelDiscriminator, self).__init__()
        
        mel_periods = [64, 128, 256, 512, 1024]
        mel_nmels = [10, 20, 40, 80, 160]

        discs = [
            DiscriminatorS(use_spectral_norm=use_spectral_norm)
        ]
        discs = discs + [
            DiscriminatorMel(period, sampling_rate=sampling_rate, n_mels=n_mel, use_spectral_norm=use_spectral_norm)
            for period, n_mel in zip(mel_periods, mel_nmels)
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
            
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
    
    
class MultiPeriodSpecDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm: bool = False) -> None:
        super(MultiPeriodSpecDiscriminator, self).__init__()
        
        periods = [3, 17, 37]
        spec_periods = [32, 64, 128, 256, 512, 1024, 2048]
        
        # discs = [
        #     DiscriminatorSpecW(period, hop_length=int(period//2/scale), window_fn=variable_hann_window, wkargs={'width': scale}, use_spectral_norm=use_spectral_norm)
        #     for period, scale in zip(spec_periods, spec_scales)
        #     # DiscriminatorSpecW(period, hop_length=int(period//2/scale), window_fn=torch.hann_window, use_spectral_norm=use_spectral_norm)
        #     # for period, scale in zip(spec_periods, spec_scales)
        # ]
        discs = [
            DiscriminatorSpec(period, use_spectral_norm=use_spectral_norm)
            for period in spec_periods
        ]
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
            noise = torch.randn_like(y) * 0.01   # counter to y of zero is unstabled
            y_d_r, fmap_r = d(y + noise, flg_train=flg_train)
            y_d_g, fmap_g = d(y_hat + noise, flg_train=flg_train)
            
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
    
    
class MultiPeriodSignalDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm: bool = False) -> None:
        super(MultiPeriodSignalDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]    # 10, 14, (first prime of < [n-1]*2)*2...

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
            
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs