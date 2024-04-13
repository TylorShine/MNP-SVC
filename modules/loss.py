import torch
import torch.nn as nn
import torchaudio
from torch.nn import functional as F
        
class SSSLoss(nn.Module):
    """
    Single-scale Spectral Loss. 
    """

    def __init__(self, n_fft=111, alpha=1.0, overlap=0, eps=1e-7, window_fn=torch.hann_window, wkwargs=None):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.eps = eps
        self.hop_length = int(n_fft * (1 - overlap))  # 25% of the length
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, power=1,
            normalized=True, center=False, window_fn=window_fn, wkwargs=wkwargs)
        
    def forward(self, x_true, x_pred):
        pad_edges = self.n_fft//2  # half of first/last frame
        x_true_pad = F.pad(x_true, (pad_edges, pad_edges), mode = 'constant')
        x_pred_pad = F.pad(x_pred, (pad_edges, pad_edges), mode = 'constant')
        true_spec = self.spec(x_true_pad)[:, :, 1:-1]
        pred_spec = self.spec(x_pred_pad)[:, :, 1:-1]
        S_true = true_spec + self.eps
        S_pred = pred_spec + self.eps
        # S_true = self.spec(x_true) + self.eps
        # S_pred = self.spec(x_pred) + self.eps
        
        converge_term = torch.mean(torch.linalg.norm(S_true - S_pred, dim = (1, 2)) / torch.linalg.norm(S_true + S_pred, dim = (1, 2)))
        
        log_term = F.l1_loss(S_true.log(), S_pred.log())

        loss = converge_term + self.alpha * log_term
        return loss
        
        
class RSSLoss(nn.Module):
    '''
    Random-scale Spectral Loss.
    '''
    
    def __init__(self, fft_min, fft_max, n_scale, alpha=1.0, overlap=0, eps=1e-7, device='cuda'):
        super().__init__()
        self.fft_min = fft_min
        self.fft_max = fft_max
        self.n_scale = n_scale
        self.lossdict = {}
        for n_fft in range(fft_min, fft_max):
            self.lossdict[n_fft] = SSSLoss(n_fft, alpha, overlap, eps).to(device)
        
    def forward(self, x_pred, x_true):
        value = 0.
        n_ffts = torch.randint(self.fft_min, self.fft_max, (self.n_scale,))
        for n_fft in n_ffts:
            loss_func = self.lossdict[int(n_fft)]
            value += loss_func(x_true, x_pred)
        return value / self.n_scale
            
        
    
class DSSLoss(nn.Module):
    '''
    Dual-scale Spectral Loss
    '''
    
    def __init__(self, fft_min, fft_max, alpha=1.0, beta=0.5, overlap=0, eps=1e-7, device='cuda'):
        super().__init__()
        self.beta = beta
        self.fft_min = fft_min
        self.fft_max = fft_max
        self.lossdict = {}
        self.lossdict[fft_max] = SSSLoss(fft_max, alpha, overlap, eps, window_fn=torch.hann_window).to(device)
        self.lossdict[fft_max-1] = SSSLoss(fft_max-1, alpha, overlap, eps, window_fn=torch.hann_window).to(device)
        self.lossdict[fft_min] = SSSLoss(fft_min, alpha, overlap, eps, window_fn=torch.blackman_window).to(device)
        self.lossdict[fft_min-1] = SSSLoss(fft_min-1, alpha, overlap, eps, window_fn=torch.blackman_window).to(device)
        
    def forward(self, x_pred, x_true):
        value = 0.
        # choose minus one fft size randomly for frequency aliasing
        n_fft_offsets = torch.randint(0, 1, (2,))
        value += self.lossdict[self.fft_min-int(n_fft_offsets[0])](x_true, x_pred)*(1. - self.beta)
        value += self.lossdict[self.fft_max-int(n_fft_offsets[1])](x_true, x_pred)*self.beta
        return value


class LF4SLoss(nn.Module):
    """
    Log-Frequency Scaled Single-Scale Spectral Loss. 
    """

    def __init__(self, n_fft=111, alpha=1.0, overlap=0, eps=1e-7, window_fn=torch.hann_window, wkwargs=None):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.eps = eps
        self.hop_length = int(n_fft * (1 - overlap))
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, power=1,
            normalized=True, center=False, window_fn=window_fn, wkwargs=wkwargs)
        self.log_freq_scale = self.log_frequency_scale(n_fft)[None, :, None]
        
    def forward(self, x_true, x_pred):
        pad_edges = self.n_fft//2  # half of first/last frame
        x_true_pad = F.pad(x_true, (pad_edges, pad_edges), mode = 'constant')
        x_pred_pad = F.pad(x_pred, (pad_edges, pad_edges), mode = 'constant')
        true_spec = self.spec(x_true_pad)[:, :, 1:-1]
        pred_spec = self.spec(x_pred_pad)[:, :, 1:-1]
        S_true = true_spec * self.log_freq_scale + self.eps
        S_pred = pred_spec * self.log_freq_scale + self.eps
        
        converge_term = torch.mean(torch.linalg.norm(S_true - S_pred, dim = (1, 2)) / torch.linalg.norm(S_true + S_pred, dim = (1, 2)))
        
        log_term = F.l1_loss(S_true.log(), S_pred.log())

        loss = converge_term + self.alpha * log_term
        return loss
    
    def log_frequency_scale(self, n_fft):
        return torch.log2(torch.arange(0, n_fft//2+1) + 2) / torch.log2(torch.tensor(n_fft))
    
    def to(self, device=None, dtype=None, non_blocking=False, memory_format=torch.preserve_format):
        super().to(device=device, dtype=dtype, non_blocking=non_blocking, memory_format=memory_format)
        self.log_freq_scale = self.log_freq_scale.to(device=device, dtype=dtype, non_blocking=non_blocking, memory_format=memory_format)
        return self


class DLFSSLoss(nn.Module):
    '''
    Dual (DFT bin aliased) Log-Frequency Scaled Spectral Loss
    '''
    
    def __init__(self, fft_min, fft_max, alpha=1.0, beta=0.5, overlap=0, eps=1e-7, device='cuda'):
        super().__init__()
        self.beta = beta
        self.fft_min = fft_min
        self.fft_max = fft_max
        self.lossdict = {}
        self.lossdict[fft_max] = LF4SLoss(fft_max, alpha, overlap, eps, window_fn=torch.hann_window).to(device)
        self.lossdict[fft_max-1] = LF4SLoss(fft_max-1, alpha, overlap, eps, window_fn=torch.hann_window).to(device)
        self.lossdict[fft_min] = LF4SLoss(fft_min, alpha, overlap, eps, window_fn=torch.blackman_window).to(device)
        self.lossdict[fft_min-1] = LF4SLoss(fft_min-1, alpha, overlap, eps, window_fn=torch.blackman_window).to(device)
        
    def forward(self, x_pred, x_true):
        value = 0.
        # choose minus one fft size randomly for frequency bin aliasing
        n_fft_offsets = torch.randint(0, 1, (2,))
        value += self.lossdict[self.fft_min-int(n_fft_offsets[0])](x_true, x_pred)*(1. - self.beta)
        value += self.lossdict[self.fft_max-int(n_fft_offsets[1])](x_true, x_pred)*self.beta
        return value


## for discriminators
def feature_loss(fmap_r, fmap_g):
    loss = 0
    
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
