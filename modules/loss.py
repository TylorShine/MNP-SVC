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
        x_true_pad = F.pad(x_true, (pad_edges, pad_edges), mode = 'reflect')
        x_pred_pad = F.pad(x_pred, (pad_edges, pad_edges), mode = 'reflect')
        true_spec = self.spec(x_true_pad)
        pred_spec = self.spec(x_pred_pad)
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
        x_true_pad = F.pad(x_true, (pad_edges, pad_edges), mode = 'reflect')
        x_pred_pad = F.pad(x_pred, (pad_edges, pad_edges), mode = 'reflect')
        true_spec = self.spec(x_true_pad)
        pred_spec = self.spec(x_pred_pad)
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
    
    
class LF4SMPLoss(nn.Module):
    """
    Log-Frequency Scaled Single-Scale Spectral Magnitude and Phase Loss. 
    """

    def __init__(self, n_fft=111, alpha=1.0, beta=0.5, overlap=0, eps=1e-7, window_fn=torch.hann_window, wkwargs=None):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.hop_length = int(n_fft * (1 - overlap))
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, power=None,
            normalized=True, center=False, window_fn=window_fn, wkwargs=wkwargs)
        self.log_freq_scale = self.log_frequency_scale(n_fft)[None, :, None]
        
    def forward(self, x_true, x_pred):
        pad_edges = self.n_fft//2  # half of first/last frame
        x_true_pad = F.pad(x_true, (pad_edges, pad_edges), mode = 'reflect')
        x_pred_pad = F.pad(x_pred, (pad_edges, pad_edges), mode = 'reflect')
        true_spec = self.spec(x_true_pad)
        pred_spec = self.spec(x_pred_pad)
        S_true = true_spec.abs() * self.log_freq_scale + self.eps
        S_pred = pred_spec.abs() * self.log_freq_scale + self.eps
        angle_true = true_spec.angle() * self.log_freq_scale
        angle_pred = pred_spec.angle() * self.log_freq_scale 
        
        
        converge_term = torch.mean(torch.linalg.norm(S_true - S_pred, dim = (1, 2)) / torch.linalg.norm(S_true + S_pred, dim = (1, 2)))
        
        log_term = F.l1_loss(S_true.log(), S_pred.log())
        
        angle_term = ((torch.cos(angle_true) - torch.cos(angle_pred)) ** 2.).mean()

        loss = converge_term + self.alpha * log_term + self.beta * angle_term
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


class DLFSSMPLoss(nn.Module):
    '''
    Dual (DFT bin aliased) Log-Frequency Scaled Spectral Magnitude and Phase Loss
    '''
    
    def __init__(self, fft_min, fft_max, alpha=1.0, beta=0.5, gamma=0.7, overlap=0, eps=1e-7, device='cuda'):
        super().__init__()
        self.beta = beta
        self.fft_min = fft_min
        self.fft_max = fft_max
        self.lossdict = {}
        self.lossdict[fft_max] = LF4SMPLoss(fft_max, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device)
        self.lossdict[fft_max-1] = LF4SMPLoss(fft_max-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device)
        self.lossdict[fft_min] = LF4SMPLoss(fft_min, alpha, gamma, overlap, eps, window_fn=torch.blackman_window).to(device)
        self.lossdict[fft_min-1] = LF4SMPLoss(fft_min-1, alpha, gamma, overlap, eps, window_fn=torch.blackman_window).to(device)
        
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
            # rl = rl.float().detach()
            rl = rl.float()
            gl = gl.float()
            loss += F.huber_loss(gl, rl)

    return loss * 2.


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr_fun, dr_dir = dr[0].float(), dr[1].float()
        dg_fun, dg_dir = dg[0].float(), dg[1].float()
        r_loss_fun = torch.mean(F.relu(1 - dr_fun)) + torch.mean(F.relu(1 + dg_fun))
        g_loss_fun = -torch.mean(dg_fun)
        r_loss_dir = torch.mean(F.relu(1 - dr_dir)) + torch.mean(F.relu(1 + dg_dir))
        g_loss_dir = -torch.mean(dg_dir)
        r_loss = r_loss_fun + r_loss_dir
        g_loss = g_loss_fun + g_loss_dir
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg))
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
