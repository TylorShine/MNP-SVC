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
            normalized=True, center=True, window_fn=window_fn, wkwargs=wkwargs)
        
    def forward(self, x_true, x_pred):
        pad_edges = self.n_fft//2  # half of first/last frame
        x_true_pad = F.pad(x_true, (pad_edges, pad_edges), mode = 'constant')
        x_pred_pad = F.pad(x_pred, (pad_edges, pad_edges), mode = 'constant')
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
            normalized=True, center=True, window_fn=window_fn, wkwargs=wkwargs)
        self.log_freq_scale = self.log_frequency_scale(n_fft)[None, :, None]
        
    def forward(self, x_true, x_pred):
        pad_edges = self.n_fft//2  # half of first/last frame
        x_true_pad = F.pad(x_true, (pad_edges, pad_edges), mode = 'constant')
        x_pred_pad = F.pad(x_pred, (pad_edges, pad_edges), mode = 'constant')
        true_spec = self.spec(x_true_pad)
        pred_spec = self.spec(x_pred_pad)
        S_true = true_spec * self.log_freq_scale + self.eps
        S_pred = pred_spec * self.log_freq_scale + self.eps
        
        converge_term = torch.mean(torch.linalg.norm(S_true - S_pred, dim = (1, 2)) / torch.linalg.norm(S_true + S_pred, dim = (1, 2)))
        
        log_term = F.l1_loss(S_true.log(), S_pred.log())

        loss = converge_term + self.alpha * log_term
        return loss
    
    # def log_frequency_scale(self, n_fft):
    #     return torch.log2(torch.arange(0, n_fft//2+1) + 2) / torch.log2(torch.tensor(n_fft))
    
    def log_frequency_scale(self, n_fft, min_clip_bins=16):
        ret = torch.log2(torch.max(torch.tensor(min_clip_bins), torch.arange(0, n_fft//2+1)) + 2) / torch.log2(torch.tensor(n_fft))
        ret[0] = 1.
        return ret
    
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
        # self.spec = torchaudio.transforms.Spectrogram(
        #     n_fft=self.n_fft, hop_length=self.hop_length, power=None,
        #     normalized=True, center=False, window_fn=window_fn, wkwargs=wkwargs)
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, power=None,
            normalized=True, center=False, window_fn=window_fn, wkwargs=wkwargs, pad_mode="constant")
        self.log_freq_scale = self.log_frequency_scale(n_fft)[None, :, None]
        
    def forward(self, x_true, x_pred):
        pad_edges = self.n_fft//2  # half of first/last frame
        x_true_pad = F.pad(x_true, (pad_edges, pad_edges), mode = 'constant')
        x_pred_pad = F.pad(x_pred, (pad_edges, pad_edges), mode = 'constant')
        true_spec = self.spec(x_true_pad)
        pred_spec = self.spec(x_pred_pad)
        # true_spec = self.spec(x_true_pad)[:, :, 1:-1]
        # pred_spec = self.spec(x_pred_pad)[:, :, 1:-1]
        S_true = true_spec.abs() * self.log_freq_scale + self.eps
        S_pred = pred_spec.abs() * self.log_freq_scale + self.eps
        # angle_true = true_spec.angle() * self.log_freq_scale
        # angle_pred = pred_spec.angle() * self.log_freq_scale 
        angle_true = true_spec.angle()
        angle_pred = pred_spec.angle()
        
        
        converge_term = torch.mean(torch.linalg.norm(S_true - S_pred, dim = (1, 2)) / torch.linalg.norm(S_true + S_pred, dim = (1, 2)))
        
        log_term = F.l1_loss(S_true.log(), S_pred.log())
        
        angle_term = ((torch.cos(angle_true) - torch.cos(angle_pred)) ** 2.).mean()

        loss = converge_term + self.alpha * log_term + self.beta * angle_term
        return loss
    
    # def log_frequency_scale(self, n_fft):
    #     return torch.log2(torch.arange(0, n_fft//2+1) + 2) / torch.log2(torch.tensor(n_fft))
    
    def log_frequency_scale(self, n_fft, min_clip_bins=16):
        ret = torch.log2(torch.max(torch.tensor(min_clip_bins), torch.arange(0, n_fft//2+1)) + 2) / torch.log2(torch.tensor(n_fft))
        ret[0] = 1.
        return ret
    
    def to(self, device=None, dtype=None, non_blocking=False, memory_format=torch.preserve_format):
        super().to(device=device, dtype=dtype, non_blocking=non_blocking, memory_format=memory_format)
        self.log_freq_scale = self.log_freq_scale.to(device=device, dtype=dtype, non_blocking=non_blocking, memory_format=memory_format)
        return self
    
    
# proposing loss: Mean Absolute Log Loss
def mall(pred_val, true_val, alpha=2.*torch.pi):
    # return (torch.log(alpha*torch.abs(true_val - pred_val) + 1.) / alpha).mean()
    return (torch.log(alpha*torch.abs(true_val - pred_val) + 1.)).mean()


# proposing loss: Mean Absolute Linear-Big Log-Small Loss
def malinblogs(pred_val, true_val, alpha=3.3, m=1., n=5.5):
    x = torch.abs(true_val - pred_val)
    alog_term = alpha*torch.log(x + 1.)
    # c = torch.min(torch.max(alog_term,m) - m,n)/n
    c = torch.clamp(alog_term - m, min=0, max=n)/n
    # c = torch.clamp(alog_term - m, min=0, max=n-m)
    v = alog_term*(1-c) + c*x
    return v.mean()
    
    
class LF4SMPMalLoss(nn.Module):
    """
    Log-Frequency Scaled Single-Scale Spectral Magnitude and Phase MAL Loss. 
    """

    def __init__(self, n_fft=111, alpha=1.0, beta=0.5, overlap=0, eps=1e-7, window_fn=torch.hann_window, wkwargs=None):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.hop_length = int(n_fft * (1 - overlap))
        # self.spec = torchaudio.transforms.Spectrogram(
        #     n_fft=self.n_fft, hop_length=self.hop_length, power=None,
        #     normalized=True, center=False, window_fn=window_fn, wkwargs=wkwargs)
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, power=None,
            normalized=True, center=False, window_fn=window_fn, wkwargs=wkwargs, pad_mode="constant")
        self.log_freq_scale = self.log_frequency_scale(n_fft)[None, :, None]
        
    def forward(self, x_true, x_pred):
        pad_edges = self.n_fft//2  # half of first/last frame
        x_true_pad = F.pad(x_true, (pad_edges, pad_edges), mode = 'constant')
        x_pred_pad = F.pad(x_pred, (pad_edges, pad_edges), mode = 'constant')
        true_spec = self.spec(x_true_pad)
        pred_spec = self.spec(x_pred_pad)
        # true_spec = self.spec(x_true_pad)[:, :, 1:-1]
        # pred_spec = self.spec(x_pred_pad)[:, :, 1:-1]
        S_true = true_spec.abs() * self.log_freq_scale + self.eps
        S_pred = pred_spec.abs() * self.log_freq_scale + self.eps
        # angle_true = true_spec.angle() * self.log_freq_scale
        # angle_pred = pred_spec.angle() * self.log_freq_scale 
        angle_true = true_spec.angle()
        angle_pred = pred_spec.angle()
        
        
        converge_term = torch.mean(torch.linalg.norm(S_true - S_pred, dim = (1, 2)) / torch.linalg.norm(S_true + S_pred, dim = (1, 2)))
        
        # log_term = F.l1_loss(S_true.log(), S_pred.log())
        # log_term = mall(S_true, S_pred)
        # log_term = mall(S_true.log(), S_pred.log())
        
        # S_true_std, S_true_mean = torch.std_mean(S_true, dim=(1,2), keepdim=True)
        # S_pred_std, S_pred_mean = torch.std_mean(S_pred, dim=(1,2), keepdim=True)
        # log_term = mall((S_true - S_true_mean)/(S_true_std + self.eps), (S_pred - S_pred_mean)/(S_pred_std + self.eps))
        
        log_term = mall(S_true, S_pred)
        
        # angle_term = ((torch.cos(angle_true) - torch.cos(angle_pred)) ** 2.).mean()
        angle_term = (F.l1_loss(torch.cos(angle_true) + torch.sin(angle_true), torch.cos(angle_pred) + torch.sin(angle_pred))).mean()

        loss = converge_term + self.alpha * log_term + self.beta * angle_term
        return loss
    
    # def log_frequency_scale(self, n_fft):
    #     return torch.log2(torch.arange(0, n_fft//2+1) + 2) / torch.log2(torch.tensor(n_fft))
    
    def log_frequency_scale(self, n_fft, min_clip_bins=16):
        ret = torch.log2(torch.max(torch.tensor(min_clip_bins), torch.arange(0, n_fft//2+1)) + 2) / torch.log2(torch.tensor(n_fft))
        ret[0] = 1.
        return ret
    
    def to(self, device=None, dtype=None, non_blocking=False, memory_format=torch.preserve_format):
        super().to(device=device, dtype=dtype, non_blocking=non_blocking, memory_format=memory_format)
        self.log_freq_scale = self.log_freq_scale.to(device=device, dtype=dtype, non_blocking=non_blocking, memory_format=memory_format)
        return self


class SSSMPMalLoss(nn.Module):
    """
    Single-Scale Spectral Magnitude and Phase MAL Loss. 
    """

    def __init__(self, n_fft=111, alpha=1.0, beta=0.5, overlap=0, eps=1e-7, window_fn=torch.hann_window, wkwargs=None, mask_bin_to=0, mask_bin_from=0):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.mask_bin_to = mask_bin_to
        self.mask_bin_from = mask_bin_from if mask_bin_from > 0 else n_fft//2+1
        self.hop_length = int(n_fft * (1 - overlap))
        # self.spec = torchaudio.transforms.Spectrogram(
        #     n_fft=self.n_fft, hop_length=self.hop_length, power=None,
        #     normalized=True, center=False, window_fn=window_fn, wkwargs=wkwargs)
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, power=None,
            normalized=True, center=False, window_fn=window_fn, wkwargs=wkwargs, pad_mode="constant")
        
    def forward(self, x_true, x_pred):
        pad_edges = self.n_fft//2  # half of first/last frame
        x_true_pad = F.pad(x_true, (pad_edges, pad_edges), mode = 'constant')
        x_pred_pad = F.pad(x_pred, (pad_edges, pad_edges), mode = 'constant')
        true_spec = self.spec(x_true_pad)[:, self.mask_bin_to:self.mask_bin_from, :]
        pred_spec = self.spec(x_pred_pad)[:, self.mask_bin_to:self.mask_bin_from, :]
        # true_spec = self.spec(x_true_pad)[:, :, 1:-1]
        # pred_spec = self.spec(x_pred_pad)[:, :, 1:-1]
        S_true = true_spec.abs() + self.eps
        S_pred = pred_spec.abs() + self.eps
        # angle_true = true_spec.angle() * self.log_freq_scale
        # angle_pred = pred_spec.angle() * self.log_freq_scale 
        angle_true = true_spec.angle()
        angle_pred = pred_spec.angle()
        
        
        converge_term = torch.mean(torch.linalg.norm(S_true - S_pred, dim = (1, 2)) / torch.linalg.norm(S_true + S_pred, dim = (1, 2)))
        
        # log_term = F.l1_loss(S_true.log(), S_pred.log())
        # S_true_std, S_true_mean = torch.std_mean(S_true, dim=(1,2), keepdim=True)
        # S_pred_std, S_pred_mean = torch.std_mean(S_pred, dim=(1,2), keepdim=True)
        # log_term = mall((S_true - S_true_mean)/(S_true_std + self.eps), (S_pred - S_pred_mean)/(S_pred_std + self.eps))
        # log_term = mall(S_true.log(), S_pred.log())
        
        log_term = mall(S_true.log(), S_pred.log())
        
        # angle_term = ((torch.cos(angle_true) - torch.cos(angle_pred)) ** 2.).mean()
        angle_term = F.l1_loss(torch.cos(angle_true) + torch.sin(angle_true), torch.cos(angle_pred) + torch.sin(angle_pred))

        loss = converge_term + self.alpha * log_term + self.beta * angle_term
        return loss
    
    # def log_frequency_scale(self, n_fft):
    #     return torch.log2(torch.arange(0, n_fft//2+1) + 2) / torch.log2(torch.tensor(n_fft))
    
    # def log_frequency_scale(self, n_fft, min_clip_bins=16):
    #     ret = torch.log2(torch.max(torch.tensor(min_clip_bins), torch.arange(0, n_fft//2+1)) + 2) / torch.log2(torch.tensor(n_fft))
    #     ret[0] = 1.
    #     return ret
    
    # def to(self, device=None, dtype=None, non_blocking=False, memory_format=torch.preserve_format):
    #     super().to(device=device, dtype=dtype, non_blocking=non_blocking, memory_format=memory_format)
    #     self.log_freq_scale = self.log_freq_scale.to(device=device, dtype=dtype, non_blocking=non_blocking, memory_format=memory_format)
    #     return self
    
    
class SSSMPL1Loss(nn.Module):
    """
    Single-Scale Spectral Magnitude and Phase L1 Loss. 
    """

    def __init__(self, n_fft=111, alpha=1.0, beta=0.5, overlap=0, eps=1e-7, window_fn=torch.hann_window, wkwargs=None, mask_bin_to=0, mask_bin_from=0):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.mask_bin_to = mask_bin_to
        self.mask_bin_from = mask_bin_from if mask_bin_from > 0 else n_fft//2+1
        self.hop_length = int(n_fft * (1 - overlap))
        # self.spec = torchaudio.transforms.Spectrogram(
        #     n_fft=self.n_fft, hop_length=self.hop_length, power=None,
        #     normalized=True, center=False, window_fn=window_fn, wkwargs=wkwargs)
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, power=None,
            normalized=True, center=False, window_fn=window_fn, wkwargs=wkwargs, pad_mode="constant")
        # self.spec = torchaudio.transforms.Spectrogram(
        #     n_fft=self.n_fft, hop_length=self.hop_length, power=None,
        #     normalized=True, center=True, window_fn=window_fn, wkwargs=wkwargs, pad_mode="reflect")
        
    def forward(self, x_true, x_pred):
        pad_edges = self.n_fft//2  # half of first/last frame
        x_true_pad = F.pad(x_true, (pad_edges, pad_edges), mode = 'constant')
        x_pred_pad = F.pad(x_pred, (pad_edges, pad_edges), mode = 'constant')
        true_spec = self.spec(x_true_pad)[:, self.mask_bin_to:self.mask_bin_from, :]
        pred_spec = self.spec(x_pred_pad)[:, self.mask_bin_to:self.mask_bin_from, :]
        # x_true = F.pad(x_true, (pad_edges, pad_edges), mode = 'constant')
        # x_pred = F.pad(x_pred, (pad_edges, pad_edges), mode = 'constant')
        # true_spec = self.spec(x_true)[:, self.mask_bin_to:self.mask_bin_from, :]
        # pred_spec = self.spec(x_pred)[:, self.mask_bin_to:self.mask_bin_from, :]
        # true_spec = self.spec(x_true_pad)[:, :, 1:-1]
        # pred_spec = self.spec(x_pred_pad)[:, :, 1:-1]
        S_true = true_spec.abs() + self.eps
        S_pred = pred_spec.abs() + self.eps
        # angle_true = true_spec.angle() * self.log_freq_scale
        # angle_pred = pred_spec.angle() * self.log_freq_scale 
        angle_true = true_spec.angle()
        angle_pred = pred_spec.angle()
        
        
        converge_term = torch.mean(torch.linalg.norm(S_true - S_pred, dim = (1, 2)) / torch.linalg.norm(S_true + S_pred, dim = (1, 2)))
        
        # # log_term = F.l1_loss(S_true.log(), S_pred.log())
        # S_true_std, S_true_mean = torch.std_mean(S_true, dim=(1,2), keepdim=True)
        # S_pred_std, S_pred_mean = torch.std_mean(S_pred, dim=(1,2), keepdim=True)
        # log_term = mall((S_true - S_true_mean)/(S_true_std + self.eps), (S_pred - S_pred_mean)/(S_pred_std + self.eps))
        # # log_term = mall(S_true.log(), S_pred.log())
        
        # log_term = malinblogs(S_true, S_pred)
        log_term = F.l1_loss(S_true, S_pred)
        
        # angle_term = ((torch.cos(angle_true) - torch.cos(angle_pred)) ** 2.).mean()
        angle_term = (F.l1_loss(torch.cos(angle_true) + torch.sin(angle_true), torch.cos(angle_pred) + torch.sin(angle_pred))).mean()

        loss = converge_term + self.alpha * log_term + self.beta * angle_term
        return loss
    
    
class SSSMPMalinblogsLoss(nn.Module):
    """
    Single-Scale Spectral Magnitude and Phase Malinblogs Loss. 
    """

    def __init__(self, n_fft=111, alpha=1.0, beta=0.5, overlap=0, eps=1e-7, window_fn=torch.hann_window, wkwargs=None, mask_bin_to=0, mask_bin_from=0):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.mask_bin_to = mask_bin_to
        self.mask_bin_from = mask_bin_from if mask_bin_from > 0 else n_fft//2+1
        self.hop_length = int(n_fft * (1 - overlap))
        # self.spec = torchaudio.transforms.Spectrogram(
        #     n_fft=self.n_fft, hop_length=self.hop_length, power=None,
        #     normalized=True, center=False, window_fn=window_fn, wkwargs=wkwargs)
        # self.spec = torchaudio.transforms.Spectrogram(
        #     n_fft=self.n_fft, hop_length=self.hop_length, power=None,
        #     normalized=True, center=False, window_fn=window_fn, wkwargs=wkwargs, pad_mode="constant")
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, power=None,
            normalized=True, center=True, window_fn=window_fn, wkwargs=wkwargs, pad_mode="constant")
        
    def forward(self, x_true, x_pred):
        pad_edges = self.n_fft//2  # half of first/last frame
        x_true_pad = F.pad(x_true, (pad_edges, pad_edges), mode = 'constant')
        x_pred_pad = F.pad(x_pred, (pad_edges, pad_edges), mode = 'constant')
        true_spec = self.spec(x_true_pad)[:, self.mask_bin_to:self.mask_bin_from, :]
        pred_spec = self.spec(x_pred_pad)[:, self.mask_bin_to:self.mask_bin_from, :]
        # x_true = F.pad(x_true, (pad_edges, pad_edges), mode = 'constant')
        # x_pred = F.pad(x_pred, (pad_edges, pad_edges), mode = 'constant')
        # true_spec = self.spec(x_true)[:, self.mask_bin_to:self.mask_bin_from, :]
        # pred_spec = self.spec(x_pred)[:, self.mask_bin_to:self.mask_bin_from, :]
        # true_spec = self.spec(x_true_pad)[:, :, 1:-1]
        # pred_spec = self.spec(x_pred_pad)[:, :, 1:-1]
        S_true = true_spec.abs() + self.eps
        S_pred = pred_spec.abs() + self.eps
        # angle_true = true_spec.angle() * self.log_freq_scale
        # angle_pred = pred_spec.angle() * self.log_freq_scale 
        angle_true = true_spec.angle()
        angle_pred = pred_spec.angle()
        
        
        converge_term = torch.mean(torch.linalg.norm(S_true - S_pred, dim = (1, 2)) / torch.linalg.norm(S_true + S_pred, dim = (1, 2)))
        
        # # log_term = F.l1_loss(S_true.log(), S_pred.log())
        # S_true_std, S_true_mean = torch.std_mean(S_true, dim=(1,2), keepdim=True)
        # S_pred_std, S_pred_mean = torch.std_mean(S_pred, dim=(1,2), keepdim=True)
        # log_term = mall((S_true - S_true_mean)/(S_true_std + self.eps), (S_pred - S_pred_mean)/(S_pred_std + self.eps))
        # # log_term = mall(S_true.log(), S_pred.log())
        
        
        # log_term = malinblogs(S_true, S_pred)
        # log_term = malinblogs(S_true.log(), S_pred.log())
        # log_term = malinblogs(S_true, S_pred)
        log_term = F.l1_loss(S_true, S_pred)
        
        # angle_term = ((torch.cos(angle_true) - torch.cos(angle_pred)) ** 2.).mean()
        angle_term = (F.l1_loss(torch.cos(angle_true) + torch.sin(angle_true), torch.cos(angle_pred) + torch.sin(angle_pred))).mean()
        # angle_term = malinblogs(torch.cos(angle_true) + torch.sin(angle_true), torch.cos(angle_pred) + torch.sin(angle_pred))

        loss = converge_term + self.alpha * log_term + self.beta * angle_term
        return loss
    
    
class LF4SMPMalinblogsLoss(nn.Module):
    """
    Log-Frequency Scaled Single-Scale Spectral Magnitude and Phase Malinblogs Loss. 
    """

    def __init__(self, n_fft=111, alpha=1.0, beta=0.5, overlap=0, eps=1e-7, window_fn=torch.hann_window, wkwargs=None, mask_bin_to=0, mask_bin_from=0):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.mask_bin_to = mask_bin_to
        self.mask_bin_from = mask_bin_from if mask_bin_from > 0 else n_fft//2+1
        self.hop_length = int(n_fft * (1 - overlap))
        # self.spec = torchaudio.transforms.Spectrogram(
        #     n_fft=self.n_fft, hop_length=self.hop_length, power=None,
        #     normalized=True, center=False, window_fn=window_fn, wkwargs=wkwargs)
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, power=None,
            normalized=True, center=False, window_fn=window_fn, wkwargs=wkwargs, pad_mode="constant")
        self.log_freq_scale = self.log_frequency_scale(n_fft)[None, :, None]
        self.rev_log_freq_scale = 1./self.log_freq_scale
        
        self.angle_mask_bin_from = min(self.mask_bin_from, n_fft//2//int(44100/2000))   # mask bins below 2kHz for angle loss
        
    def forward(self, x_true, x_pred):
        pad_edges = self.n_fft//2  # half of first/last frame
        x_true_pad = F.pad(x_true, (pad_edges, pad_edges), mode = 'constant')
        x_pred_pad = F.pad(x_pred, (pad_edges, pad_edges), mode = 'constant')
        true_spec = self.spec(x_true_pad)[:, self.mask_bin_to:self.mask_bin_from, :]
        pred_spec = self.spec(x_pred_pad)[:, self.mask_bin_to:self.mask_bin_from, :]
        # true_spec = self.spec(x_true_pad)[:, :, 1:-1]
        # pred_spec = self.spec(x_pred_pad)[:, :, 1:-1]
        S_true = true_spec.abs() * self.log_freq_scale[:, self.mask_bin_to:self.mask_bin_from, :] + self.eps
        S_pred = pred_spec.abs() * self.log_freq_scale[:, self.mask_bin_to:self.mask_bin_from, :] + self.eps
        # angle_true = true_spec.angle() * self.log_freq_scale
        # angle_pred = pred_spec.angle() * self.log_freq_scale 
        angle_true = true_spec.angle()
        angle_pred = pred_spec.angle()
        
        
        converge_term = torch.mean(torch.linalg.norm(S_true - S_pred, dim = (1, 2)) / torch.linalg.norm(S_true + S_pred, dim = (1, 2)))
        
        # log_term = F.l1_loss(S_true.log(), S_pred.log())
        # log_term = mall(S_true, S_pred)
        # log_term = mall(S_true.log(), S_pred.log())
        
        # S_true_std, S_true_mean = torch.std_mean(S_true, dim=(1,2), keepdim=True)
        # S_pred_std, S_pred_mean = torch.std_mean(S_pred, dim=(1,2), keepdim=True)
        # log_term = malinblogs((S_true - S_true_mean)/(S_true_std + self.eps), (S_pred - S_pred_mean)/(S_pred_std + self.eps))
        
        # log_term = malinblogs(S_true, S_pred)
        log_term = F.l1_loss(S_true, S_pred)
        
        # angle_term = ((torch.cos(angle_true) - torch.cos(angle_pred)) ** 2.).mean()
        # angle_sincos_true =  torch.cat((torch.cos(angle_true)*self.rev_log_freq_scale[:, self.mask_bin_to:self.mask_bin_from, :],
        #                                 torch.sin(angle_true)*self.rev_log_freq_scale[:, self.mask_bin_to:self.mask_bin_from, :]), dim=1)
        # angle_sincos_pred = torch.cat((torch.cos(angle_pred)*self.rev_log_freq_scale[:, self.mask_bin_to:self.mask_bin_from, :],
        #                                torch.sin(angle_pred)*self.rev_log_freq_scale[:, self.mask_bin_to:self.mask_bin_from, :]), dim=1)
        
        # angle_sincos_true =  torch.cat((
        #     torch.cos(angle_true[:, self.mask_bin_to:self.angle_mask_bin_from, :])*self.rev_log_freq_scale[:, self.mask_bin_to:self.angle_mask_bin_from, :],
        #     torch.sin(angle_true[:, self.mask_bin_to:self.angle_mask_bin_from, :])*self.rev_log_freq_scale[:, self.mask_bin_to:self.angle_mask_bin_from, :]), dim=1)
        # angle_sincos_pred = torch.cat((
        #     torch.cos(angle_pred[:, self.mask_bin_to:self.angle_mask_bin_from, :])*self.rev_log_freq_scale[:, self.mask_bin_to:self.angle_mask_bin_from, :],
        #     torch.sin(angle_pred[:, self.mask_bin_to:self.angle_mask_bin_from, :])*self.rev_log_freq_scale[:, self.mask_bin_to:self.angle_mask_bin_from, :]), dim=1)
        
        # angle_term = F.l1_loss(angle_sincos_true, angle_sincos_pred)
        
        # angle_term = (F.l1_loss(torch.cos(angle_true) + torch.sin(angle_true), torch.cos(angle_pred) + torch.sin(angle_pred))).mean()
        # angle_term = F.l1_loss(
        #     torch.cos(angle_true[:, self.mask_bin_to:self.mask_bin_from, :]) + torch.sin(angle_true[:, self.mask_bin_to:self.mask_bin_from, :]),
        #     torch.cos(angle_pred[:, self.mask_bin_to:self.mask_bin_from, :]) + torch.sin(angle_pred[:, self.mask_bin_to:self.mask_bin_from, :]))
        angle_term = F.l1_loss(
            torch.cos(angle_true[:, self.mask_bin_to:self.mask_bin_from, :]) + torch.sin(angle_true[:, self.mask_bin_to:self.mask_bin_from, :]),
            torch.cos(angle_pred[:, self.mask_bin_to:self.mask_bin_from, :]) + torch.sin(angle_pred[:, self.mask_bin_to:self.mask_bin_from, :]))

        loss = converge_term + self.alpha * log_term + self.beta * angle_term
        return loss
    
    # def log_frequency_scale(self, n_fft):
    #     return torch.log2(torch.arange(0, n_fft//2+1) + 2) / torch.log2(torch.tensor(n_fft))
    
    def log_frequency_scale(self, n_fft, min_clip_bins=16):
        ret = torch.log2(torch.max(torch.tensor(min_clip_bins), torch.arange(0, n_fft//2+1)) + 2) / torch.log2(torch.tensor(n_fft))
        ret[0] = 1.
        return ret
    
    def to(self, device=None, dtype=None, non_blocking=False, memory_format=torch.preserve_format):
        super().to(device=device, dtype=dtype, non_blocking=non_blocking, memory_format=memory_format)
        self.log_freq_scale = self.log_freq_scale.to(device=device, dtype=dtype, non_blocking=non_blocking, memory_format=memory_format)
        self.rev_log_freq_scale = self.rev_log_freq_scale.to(device=device, dtype=dtype, non_blocking=non_blocking, memory_format=memory_format)
        return self
    
    
class LF4SMalinblogsLoss(nn.Module):
    """
    Log-Frequency Scaled Single-Scale Spectral Magnitude and Phase Malinblogs Loss. 
    """

    def __init__(self, n_fft=111, alpha=1.0, beta=0.5, overlap=0, eps=1e-7, window_fn=torch.hann_window, wkwargs=None):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.hop_length = int(n_fft * (1 - overlap))
        # self.spec = torchaudio.transforms.Spectrogram(
        #     n_fft=self.n_fft, hop_length=self.hop_length, power=None,
        #     normalized=True, center=False, window_fn=window_fn, wkwargs=wkwargs)
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, power=None,
            normalized=True, center=False, window_fn=window_fn, wkwargs=wkwargs, pad_mode="constant")
        self.log_freq_scale = self.log_frequency_scale(n_fft)[None, :, None]
        
    def forward(self, x_true, x_pred):
        pad_edges = self.n_fft//2  # half of first/last frame
        x_true_pad = F.pad(x_true, (pad_edges, pad_edges), mode = 'constant')
        x_pred_pad = F.pad(x_pred, (pad_edges, pad_edges), mode = 'constant')
        true_spec = self.spec(x_true_pad)
        pred_spec = self.spec(x_pred_pad)
        # true_spec = self.spec(x_true_pad)[:, :, 1:-1]
        # pred_spec = self.spec(x_pred_pad)[:, :, 1:-1]
        S_true = true_spec.abs() * self.log_freq_scale + self.eps
        S_pred = pred_spec.abs() * self.log_freq_scale + self.eps
        # angle_true = true_spec.angle() * self.log_freq_scale
        # angle_pred = pred_spec.angle() * self.log_freq_scale 
        # angle_true = true_spec.angle()
        # angle_pred = pred_spec.angle()
        
        
        converge_term = torch.mean(torch.linalg.norm(S_true - S_pred, dim = (1, 2)) / torch.linalg.norm(S_true + S_pred, dim = (1, 2)))
        
        # log_term = F.l1_loss(S_true.log(), S_pred.log())
        # log_term = mall(S_true, S_pred)
        # log_term = mall(S_true.log(), S_pred.log())
        
        S_true_std, S_true_mean = torch.std_mean(S_true, dim=(1,2), keepdim=True)
        S_pred_std, S_pred_mean = torch.std_mean(S_pred, dim=(1,2), keepdim=True)
        log_term = malinblogs((S_true - S_true_mean)/(S_true_std + self.eps), (S_pred - S_pred_mean)/(S_pred_std + self.eps))
        
        # angle_term = ((torch.cos(angle_true) - torch.cos(angle_pred)) ** 2.).mean()
        # angle_term = (F.l1_loss(torch.cos(angle_true), torch.cos(angle_pred))).mean()

        loss = converge_term + self.alpha * log_term
        return loss
    
    # def log_frequency_scale(self, n_fft):
    #     return torch.log2(torch.arange(0, n_fft//2+1) + 2) / torch.log2(torch.tensor(n_fft))
    
    def log_frequency_scale(self, n_fft, min_clip_bins=16):
        ret = torch.log2(torch.max(torch.tensor(min_clip_bins), torch.arange(0, n_fft//2+1)) + 2) / torch.log2(torch.tensor(n_fft))
        ret[0] = 1.
        return ret
    
    def to(self, device=None, dtype=None, non_blocking=False, memory_format=torch.preserve_format):
        super().to(device=device, dtype=dtype, non_blocking=non_blocking, memory_format=memory_format)
        self.log_freq_scale = self.log_freq_scale.to(device=device, dtype=dtype, non_blocking=non_blocking, memory_format=memory_format)
        return self
    
    
class SSSMalinblogsLoss(nn.Module):
    """
    Single-Scale Spectral Malinblogs Loss. 
    """

    def __init__(self, n_fft=111, alpha=1.0, beta=0.5, overlap=0, eps=1e-7, window_fn=torch.hann_window, wkwargs=None, mask_bin_to=0, mask_bin_from=0):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        # self.beta = beta
        self.eps = eps
        self.mask_bin_to = mask_bin_to
        self.mask_bin_from = mask_bin_from if mask_bin_from > 0 else n_fft//2+1
        self.hop_length = int(n_fft * (1 - overlap))
        # self.spec = torchaudio.transforms.Spectrogram(
        #     n_fft=self.n_fft, hop_length=self.hop_length, power=None,
        #     normalized=True, center=False, window_fn=window_fn, wkwargs=wkwargs)
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, power=None,
            normalized=True, center=False, window_fn=window_fn, wkwargs=wkwargs, pad_mode="constant")
        # self.spec = torchaudio.transforms.Spectrogram(
        #     n_fft=self.n_fft, hop_length=self.hop_length, power=None,
        #     normalized=True, center=True, window_fn=window_fn, wkwargs=wkwargs, pad_mode="reflect")
        
    def forward(self, x_true, x_pred):
        pad_edges = self.n_fft//2  # half of first/last frame
        x_true_pad = F.pad(x_true, (pad_edges, pad_edges), mode = 'constant')
        x_pred_pad = F.pad(x_pred, (pad_edges, pad_edges), mode = 'constant')
        true_spec = self.spec(x_true_pad)[:, self.mask_bin_to:self.mask_bin_from, :]
        pred_spec = self.spec(x_pred_pad)[:, self.mask_bin_to:self.mask_bin_from, :]
        # x_true = F.pad(x_true, (pad_edges, pad_edges), mode = 'constant')
        # x_pred = F.pad(x_pred, (pad_edges, pad_edges), mode = 'constant')
        # true_spec = self.spec(x_true)[:, self.mask_bin_to:self.mask_bin_from, :]
        # pred_spec = self.spec(x_pred)[:, self.mask_bin_to:self.mask_bin_from, :]
        # true_spec = self.spec(x_true_pad)[:, :, 1:-1]
        # pred_spec = self.spec(x_pred_pad)[:, :, 1:-1]
        S_true = true_spec.abs() + self.eps
        S_pred = pred_spec.abs() + self.eps
        # angle_true = true_spec.angle() * self.log_freq_scale
        # angle_pred = pred_spec.angle() * self.log_freq_scale 
        
        # angle_true = true_spec.angle()
        # angle_pred = pred_spec.angle()
        
        
        converge_term = torch.mean(torch.linalg.norm(S_true - S_pred, dim = (1, 2)) / torch.linalg.norm(S_true + S_pred, dim = (1, 2)))
        
        # # log_term = F.l1_loss(S_true.log(), S_pred.log())
        # S_true_std, S_true_mean = torch.std_mean(S_true, dim=(1,2), keepdim=True)
        # S_pred_std, S_pred_mean = torch.std_mean(S_pred, dim=(1,2), keepdim=True)
        # log_term = mall((S_true - S_true_mean)/(S_true_std + self.eps), (S_pred - S_pred_mean)/(S_pred_std + self.eps))
        # # log_term = mall(S_true.log(), S_pred.log())
        
        log_term = malinblogs(S_true, S_pred)
        # log_term = F.l1_loss(S_true, S_pred)
        
        # angle_term = ((torch.cos(angle_true) - torch.cos(angle_pred)) ** 2.).mean()
        # angle_term = (F.l1_loss(torch.cos(angle_true) + torch.sin(angle_true), torch.cos(angle_pred) + torch.sin(angle_pred))).mean()

        loss = converge_term + self.alpha * log_term
        return loss


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
    
    
class DSMPMalLoss(nn.Module):
    '''
    Dual (DFT bin aliased) Spectral Magnitude and Phase MAL Loss
    '''
    
    def __init__(self, fft_min, fft_max, alpha=1.0, beta=0.5, gamma=0.7, overlap=0, eps=1e-7, device='cuda'):
        super().__init__()
        self.beta = beta
        self.fft_min = fft_min
        self.fft_max = fft_max
        self.lossdict = {}
        self.lossdict[fft_max] = SSSMPMalLoss(fft_max, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device)
        self.lossdict[fft_max-1] = SSSMPMalLoss(fft_max-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device)
        self.lossdict[fft_min] = SSSMPMalLoss(fft_min, alpha, gamma, overlap, eps, window_fn=torch.blackman_window).to(device)
        self.lossdict[fft_min-1] = SSSMPMalLoss(fft_min-1, alpha, gamma, overlap, eps, window_fn=torch.blackman_window).to(device)
        
    def forward(self, x_pred, x_true):
        value = 0.
        # choose minus one fft size randomly for frequency bin aliasing
        n_fft_offsets = torch.randint(0, 1, (2,))
        value += self.lossdict[self.fft_min-int(n_fft_offsets[0])](x_true, x_pred)*(1. - self.beta)
        value += self.lossdict[self.fft_max-int(n_fft_offsets[1])](x_true, x_pred)*self.beta
        return value


class DLFSSMPMalLoss(nn.Module):
    '''
    Dual (DFT bin aliased) Log-Frequency Scaled Spectral Magnitude and Phase MAL Loss
    '''
    
    def __init__(self, fft_min, fft_max, alpha=1.0, beta=0.5, gamma=0.7, overlap=0, eps=1e-7, device='cuda'):
        super().__init__()
        self.beta = beta
        self.fft_min = fft_min
        self.fft_max = fft_max
        self.lossdict = {}
        self.lossdict[fft_max] = LF4SMPMalLoss(fft_max, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device)
        self.lossdict[fft_max-1] = LF4SMPMalLoss(fft_max-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device)
        self.lossdict[fft_min] = LF4SMPMalLoss(fft_min, alpha, gamma, overlap, eps, window_fn=torch.blackman_window).to(device)
        self.lossdict[fft_min-1] = LF4SMPMalLoss(fft_min-1, alpha, gamma, overlap, eps, window_fn=torch.blackman_window).to(device)
        
    def forward(self, x_pred, x_true):
        value = 0.
        # choose minus one fft size randomly for frequency bin aliasing
        n_fft_offsets = torch.randint(0, 1, (2,))
        value += self.lossdict[self.fft_min-int(n_fft_offsets[0])](x_true, x_pred)*(1. - self.beta)
        value += self.lossdict[self.fft_max-int(n_fft_offsets[1])](x_true, x_pred)*self.beta
        return value
    
    
class DLFSSMPMalinblogsLoss(nn.Module):
    '''
    Dual (DFT bin aliased) Log-Frequency Scaled Spectral Magnitude and Phase Malinblogs Loss
    '''
    
    def __init__(self, fft_min, fft_max, alpha=1.0, beta=0.5, gamma=0.7, overlap=0, eps=1e-7, device='cuda'):
        super().__init__()
        self.beta = beta
        self.fft_min = fft_min
        self.fft_max = fft_max
        self.lossdict = {}
        self.lossdict[fft_max] = LF4SMPMalinblogsLoss(fft_max, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device)
        self.lossdict[fft_max-1] = LF4SMPMalinblogsLoss(fft_max-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device)
        self.lossdict[fft_min] = LF4SMPMalinblogsLoss(fft_min, alpha, gamma, overlap, eps, window_fn=torch.blackman_window).to(device)
        self.lossdict[fft_min-1] = LF4SMPMalinblogsLoss(fft_min-1, alpha, gamma, overlap, eps, window_fn=torch.blackman_window).to(device)
        
    def forward(self, x_pred, x_true):
        value = 0.
        # choose minus one fft size randomly for frequency bin aliasing
        n_fft_offsets = torch.randint(0, 1, (2,))
        value += self.lossdict[self.fft_min-int(n_fft_offsets[0])](x_true, x_pred)*(1. - self.beta)
        value += self.lossdict[self.fft_max-int(n_fft_offsets[1])](x_true, x_pred)*self.beta
        return value
    
    
class DLFSSMalinblogsLoss(nn.Module):
    '''
    Dual (DFT bin aliased) Log-Frequency Scaled Spectral Malinblogs Loss
    '''
    
    def __init__(self, fft_min, fft_max, alpha=1.0, beta=0.5, gamma=0.7, overlap=0, eps=1e-7, device='cuda'):
        super().__init__()
        self.beta = beta
        self.fft_min = fft_min
        self.fft_max = fft_max
        self.lossdict = {}
        self.lossdict[fft_max] = LF4SMalinblogsLoss(fft_max, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device)
        self.lossdict[fft_max-1] = LF4SMalinblogsLoss(fft_max-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device)
        self.lossdict[fft_min] = LF4SMalinblogsLoss(fft_min, alpha, gamma, overlap, eps, window_fn=torch.blackman_window).to(device)
        self.lossdict[fft_min-1] = LF4SMalinblogsLoss(fft_min-1, alpha, gamma, overlap, eps, window_fn=torch.blackman_window).to(device)
        
    def forward(self, x_pred, x_true):
        value = 0.
        # choose minus one fft size randomly for frequency bin aliasing
        n_fft_offsets = torch.randint(0, 1, (2,))
        value += self.lossdict[self.fft_min-int(n_fft_offsets[0])](x_true, x_pred)*(1. - self.beta)
        value += self.lossdict[self.fft_max-int(n_fft_offsets[1])](x_true, x_pred)*self.beta
        return value
    
    
class MRLFSSMPMalinblogsLoss(nn.Module):
    '''
    Multi-Resolution Log-Frequency Scaled Spectral Magnitude and Phase Malinblogs Loss
    '''
    
    def __init__(self, n_fft, n_div, alpha=1.0, beta=0.5, gamma=0.7, overlap=0, eps=1e-7, device='cuda'):
        super().__init__()
        self.beta = beta
        self.n_fft = n_fft
        self.n_div = n_div
        self.fftnums = [int(self.n_fft/(2 ** powr)) for powr in range(0,self.n_div)]
        self.lossdict = {
            n_fft: LF4SMPMalinblogsLoss(n_fft, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_from=(n_fft//2+1) - n_fft//2//n_div*(n_div-1)).to(device),
            n_fft-1: LF4SMPMalinblogsLoss(n_fft-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_from=((n_fft-1)//2+1) - (n_fft-1)//2//n_div*(n_div-1)).to(device)
        }
        self.lossdict.update(
            {
                n: LF4SMPMalinblogsLoss(n, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_to=(n//2+1) - n//2//n_div*(n_div-(i+1))).to(device)
                for i, n in enumerate(self.fftnums[1:])
            }
        )
        self.lossdict.update(
            {
                n-1: LF4SMPMalinblogsLoss(n-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_to=((n-1)//2+1) - (n-1)//2//n_div*(n_div-(i+1))).to(device)
                for i, n in enumerate(self.fftnums[1:])
            }
        )
        # self.lossdict[fft_max] = LF4SMPMalinblogsLoss(fft_max, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device)
        # self.lossdict[fft_max-1] = LF4SMPMalinblogsLoss(fft_max-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device)
        # self.lossdict[fft_min] = LF4SMPMalinblogsLoss(fft_min, alpha, gamma, overlap, eps, window_fn=torch.blackman_window).to(device)
        # self.lossdict[fft_min-1] = LF4SMPMalinblogsLoss(fft_min-1, alpha, gamma, overlap, eps, window_fn=torch.blackman_window).to(device)
        
    def forward(self, x_pred, x_true):
        # choose minus one fft size randomly for frequency bin aliasing
        # n_fft_offsets = torch.randint(0, 1, (2,))
        n_fft_offset = torch.randint(0, 1, (1,))[0]
        return torch.stack([self.lossdict[n-int(n_fft_offset)](x_true, x_pred) for n in self.fftnums], dim=0).mean()
        # value += self.lossdict[self.fft_min-int(n_fft_offsets[0])](x_true, x_pred)*(1. - self.beta)
        # value += self.lossdict[self.fft_max-int(n_fft_offsets[1])](x_true, x_pred)*self.beta
        # return value
        

def freq_hann_window(window_lengh, freq=1.0):
    n = torch.linspace(0.0, 1.0, window_lengh)
    mask = torch.where(n*freq + 0.5-freq*0.5 < 0.0, 0.0, 1.0)
    mask = mask * torch.where(n*freq + 0.5-freq*0.5 > 1.0, 0.0, 1.0)
    window = (0.5 + 0.5*torch.cos(2.*torch.pi*(n - 0.5)*freq))*mask
    return window


def freq_bartlett_window(window_lengh, freq=1.0):
    n = torch.linspace(0.0, 1.0, window_lengh)
    mask = torch.where(n*freq + 0.5-freq*0.5 < 0.0, 0.0, 1.0)
    mask = mask * torch.where(n*freq + 0.5-freq*0.5 > 1.0, 0.0, 1.0)
    window = (1. - torch.abs(2.*(n - 0.5)*freq))*mask
    return window

        
class MRSMPMalinblogsLoss(nn.Module):
    '''
    Multi-Resolution Spectral Magnitude and Phase Malinblogs Loss
    '''
    
    def __init__(self, n_fft, n_div, alpha=1.0, beta=0.5, gamma=0.7, n_harm_width=8, overlap=0, eps=1e-7, device='cuda'):
        super().__init__()
        self.beta = beta
        self.n_fft = n_fft
        self.n_div = n_div
        self.fftnums = [int(self.n_fft/(2 ** powr)) for powr in range(0,self.n_div)]
        # self.lossdict = {
        #     n_fft: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_from=(n_fft//2+1) - n_fft//2//n_div*(n_div-1)).to(device),
        #     n_fft-1: SSSMPMalinblogsLoss(n_fft-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_from=((n_fft-1)//2+1) - (n_fft-1)//2//n_div*(n_div-1)).to(device)
        # }
        # self.lossdict.update(
        #     {
        #         n: SSSMPMalinblogsLoss(n, alpha, gamma, overlap, eps, window_fn=torch.hann_window,
        #                                mask_bin_to=(n//2+1) - n//2//n_div*(n_div-(i+1)),
        #                                mask_bin_from=(n//2+1) - n//2//n_div*(n_div-(i+1+1))).to(device)
        #         for i, n in enumerate(self.fftnums[1:])
        #     }
        # )
        # self.lossdict.update(
        #     {
        #         n-1: SSSMPMalinblogsLoss(n-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window,
        #                                  mask_bin_to=((n-1)//2+1) - (n-1)//2//n_div*(n_div-(i+1)),
        #                                  mask_bin_from=((n-1)//2+1) - (n-1)//2//n_div*(n_div-(i+1+1))).to(device)
        #         for i, n in enumerate(self.fftnums[1:])
        #     }
        # )
        
        # self.lossdict = {
        #     n_fft: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap, eps, window_fn=freq_hann_window, mask_bin_from=(n_fft//2+1) - n_fft//2//n_div*(n_div-1)).to(device),
        #     n_fft-1: SSSMPMalinblogsLoss(n_fft-1, alpha, gamma, overlap, eps, window_fn=freq_hann_window, mask_bin_from=((n_fft-1)//2+1) - (n_fft-1)//2//n_div*(n_div-1)).to(device)
        # }
        # self.lossdict.update(
        #     {
        #         n: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap + (1.-1./(n_fft/n))*(1.-overlap), eps, window_fn=freq_hann_window,
        #                                mask_bin_to=(n_fft//2+1) - n_fft//2//n_div*(n_div-(i+1)),
        #                                mask_bin_from=(n_fft//2+1) - n_fft//2//n_div*(n_div-(i+1+1)),
        #                                wkwargs={'freq': n_fft/n}).to(device)
        #         for i, n in enumerate(self.fftnums[1:])
        #     }
        # )
        # self.lossdict.update(
        #     {
        #         n-1: SSSMPMalinblogsLoss(n_fft-1, alpha, gamma, overlap + (1.-1./(n_fft/n))*(1.-overlap), eps, window_fn=freq_hann_window,
        #                                  mask_bin_to=((n_fft-1)//2+1) - (n_fft-1)//2//n_div*(n_div-(i+1)),
        #                                  mask_bin_from=((n_fft-1)//2+1) - (n_fft-1)//2//n_div*(n_div-(i+1+1)),
        #                                  wkwargs={'freq': n_fft/n}).to(device)
        #         for i, n in enumerate(self.fftnums[1:])
        #     }
        # )
        
        # self.lossdict = {
        #     n_fft: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_from=n_harm_width+1).to(device),
        #     # n_fft-1: SSSMPMalinblogsLoss(n_fft-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_from=n_harm_width+1).to(device),
        #     # n_fft: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device),
        #     # n_fft-1: SSSMPMalinblogsLoss(n_fft-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device),
        #     self.fftnums[-1]: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap + (1.-1./(n_fft/self.fftnums[-1]))*(1.-overlap), eps,
        #                                           window_fn=freq_hann_window,
        #                                         #   mask_bin_to=n_harm_width//2,
        #                                           wkwargs={'freq': n_fft/self.fftnums[-1]}).to(device),
        #     # self.fftnums[-1]-1: SSSMPMalinblogsLoss(n_fft-1, alpha, gamma, overlap + (1.-1./(n_fft/(self.fftnums[-1]-1)))*(1.-overlap), eps,
        #     #                                         window_fn=freq_hann_window, 
        #     #                                         mask_bin_to=n_harm_width//2,
        #     #                                         wkwargs={'freq': n_fft/(self.fftnums[-1]-1)}).to(device)
        # }
        # self.lossdict.update(
        #     {
        #         n: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap + (1.-1./(n_fft/n))*(1.-overlap), eps, window_fn=freq_hann_window,
        #                                mask_bin_to=(n_harm_width*(i+1))//2,
        #                                mask_bin_from=n_harm_width*(i+1)+1,
        #                                wkwargs={'freq': n_fft/n}).to(device)
        #         for i, n in enumerate(self.fftnums[1:-1])
        #     }
        # )
        
        self.lossdict = {
            self.fftnums[-1]: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap + (1.-1./(n_fft/self.fftnums[-1]))*(1.-overlap), eps,
                                                  window_fn=freq_hann_window,
                                                #   mask_bin_to=n_harm_width//2,
                                                  wkwargs={'freq': n_fft/self.fftnums[-1]}).to(device),
        }
        
        # self.lossdict.update(
        #     {
        #         n-1: SSSMPMalinblogsLoss(n_fft-1, alpha, gamma, overlap + (1.-1./(n_fft/(n-1)))*(1.-overlap), eps, window_fn=freq_hann_window,
        #                                  mask_bin_to=(n_harm_width*(i+1))//2,
        #                                  mask_bin_from=n_harm_width*(i+1)+1,
        #                                  wkwargs={'freq': n_fft/(n-1)}).to(device)
        #         for i, n in enumerate(self.fftnums[1:])
        #     }
        # )
        min_fft = self.fftnums[-1]
        min_fft_2 = self.fftnums[-2]
        # min_half_fft = self.fftnums[-1]+self.fftnums[-1]//2
        # self.fftnums.extend([min_fft+min_fft//2, min_fft+min_fft//2-1])
        # self.lossdict.update(
        #     {
        #         n: SSSMPMalinblogsLoss(n, alpha, gamma, overlap, eps, window_fn=torch.hann_window,
        #                                mask_bin_to=n_harm_width//2).to(device)
        #         for n in [min_fft+min_fft//2, min_fft+min_fft//2-1]
        #     }
        # )
        
        # self.fftnums_rand = [n for n in range(min_fft+1,min_fft_2*n_harm_width, 4)]
        # self.fftnums_rand = [n for n in range(min_fft+1,min_fft_2*(n_harm_width//2))]
        # fftnums_rands = [(2 ** powr) for powr in range(2, int(torch.log2(min_fft)))]
        # self.fftnums_rand = [n for n in fftnums_rands]
        # self.fftnums_rand = [(2 ** powr) for powr in range(2, int(torch.log2(torch.tensor(min_fft))))]
        
        # self.fftnums_rand = [n for n in range(min_fft+1,min_fft_2*(n_harm_width//2))]
        # self.lossdict.update(
        #     {
        #         n: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap + (1.-1./(n_fft/n))*(1.-overlap), eps, window_fn=freq_hann_window,
        #                                mask_bin_to=(n_harm_width*int(n_fft/n))//2,
        #                                wkwargs={'freq': n_fft/n}).to(device)
        #         for n in self.fftnums_rand
        #     }
        # )
        
        # self.lossdict = {
        #     n_fft: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_from=n_harm_width+1).to(device),
        #     n_fft-1: SSSMPMalinblogsLoss(n_fft-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_from=n_harm_width+1).to(device),
        #     # n_fft: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device),
        #     # n_fft-1: SSSMPMalinblogsLoss(n_fft-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device),
        #     self.fftnums[-1]: SSSMPMalinblogsLoss(self.fftnums[-1], alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_to=n_harm_width//2).to(device),
        #     self.fftnums[-1]-1: SSSMPMalinblogsLoss(self.fftnums[-1]-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_to=n_harm_width//2).to(device)
        # }
        # self.lossdict.update(
        #     {
        #         n: SSSMPMalinblogsLoss(n, alpha, gamma, overlap, eps, window_fn=torch.hann_window,
        #                                mask_bin_to=n_harm_width//2,
        #                                mask_bin_from=n_harm_width+1).to(device)
        #         for i, n in enumerate(self.fftnums[1:-1])
        #     }
        # )
        # self.lossdict.update(
        #     {
        #         n-1: SSSMPMalinblogsLoss(n-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window,
        #                                  mask_bin_to=n_harm_width//2,
        #                                  mask_bin_from=n_harm_width+1).to(device)
        #         for i, n in enumerate(self.fftnums[1:-1])
        #     }
        # )
        # min_fft = self.fftnums[-1]
        # min_fft_2 = self.fftnums[-2]
        # # min_half_fft = self.fftnums[-1]+self.fftnums[-1]//2
        # # self.fftnums.extend([min_fft+min_fft//2, min_fft+min_fft//2-1])
        # # self.lossdict.update(
        # #     {
        # #         n: SSSMPMalinblogsLoss(n, alpha, gamma, overlap, eps, window_fn=torch.hann_window,
        # #                                mask_bin_to=n_harm_width//2).to(device)
        # #         for n in [min_fft+min_fft//2, min_fft+min_fft//2-1]
        # #     }
        # # )
        
        # # self.fftnums_rand = [n for n in range(min_fft+1,min_fft_2*n_harm_width, 4)]
        # self.fftnums_rand = [n for n in range(min_fft+1,min_fft_2*(n_harm_width//2))]
        # self.lossdict.update(
        #     {
        #         n: SSSMPMalinblogsLoss(n, alpha, gamma, overlap, eps, window_fn=torch.hann_window,
        #                                mask_bin_to=n_harm_width//2).to(device)
        #         for n in self.fftnums_rand
        #     }
        # )
        
        # self.lossdict[fft_max] = LF4SMPMalinblogsLoss(fft_max, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device)
        # self.lossdict[fft_max-1] = LF4SMPMalinblogsLoss(fft_max-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device)
        # self.lossdict[fft_min] = LF4SMPMalinblogsLoss(fft_min, alpha, gamma, overlap, eps, window_fn=torch.blackman_window).to(device)
        # self.lossdict[fft_min-1] = LF4SMPMalinblogsLoss(fft_min-1, alpha, gamma, overlap, eps, window_fn=torch.blackman_window).to(device)
        
    def forward(self, x_pred, x_true):
        # choose minus one fft size randomly for frequency bin aliasing
        # n_fft_offsets = torch.randint(0, 1, (2,))
        # n_fft_offset = torch.randint(0, 2, (1,))[0]
        # n_fft_rand = torch.randint(0, len(self.fftnums_rand), (1,))[0]
        # return torch.stack([
        #     *[self.lossdict[n-int(n_fft_offset)](x_true, x_pred) for n in self.fftnums],
        #     # self.lossdict[self.fftnums_rand[n_fft_rand]](x_true, x_pred)
        #     # *[self.lossdict[n](x_true, x_pred) for n in self.fftnums_rand]
        # ], dim=0).mean()
        
        # values = 0.
        # for n in self.fftnums:
        #     # values += self.lossdict[n-int(n_fft_offset)](x_true, x_pred)
        #     values += self.lossdict[n](x_true, x_pred)
        # # values += self.lossdict[self.fftnums_rand[n_fft_rand]](x_true, x_pred)
        # # return values/(len(self.fftnums) + 1)
        # return values/(len(self.fftnums))
        
        return self.lossdict[self.fftnums[-1]](x_true, x_pred)
        
        
class MRSMPL1Loss(nn.Module):
    '''
    Multi-Resolution Spectral Magnitude and Phase L1 Loss
    '''
    
    def __init__(self, n_fft, n_div, alpha=1.0, beta=0.5, gamma=0.7, n_harm_width=8, overlap=0, eps=1e-7, device='cuda'):
        super().__init__()
        self.beta = beta
        self.n_fft = n_fft
        self.n_div = n_div
        self.fftnums = [int(self.n_fft/(2 ** powr)) for powr in range(0,self.n_div)]
        
        self.lossdict = {
            n_fft: SSSMPL1Loss(n_fft, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_from=n_harm_width+1).to(device),
            # n_fft-1: SSSMPMalinblogsLoss(n_fft-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_from=n_harm_width+1).to(device),
            # n_fft: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device),
            # n_fft-1: SSSMPMalinblogsLoss(n_fft-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device),
            self.fftnums[-1]: SSSMPL1Loss(n_fft, alpha, gamma, overlap + (1.-1./(n_fft/self.fftnums[-1]))*(1.-overlap), eps,
                                                  window_fn=freq_hann_window,
                                                  mask_bin_to=n_harm_width//2,
                                                  wkwargs={'freq': n_fft/self.fftnums[-1]}).to(device),
            # self.fftnums[-1]-1: SSSMPMalinblogsLoss(n_fft-1, alpha, gamma, overlap + (1.-1./(n_fft/(self.fftnums[-1]-1)))*(1.-overlap), eps,
            #                                         window_fn=freq_hann_window, 
            #                                         mask_bin_to=n_harm_width//2,
            #                                         wkwargs={'freq': n_fft/(self.fftnums[-1]-1)}).to(device)
        }
        self.lossdict.update(
            {
                n: SSSMPL1Loss(n_fft, alpha, gamma, overlap + (1.-1./(n_fft/n))*(1.-overlap), eps, window_fn=freq_hann_window,
                                       mask_bin_to=(n_harm_width*(i+1))//2,
                                       mask_bin_from=n_harm_width*(i+1)+1,
                                       wkwargs={'freq': n_fft/n}).to(device)
                for i, n in enumerate(self.fftnums[1:-1])
            }
        )
        # self.lossdict.update(
        #     {
        #         n-1: SSSMPMalinblogsLoss(n_fft-1, alpha, gamma, overlap + (1.-1./(n_fft/(n-1)))*(1.-overlap), eps, window_fn=freq_hann_window,
        #                                  mask_bin_to=(n_harm_width*(i+1))//2,
        #                                  mask_bin_from=n_harm_width*(i+1)+1,
        #                                  wkwargs={'freq': n_fft/(n-1)}).to(device)
        #         for i, n in enumerate(self.fftnums[1:])
        #     }
        # )
        min_fft = self.fftnums[-1]
        min_fft_2 = self.fftnums[-2]
        
        self.fftnums_rand = [n for n in range(min_fft+1,min_fft_2*(n_harm_width//2))]
        self.lossdict.update(
            {
                n: SSSMPL1Loss(n_fft, alpha, gamma, overlap + (1.-1./(n_fft/n))*(1.-overlap), eps, window_fn=freq_hann_window,
                                       mask_bin_to=(n_harm_width*int(n_fft/n))//2,
                                       wkwargs={'freq': n_fft/n}).to(device)
                for n in self.fftnums_rand
            }
        )
        
    def forward(self, x_pred, x_true):
        # choose minus one fft size randomly for frequency bin aliasing
        # n_fft_offsets = torch.randint(0, 1, (2,))
        # n_fft_offset = torch.randint(0, 2, (1,))[0]
        n_fft_rand = torch.randint(0, len(self.fftnums_rand), (1,))[0]
        # return torch.stack([
        #     *[self.lossdict[n-int(n_fft_offset)](x_true, x_pred) for n in self.fftnums],
        #     # self.lossdict[self.fftnums_rand[n_fft_rand]](x_true, x_pred)
        #     # *[self.lossdict[n](x_true, x_pred) for n in self.fftnums_rand]
        # ], dim=0).mean()
        values = 0.
        for n in self.fftnums:
            # values += self.lossdict[n-int(n_fft_offset)](x_true, x_pred)
            values += self.lossdict[n](x_true, x_pred)
        values += self.lossdict[self.fftnums_rand[n_fft_rand]](x_true, x_pred)
        return values/(len(self.fftnums) + 1)
        # return values/(len(self.fftnums))


class MRLF4SMPMalinblogsLoss(nn.Module):
    '''
    Multi-Resolution Spectral Magnitude and Phase Malinblogs Loss
    '''
    
    def __init__(self, n_fft, n_div, alpha=1.0, beta=0.5, gamma=0.7, n_harm_width=16, overlap=0, eps=1e-7, device='cuda'):
        super().__init__()
        self.beta = beta
        self.n_fft = n_fft
        self.n_div = n_div
        self.fftnums = [int(self.n_fft/(2 ** powr)) for powr in range(0,self.n_div)]
        
        self.lossdict = {
            n_fft: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap, eps,
                                                    window_fn=torch.hann_window,
                                                    mask_bin_from=n_harm_width).to(device),
            self.fftnums[-1]: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap + (1.-1./(n_fft/self.fftnums[-1]))*(1.-overlap), eps,
                                                    window_fn=freq_hann_window,
                                                    #   mask_bin_to=n_harm_width//2,
                                                    mask_bin_to=n_harm_width,
                                                    wkwargs={'freq': n_fft/self.fftnums[-1]}).to(device),
        }
        
    def forward(self, x_pred, x_true):
        value = self.lossdict[self.n_fft](x_true, x_pred)
        return (value + self.lossdict[self.fftnums[-1]](x_true, x_pred))*0.5
    
    
class MRRSMalinblogsLoss(nn.Module):
    '''
    Multi-Resolution Resampled Spectral Malinblogs Loss
    '''
    
    def __init__(self, n_fft, n_div, alpha=1.0, beta=0.5, gamma=0.7, n_harm_width=8, overlap=0, eps=1e-7, device='cuda'):
        super().__init__()
        self.beta = beta
        self.n_fft = n_fft
        self.n_div = n_div
        self.fftnums = [int(self.n_fft/(2 ** powr)) for powr in range(0,self.n_div)]
        
        # self.lossdict = {
        #     n_fft: SSSMalinblogsLoss(n_fft, alpha, gamma, overlap, eps,
        #                                             window_fn=torch.hann_window,
        #                                             mask_bin_from=n_harm_width).to(device),
        #     self.fftnums[-1]: SSSMalinblogsLoss(n_fft, alpha, gamma, overlap + (1.-1./(n_fft/self.fftnums[-1]))*(1.-overlap), eps,
        #                                             window_fn=freq_hann_window,
        #                                             #   mask_bin_to=n_harm_width//2,
        #                                             mask_bin_to=n_harm_width,
        #                                             wkwargs={'freq': n_fft/self.fftnums[-1]}).to(device),
        # }
        
        # self.lossdict = {
        #     n_fft: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_from=(n_fft//2+1) - n_fft//2//n_div*(n_div-1)).to(device),
        #     n_fft-1: SSSMPMalinblogsLoss(n_fft-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_from=((n_fft-1)//2+1) - (n_fft-1)//2//n_div*(n_div-1)).to(device)
        # }
        # self.lossdict.update(
        #     {
        #         n: SSSMPMalinblogsLoss(n, alpha, gamma, overlap, eps, window_fn=torch.hann_window,
        #                                mask_bin_to=(n//2+1) - n//2//n_div*(n_div-(i+1)),
        #                                mask_bin_from=(n//2+1) - n//2//n_div*(n_div-(i+1+1))).to(device)
        #         for i, n in enumerate(self.fftnums[1:])
        #     }
        # )
        # self.lossdict.update(
        #     {
        #         n-1: SSSMPMalinblogsLoss(n-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window,
        #                                  mask_bin_to=((n-1)//2+1) - (n-1)//2//n_div*(n_div-(i+1)),
        #                                  mask_bin_from=((n-1)//2+1) - (n-1)//2//n_div*(n_div-(i+1+1))).to(device)
        #         for i, n in enumerate(self.fftnums[1:])
        #     }
        # )
        # self.lossdict.update(
        #     {
        #         n: SSSMPMalinblogsLoss(n, alpha, gamma, overlap + (1.-1./(n/n))*(1.-overlap), eps, window_fn=freq_hann_window,
        #                                mask_bin_to=(n//2+1) - n//2//n_div*(n_div-(i+1)),
        #                                mask_bin_from=(n//2+1) - n//2//n_div*(n_div-(i+1+1)),
        #                                wkwargs={'freq': 1.}).to(device)
        #         for i, n in enumerate(self.fftnums[1:])
        #     }
        # )
        # self.lossdict.update(
        #     {
        #         n-1: SSSMPMalinblogsLoss(n-1, alpha, gamma, overlap + (1.-1./(n/n))*(1.-overlap), eps, window_fn=freq_hann_window,
        #                                  mask_bin_to=((n-1)//2+1) - (n-1)//2//n_div*(n_div-(i+1)),
        #                                  mask_bin_from=((n-1)//2+1) - (n-1)//2//n_div*(n_div-(i+1+1)),
        #                                  wkwargs={'freq': 1.}).to(device)
        #         for i, n in enumerate(self.fftnums[1:])
        #     }
        # )
        
        # self.lossdict = {
        #     n_fft: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_from=n_harm_width+1).to(device),
        #     # n_fft-1: SSSMPMalinblogsLoss(n_fft-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_from=n_harm_width+1).to(device),
        #     # n_fft: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device),
        #     # n_fft-1: SSSMPMalinblogsLoss(n_fft-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device),
        #     self.fftnums[-1]: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap + (1.-1./(n_fft/self.fftnums[-1]))*(1.-overlap), eps,
        #                                           window_fn=freq_hann_window,
        #                                           mask_bin_to=n_harm_width//2,
        #                                           wkwargs={'freq': n_fft/self.fftnums[-1]}).to(device),
        #     # self.fftnums[-1]-1: SSSMPMalinblogsLoss(n_fft-1, alpha, gamma, overlap + (1.-1./(n_fft/(self.fftnums[-1]-1)))*(1.-overlap), eps,
        #     #                                         window_fn=freq_hann_window, 
        #     #                                         mask_bin_to=n_harm_width//2,
        #     #                                         wkwargs={'freq': n_fft/(self.fftnums[-1]-1)}).to(device)
        # }
        # self.lossdict.update(
        #     {
        #         n: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap + (1.-1./(n_fft/n))*(1.-overlap), eps, window_fn=freq_hann_window,
        #                                mask_bin_to=(n_harm_width*(i+1))//2,
        #                                mask_bin_from=n_harm_width*(i+1)+1,
        #                                wkwargs={'freq': n_fft/n}).to(device)
        #         for i, n in enumerate(self.fftnums[1:-1])
        #     }
        # )
        
        # self.lossdict = {
        #     n_fft: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_from=n_harm_width+1).to(device),
        #     # n_fft-1: SSSMPMalinblogsLoss(n_fft-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_from=n_harm_width+1).to(device),
        #     # n_fft: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device),
        #     # n_fft-1: SSSMPMalinblogsLoss(n_fft-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device),
        #     self.fftnums[-1]: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap + (1.-1./(n_fft/self.fftnums[-1]))*(1.-overlap), eps,
        #                                           window_fn=freq_hann_window,
        #                                         #   mask_bin_to=n_harm_width//2,
        #                                           wkwargs={'freq': n_fft/self.fftnums[-1]}).to(device),
        #     # self.fftnums[-1]-1: SSSMPMalinblogsLoss(n_fft-1, alpha, gamma, overlap + (1.-1./(n_fft/(self.fftnums[-1]-1)))*(1.-overlap), eps,
        #     #                                         window_fn=freq_hann_window, 
        #     #                                         mask_bin_to=n_harm_width//2,
        #     #                                         wkwargs={'freq': n_fft/(self.fftnums[-1]-1)}).to(device)
        # }
        # self.lossdict.update(
        #     {
        #         n: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap + (1.-1./(n_fft/n))*(1.-overlap), eps, window_fn=freq_hann_window,
        #                                mask_bin_to=(n_harm_width*(i+1))//2,
        #                                mask_bin_from=n_harm_width*(i+1)+1,
        #                                wkwargs={'freq': n_fft/n}).to(device)
        #         for i, n in enumerate(self.fftnums[1:-1])
        #     }
        # )
        
        # self.lossdict = {
        #     n_fft: SSSMalinblogsLoss(n_fft, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_from=(n_fft//2+1) - n_fft//2//n_div*(n_div-1)).to(device),
        #     n_fft-1: SSSMalinblogsLoss(n_fft-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_from=((n_fft-1)//2+1) - (n_fft-1)//2//n_div*(n_div-1)).to(device)
        # }
        # self.lossdict.update(
        #     {
        #         n: SSSMalinblogsLoss(n, alpha, gamma, overlap, eps, window_fn=torch.hann_window,
        #                                mask_bin_to=(n//2+1) - n//2//n_div*(n_div-(i+1)),
        #                                mask_bin_from=(n//2+1) - n//2//n_div*(n_div-(i+1+1))).to(device)
        #         for i, n in enumerate(self.fftnums[1:])
        #     }
        # )
        # self.lossdict.update(
        #     {
        #         n-1: SSSMalinblogsLoss(n-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window,
        #                                  mask_bin_to=((n-1)//2+1) - (n-1)//2//n_div*(n_div-(i+1)),
        #                                  mask_bin_from=((n-1)//2+1) - (n-1)//2//n_div*(n_div-(i+1+1))).to(device)
        #         for i, n in enumerate(self.fftnums[1:])
        #     }
        # )
        
        # self.resamples_num = self.fftnums[0] // self.fftnums[-1]
        
        # self.resamples = {
        #     f"{n_fft}-{n_fft_to}": torchaudio.transforms.Resample(n_fft, n_fft_to, lowpass_filter_width=n_fft//4).to(device)
        #     for i, n_fft in enumerate(self.fftnums[:-1])
        #     for n_fft_to in [self.fftnums[i+1] + int(self.fftnums[i+1]/self.resamples_num*v) for v in range(self.resamples_num)]
        # }
        
        #  + (1.-1./(n_fft/n))*(1.-overlap)
        
        # self.lossdict = {
        #     n_fft: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_from=(n_fft//2+1) - n_fft//2//n_div*(n_div-1)).to(device),
        #     # n_fft-1: SSSMalinblogsLoss(n_fft-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_from=((n_fft-1)//2+1) - (n_fft-1)//2//n_div*(n_div-1)).to(device)
        #     # n_fft: LF4SMPMalLoss(n_fft, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device),
        #     # n_fft: SSSMPMalLoss(n_fft, alpha, gamma, overlap + (1.-1./(n_fft/self.fftnums[-1]))*(1.-overlap), eps, window_fn=freq_hann_window,
        #     #                     mask_bin_from=(n_fft//2+1) - n_fft//2//n_div*(n_div-1),
        #     #                     wkwargs={'freq': n_fft/self.fftnums[-1]}).to(device),
        # }
        
        self.lossdict = {
            # n_fft: SSSMalinblogsLoss(n_fft, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_from=(n_fft//2+1) - n_fft//2//n_div*(n_div-1)).to(device),
            # n_fft-1: SSSMalinblogsLoss(n_fft-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_from=((n_fft-1)//2+1) - (n_fft-1)//2//n_div*(n_div-1)).to(device)
            # n_fft: LF4SMPMalLoss(n_fft, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device),
            # n_fft: SSSMPMalLoss(n_fft, alpha, gamma, overlap + (1.-1./(n_fft/self.fftnums[-1]))*(1.-overlap), eps, window_fn=freq_hann_window,
            #                     mask_bin_from=(n_fft//2+1) - n_fft//2//n_div*(n_div-1),
            #                     wkwargs={'freq': n_fft/self.fftnums[-1]}).to(device),
            # n_fft: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap + (1.-1./(n_fft/self.fftnums[-1]))*(1.-overlap), eps, window_fn=freq_hann_window,
            #                     wkwargs={'freq': n_fft/self.fftnums[-1]}).to(device),
            # n_fft: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap, eps, window_fn=torch.hann_window, mask_bin_from=(n_fft//2+1) - n_fft//2//n_div*(n_div-1)).to(device),
            n_fft: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device),
            # n_fft: LF4SMPMalinblogsLoss(n_fft, alpha, gamma, overlap, eps, window_fn=torch.hann_window).to(device),
        }
        
        # self.lossdict.update(
        #     {
        #         # m: SSSMalinblogsLoss(n, alpha, gamma, overlap + (1.-1./(n/m))*(1.-overlap), eps, window_fn=freq_hann_window,
        #         #                        mask_bin_to=(n//2+1) - n//2//n_div*(n_div-(i+1)),
        #         #                        mask_bin_from=(n//2+1) - n//2//n_div*(n_div-(i+1+1)),
        #         #                        wkwargs={'freq': n/m}).to(device)
        #         # m: LF4SMPMalLoss(n, alpha, gamma, overlap + (1.-1./(n/m))*(1.-overlap), eps, window_fn=freq_hann_window,
        #         #                        wkwargs={'freq': n/m}).to(device)
        #         m: SSSMPMalinblogsLoss(n, alpha, gamma, overlap + (1.-1./(n/self.fftnums[-1]))*(1.-overlap), eps, window_fn=freq_hann_window,
        #                         mask_bin_to=((n_fft-1)//2+1) - (n_fft-1)//2//n_div*(n_div-(i+1)),
        #                         mask_bin_from=((n_fft-1)//2+1) - (n_fft-1)//2//n_div*(n_div-(i+1+1)),
        #                         wkwargs={'freq': n/self.fftnums[-1]}).to(device)
        #         for i, n in enumerate(self.fftnums[:-1])
        #         for m in [v for v in range(self.fftnums[i+1], n)]
        #     }
        # )
        
        self.lossdict.update(
            {
                # m: SSSMalinblogsLoss(n, alpha, gamma, overlap + (1.-1./(n/m))*(1.-overlap), eps, window_fn=freq_hann_window,
                #                        mask_bin_to=(n//2+1) - n//2//n_div*(n_div-(i+1)),
                #                        mask_bin_from=(n//2+1) - n//2//n_div*(n_div-(i+1+1)),
                #                        wkwargs={'freq': n/m}).to(device)
                # m: LF4SMPMalLoss(n, alpha, gamma, overlap + (1.-1./(n/m))*(1.-overlap), eps, window_fn=freq_hann_window,
                #                        wkwargs={'freq': n/m}).to(device)
                # n: SSSMPMalLoss(n_fft, alpha, gamma, overlap + (1.-1./(n_fft/n))*(1.-overlap), eps, window_fn=freq_hann_window,
                #                 mask_bin_to=((n_fft-1)//2+1) - (n_fft-1)//2//n_div*(n_div-(i+1)),
                #                 mask_bin_from=((n_fft-1)//2+1) - (n_fft-1)//2//n_div*(n_div-(i+1+1)),
                #                 wkwargs={'freq': n_fft/n}).to(device)
                # for i, n in enumerate(self.fftnums[1:])
                # n: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap + (1.-1./(n_fft/n))*(1.-overlap), eps, window_fn=freq_hann_window,
                #                 mask_bin_to=((n_fft-1)//2+1) - (n_fft-1)//2//n_div*(n_div-(i+1)),
                #                 mask_bin_from=((n_fft-1)//2+1) - (n_fft-1)//2//n_div*(n_div-(i+1+1)),
                #                 wkwargs={'freq': n_fft/n}).to(device)
                # for i, n in enumerate(self.fftnums[1:])
                # n: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap + (1.-1./(n_fft/n))*(1.-overlap), eps, window_fn=freq_hann_window,
                #                 mask_bin_to=((n_fft-1)//2+1) - (n_fft-1)//2//n_div*(n_div-(i+1)),
                #                 mask_bin_from=((n_fft-1)//2+1) - (n_fft-1)//2//n_div*(n_div-(i+1+1)),
                #                 wkwargs={'freq': n_fft/n}).to(device)
                # n: SSSMPMalinblogsLoss(n, alpha, gamma, overlap, eps, window_fn=torch.hann_window,
                #                         mask_bin_to=((n-1)//2+1) - (n-1)//2//n_div*(n_div-(i+1)),
                #                         mask_bin_from=((n-1)//2+1) - (n-1)//2//n_div*(n_div-(i+1+1))).to(device)
                # for i, n in enumerate(self.fftnums[1:])
                n: SSSMPMalinblogsLoss(n_fft, alpha, gamma, overlap + (1.-1./(n_fft/n))*(1.-overlap), eps, window_fn=freq_hann_window,
                # n: LF4SMPMalinblogsLoss(n_fft, alpha, gamma, overlap + (1.-1./(n_fft/n))*(1.-overlap), eps, window_fn=freq_hann_window,
                                wkwargs={'freq': n_fft/n}).to(device)
                for i, n in enumerate(self.fftnums[1:])
            }
        )
        
        
        # self.lossdict.update(
        #     {
        #         m: SSSMPMalinblogsLoss(n, alpha, gamma, overlap + (1.-1./((n)/m))*(1.-overlap), eps, window_fn=freq_hann_window,
        #         # m-1: SSSMalinblogsLoss(n-1, alpha, gamma, overlap, eps, window_fn=torch.hann_window,
        #                                  mask_bin_to=((n)//2+1) - (n)//2//n_div*(n_div-(i)),
        #                                  mask_bin_from=((n)//2+1) - (n)//2//n_div*(n_div-(i+1)),
        #                                  wkwargs={'freq': n_fft/n}).to(device)
        #         for i, n in enumerate(self.fftnums[:-1])
        #         for m in [v for v in range(self.fftnums[i+1]+1, n)]
        #     }
        # )
        # self.fftnums.extend(
        #     [
        #         m for i, n in enumerate(self.fftnums[:-1])
        #         for m in [v for v in range(self.fftnums[i+1]+1, n)]
        #     ])
        
    def forward(self, x_pred, x_true):
        # n_fft_offset = int(torch.randint(0, 2, (1,))[0])
        
        # # value = self.lossdict[self.n_fft](x_true, x_pred)
        # # return (value + self.lossdict[self.fftnums[-1]](x_true, x_pred))*0.5
        
        # # x_pred_resample = torchaudio.functional.resample(x_pred, self.n_fft, self.fftnums[-1], lowpass_filter_width=32)
        
        # # values = 0.
        # # resample_rate = int(torch.randint(self.fftnums[1], self.fftnums[0], (1,))[0])
        # # x_pred_resample = torchaudio.functional.resample(x_pred, self.fftnums[0], resample_rate, lowpass_filter_width=self.fftnums[0]//4)
        # # x_true_resample = torchaudio.functional.resample(x_true, self.fftnums[0], resample_rate, lowpass_filter_width=self.fftnums[0]//4)
        # # values = self.lossdict[self.fftnums[0]-n_fft_offset](x_true_resample, x_pred_resample)
        # # resample_rate = int(torch.randint(self.fftnums[-1], self.fftnums[-2], (1,))[0])
        # # x_pred_resample = torchaudio.functional.resample(x_pred, self.fftnums[-1], resample_rate, lowpass_filter_width=self.fftnums[-1]//4)
        # # x_true_resample = torchaudio.functional.resample(x_true, self.fftnums[-1], resample_rate, lowpass_filter_width=self.fftnums[-1]//4)
        # # values = self.lossdict[self.fftnums[-1]-n_fft_offset](x_true_resample, x_pred_resample)
        # # values = self.lossdict[self.fftnums[0]-n_fft_offset](x_true, x_pred)
        # # fft_randidx = int(torch.randint(1, len(self.fftnums), (1,))[0])
        # values = self.lossdict[self.fftnums[-1]-n_fft_offset](x_true, x_pred)
        # fft_randidx = int(torch.randint(0, len(self.fftnums)-1, (1,))[0])
        # # resample_rate = int(torch.randint(self.fftnums[fft_randidx], self.fftnums[fft_randidx-1], (1,))[0])
        # # resample_rate = int(torch.randint(self.fftnums[fft_randidx+1], self.fftnums[fft_randidx], (1,))[0])
        # resample_rate = self.fftnums[fft_randidx+1] + int(self.fftnums[fft_randidx+1]/self.resamples_num*torch.randint(0, self.resamples_num, (1,))[0])
        # # x_pred_resample = torchaudio.functional.resample(x_pred, self.fftnums[fft_randidx], resample_rate, lowpass_filter_width=self.fftnums[fft_randidx]//4)
        # # x_true_resample = torchaudio.functional.resample(x_true, self.fftnums[fft_randidx], resample_rate, lowpass_filter_width=self.fftnums[fft_randidx]//4)
        # x_pred_resample = self.resamples[f"{self.fftnums[fft_randidx]}-{resample_rate}"](x_pred)
        # x_true_resample = self.resamples[f"{self.fftnums[fft_randidx]}-{resample_rate}"](x_true.float())
        # values += self.lossdict[self.fftnums[fft_randidx]-n_fft_offset](x_true_resample, x_pred_resample)
        # for i, n in enumerate(self.fftnums[:-1]):
        # for i, n in enumerate(self.fftnums[1:], start=1):
        #     values += self.lossdict[n-n_fft_offset](x_true, x_pred)
            # resample_rate = int(torch.randint(self.fftnums[i+1], n+1, (1,))[0])
            # resample_rate = int(torch.randint(n, self.fftnums[i-1]+1, (1,))[0])
            # x_pred_resample = torchaudio.functional.resample(x_pred, n, resample_rate, lowpass_filter_width=n//4)
            # x_true_resample = torchaudio.functional.resample(x_true, n, resample_rate, lowpass_filter_width=n//4)
            # values += self.lossdict[n-n_fft_offset](x_true_resample, x_pred_resample)
        # values += self.lossdict[self.fftnums_rand[n_fft_rand]](x_true, x_pred)
        # return values/(len(self.fftnums) + 1)
        # return values/(len(self.fftnums))
        # return values*0.5
        
        # n_fft_offset = int(torch.randint(0, 2, (1,))[0])
        
        # values = 0.
        # for n in self.fftnums:
        #     values += self.lossdict[n](x_true, x_pred)
        # return values/len(self.fftnums)
        
        # return self.lossdict[self.n_fft](x_true, x_pred)
        
        # return self.lossdict[int(torch.randint(self.fftnums[-1], self.fftnums[0]+1, (1,))[0])](x_true, x_pred)
        
        # fft_num_rand = int(torch.randint(0, len(self.fftnums), (1,))[0])
        # return self.lossdict[self.fftnums[fft_num_rand]](x_true, x_pred)
        # return self.lossdict[self.fftnums[0]](x_true, x_pred)
        # return self.lossdict[self.fftnums[-1]](x_true, x_pred)
        return self.lossdict[self.fftnums[2]](x_true, x_pred)
    
        # values = 0.
        # for i, n in enumerate(self.fftnums[:-1]):
        #     values += self.lossdict[int(torch.randint(self.fftnums[i+1], n+1, (1,))[0])](x_true, x_pred)
        # return values/(len(self.fftnums) - 1)
        
        
class MelLoss(nn.Module):
    '''
    Mel-Spectrogram Loss
    '''
    
    def __init__(self, n_fft, n_mels, alpha=1.0, beta=0.5, gamma=0.7, n_harm_width=8, overlap=0, sample_rate=44100, eps=1e-7, device='cuda'):
        super().__init__()
        self.beta = beta
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = int(self.n_fft/2)
        
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=n_mels, mel_scale='slaney',
            center=False, power=1, pad_mode="constant")
        
    def forward(self, x_pred, x_true):
        mel_pred = self.mel(F.pad(x_pred, (self.hop_length//2, self.hop_length//2), mode="constant"))[:, :, 1:-1]
        mel_true = self.mel(F.pad(x_true, (self.hop_length//2, self.hop_length//2), mode="constant"))[:, :, 1:-1]
        # Calculate loss
        mel_loss = F.l1_loss(mel_pred, mel_true)
        return mel_loss
    

## for discriminators
def feature_loss(fmap_r, fmap_g):
    loss = 0
    
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            # rl = rl.float().detach()
            rl = rl.float()
            gl = gl.float()
            # loss += F.huber_loss(gl, rl)
            loss += torch.mean(torch.abs(rl - gl))

    # return loss/len(fmap_r) * 2.
    return loss * 2.


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr_fun, dr_dir = dr[0].float(), dr[1].float()
        dg_fun, dg_dir = dg[0].float(), dg[1].float()
        # r_loss_fun = torch.mean(F.softplus(1 - dr_fun) ** 2.)
        # g_loss_fun = torch.mean(F.softplus(dg_fun) ** 2.)
        # r_loss_dir = torch.mean( F.softplus(1 - dr_dir) ** 2.)
        # g_loss_dir = torch.mean(-F.softplus(1 - dg_dir) ** 2.)
        # r_loss_fun = torch.mean((dr_fun - 1) ** 2.) + torch.mean((dg_fun + 1) ** 2.)
        # g_loss_fun = torch.mean((dg_fun) ** 2.)
        # r_loss_dir = torch.mean( F.softplus(1 - dr_dir) ** 2.)
        # g_loss_dir = torch.mean(-F.softplus(1 - dg_dir) ** 2.)
        # r_loss_fun = torch.mean((dr_fun - 1) ** 2.)
        # g_loss_fun = torch.mean((dg_fun) ** 2.)
        r_loss_fun = torch.mean((1 - dr_fun) ** 2.)
        g_loss_fun = torch.mean((dg_fun) ** 2.)
        r_loss_dir = torch.mean( F.softplus(1 - dr_dir) ** 2.)
        g_loss_dir = torch.mean(-F.softplus(1 - dg_dir) ** 2.)
        # r_loss_dir = torch.mean((dr_dir - 1) ** 2.)
        # g_loss_dir = torch.mean((dg_dir) ** 2.)
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
        # l = torch.mean(F.softplus(1 - dg) ** 2.)
        # l = torch.mean((dg - 1) ** 2.)
        l = torch.mean((1 - dg) ** 2.)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
