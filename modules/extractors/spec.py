import torch
from torchaudio.transforms import Resample, Spectrogram, MelSpectrogram

class SpecExtractor:
    def __init__(self, n_fft, n_mels, hop_length, sample_rate, device = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        
        # self.spectrogram = Spectrogram(self.n_fft, win_length=self.hop_length, hop_length=self.hop_length, center=False, power=1).to(self.device)
        # self.spectrogram = Spectrogram(self.n_fft, win_length=self.hop_length, hop_length=self.hop_length, power=1).to(self.device)
        self.spectrogram = MelSpectrogram(sample_rate, self.n_fft, n_mels=n_mels, win_length=self.hop_length, hop_length=self.hop_length, center=False, power=1, mel_scale='slaney').to(self.device)
        # self.spectrogram = MelSpectrogram(sample_rate, self.n_fft, n_mels=n_mels, win_length=self.n_fft//2, hop_length=self.hop_length, center=False, power=1, mel_scale='slaney').to(self.device)
        
        self.resample_kernel = {}
        
    def log2freq_fbank(self, n_filters, sample_rate):
        # calculate log-frequency filter banks
        frequencies = torch.linspace(0, sample_rate/2, n_filters)
        
        
    def extract(self, audio, sample_rate=0):
        # resample
        if sample_rate == self.sample_rate or sample_rate == 0:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.sample_rate, lowpass_filter_width = 128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)
            
        # if audio_res.size(-1) % self.hop_length != 0:
        #     audio_res = torch.nn.functional.pad(audio_res, (0, self.hop_length - audio_res.size(-1) % self.hop_length))
        
        # spec = self.spectrogram(audio_res).transpose(1, 2)
        
        # # calc log-frequency spectrum from spec
        # spec = torch.log(spec + 1e-6)
        
        
        # # interpolate bins by log2
        # bins_l = torch.linspace(0, self.n_fft//2, self.n_fft//2+1)
        # bins_l_log2 = bins_l[1:].log2()
        # bins_l_log2 = torch.cat((torch.tensor([0]), bins_l_log2))
        # spec = torch.nn.functional.interpolate(spec, scale_factor=bins_l_log2.numel()//bins_l.numel())[:, :, :bins_l_log2.numel()]
            
        # return spec
            
        return self.spectrogram(audio_res).transpose(1, 2)
    