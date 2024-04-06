import os
import numpy as np
import yaml
import torch
import torch.nn.functional as F
import torchcrepe
import pyworld as pw
from torchaudio.transforms import Resample

from .common import MaskedAvgPool1d, MedianPool1d

CREPE_RESAMPLE_KERNEL = {}
F0_KERNEL = {}

class F0Extractor:
    def __init__(self, f0_extractor, sample_rate = 44100, hop_size = 512, f0_min = 65, f0_max = 800):
        self.f0_extractor = f0_extractor
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.f0_min = f0_min
        self.f0_max = f0_max
        
        if f0_extractor == 'crepe':
            key_str = str(sample_rate)
            if key_str not in CREPE_RESAMPLE_KERNEL:
                CREPE_RESAMPLE_KERNEL[key_str] = Resample(sample_rate, 16000, lowpass_filter_width = 128)
            self.resample_kernel = CREPE_RESAMPLE_KERNEL[key_str]
        if f0_extractor == 'rmvpe':
            if 'rmvpe' not in F0_KERNEL:
                from modules.encoders.rmvpe import RMVPE
                F0_KERNEL['rmvpe'] = RMVPE('models/pretrained/rmvpe/model.pt', hop_length=160)
            self.rmvpe = F0_KERNEL['rmvpe']
        if f0_extractor == 'fcpe':
            self.device_fcpe = 'cuda' if torch.cuda.is_available() else 'cpu'
            if 'fcpe' not in F0_KERNEL:
                from torchfcpe import spawn_bundled_infer_model
                F0_KERNEL['fcpe'] = spawn_bundled_infer_model(device=self.device_fcpe)
            self.fcpe = F0_KERNEL['fcpe']
                
    def extract(self, audio, uv_interp = False, device = None, silence_front = 0): # audio: 1d numpy array
        # extractor start time
        n_frames = int(len(audio) // self.hop_size) + 1
                
        start_frame = int(silence_front * self.sample_rate / self.hop_size)
        real_silence_front = start_frame * self.hop_size / self.sample_rate
        audio = audio[int(np.round(real_silence_front * self.sample_rate)) : ]
            
        # extract f0 using dio
        if self.f0_extractor == 'dio':
            _f0, t = pw.dio(
                audio.astype('double'), 
                self.sample_rate, 
                f0_floor = self.f0_min, 
                f0_ceil = self.f0_max, 
                channels_in_octave=2, 
                frame_period = (1000 * self.hop_size / self.sample_rate))
            f0 = pw.stonemask(audio.astype('double'), _f0, t, self.sample_rate)
            f0 = np.pad(f0.astype('float'), (start_frame, n_frames - len(f0) - start_frame))
        
        # extract f0 using harvest
        elif self.f0_extractor == 'harvest':
            f0, _ = pw.harvest(
                audio.astype('double'), 
                self.sample_rate, 
                f0_floor = self.f0_min, 
                f0_ceil = self.f0_max, 
                frame_period = (1000 * self.hop_size / self.sample_rate))
            f0 = np.pad(f0.astype('float'), (start_frame, n_frames - len(f0) - start_frame))
        
        # extract f0 using crepe        
        elif self.f0_extractor == 'crepe':
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            resample_kernel = self.resample_kernel.to(device)
            wav16k_torch = resample_kernel(torch.FloatTensor(audio).unsqueeze(0).to(device))
            
            f0, pd = torchcrepe.predict(wav16k_torch, 16000, 80, self.f0_min, self.f0_max, pad=True, model='full', batch_size=512, device=device, return_periodicity=True)
            pd = MedianPool1d(pd, 4)
            f0 = torchcrepe.threshold.At(0.05)(f0, pd)
            f0 = MaskedAvgPool1d(f0, 4)
            
            f0 = f0.squeeze(0).cpu().numpy()
            f0 = np.array([f0[int(min(int(np.round(n * self.hop_size / self.sample_rate / 0.005)), len(f0) - 1))] for n in range(n_frames - start_frame)])
            f0 = np.pad(f0, (start_frame, 0))
        
        # extract f0 using rmvpe
        elif self.f0_extractor == "rmvpe":
            f0 = self.rmvpe.infer_from_audio(audio, self.sample_rate, device=device, thred=0.03, use_viterbi=False)
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            origin_time = 0.01 * np.arange(len(f0))
            target_time = self.hop_size / self.sample_rate * np.arange(n_frames - start_frame)
            f0 = np.interp(target_time, origin_time, f0)
            uv = np.interp(target_time, origin_time, uv.astype(float)) > 0.5
            f0[uv] = 0
            f0 = np.pad(f0, (start_frame, 0))
        
        # extract f0 using fcpe
        elif self.f0_extractor == "fcpe":
            _audio = torch.from_numpy(audio).to(self.device_fcpe).unsqueeze(0)
            f0 = self.fcpe(_audio, sr=self.sample_rate, decoder_mode="local_argmax", threshold=0.006)
            f0 = f0.squeeze().cpu().numpy()
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            origin_time = 0.01 * np.arange(len(f0))
            target_time = self.hop_size / self.sample_rate * np.arange(n_frames - start_frame)
            f0 = np.interp(target_time, origin_time, f0)
            uv = np.interp(target_time, origin_time, uv.astype(float)) > 0.5
            f0[uv] = 0
            f0 = np.pad(f0, (start_frame, 0))
            
        else:
            raise ValueError(f" [x] Unknown f0 extractor: {self.f0_extractor}")
                    
        # interpolate the unvoiced f0 
        if uv_interp:
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            f0[f0 < self.f0_min] = self.f0_min
        return f0