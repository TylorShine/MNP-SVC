import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..nsf_hifigan.nvSTFT import STFT
from ..nsf_hifigan.models import load_model,load_config
from torchaudio.transforms import Resample
from .diffusion import GaussianDiffusion
from .wavenet import WaveNet
from .naive_v2_diff import NaiveV2Diff
from ..vocoder import CombSubMinimumNoisedPhase

class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__


def load_model_vocoder(
        model_path,
        device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    # load vocoder
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
    
    # load model
    if args.model.type == 'Diffusion':
        model = Unit2Mel(
                    args.data.encoder_out_channels, 
                    args.model.n_spk,
                    args.model.use_pitch_aug,
                    vocoder.dimension,
                    args.model.n_layers,
                    args.model.n_chans,
                    args.model.n_hidden)
                    
    elif args.model.type == 'DiffusionNew':
        model = Unit2Wav(
                args.data.sampling_rate,
                args.data.block_size,
                args.data.encoder_out_channels, 
                args.model.n_spk,
                args.model.use_pitch_aug,
                vocoder.dimension,
                args.model.n_layers,
                args.model.n_chans,
                pcmer_norm=args.model.pcmer_norm)
    
    elif args.model.type == 'DiffusionFast':
        model = Unit2WavFast(
                args.data.sampling_rate,
                args.data.block_size,
                args.model.win_length,
                args.data.encoder_out_channels, 
                args.model.n_spk,
                args.model.use_pitch_aug,
                vocoder.dimension,
                args.model.n_layers,
                args.model.n_chans)
        
    elif args.model.type == 'DiffusionMinimumNoisedPhase':
        model = Unit2WavMinimumNoisedPhase(
                args.data.sampling_rate,
                args.data.block_size,
                args.model.win_length,
                args.data.encoder_out_channels, 
                args.model.n_spk,
                args.model.use_pitch_aug,
                vocoder.dimension,
                args.model.n_layers,
                args.model.n_chans)
                
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
        
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, vocoder, args
    
class Vocoder:
    def __init__(self, vocoder_type, vocoder_ckpt, device = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        if vocoder_type == 'nsf-hifigan':
            self.vocoder = NsfHifiGAN(vocoder_ckpt, device = device)
        elif vocoder_type == 'nsf-hifigan-log10':
            self.vocoder = NsfHifiGANLog10(vocoder_ckpt, device = device)
        else:
            raise ValueError(f" [x] Unknown vocoder: {vocoder_type}")
            
        self.resample_kernel = {}
        self.vocoder_sample_rate = self.vocoder.sample_rate()
        self.vocoder_hop_size = self.vocoder.hop_size()
        self.dimension = self.vocoder.dimension()
        
    def extract(self, audio, sample_rate=0, keyshift=0):
                
        # resample
        if sample_rate == self.vocoder_sample_rate or sample_rate == 0:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.vocoder_sample_rate, lowpass_filter_width = 128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)    
        
        # extract
        mel = self.vocoder.extract(audio_res, keyshift=keyshift) # B, n_frames, bins
        return mel
   
    def infer(self, mel, f0):
        f0 = f0[:,:mel.size(1),0] # B, n_frames
        audio = self.vocoder(mel, f0)
        return audio
        
        
class NsfHifiGAN(torch.nn.Module):
    def __init__(self, model_path, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model_path = model_path
        self.model = None
        self.h = load_config(model_path)
        self.stft = STFT(
                self.h.sampling_rate, 
                self.h.num_mels, 
                self.h.n_fft, 
                self.h.win_size, 
                self.h.hop_size, 
                self.h.fmin, 
                self.h.fmax)
    
    def sample_rate(self):
        return self.h.sampling_rate
        
    def hop_size(self):
        return self.h.hop_size
    
    def dimension(self):
        return self.h.num_mels
        
    def extract(self, audio, keyshift=0):       
        mel = self.stft.get_mel(audio, keyshift=keyshift).transpose(1, 2) # B, n_frames, bins
        return mel
    
    def forward(self, mel, f0):
        if self.model is None:
            print('| Load HifiGAN: ', self.model_path)
            self.model, self.h = load_model(self.model_path, device=self.device)
        with torch.no_grad():
            c = mel.transpose(1, 2)
            audio = self.model(c, f0)
            return audio


class NsfHifiGANLog10(NsfHifiGAN):    
    def forward(self, mel, f0):
        if self.model is None:
            print('| Load HifiGAN: ', self.model_path)
            self.model, self.h = load_model(self.model_path, device=self.device)
        with torch.no_grad():
            c = 0.434294 * mel.transpose(1, 2)
            audio = self.model(c, f0)
            return audio


class Unit2Mel(nn.Module):
    def __init__(
            self,
            input_channel,
            n_spk,
            use_pitch_aug=False,
            out_dims=128,
            n_layers=20, 
            n_chans=384, 
            n_hidden=256):
        super().__init__()
        self.unit_embed = nn.Linear(input_channel, n_hidden)
        self.f0_embed = nn.Linear(1, n_hidden)
        self.volume_embed = nn.Linear(1, n_hidden)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        else:
            self.aug_shift_embed = None
        self.n_spk = n_spk
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, n_hidden)
            
        # diffusion
        self.decoder = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden), out_dims=out_dims)

    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None, vocoder=None,
                gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=300, use_tqdm=True):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        x = self.unit_embed(units) + self.f0_embed((1+ f0 / 700).log()) + self.volume_embed(volume)
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
                    x = x + v * self.spk_embed(spk_id_torch - 1)
            else:
                x = x + self.spk_embed(spk_id - 1)
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5) 
        x = self.decoder(x, gt_spec=gt_spec, infer=infer, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
    
        return x

        
class Unit2Wav(nn.Module):
    def __init__(
            self,
            sampling_rate,
            block_size,
            n_unit,
            n_spk,
            use_pitch_aug=False,
            out_dims=128,
            n_layers=20, 
            n_chans=512,
            pcmer_norm=False):
        super().__init__()
        self.ddsp_model = CombSubFast(sampling_rate, block_size, n_unit, n_spk, use_pitch_aug, pcmer_norm=pcmer_norm)
        self.diff_model = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, 256), out_dims=out_dims)

    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None, vocoder=None,
                gt_spec=None, infer=True, return_wav=False, infer_speedup=10, method='dpm-solver', k_step=None, use_tqdm=True):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        ddsp_wav, hidden, (_, _) = self.ddsp_model(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift, infer=infer)
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav)
        else:
            ddsp_mel = None
            
        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            diff_loss = self.diff_model(hidden, gt_spec=gt_spec, k_step=k_step, infer=False)
            return ddsp_loss, diff_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if k_step > 0:
                mel = self.diff_model(hidden, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0)
            else:
                return mel
                

class Unit2WavFast(nn.Module):
    def __init__(
            self,
            sampling_rate,
            block_size,
            win_length,
            n_unit,
            n_spk,
            use_pitch_aug=False,
            out_dims=128,
            n_layers=6, 
            n_chans=512):
        super().__init__()
        self.ddsp_model = CombSubSuperFast(sampling_rate, block_size, win_length, n_unit, n_spk, use_pitch_aug)
        self.diff_model = GaussianDiffusion(NaiveV2Diff(mel_channels=out_dims, dim=n_chans, num_layers=n_layers, condition_dim=out_dims, use_mlp=False), out_dims=out_dims)

    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None, vocoder=None,
                gt_spec=None, infer=True, return_wav=False, infer_speedup=10, method='dpm-solver', k_step=None, use_tqdm=True):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        ddsp_wav, hidden, (_, _) = self.ddsp_model(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift, infer=infer)
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav)
        else:
            ddsp_mel = None
            
        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            diff_loss = self.diff_model(ddsp_mel, gt_spec=gt_spec, k_step=k_step, infer=False)
            return ddsp_loss, diff_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if k_step > 0:
                mel = self.diff_model(ddsp_mel, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0)
            else:
                return mel
            
            
            
class Unit2WavMinimumNoisedPhase(nn.Module):
    def __init__(
            self,
            sampling_rate,
            block_size,
            win_length,
            n_unit,
            n_spk,
            use_pitch_aug=False,
            out_dims=128,
            n_layers=6, 
            n_chans=512,
            n_hidden_channels=256,
            use_speaker_embed=True,
            use_embed_conv=True,
            spk_embed_channels=256,
            f0_input_variance=0.1,
            f0_offset_size_downsamples=1,
            noise_env_size_downsamples=4,
            harmonic_env_size_downsamples=4,
            use_harmonic_env=False,
            use_noise_env=True,
            use_add_noise_env=True,
            noise_to_harmonic_phase=False,
            add_noise=False,
            use_phase_offset=False,
            use_f0_offset=False,
            use_short_filter=False,
            use_noise_short_filter=False,
            noise_seed=289,
            onnx_unit2ctrl=None,
            export_onnx=False,
            device=None):
        super().__init__()
        # sampling_rate=args.data.sampling_rate,
        #     block_size=args.data.block_size,
        #     win_length=args.model.win_length,
        #     n_unit=args.data.encoder_out_channels,
        #     n_hidden_channels=args.model.units_hidden_channels,
        #     n_spk=args.model.n_spk,
        #     use_speaker_embed=args.model.use_speaker_embed,
        #     use_embed_conv=not args.model.no_use_embed_conv,
        #     spk_embed_channels=args.data.spk_embed_channels,
        #     f0_input_variance=args.model.f0_input_variance,
        #     f0_offset_size_downsamples=args.model.f0_offset_size_downsamples,
        #     noise_env_size_downsamples=args.model.noise_env_size_downsamples,
        #     harmonic_env_size_downsamples=args.model.harmonic_env_size_downsamples,
        #     use_harmonic_env=args.model.use_harmonic_env,
        #     use_noise_env=args.model.use_noise_env,
        #     use_add_noise_env=args.model.use_add_noise_env,
        #     noise_to_harmonic_phase=args.model.noise_to_harmonic_phase,
        #     add_noise=args.model.add_noise,
        #     use_phase_offset=args.model.use_phase_offset,
        #     use_f0_offset=args.model.use_f0_offset,
        #     use_short_filter=args.model.use_short_filter,
        #     use_noise_short_filter=args.model.use_noise_short_filter,
        #     use_pitch_aug=args.model.use_pitch_aug,
        #     noise_seed=args.model.noise_seed,
        self.ddsp_model = CombSubMinimumNoisedPhase(
            sampling_rate, block_size, win_length, n_unit, n_hidden_channels, n_spk,
            use_speaker_embed, use_embed_conv, spk_embed_channels,
            f0_input_variance, f0_offset_size_downsamples, noise_env_size_downsamples, harmonic_env_size_downsamples,
            use_harmonic_env,
            use_noise_env,
            use_add_noise_env,
            noise_to_harmonic_phase,
            add_noise,
            use_phase_offset,
            use_f0_offset,
            use_short_filter,
            use_noise_short_filter,
            use_pitch_aug,
            noise_seed,
            onnx_unit2ctrl=None,
            export_onnx=False,
            device=device)
        self.diff_model = GaussianDiffusion(NaiveV2Diff(mel_channels=out_dims, dim=n_chans, num_layers=n_layers, condition_dim=out_dims, use_mlp=False), out_dims=out_dims)

    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None, vocoder=None,
                gt_spec=None, infer=True, return_wav=False, infer_speedup=10, method='dpm-solver', k_step=None, use_tqdm=True):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        ddsp_wav = self.ddsp_model(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift, infer=infer)
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav)
        else:
            ddsp_mel = None
            
        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            diff_loss = self.diff_model(ddsp_mel, gt_spec=gt_spec, k_step=k_step, infer=False)
            return ddsp_loss, diff_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if k_step > 0:
                mel = self.diff_model(ddsp_mel, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0)
            else:
                return mel