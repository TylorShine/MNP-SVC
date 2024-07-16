import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..nsf_hifigan.nvSTFT import STFT
from ..nsf_hifigan.models import load_model,load_config
from torchaudio.transforms import Resample
from .reflow import RectifiedFlow
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
    vocoder = Vocoder(args.model.vocoder.type, args.model.vocoder.ckpt, device=device)
    
    # load model
    if args.model.type == 'RectifiedFlow':
        model = Unit2Wav(
                    args.data.sampling_rate,
                    args.data.block_size,
                    args.model.win_length,
                    args.data.encoder_out_channels, 
                    args.model.n_spk,
                    args.model.use_pitch_aug,
                    vocoder.dimension,
                    args.model.n_layers,
                    args.model.n_chans)
    elif args.model.type == 'ReflowMinimumNoisedPhase':
        model = Unit2WavMinimumNoisedPhase(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            win_length=args.model.win_length,
            n_unit=args.data.encoder_out_channels,
            n_hidden_channels=args.model.units_hidden_channels,
            n_spk=args.model.n_spk,
            use_speaker_embed=args.model.use_speaker_embed,
            use_embed_conv=not args.model.no_use_embed_conv,
            spk_embed_channels=args.data.spk_embed_channels,
            f0_input_variance=args.model.f0_input_variance,
            f0_offset_size_downsamples=args.model.f0_offset_size_downsamples,
            noise_env_size_downsamples=args.model.noise_env_size_downsamples,
            harmonic_env_size_downsamples=args.model.harmonic_env_size_downsamples,
            use_harmonic_env=args.model.use_harmonic_env,
            use_noise_env=args.model.use_noise_env,
            use_add_noise_env=args.model.use_add_noise_env,
            noise_to_harmonic_phase=args.model.noise_to_harmonic_phase,
            add_noise=args.model.add_noise,
            use_phase_offset=args.model.use_phase_offset,
            use_f0_offset=args.model.use_f0_offset,
            use_short_filter=args.model.use_short_filter,
            use_noise_short_filter=args.model.use_noise_short_filter,
            use_pitch_aug=args.model.use_pitch_aug,
            noise_seed=args.model.noise_seed,
            out_dims=vocoder.dimension,
            n_layers=args.model.n_layers,
            n_chans=args.model.n_chans,
        )
        
    elif args.model.type == 'ReflowDirectMinimumNoisedPhaseHidden':
        model = Unit2WavMinimumNoisedPhaseHidden(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            win_length=args.model.win_length,
            n_unit=args.data.encoder_out_channels,
            n_hidden_channels=args.model.units_hidden_channels,
            n_spk=args.model.n_spk,
            use_speaker_embed=args.model.use_speaker_embed,
            use_embed_conv=not args.model.no_use_embed_conv,
            spk_embed_channels=args.data.spk_embed_channels,
            f0_input_variance=args.model.f0_input_variance,
            f0_offset_size_downsamples=args.model.f0_offset_size_downsamples,
            noise_env_size_downsamples=args.model.noise_env_size_downsamples,
            harmonic_env_size_downsamples=args.model.harmonic_env_size_downsamples,
            use_harmonic_env=args.model.use_harmonic_env,
            use_noise_env=args.model.use_noise_env,
            use_add_noise_env=args.model.use_add_noise_env,
            noise_to_harmonic_phase=args.model.noise_to_harmonic_phase,
            add_noise=args.model.add_noise,
            use_phase_offset=args.model.use_phase_offset,
            use_f0_offset=args.model.use_f0_offset,
            use_short_filter=args.model.use_short_filter,
            use_noise_short_filter=args.model.use_noise_short_filter,
            use_pitch_aug=args.model.use_pitch_aug,
            noise_seed=args.model.noise_seed,
            out_dims=vocoder.dimension,
            # out_dims=args.model.units_hidden_channels,
            n_layers=args.model.n_layers,
            n_chans=args.model.n_chans,
        )
                   
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
        
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    if args.model.use_speaker_embed:
        spk_info_path = os.path.join(os.path.split(model_path)[0], 'spk_info.npz')
        if os.path.isfile(spk_info_path):
            spk_info = np.load(spk_info_path, allow_pickle=True)
        else:
            print(' [Warning] spk_info.npz not found but model seems to setup with speaker embed')
            spk_info = None
    else:
        spk_info = None
    
    return model, vocoder, args, spk_info


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


class Unit2Wav(nn.Module):
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
        self.reflow_model = RectifiedFlow(NaiveV2Diff(mel_channels=out_dims, dim=n_chans, num_layers=n_layers, condition_dim=out_dims, use_mlp=False), out_dims=out_dims)

    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None, vocoder=None,
                gt_spec=None, infer=True, return_wav=False, infer_step=10, method='euler', t_start=0.0, use_tqdm=True):
        
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
            reflow_loss = self.reflow_model(ddsp_mel, gt_spec=gt_spec, t_start=t_start, infer=False)
            return ddsp_loss, reflow_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if t_start < 1.0:
                mel = self.reflow_model(ddsp_mel, gt_spec=ddsp_mel, infer=True, infer_step=infer_step, method=method, t_start=t_start, use_tqdm=use_tqdm)
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
        self.reflow_model = RectifiedFlow(NaiveV2Diff(mel_channels=out_dims, dim=n_chans, num_layers=n_layers, condition_dim=out_dims, use_mlp=False), out_dims=out_dims)

    def forward(self, units, f0, volume, spk_id=None, spk_mix=None, aug_shift=None, vocoder=None,
                gt_spec=None, gt_wav=None, infer=True, return_wav=False, infer_step=10, method='euler', t_start=0.0, loss_func=None, use_tqdm=True):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        ddsp_wav, f0 = self.ddsp_model(units, f0, volume, spk_id=spk_id, spk_mix=spk_mix, aug_shift=aug_shift, infer=infer)
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav)
        else:
            ddsp_mel = None
            
        if not infer:
            if loss_func is None:
                ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            else:
                ddsp_loss = loss_func(ddsp_wav, gt_wav)
            reflow_loss = self.reflow_model(ddsp_mel, gt_spec=gt_spec, t_start=t_start, infer=False)
            return ddsp_loss, reflow_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if t_start < 1.0:
                mel = self.reflow_model(ddsp_mel, gt_spec=ddsp_mel, infer=True, infer_step=infer_step, method=method, t_start=t_start, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0)
            else:
                return mel
            
            
class Unit2WavMinimumNoisedPhaseDirect(nn.Module):
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
        self.block_size = block_size
        self.out_dims = out_dims
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
        # self.reflow_model = RectifiedFlow(NaiveV2Diff(mel_channels=out_dims, dim=n_chans, num_layers=n_layers, condition_dim=out_dims, use_mlp=False), out_dims=out_dims, spec_min=-(1. - 0.7), spec_max=1. - 0.7)
        self.reflow_model = RectifiedFlow(NaiveV2Diff(mel_channels=out_dims, dim=n_chans, num_layers=n_layers, condition_dim=out_dims, use_mlp=False), out_dims=out_dims, spec_min=-0.7, spec_max=0.7)
        # self.reflow_model = RectifiedFlow(NaiveV2Diff(mel_channels=block_size, dim=n_chans, num_layers=n_layers, condition_dim=block_size, use_mlp=False), out_dims=block_size, spec_min=-1., spec_max=1.)
        self.window = None

    def forward(self, units, f0, volume, spk_id=None, spk_mix=None, aug_shift=None, vocoder=None,
                gt_spec=None, gt_wav=None, infer=True, return_wav=False, infer_step=10, method='euler', t_start=0.0, loss_func=None, use_tqdm=True):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        ddsp_wav = self.ddsp_model(units, f0, volume, spk_id=spk_id, spk_mix=spk_mix, aug_shift=aug_shift, infer=infer)
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav)
        else:
            ddsp_mel = None
            
        if self.window is None:
            self.window = torch.hann_window(self.out_dims*2, device=ddsp_wav.device)
        ddsp_stft = torch.stft(
            ddsp_wav,
            n_fft=self.out_dims*2,
            win_length=self.out_dims*2,
            hop_length=self.out_dims,
            window=self.window,
            return_complex=True)
        
        gt_stft = torch.stft(
            gt_wav,
            n_fft=self.out_dims*2,
            win_length=self.out_dims*2,
            hop_length=self.out_dims,
            window=self.window,
            return_complex=True)
        
        ddsp_stft_real = ddsp_stft.real.transpose(2, 1)[:, :, 1:]
        ddsp_stft_real_std_mean = torch.std_mean(ddsp_stft_real, dim=2, keepdim=True)
        # ddsp_stft_real = (ddsp_stft_real - ddsp_stft_real_std_mean[1])/torch.clamp(ddsp_stft_real_std_mean[0], min=1e-2)
        # ddsp_stft_real = (ddsp_stft_real - ddsp_stft_real_std_mean[1])/torch.clamp(ddsp_stft_real_std_mean[0], min=1e-2) + ddsp_stft_real_std_mean[1]
        ddsp_stft_real = (ddsp_stft_real)/torch.clamp(ddsp_stft_real_std_mean[0], min=1e-2)
            
        if not infer:
            if loss_func is None:
                ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            else:
                ddsp_loss = loss_func(ddsp_wav, gt_wav)
            gt_stft_real = gt_stft.real.transpose(2, 1)[:, :, 1:]
            gt_stft_real_std_mean = torch.std_mean(gt_stft_real, dim=2, keepdim=True)
            # gt_stft_real = (gt_stft_real - gt_stft_real_std_mean[1])/torch.clamp(gt_stft_real_std_mean[0], min=1e-2)
            # gt_stft_real = (gt_stft_real - gt_stft_real_std_mean[1])/torch.clamp(gt_stft_real_std_mean[0], min=1e-2) + gt_stft_real_std_mean[1]
            gt_stft_real = (gt_stft_real)/torch.clamp(gt_stft_real_std_mean[0], min=1e-2)
            # print(ddsp_stft_real.shape, gt_stft_real.shape)
            reflow_loss = self.reflow_model(
                # ddsp_wav.unsqueeze(-1),
                # gt_spec=gt_wav.unsqueeze(-1),
                # ddsp_wav.view(ddsp_wav.shape[0], -1, self.block_size),
                # gt_spec=gt_wav.view(gt_wav.shape[0], -1, self.block_size),
                # gt_spec=(gt_wav - ddsp_wav).view(gt_wav.shape[0], -1, self.block_size),
                ddsp_stft_real,
                gt_spec=gt_stft_real,
                t_start=t_start, infer=False)
            # print(ddsp_mel.shape, ddsp_wav.view(ddsp_wav.shape[0], -1, self.block_size).shape)
            return ddsp_loss, reflow_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if t_start < 1.0:
                wav = self.reflow_model(
                    # ddsp_wav.unsqueeze(-1),
                    # gt_spec=gt_wav.unsqueeze(-1),
                    # ddsp_wav.view(ddsp_wav.shape[0], -1, self.block_size),
                    # gt_spec=ddsp_wav.view(ddsp_wav.shape[0], -1, self.block_size),
                    ddsp_stft_real,
                    gt_spec=ddsp_stft_real,
                    infer=True, infer_step=infer_step, method=method, t_start=t_start, use_tqdm=use_tqdm)
                # ddsp_stft.real[:, 1:, :] = (wav*torch.clamp(ddsp_stft_real_std_mean[0], min=1e-2) + ddsp_stft_real_std_mean[1]).transpose(2, 1)
                # ddsp_stft.real[:, 1:, :] = ((wav - ddsp_stft_real_std_mean[1])*torch.clamp(ddsp_stft_real_std_mean[0], min=1e-2) + ddsp_stft_real_std_mean[1]).transpose(2, 1)
                ddsp_stft.real[:, 1:, :] = (wav*torch.clamp(ddsp_stft_real_std_mean[0], min=1e-2)).transpose(2, 1)
                # ddsp_stft.real[:, 1:, :] = wav.transpose(2, 1)
                wav = torch.istft(
                    ddsp_stft,
                    n_fft=self.out_dims*2,
                    win_length=self.out_dims*2,
                    hop_length=self.out_dims,
                    window=self.window,
                )
            else:
                wav = ddsp_wav
            return wav.flatten(1)
            # if return_wav:
            #     return vocoder.infer(mel, f0)
            # else:
            #     return mel
            
            
class Unit2WavMinimumNoisedPhaseHidden(nn.Module):
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
        self.block_size = block_size
        self.out_dims = out_dims
        self.n_hidden_channels = n_hidden_channels
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
        # self.reflow_model = RectifiedFlow(NaiveV2Diff(mel_channels=out_dims, dim=n_chans, num_layers=n_layers, condition_dim=out_dims, use_mlp=False), out_dims=out_dims, spec_min=-(1. - 0.7), spec_max=1. - 0.7)
        # self.reflow_model = RectifiedFlow(NaiveV2Diff(mel_channels=n_hidden_channels, dim=n_chans, num_layers=n_layers, condition_dim=n_hidden_channels, use_mlp=False), out_dims=n_hidden_channels, spec_min=-0.7, spec_max=0.7)
        self.reflow_model = RectifiedFlow(NaiveV2Diff(mel_channels=out_dims, dim=n_chans, num_layers=n_layers, condition_dim=out_dims, use_mlp=False), out_dims=out_dims, spec_min=-1.0, spec_max=1.0)
        # self.reflow_model = RectifiedFlow(NaiveV2Diff(mel_channels=block_size, dim=n_chans, num_layers=n_layers, condition_dim=block_size, use_mlp=False), out_dims=block_size, spec_min=-1., spec_max=1.)
        self.window = None

    def forward(self, units, f0, volume, spk_id=None, spk_mix=None, aug_shift=None, vocoder=None,
                gt_spec=None, gt_wav=None, infer=True, return_wav=False, infer_step=10, method='euler', t_start=0.0, loss_func=None, use_tqdm=True):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        ddsp_wav, ddsp_hidden = self.ddsp_model(units, f0, volume, spk_id=spk_id, spk_mix=spk_mix, aug_shift=aug_shift, infer=infer)
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav)
        else:
            ddsp_mel = None
            
        if self.window is None:
            # self.window = torch.hann_window(self.out_dms*2, device=ddsp_wav.device)
            self.window = torch.hann_window(self.n_hidden_channels*2, device=ddsp_wav.device)
        # ddsp_stft = torch.stft(
        #     ddsp_wav,
        #     n_fft=self.out_dims*2,
        #     win_length=self.out_dims*2,
        #     hop_length=self.out_dims,
        #     window=self.window,
        #     return_complex=True)
        
        # ddsp_stft = torch.stft(
        #     ddsp_wav,
        #     n_fft=self.n_hidden_channels*2,
        #     win_length=self.n_hidden_channels*2,
        #     hop_length=self.n_hidden_channels,
        #     window=self.window,
        #     return_complex=True)
        
        ddsp_mel_std_mean = torch.std_mean(ddsp_mel, dim=2, keepdim=True)
        ddsp_mel = (ddsp_mel)/torch.clamp(ddsp_mel_std_mean[0], min=1e-2)
        
        # gt_stft = torch.stft(
        #     gt_wav,
        #     n_fft=self.out_dims*2,
        #     win_length=self.out_dims*2,
        #     hop_length=self.out_dims,
        #     window=self.window,
        #     return_complex=True)
        
        # ddsp_stft_real = ddsp_stft.real.transpose(2, 1)[:, :-1, 1:]
        # ddsp_stft_real_std_mean = torch.std_mean(ddsp_stft_real, dim=2, keepdim=True)
        # # ddsp_stft_real = (ddsp_stft_real - ddsp_stft_real_std_mean[1])/torch.clamp(ddsp_stft_real_std_mean[0], min=1e-2)
        # # ddsp_stft_real = (ddsp_stft_real - ddsp_stft_real_std_mean[1])/torch.clamp(ddsp_stft_real_std_mean[0], min=1e-2) + ddsp_stft_real_std_mean[1]
        # ddsp_stft_real = (ddsp_stft_real)/torch.clamp(ddsp_stft_real_std_mean[0], min=1e-2)
        
        # ddsp_hidden = ddsp_hidden.repeat_interleave(2, dim=1)
            
        if not infer:
            if loss_func is None:
                ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            else:
                ddsp_loss = loss_func(ddsp_wav, gt_wav)
                
            # gt_stft = torch.stft(
            #     gt_wav,
            #     n_fft=self.out_dims*2,
            #     win_length=self.out_dims*2,
            #     hop_length=self.out_dims,
            #     window=self.window,
            #     return_complex=True)
            
            # gt_stft = torch.stft(
            #     gt_wav,
            #     n_fft=self.n_hidden_channels*2,
            #     win_length=self.n_hidden_channels*2,
            #     hop_length=self.n_hidden_channels,
            #     window=self.window,
            #     return_complex=True)
            
            # gt_stft_real = gt_stft.real.transpose(2, 1)[:, :-1, 1:]
            # gt_stft_real_std_mean = torch.std_mean(gt_stft_real, dim=2, keepdim=True)
            # # gt_stft_real = (gt_stft_real - gt_stft_real_std_mean[1])/torch.clamp(gt_stft_real_std_mean[0], min=1e-2)
            # # gt_stft_real = (gt_stft_real - gt_stft_real_std_mean[1])/torch.clamp(gt_stft_real_std_mean[0], min=1e-2) + gt_stft_real_std_mean[1]
            # gt_stft_real = (gt_stft_real)/torch.clamp(gt_stft_real_std_mean[0], min=1e-2)
            
            # print(ddsp_stft_real.shape, gt_stft_real.shape)
            # print(gt_stft_real.shape, ddsp_hidden.shape)
            reflow_loss = self.reflow_model(
                # ddsp_wav.unsqueeze(-1),
                # gt_spec=gt_wav.unsqueeze(-1),
                # ddsp_wav.view(ddsp_wav.shape[0], -1, self.block_size),
                # gt_spec=gt_wav.view(gt_wav.shape[0], -1, self.block_size),
                # gt_spec=(gt_wav - ddsp_wav).view(gt_wav.shape[0], -1, self.block_size),
                # ddsp_hidden,
                # gt_spec=gt_stft_real,
                ddsp_mel,
                gt_spec=gt_spec,
                t_start=t_start, infer=False)
                # t_start=0.0, infer=False)
            # print(ddsp_mel.shape, ddsp_wav.view(ddsp_wav.shape[0], -1, self.block_size).shape)
            return ddsp_loss, reflow_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if t_start < 1.0:
                mel = self.reflow_model(
                    # ddsp_wav.unsqueeze(-1),
                    # gt_spec=gt_wav.unsqueeze(-1),
                    # ddsp_wav.view(ddsp_wav.shape[0], -1, self.block_size),
                    # gt_spec=ddsp_wav.view(ddsp_wav.shape[0], -1, self.block_size),
                    # ddsp_stft_real,
                    # ddsp_hidden,
                    ddsp_mel,
                    # gt_spec=ddsp_stft_real,
                    # gt_spec=ddsp_stft_real,
                    gt_spec=ddsp_mel,
                    t_start=t_start,
                    # t_start=0.0,
                    infer=True, infer_step=infer_step, method=method, use_tqdm=use_tqdm)
                # ddsp_stft.real[:, 1:, :] = (wav*torch.clamp(ddsp_stft_real_std_mean[0], min=1e-2) + ddsp_stft_real_std_mean[1]).transpose(2, 1)
                # ddsp_stft.real[:, 1:, :] = ((wav - ddsp_stft_real_std_mean[1])*torch.clamp(ddsp_stft_real_std_mean[0], min=1e-2) + ddsp_stft_real_std_mean[1]).transpose(2, 1)
                # ddsp_stft.real[:, 1:, :-1] = (wav*torch.clamp(ddsp_stft_real_std_mean[0], min=1e-2)).transpose(2, 1)
                mel = (mel*torch.clamp(ddsp_mel_std_mean[0], min=1e-2))
                # ddsp_stft.real[:, 1:, :-1] = wav.transpose(2, 1)
                # wav = torch.istft(
                #     ddsp_stft,
                #     n_fft=self.out_dims*2,
                #     win_length=self.out_dims*2,
                #     hop_length=self.out_dims,
                #     window=self.window,
                # )
                # print(wav.shape, ddsp_stft.shape)
                # wav = torch.istft(
                #     ddsp_stft,
                #     n_fft=self.n_hidden_channels*2,
                #     win_length=self.n_hidden_channels*2,
                #     hop_length=self.n_hidden_channels,
                #     window=self.window,
                # )
            else:
                mel = ddsp_mel
            # return wav.flatten(1)
            if return_wav:
                return vocoder.infer(mel, f0).flatten(1)
            else:
                return mel