import os

import onnxruntime
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from modules.common import DotDict


def load_model(
        model_path,
        device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    # load model
    model = None
    if args.model.type == 'CombSubMinimumNoisedPhase':
        model = CombSubMinimumNoisedPhase(
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
            noise_to_harmonic_phase=args.model.noise_to_harmonic_phase,
            add_noise=args.model.add_noise,
            use_f0_offset=args.model.use_f0_offset,
            use_pitch_aug=args.model.use_pitch_aug,
            noise_seed=args.model.noise_seed,
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
    
    return model, args, spk_info


def load_onnx_model(
            model_path,
            providers=['CPUExecutionProvider'],
            device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    sess = onnxruntime.InferenceSession(
        model_path,
        providers=providers)
    
    # load model
    model = None
    if args.model.type == 'CombSubMinimumNoisedPhase':
        model = CombSubMinimumNoisedPhase(
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
            noise_to_harmonic_phase=args.model.noise_to_harmonic_phase,
            use_f0_offset=args.model.use_f0_offset,
            use_pitch_aug=args.model.use_pitch_aug,
            noise_seed=args.model.noise_seed,
            onnx_unit2ctrl=sess,
            device=device)
            
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
    
    if args.model.use_speaker_embed:
        spk_info_path = os.path.join(os.path.split(model_path)[0], 'spk_info.npz')
        if os.path.isfile(spk_info_path):
            spk_info = np.load(spk_info_path, allow_pickle=True)
        else:
            print(' [Warning] spk_info.npz not found but model seems to setup with speaker embed')
            spk_info = None
    else:
        spk_info = None
    
    return model, args, spk_info


class CombSubMinimumNoisedPhaseStackOnly(torch.nn.Module):
    def __init__(self, 
            n_unit=256,
            n_hidden_channels=256):
        super().__init__()

        print(' [DDSP Model] Minimum-Phase harmonic Source Combtooth Subtractive Synthesiser (u2c stack only)')
        
        from .unit2control import Unit2ControlStackOnly
        self.unit2ctrl = Unit2ControlStackOnly(n_unit,
                                                n_hidden_channels=n_hidden_channels,
                                                conv_stack_middle_size=32)
        
    def forward(self, units_frames, **kwargs):
        '''
            units_frames: B x n_frames x n_unit
            f0_frames: B x n_frames x 1
            volume_frames: B x n_frames x 1 
            spk_id: B x 1
        '''
        
        return self.unit2ctrl(units_frames)


class CombSubMinimumNoisedPhase(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            win_length,
            n_unit=256,
            n_hidden_channels=256,
            n_spk=1,
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
            use_pitch_aug=False,
            noise_seed=289,
            onnx_unit2ctrl=None,
            export_onnx=False,
            device=None):
        super().__init__()

        print(' [DDSP Model] Minimum-Phase harmonic Source Combtooth Subtractive Synthesiser')
        
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("win_length", torch.tensor(win_length))
        self.register_buffer("window", torch.hann_window(win_length))
        self.register_buffer("f0_input_variance", torch.tensor(f0_input_variance))
        self.register_buffer("f0_offset_size_downsamples", torch.tensor(f0_offset_size_downsamples))
        self.register_buffer("noise_env_size_downsamples", torch.tensor(noise_env_size_downsamples))
        self.register_buffer("harmonic_env_size_downsamples", torch.tensor(harmonic_env_size_downsamples))
        self.register_buffer("use_harmonic_env", torch.tensor(use_harmonic_env))
        self.register_buffer("use_noise_env", torch.tensor(use_noise_env))
        self.register_buffer("use_add_noise_env", torch.tensor(use_add_noise_env))
        self.register_buffer("noise_to_harmonic_phase", torch.tensor(noise_to_harmonic_phase))
        # self.register_buffer("add_noise", torch.tensor(add_noise))
        self.register_buffer("use_f0_offset", torch.tensor(use_f0_offset))
        self.register_buffer("use_speaker_embed", torch.tensor(use_speaker_embed))
        self.register_buffer("use_embed_conv", torch.tensor(use_embed_conv))
        self.register_buffer("noise_seed", torch.tensor(noise_seed))
        
        if add_noise is None:
            self.add_noise = add_noise
        else:
            self.register_buffer("add_noise", torch.tensor(add_noise))
        
        if use_phase_offset is None:
            self.use_phase_offset = use_phase_offset
        else:
            self.register_buffer("use_phase_offset", torch.tensor(use_phase_offset))

        self.pred_filter_size = win_length // 2 + 1
        
        
        #Unit2Control
        split_map = {
            'harmonic_magnitude': self.pred_filter_size,
            'harmonic_phase': self.pred_filter_size,
            'noise_magnitude': self.pred_filter_size,
            'noise_phase': self.pred_filter_size,
        }
        if use_noise_env:
            split_map['noise_envelope_magnitude'] = block_size//noise_env_size_downsamples
        if use_harmonic_env:
            split_map['harmonic_envelope_magnitude'] = block_size//harmonic_env_size_downsamples
        if use_f0_offset:
            split_map['f0_offset'] = block_size//f0_offset_size_downsamples
        if add_noise:
            split_map['add_noise_magnitude'] = self.pred_filter_size
            split_map['add_noise_phase'] = self.pred_filter_size
            if use_add_noise_env:
                split_map['add_noise_envelope_magnitude'] = block_size//noise_env_size_downsamples
        if use_phase_offset:
            split_map['phase_offset'] = self.pred_filter_size
            
        
        if onnx_unit2ctrl is not None:
            from .unit2control import Unit2ControlGE2E_onnx
            self.unit2ctrl = Unit2ControlGE2E_onnx(onnx_unit2ctrl, split_map)
        elif export_onnx:
            from .unit2control import Unit2ControlGE2E_export
            self.unit2ctrl = Unit2ControlGE2E_export(n_unit, spk_embed_channels, split_map,
                                            n_hidden_channels=n_hidden_channels,
                                            n_spk=n_spk,
                                            use_pitch_aug=use_pitch_aug,
                                            use_spk_embed=use_speaker_embed,
                                            use_embed_conv=use_embed_conv,
                                            embed_conv_channels=64, conv_stack_middle_size=32)
        else:
            from .unit2control import Unit2ControlGE2E
            self.unit2ctrl = Unit2ControlGE2E(n_unit, spk_embed_channels, split_map,
                                            n_hidden_channels=n_hidden_channels,
                                            n_spk=n_spk,
                                            use_pitch_aug=use_pitch_aug,
                                            use_spk_embed=use_speaker_embed,
                                            use_embed_conv=use_embed_conv,
                                            embed_conv_channels=64, conv_stack_middle_size=32)
        
        # generate static noise
        self.gen = torch.Generator()
        self.gen.manual_seed(noise_seed)
        static_noise_t = (torch.rand([
            win_length*127  # about 5.9sec when sampling_rate=44100 and win_length=2048
        ], generator=self.gen)*2.0-1.0)
        # if self.noise_to_harmonic_phase:
        #     dump = torch.zeros_like(static_noise_t)
        #     dump_samples = int(sampling_rate / 300.0)    # samples of 300Hz
        #     dump[:dump_samples] = torch.linspace(1.0, 0.0, dump_samples) ** 2.
        #     static_noise_t *= dump
        self.register_buffer('static_noise_t', static_noise_t)

        if add_noise:
            static_add_noise_t = (torch.rand([
            win_length*127  # about 5.9sec when sampling_rate=44100 and win_length=2048
            ], generator=self.gen)*2.0-1.0)
            self.register_buffer('static_add_noise_t', static_add_noise_t)
        
        # generate minimum-phase windowed sinc signal for harmonic source
        ## TODO: functional
        minphase_wsinc_w = 0.5
        phase = torch.linspace(-(win_length-1)*minphase_wsinc_w, win_length*minphase_wsinc_w, win_length)
        windowed_sinc = torch.sinc(
                phase
        ) * torch.blackman_window(win_length)
        log_freq_windowed_sinc = torch.log(torch.fft.fft(
            F.pad(windowed_sinc, (win_length//2, win_length//2)),
            n = win_length*2,
        ) + 1e-7)
        ceps_windowed_sinc = torch.fft.ifft(torch.cat([
            log_freq_windowed_sinc[:log_freq_windowed_sinc.shape[0]//2+1],
            torch.flip(log_freq_windowed_sinc[1:log_freq_windowed_sinc.shape[0]//2], (0,))
        ]))
        ceps_windowed_sinc.imag[0] *= -1.
        ceps_windowed_sinc.real[1:ceps_windowed_sinc.shape[0]//2] *= 2.
        ceps_windowed_sinc.imag[1:ceps_windowed_sinc.shape[0]//2] *= -2.
        ceps_windowed_sinc.imag[ceps_windowed_sinc.shape[0]//2] *= -1.
        ceps_windowed_sinc[ceps_windowed_sinc.shape[0]//2+1:] = 0.
        
        static_freq_minphase_wsinc = torch.fft.rfft( torch.fft.ifft(
            torch.exp(torch.fft.fft(ceps_windowed_sinc))
            ).real.roll(ceps_windowed_sinc.shape[0]//2-1)[:ceps_windowed_sinc.shape[0]//2]
                * torch.hann_window(win_length*2)[win_length:],
        )
        self.register_buffer('static_freq_minphase_wsinc', static_freq_minphase_wsinc)
        
        ## TODO: necessary?
        minphase_wsinc_w_env = 0.5
        windowed_sinc = torch.sinc(
            torch.linspace(-(block_size//2-1)*minphase_wsinc_w_env, block_size//2*minphase_wsinc_w_env, block_size//2)
        ) * torch.blackman_window(block_size//2)
            
        log_freq_windowed_sinc = torch.log(torch.fft.fft(
            F.pad(windowed_sinc, (block_size//4, block_size//4)),
            n = block_size,
        ) + 1e-7)
        ceps_windowed_sinc = torch.fft.ifft(torch.cat([
            log_freq_windowed_sinc[:log_freq_windowed_sinc.shape[0]//2+1],
            torch.flip(log_freq_windowed_sinc[1:log_freq_windowed_sinc.shape[0]//2], (0,))
        ]))
        ceps_windowed_sinc.imag[0] *= -1.
        ceps_windowed_sinc.real[1:ceps_windowed_sinc.shape[0]//2] *= 2.
        ceps_windowed_sinc.imag[1:ceps_windowed_sinc.shape[0]//2] *= -2.
        ceps_windowed_sinc.imag[ceps_windowed_sinc.shape[0]//2] *= -1.
        ceps_windowed_sinc[ceps_windowed_sinc.shape[0]//2+1:] = 0.
        static_freq_minphase_wsinc_env = torch.fft.rfft( F.pad(torch.fft.ifft(
            torch.exp(torch.fft.fft(ceps_windowed_sinc))
            ).real.roll(ceps_windowed_sinc.shape[0]//2-1)[:ceps_windowed_sinc.shape[0]//2],
            (0, win_length - (block_size//2))
        ) )
        
        self.register_buffer('static_freq_minphase_wsinc_env', static_freq_minphase_wsinc_env)
        

    def forward(self, units_frames, f0_frames, volume_frames,
                spk_id=None, spk_mix=None, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        '''
            units_frames: B x n_frames x n_unit
            f0_frames: B x n_frames x 1
            volume_frames: B x n_frames x 1 
            spk_id: B x 1
        '''
        
        # f0 = upsample(f0_frames, self.block_size)
        # we just enough simply repeat to that because block size is short enough, it is sufficient I think.
        f0 = f0_frames.repeat(1,1,self.block_size).flatten(1).unsqueeze(-1)
        # add f0 variance
        # this expected suppress leakage of original speaker's f0 features
        # but get blurred pitch and I don't think that necessary.
        f0_variance = torch.rand_like(f0)*self.f0_input_variance
        f0 = (f0 * 2.**(-self.f0_input_variance/12.))*(2.**(f0_variance/12.))   # semitone units
        if infer:
            # TODO: maybe this is for precision, but necessary?
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / torch.pi
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2. * torch.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        ctrls, hidden = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames,
                                       spk_id=spk_id, spk_mix=spk_mix, aug_shift=aug_shift)
        
        if self.use_f0_offset:
            # apply predicted f0 offset
            # NOTE: use f0 offset can be generalize inputs and fit target more, but pitch tracking is get blurry (oscillated) and feeling softer depending on the target
            f0 += (((ctrls['f0_offset'] - 2.)*0.5)).unsqueeze(-1).repeat(1,1,1,self.f0_offset_size_downsamples).flatten(2).reshape(f0.shape[0], f0.shape[1], 1)
            
            if infer:
                x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
            else:
                x = torch.cumsum(f0 / self.sampling_rate, axis=1)
            if initial_phase is not None:
                x += initial_phase.to(x) / 2 / torch.pi
            x = x - torch.round(x)
            x = x.to(f0)
        
        
        # exciter phase
        
        # filters
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * torch.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1).permute(0, 2, 1)
        
        noise_filter = torch.exp(ctrls['noise_magnitude'] + 1.j * torch.pi * ctrls['noise_phase'])/self.block_size
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1).permute(0, 2, 1)

        if self.add_noise:
            add_noise_filter = torch.exp(ctrls['add_noise_magnitude'] + 1.j * torch.pi * ctrls['add_noise_phase'])/self.block_size
            add_noise_filter = torch.cat((add_noise_filter, add_noise_filter[:,-1:,:]), 1).permute(0, 2, 1)
        
        
        # Dirac delta
        # note that its not accurate at boundary but fast and almost enough
        combtooth = torch.where(x.roll(1) - x < 0., 1., 0.)
        combtooth = combtooth.squeeze(-1)
        
        if self.use_harmonic_env:
            # TODO: scale by log10 now, but necessary?
            harmonic_env_flat = (torch.log10(torch.clamp(ctrls['harmonic_envelope_magnitude'], min=0.0)*9. + 1.) + 1.).flatten(1)
            
            # lazy, not accurate, but fast linear interpolation
            # repeat a last sample
            harmonic_env_exp = torch.cat((harmonic_env_flat, harmonic_env_flat[:, -1:]), 1)
            # calc slopes (differentials) sample by sample, then repeat
            harmonic_env_slopes = (harmonic_env_exp[:, 1:] - harmonic_env_exp[:, :-1]).unsqueeze(-1).repeat(1, 1, self.harmonic_env_size_downsamples).flatten(1)
            # repeat indice, this is lerp coefficients
            harmonic_env_repeat_idx = (torch.arange(self.harmonic_env_size_downsamples).to(harmonic_env_flat)/self.harmonic_env_size_downsamples).unsqueeze(-1).transpose(1, 0).repeat(harmonic_env_flat.shape)
            # repeat original values, then interpolate
            harmonic_env = harmonic_env_flat.unsqueeze(-1).repeat(1, 1, self.harmonic_env_size_downsamples).flatten(1) + harmonic_env_slopes*harmonic_env_repeat_idx/self.harmonic_env_size_downsamples
            
            # apply it to source
            combtooth *= harmonic_env

        pad_mode = 'constant'
        
        
        # noise exciter
        noise_t = self.static_noise_t.unsqueeze(0).repeat(combtooth.shape[0], combtooth.shape[1]//self.static_noise_t.shape[0] + 1)[:, :combtooth.shape[1]]
        # if self.noise_to_harmonic_phase:
        #     # f0_mean = torch.mean(f0, dim=1)
        #     dump = torch.zeros_like(noise_t)
        #     # print(f0.shape, ctrls['noise_scale'].shape)
        #     dump_samples = self.sampling_rate / f0[:, ::self.block_size, :] * torch.clamp(ctrls['noise_scale'], min=0.0, max=1.0)
        #     # dump_samples = self.sampling_rate / f0[:, ::self.block_size, :] * 0.707
        #     # dump_samples = self.sampling_rate / f0_mean * 0.707
        #     dump_samples = torch.mean(dump_samples, dim=1)
        #     for b in range(dump_samples.shape[0]):
        #         dump[b, :int(dump_samples[b])] = torch.linspace(1.0, 0.0, int(dump_samples[b])) ** 2.
        #         dump[b, 0] = 1.
        #     noise_t *= dump
        if self.use_noise_env:
            if not self.noise_to_harmonic_phase:
                # TODO: log10 necessary?
                noise_env_flat = (torch.log10(torch.clamp(ctrls['noise_envelope_magnitude'], min=0.0)*9. + 1.)).flatten(1)
            else:
                noise_env_flat = (torch.clamp(ctrls['noise_envelope_magnitude'], min=0.0)).flatten(1)
            noise_env_exp = torch.cat((noise_env_flat, noise_env_flat[:, -1:]), 1)
            noise_env_slopes = (noise_env_exp[:, 1:] - noise_env_exp[:, :-1]).unsqueeze(-1).repeat(1, 1, self.noise_env_size_downsamples).flatten(1)
            noise_env_repeat_idx = (torch.arange(self.noise_env_size_downsamples).to(noise_env_flat)/self.noise_env_size_downsamples).unsqueeze(-1).transpose(1, 0).repeat(noise_env_flat.shape)
            noise_env = noise_env_flat.unsqueeze(-1).repeat(1, 1, self.noise_env_size_downsamples).flatten(1) + noise_env_slopes*noise_env_repeat_idx/self.noise_env_size_downsamples
            noise_t *= noise_env
        
        
        combtooth_stft = torch.stft(
                            combtooth,
                            n_fft = self.win_length,
                            win_length = self.win_length,
                            hop_length = self.block_size,
                            window = self.window,
                            center = True,
                            return_complex = True,
                            pad_mode = pad_mode) * self.static_freq_minphase_wsinc.unsqueeze(-1).repeat(1, combtooth.shape[1]//self.block_size + 1)
        # and apply predicted filter
        combtooth_stft = combtooth_stft * src_filter
        
            
        # TODO: can precalculate if noise env is not using
        noise_stft = torch.stft(
            noise_t,
            n_fft = self.win_length,
            win_length = self.win_length,
            hop_length = self.block_size,
            window = self.window,
            center = True,
            return_complex = True,
            pad_mode = pad_mode)
            
        if self.use_noise_env:
            noise_stft *= self.static_freq_minphase_wsinc_env.unsqueeze(-1).repeat(1, combtooth_stft.shape[2])    # TODO: should this and necessary?
            # noise_stft *= self.static_freq_minphase_wsinc.unsqueeze(-1).repeat(1, combtooth_stft.shape[2])
            
        # apply predicted filter
        noise_stft *= noise_filter
            
        if self.use_phase_offset:
            signal_stft = combtooth_stft
            phase_offset = torch.clamp(torch.cat((ctrls['phase_offset'], ctrls['phase_offset'][:,-1:,:]), 1), min=-torch.pi*0.5, max=torch.pi*0.5).permute(0, 2, 1)
            signal_stft.imag += phase_offset
            # signal_stft += noise_stft
        elif self.noise_to_harmonic_phase:
            signal_stft = combtooth_stft
            # we apply real part of noise to just imaginary part.
            # it expected that learning to amount of per-frequency phase modulation by noise
            # TODO: could be simplify more?
            signal_stft.imag += noise_stft.real
            if self.add_noise:
                add_noise_t = self.static_add_noise_t.unsqueeze(0).repeat(combtooth.shape[0], combtooth.shape[1]//self.static_add_noise_t.shape[0] + 1)[:, :combtooth.shape[1]]
                if self.use_add_noise_env:
                    noise_env_flat = (torch.log10(torch.clamp(ctrls['add_noise_envelope_magnitude'], min=0.0)*9. + 1.)).flatten(1)
                    noise_env_exp = torch.cat((noise_env_flat, noise_env_flat[:, -1:]), 1)
                    noise_env_slopes = (noise_env_exp[:, 1:] - noise_env_exp[:, :-1]).unsqueeze(-1).repeat(1, 1, self.noise_env_size_downsamples).flatten(1)
                    noise_env_repeat_idx = (torch.arange(self.noise_env_size_downsamples).to(noise_env_flat)/self.noise_env_size_downsamples).unsqueeze(-1).transpose(1, 0).repeat(noise_env_flat.shape)
                    noise_env = noise_env_flat.unsqueeze(-1).repeat(1, 1, self.noise_env_size_downsamples).flatten(1) + noise_env_slopes*noise_env_repeat_idx/self.noise_env_size_downsamples
                    add_noise_t *= noise_env
                add_noise_stft = torch.stft(
                    add_noise_t,
                    n_fft = self.win_length,
                    win_length = self.win_length,
                    hop_length = self.block_size,
                    window = self.window,
                    center = True,
                    return_complex = True,
                    pad_mode = pad_mode)
                signal_stft += add_noise_stft * add_noise_filter
        else:
            signal_stft = combtooth_stft + noise_stft
        
        
        # take the istft to resynthesize audio.
        signal = torch.istft(
            signal_stft,
            n_fft = self.win_length,
            win_length = self.win_length,
            hop_length = self.block_size,
            window = self.window,
            center = True)
        
        return signal, hidden
