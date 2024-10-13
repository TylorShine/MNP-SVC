import torch
import torchaudio

class UnitsEncoder:
    def __init__(self, encoder, encoder_ckpt, encoder_sample_rate = 16000, encoder_hop_size = 320, device = None,
                 skip_frames=0, extract_layers=None, no_alignment=False):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        is_loaded_encoder = False
        if encoder == 'wavlmbase':
            self.model = Audio2WavLM(encoder_ckpt, device=device, extract_layer=max(max(extract_layers)), extract_layers=extract_layers)
            is_loaded_encoder = True
        if encoder == 'dpwavlmbase':
            self.model = Audio2DPWavLM(encoder_ckpt, device=device, extract_layer=max(max(extract_layers)), extract_layers=extract_layers)
            is_loaded_encoder = True
        if encoder == 'phrex':
            self.model = Audio2Phrex(encoder_ckpt, encoder_sample_rate, device=device)
            is_loaded_encoder = True
        if not is_loaded_encoder:
            raise ValueError(f" [x] Unknown units encoder: {encoder}")
            
        self.resample_kernel = {}
        self.encoder_sample_rate = encoder_sample_rate
        self.encoder_hop_size = encoder_hop_size
        
        self.skip_frames = skip_frames
        self.no_alignment = no_alignment
        
    def encode(self, 
                audio, # B, T
                sample_rate,
                hop_size,
                f0=None):
        
        # resample
        if sample_rate == self.encoder_sample_rate:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = torchaudio.transforms.Resample(sample_rate, self.encoder_sample_rate, lowpass_filter_width = 128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)
                    
        # encode
        if audio_res.size(-1) < 400:
            audio_res = torch.nn.functional.pad(audio, (0, 400 - audio_res.size(-1)))
        if f0 is None:
            units = self.model(audio_res)
        else:
            units = self.model(audio_res, f0)
        
        if self.no_alignment:
            return units
        
        # alignment
        n_frames = audio.size(-1) // hop_size + 1
        ratio = (hop_size / sample_rate) / (self.encoder_hop_size / self.encoder_sample_rate)
        index = torch.clamp(torch.round(ratio * torch.arange(n_frames).to(self.device)).long(), max = units.size(1) - 1)
        repeats = [1, 1, units.size(-1)]
        orig_dim = None
        if units.ndim == 4:
            orig_dim = units.size(2)
            units = units.flatten(2)
            repeats[-1] = units.size(-1)
        index = index.unsqueeze(0).unsqueeze(-1).repeat(repeats)
        units_aligned = torch.gather(units, 1, index)
        if orig_dim is not None:
            units_aligned = units_aligned.view(*units_aligned.shape[0:2], orig_dim, units_aligned.shape[2]//orig_dim)
        if self.skip_frames > 0:
            units_aligned = torch.repeat_interleave(units_aligned[:, ::self.skip_frames+1], self.skip_frames+1, dim=1)[:, :units_aligned.shape[1]]
        
        
        return units_aligned
    
    
class Audio2WavLM():
    def __init__(self, path, device='cpu', extract_layer=12, extract_layers=None):
        from ..encoders.wavlm.WavLM import WavLM, WavLMConfig
        self.device = device
        print(' [Encoder Model] WavLM Base')
        print(' [Loading] ' + path)
        ckpt = torch.load(path)
        self.cfg = WavLMConfig(ckpt['cfg'])
        self.model = WavLM(self.cfg)
        self.model.load_state_dict(ckpt['model'])
        self.model.to(self.device)
        self.model.eval()
        self.extract_layer = extract_layer
        self.extract_layers = extract_layers

    def __call__(self,
                 audio):  # B, T
        with torch.no_grad():
            if self.cfg.normalize:
                audio_in = torch.nn.functional.layer_norm(audio, audio.shape)
            else:
                audio_in = audio
            if self.extract_layer == 0:
                units = self.model.extract_features(audio_in, output_layer=None, ret_conv=True)[0]
            elif self.extract_layers is not None:
                layers = self.model.extract_features(audio_in, output_layer=self.extract_layer, ret_layer_results=True)[0][1]
                units = []
                for accum_layers in self.extract_layers:
                    units.append(
                        torch.mean(torch.stack([layers[l][0] for l in accum_layers]), dim=0).squeeze(1))
                units = torch.stack(units, dim=1).unsqueeze(0)
            else:
                units = self.model.extract_features(audio_in, output_layer=self.extract_layer)[0]
            return units
        
        
class Audio2DPWavLM():
    def __init__(self, path, device='cpu', extract_layer=None, extract_layers=None):
        from ..encoders.dphubert.model import wavlm_model
        self.device = device
        print(' [Encoder Model] DPWavLM Base')
        print(' [Loading] ' + path)
        ckpt = torch.load(path)
        self.model = wavlm_model(**ckpt['config'])
        self.model.load_state_dict(ckpt['state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.extract_layer = extract_layer
        self.extract_layers = extract_layers

    def __call__(self,
                 audio):  # B, T
        with torch.no_grad():
            if self.extract_layers is not None:
                layers = self.model.extract_features(audio, num_layers=self.extract_layer)[0]
                units = []
                for accum_layers in self.extract_layers:
                    units.append(
                        torch.mean(torch.stack([layers[l] for l in accum_layers]), dim=0).squeeze(1))
                units = torch.stack(units, dim=1).squeeze(0)
            else:
                units = self.model.extract_features(audio, num_layers=self.extract_layer)[0][0]
            return units
        
        
class Audio2Phrex():
    def __init__(self, path, sample_rate, device='cpu'):
        from ..encoders.phrex.decoder import load_model, get_normalized_spectrogram
        from ..extractors.spec import SpecExtractor
        self.sample_rate = sample_rate
        self.device = device
        print(' [Encoder Model] Phrex')
        print(' [Loading] ' + path)
        self.model, self.args = load_model(path, device=device)
        
        self.spec_extractor = SpecExtractor(
            self.args.model.spec_n_fft,
            self.args.model.in_channels,
            self.args.data.block_size,
            # self.args.data.sampling_rate,
            sample_rate,
            device=device)

    def __call__(self,
                 audio): # B, T
        with torch.no_grad():
            # print(audio.shape)
            spec = self.spec_extractor.extract(audio.float().to(self.spec_extractor.device)).squeeze()
            spec: torch.Tensor
            spec = (spec / (spec.max(dim=-1, keepdim=True).values + 1e-3))[:, :self.args.model.in_channels].unsqueeze(0)
            
            # print(spec.shape)
            
            units = self.model.infer(spec, self.sample_rate)
            return units