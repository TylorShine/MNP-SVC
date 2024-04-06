import torch
import torchaudio

class SpeakerEmbedEncoder:
    def __init__(self, encoder, encoder_ckpt,
                 encoder_sample_rate = 16000, device = None,):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        is_loaded_encoder = False
        if encoder == 'pyannote.audio':
            self.model = Audio2PyannoteAudio(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'resemblyzer':
            self.model = Audio2Resemblyzer(encoder_ckpt, device=device)
            is_loaded_encoder = True
            
        if not is_loaded_encoder:
            raise ValueError(f" [x] Unknown speaker embedding encoder: {encoder}")
        
        self.resample_kernel = {}
        self.encoder_sample_rate = encoder_sample_rate
        
    def encode(self, 
                audio, # B, T
                sample_rate): 
        
        audio_sample_rate = sample_rate
        # resample
        if sample_rate == self.encoder_sample_rate or sample_rate is None:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = torchaudio.transforms.Resample(sample_rate, self.encoder_sample_rate, lowpass_filter_width = 128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)
            audio_sample_rate = self.encoder_sample_rate
        
        # encode
        if audio_res.size(-1) < 400:
            audio_res = torch.nn.functional.pad(audio, (0, 400 - audio_res.size(-1)))
        embed = self.model(audio_res, audio_sample_rate)
        
        return embed
    
    def encode_speaker(self, 
                audio_files, # [Filenames]...
                ): 
        return self.model.embed_speaker(audio_files)
    
    
class Audio2PyannoteAudio():
    def __init__(self, path, device='cpu'):
        from pyannote.audio import Model as PyannoteModel, Inference as PyannoteInference
        self.device = device
        print(' [Speaker Encoder Model] pyannote.audio')
        print(' [Loading] ' + path)
        self.model = PyannoteModel.from_pretrained(path, map_location=torch.device(device))
        self.inference = PyannoteInference(self.model, window="whole")

    def __call__(self,
                 audio,
                 input_sample_rate):  # B, T
        wav_tensor = audio
        with torch.no_grad():
            encoded_frames = []
            for i in range(audio.shape[0]):
                encoded_frames.append(
                    torch.from_numpy(self.inference({"waveform": wav_tensor[i].unsqueeze(0), "sample_rate": input_sample_rate})))
        
        embed = torch.cat(encoded_frames, dim=0)
        
        return embed
    
    def embed_utterance_from_files(self, audio_files):
        encoded_frames = []
        with torch.no_grad():
            for i in range(len(audio_files)):
                encoded_frames.append(
                    torch.from_numpy(self.inference(audio_files[i])))
        return encoded_frames
    
    def embed_speaker(self,
                      audio_files):  # [T]...
        with torch.no_grad():
            processed_embeds = self.embed_utterance_from_files(audio_files)
            embed = torch.mean(torch.stack(processed_embeds), dim=0)
            
        return embed / torch.linalg.norm(embed, ord=2)
    
    
class Audio2Resemblyzer():
    def __init__(self, path, device='cpu'):
        from resemblyzer import VoiceEncoder, preprocess_wav as resemblyzer_preprocess_wav
        self.device = device
        print(' [Speaker Encoder Model] Resemblyzer')
        print(' [Loading] ' + path)
        self.model = VoiceEncoder(device=self.device, weights_fpath=path)

    def __call__(self,
                 audio,
                 input_sample_rate):  # B, T
        wav_tensor = audio
        with torch.no_grad():
            encoded_frames = []
            for i in range(audio.shape[0]):
                processed_wav = resemblyzer_preprocess_wav(wav_tensor[i].cpu().numpy(), source_sr=input_sample_rate)
                encoded_frames.append(torch.from_numpy(self.model.embed_utterance(processed_wav)))
        
        embed = torch.cat(encoded_frames, dim=0)
        
        return embed
    
    def embed_speaker(self,
                      audio_files):  # [T]...
        processed_wavs = [ resemblyzer_preprocess_wav(audio) for audio in audio_files ]
        with torch.no_grad():
            embed = torch.from_numpy(self.model.embed_speaker(processed_wavs))
            
        return embed