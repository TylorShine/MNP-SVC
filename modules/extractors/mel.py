import torch

from ..diffusion.vocoder import Vocoder

class MelExtractor:
    def __init__(self, vocoder_type, vocoder_ckpt, device = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
                
        self.vocoder = Vocoder(vocoder_type, vocoder_ckpt, device)
        
    def extract(self, audio, sample_rate=0, keyshift=0):
        return self.vocoder.extract(audio, sample_rate, keyshift)
    