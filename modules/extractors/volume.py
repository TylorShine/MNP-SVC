import numpy as np

class VolumeExtractor:
    def __init__(self, hop_size=512, window_size=1):
        self.hop_size = hop_size
        self.window_size = window_size
        
    def extract(self, audio): # audio: 1d numpy array
        n_frames = int(len(audio) // self.hop_size) + 1
        audio2 = audio ** 2
        audio2 = np.pad(audio2, (int(self.window_size - 1), int((self.hop_size-self.window_size + 1))), mode = 'reflect')
        volume = np.array([np.max(audio2[int(n * self.hop_size) : int((n + 1) * self.hop_size)]) for n in range(n_frames)])
        volume = np.sqrt(volume)
        return volume