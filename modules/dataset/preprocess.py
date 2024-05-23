import os
import torch
import numpy as np

import librosa
from tqdm import tqdm

from modules.extractors import F0Extractor, SpeakerEmbedEncoder, UnitsEncoder, VolumeExtractor


class PreprocessorParameters:
    def __init__(self,
                 data_dir: str,
                 sample_rate: int = 44100,
                 block_size: int = 512,
                 use_f0: bool = True,
                 f0_extractor: str = 'rmvpe',
                 f0_min: float = 65.,
                 f0_max: float = 1200.,
                 use_speaker_embed: bool = True,
                 speaker_embed_encoder: str = 'pyannote.audio',
                 speaker_embed_encoder_path: str = 'pyannote/wespeaker-voxceleb-resnet34-LM',
                 speaker_embed_encoder_sample_rate: int = 16000,
                 per_file_speaker_embed = False,
                 units_encoder: str = 'dpwavlmbase',
                 units_encoder_path: str = 'models/pretrained/dphubert/DPWavLM-sp0.75.pth',
                 units_encoder_sample_rate: int = 16000,
                 units_encoder_hop_size: int = 320,
                 units_encoder_skip_frames: int = 0,
                 units_encoder_extract_layers: list[list[int]] = 0,
                 units_encoder_no_alignment: bool = False,
                 volume_extractor_window_size: int = 8,
                 device: str | torch.device = 'cpu',
                 ):
        
        self.common = {
            'data_dir': data_dir,
            'sample_rate': sample_rate,
            'block_size': block_size,
            'per_file_speaker_embed': per_file_speaker_embed,
            'device': device,
        }
        
        self.units_encoder = {
            'encoder': units_encoder,
            'encoder_ckpt': units_encoder_path,
            'encoder_sample_rate': units_encoder_sample_rate,
            'encoder_hop_size': units_encoder_hop_size,
            'skip_frames': units_encoder_skip_frames,
            'extract_layers': units_encoder_extract_layers,
            'no_alignment': units_encoder_no_alignment,
            'device': device,
        }
        
        self.volume_extractor = {
            'hop_size': block_size,
            'window_size': volume_extractor_window_size,
        }
        
        self.f0_extractor = None
        self.speaker_embed_encoder = None
        
        if use_f0:
            self.f0_extractor = {
                'f0_extractor': f0_extractor,
                'sample_rate': sample_rate,
                'hop_size': block_size,
                'f0_min': f0_min,
                'f0_max': f0_max,
            }
            
        if use_speaker_embed:
            self.speaker_embed_encoder = {
                'encoder': speaker_embed_encoder,
                'encoder_ckpt': speaker_embed_encoder_path,
                'encoder_sample_rate': speaker_embed_encoder_sample_rate,
                'device': device,
            }
            

PREPROCESSOR_PARAMS: PreprocessorParameters = None

def preprocess_main(root_path: str, dataset: dict[str, dict[str, str]], params: PreprocessorParameters = PREPROCESSOR_PARAMS):
    data_dir = os.path.join(params.common['data_dir'], 'data')
    
    # units
    units_encoder = UnitsEncoder(**params.units_encoder)
    units_dir = os.path.join(params.common['data_dir'], 'units')
    for path in tqdm(dataset.keys(), desc='Extract units'):
        audio, sr = librosa.load(os.path.join(root_path, path), sr=None)
        units = units_encoder.encode(
            torch.from_numpy(audio).to(params.common['device'], dtype=torch.float32).unsqueeze(0), sr, params.common['block_size'])
        units_path = os.path.join(units_dir, os.path.relpath(path, start='data'))
        os.makedirs(os.path.dirname(units_path), exist_ok=True)
        np.savez_compressed(f'{units_path}.npz', units=units.squeeze().cpu().numpy())
    del units_encoder
    units_encoder = None
    
    # volume
    volume_extractor = VolumeExtractor(**params.volume_extractor)
    volume_dir = os.path.join(params.common['data_dir'], 'volume')
    for path in tqdm(dataset.keys(), desc='Extract volume'):
        audio, sr = librosa.load(os.path.join(root_path, path), sr=None)
        volume = volume_extractor.extract(audio)
        # volume_path = os.path.join(volume_dir, os.path.relpath(path, start=data_dir))
        volume_path = os.path.join(volume_dir, os.path.relpath(path, start='data'))
        os.makedirs(os.path.dirname(volume_path), exist_ok=True)
        np.savez_compressed(f'{volume_path}.npz', volume=volume)
    
    # f0
    if params.f0_extractor is not None:
        f0_extractor = F0Extractor(**params.f0_extractor)
        f0_dir = os.path.join(params.common['data_dir'], 'f0')
        for path in tqdm(dataset.keys(), desc='Extract f0'):
            audio, sr = librosa.load(os.path.join(root_path, path), sr=None)
            f0 = f0_extractor.extract(audio, device=params.common['device'])
            f0_uv = f0 == 0
            f0[f0_uv] = np.random.rand(*f0[f0_uv].shape)*float(params.common['sample_rate']/params.common['block_size']) + float(params.common['sample_rate']/params.common['block_size'])
            # f0_path = os.path.join(f0_dir, os.path.relpath(path, start=data_dir))
            f0_path = os.path.join(f0_dir, os.path.relpath(path, start='data'))
            os.makedirs(os.path.dirname(f0_path), exist_ok=True)
            np.savez_compressed(f'{f0_path}.npz', f0=f0)
        del f0_extractor
        f0_extractor = None


def preprocess_spkinfo(root_path: str, dataset: dict[str, dict[str, str]], split: str = "", params: PreprocessorParameters = PREPROCESSOR_PARAMS):
    speaker_embed_encoder = SpeakerEmbedEncoder(**params.speaker_embed_encoder)
    speaker_ids = set([c['spk_id'] for c in dataset.values()])
    
    speaker_path_dict = {
        # i: [p['path'] for p, id in zip(examples['audio'], examples['spk_id']) if id == i]
        i: [os.path.join(root_path, p) for p, c in dataset.items() if c['spk_id'] == i]
        for i in speaker_ids
    }
    speaker_embed_dict = {
        str(i): {
            'name': [c['spk_name'] for c in dataset.values() if c['spk_id'] == i][0],
            'spk_embed': speaker_embed_encoder.encode_speaker(speaker_path_dict[i]).numpy()
        }
        for i in tqdm(speaker_path_dict.keys(), desc="Extract speaker embed")
    }
    
    spk_info_name = 'spk_info.npz'
    if split != "":
        spk_info_name = f'spk_info_{split}.npz'
    
    np.savez_compressed(os.path.join(params.common['data_dir'], spk_info_name), **speaker_embed_dict)
    
    if params.common['per_file_speaker_embed']:
        spk_embed_dir = os.path.join(params.common['data_dir'], 'spk_embed')
        for path in tqdm(dataset.keys(), desc='Extract speaker embed'):
            audio, sr = librosa.load(os.path.join(root_path, path), sr=None)
            spk_embed = speaker_embed_encoder.encode(torch.from_numpy(audio).unsqueeze(0).to(params.common['device']), sr)
            # f0_path = os.path.join(f0_dir, os.path.relpath(path, start=data_dir))
            spk_embed_path = os.path.join(spk_embed_dir, os.path.relpath(path, start='data'))
            os.makedirs(os.path.dirname(spk_embed_path), exist_ok=True)
            np.savez_compressed(f'{spk_embed_path}.npz', spk_embed=spk_embed)
    
    
if __name__ == '__main__':
    import sys
    from modules.dataset import loader
    
    PREPROCESSOR_PARAMS = PreprocessorParameters(
        data_dir=sys.argv[1],
        units_encoder_extract_layers=[[10, 11]],
        speaker_embed_encoder_path='./models/pretrained/pyannote.audio/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin',
        device='cuda')
    
    splits = map(os.path.basename, os.listdir(os.path.join(sys.argv[1], 'data')))
    
    ds_train = loader.get_datasets(os.path.join(sys.argv[1], 'train.csv'))
    ds_test = loader.get_datasets(os.path.join(sys.argv[1], 'test.csv'))
    
    preprocess_spkinfo(sys.argv[1], ds_train)
    preprocess_main(sys.argv[1], ds_train)
    preprocess_main(sys.argv[1], ds_test)
    
    
    