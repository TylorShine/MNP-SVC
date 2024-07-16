import os
import random
import torch
import numpy as np

import csv

import librosa

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset


def get_datasets(data_csv: str):
    if not os.path.isfile(data_csv):
        raise FileNotFoundError(f'metadata csv not found: {data_csv}')
    
    with open(data_csv, encoding='utf-8') as f:
        reader = csv.reader(f)
        lines = [row for row in reader]
        headers = lines[0]
        datas = {}
        for line in lines[1:]:
            datas[line[0]] = {}
            for h, v in enumerate(line[1:], start=1):
                datas[line[0]][headers[h]] = v
    return datas


class AudioDataset(TorchDataset):
    def __init__(
        self,
        root_path,
        metadatas: dict,
        crop_duration,
        hop_size,
        sampling_rate,
        whole_audio=False,
        # extensions=['wav'],
        # n_spk=1,
        cache_all_data=True,
        device='cpu',
        fp16=False,
        use_aug=False,
        use_spk_embed=False,
        per_file_spk_embed=False,
        use_mel=False,
        units_only=False,
    ):
        super().__init__()
        
        self.root_path = root_path
        self.crop_duration = crop_duration
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.whole_audio = whole_audio
        self.use_aug = use_aug
        self.use_mel = use_mel,
        self.use_spk_embed = use_spk_embed
        self.per_file_spk_embed = per_file_spk_embed
        
        self.units_only = units_only
        
        self.paths = list(metadatas.keys())
        
        self.data_buffer = {}
        self.spk_embeds = {}
        
        skip_index = []
        if units_only:
            data_dir = os.path.join(root_path, 'data')
            for file in tqdm(metadatas.keys(), desc='loading data'):
                audio_path = os.path.join(root_path, file)
                duration = librosa.get_duration(path=audio_path, sr=sampling_rate)
                file_dir, file_name = os.path.split(file)
                # file_rel = os.path.relpath(file_dir, start=data_dir)
                file_rel = os.path.relpath(file_dir, start='data')
                
                # load units
                units_dir = os.path.join(self.root_path, 'units', file_rel, file_name) + '.npz'
                units = np.load(units_dir)['units']
                units = torch.from_numpy(units).to(device)
                
                if fp16:
                    units = units.half()
                
                self.data_buffer[file] = {
                    'duration': duration,
                    'units': units,
                    'spk_id': torch.LongTensor(np.array([int(metadatas[file]['spk_id'])])).to(device),
                }
        else:
            if use_spk_embed:
                if not self.per_file_spk_embed:
                    spk_info_path = os.path.join(root_path, 'spk_info.npz')
                    spk_infos = np.load(spk_info_path, allow_pickle=True)
                    spk_ids = set((i for i in spk_infos.files))
                    self.spk_embeds = {
                        k: spk_infos[k].item()['spk_embed']
                        for k in spk_ids
                    }
                
            for idx, file in enumerate(tqdm(self.paths, total=len(self.paths), desc='loading data')):
                audio_path = os.path.join(root_path, file)
                duration = librosa.get_duration(path=audio_path, sr=sampling_rate)
                
                if duration < crop_duration + 0.1 and not whole_audio:
                    print(f"skip loading file {file}, because length {duration:.2f}sec is too short.")
                    skip_index.append(idx)
                    continue
                
                file_dir, file_name = os.path.split(file)
                # file_rel = os.path.relpath(file_dir, start=data_dir)
                file_rel = os.path.relpath(file_dir, start='data')
                
                # load f0
                f0_path = os.path.join(self.root_path, 'f0', file_rel, file_name) + '.npz'
                f0 = np.load(f0_path)['f0']
                f0 = torch.from_numpy(f0).float().unsqueeze(-1).to(device)
                
                if cache_all_data:
                    # load audio
                    audio, sr = librosa.load(audio_path, sr=sampling_rate)
                    audio = torch.from_numpy(audio).to(device)
                    
                    if use_mel:
                        # load mel
                        path_mel = os.path.join(self.root_path, 'mel', file_rel, file_name) + '.npz'
                        mel = np.load(path_mel)['mel']
                        mel = torch.from_numpy(mel).to(device)
                        
                        # path_augmel = os.path.join(self.path_root, 'aug_mel', file_rel, file_name) + '.npz'
                        # aug_mel = np.load(path_augmel)['aug_mel']
                        # aug_mel = torch.from_numpy(aug_mel).to(device)
                
                # load volume
                volume_path = os.path.join(self.root_path, 'volume', file_rel, file_name) + '.npz'
                volume = np.load(volume_path)['volume']
                volume = torch.from_numpy(volume).float().unsqueeze(-1).to(device)
                
                if cache_all_data:
                    # load units
                    units_dir = os.path.join(self.root_path, 'units', file_rel, file_name) + '.npz'
                    units = np.load(units_dir)['units']
                    units = torch.from_numpy(units).to(device)
                
                if cache_all_data and fp16:
                    audio = audio.half()
                    units = units.half()
                    if use_mel:
                        mel = mel.half()
                    # aug_mel = aug_mel.half()
                
                self.data_buffer[file] = {
                    'duration': duration,
                    'f0': f0,
                    'volume': volume,
                    'spk_id': torch.LongTensor(np.array([int(metadatas[file]['spk_id'])])).to(device),
                }
                
                if cache_all_data:
                    self.data_buffer[file]['audio'] = audio
                    self.data_buffer[file]['units'] = units
                    if use_mel:
                        self.data_buffer[file]['mel'] = mel
                        # self.data_buffer[file]['aug_mel'] = aug_mel
                
                if use_spk_embed:
                    if not self.per_file_spk_embed:
                        self.data_buffer[file]['spk_embed'] = self.spk_embeds[metadatas[file]['spk_id']]
                    else:
                        # load speaker embed
                        spk_embed_path = os.path.join(self.root_path, 'spk_embed', file_rel, file_name) + '.npz'
                        spk_embed = np.load(spk_embed_path)['spk_embed']
                        self.data_buffer[file]['spk_embed'] = torch.from_numpy(spk_embed).float().unsqueeze(-1).to(device)
                    
        if len(skip_index) > 0:
            print(f"skip {len(skip_index)} files.")
            self.paths = [v for i, v in enumerate(self.paths) if i not in skip_index]
                    
    def __getitem__(self, file_idx):
        file = self.paths[file_idx]
        data_buffer = self.data_buffer[file]
        
        # # check duration. if too short, then skip
        if data_buffer['duration'] < (self.crop_duration + 0.1):
            return self.__getitem__( (file_idx + 1) % len(self.paths))
            
        # get item
        return self.get_data(file, data_buffer)
    
    def __len__(self):
        return len(self.paths)
    
    def get_data(self, file, data_buffer):
        name = os.path.splitext(file)[0]
        frame_resolution = self.hop_size / self.sampling_rate
        duration = data_buffer['duration']
        crop_duration = duration if self.whole_audio else self.crop_duration
        
        idx_from = 0 if self.whole_audio else random.uniform(0, duration - crop_duration - 0.1)
        start_frame = int(idx_from / frame_resolution)
        units_frame_len = int(crop_duration / frame_resolution)
        
        aug_flag = random.choice([True, False]) and self.use_aug
        file_dir, file_name = os.path.split(file)
        file_rel = os.path.relpath(file_dir, start='data')
        
        # load units
        if 'units' in data_buffer.keys():
            units = data_buffer['units'][start_frame : start_frame + units_frame_len]
        else:
            # load units
            # file_dir, file_name = os.path.split(file)
            # file_rel = os.path.relpath(file_dir, start='data')
            units_dir = os.path.join(self.root_path, 'units', file_rel, file_name) + '.npz'
            units = np.load(units_dir)['units'][start_frame : start_frame + units_frame_len]
            units = torch.from_numpy(units).to(self.device)
            
        # load spk_id
        spk_id = data_buffer['spk_id']
        
        if self.units_only:
            return dict(spk_id=spk_id, units=units)
        
        if self.use_mel == True:
            # load mel
            # mel_key = 'aug_mel' if aug_flag else 'mel'
            mel_key = 'mel'
            mel = data_buffer.get(mel_key)
            if mel is None:
                mel = os.path.join(self.root_path, mel_key, file_rel, file_name) + '.npz'
                mel = np.load(mel)[mel_key]
                mel = mel[start_frame : start_frame + units_frame_len]
                mel = torch.from_numpy(mel).float()
            else:
                mel = mel[start_frame : start_frame + units_frame_len]
        
        # load audio
        if 'audio' in data_buffer.keys():
            audio = data_buffer['audio'][start_frame*self.hop_size:(start_frame + units_frame_len)*self.hop_size]
        else:
            # load audio
            audio_path = os.path.join(self.root_path, file)
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
            audio = torch.from_numpy(audio[start_frame*self.hop_size:(start_frame + units_frame_len)*self.hop_size]).to(self.device)
        
        # load f0
        f0 = data_buffer['f0'][start_frame : start_frame + units_frame_len]
        
        # load volume
        volume = data_buffer['volume'][start_frame : start_frame + units_frame_len]
        
        # volume augumentation
        if self.use_aug:
            max_gain = torch.max(torch.abs(audio)) + 1e-5
            max_shift = min(1.5, torch.log10(1./max_gain))
            log10_vol_shift = random.uniform(-1.5, max_shift)
            aug_audio = audio*(10 ** log10_vol_shift)
            aug_volume = volume*(10 ** log10_vol_shift)
        else:
            aug_audio = audio
            aug_volume = volume
            
        data = {
            'audio': aug_audio,
            'f0': f0,
            'volume': aug_volume,
            'units': units,
            'spk_id': spk_id,
            'name': name
        }
        
        if self.use_spk_embed:
            data['spk_embed'] = data_buffer['spk_embed']
            
        if self.use_mel == True:
            data['mel'] = mel
            
        return data
            
      
        
class AudioCrop:
    def __init__(self, block_size, sampling_rate, crop_duration):
        self.block_size = block_size
        self.sampling_rate = sampling_rate
        self.crop_duration = crop_duration
        
    def crop_audio(self, batch):
        frame_resolution = self.block_size / self.sampling_rate
        units_frame_len = int(self.crop_duration / frame_resolution)
        # print(batch['units'].shape, batch['audio'].shape)
        # print(batch['units'][0][0].shape, len(batch['units']), len(batch['units'][0]))
        # print(len(batch['units']), len(batch['units'][0]), len(batch['units'][0][0]))
        for b in range(len(batch['audio'])):
            duration = len(batch['audio'][b]) / self.sampling_rate
            idx_from = random.uniform(0, duration - self.crop_duration - 0.1)
            start_frame = int(idx_from / frame_resolution)
        
            batch['units'][b] = batch['units'][b][0][start_frame:start_frame+units_frame_len]
            batch['f0'][b] = batch['f0'][b][start_frame:start_frame+units_frame_len]
            batch['volume'][b] = batch['volume'][b][start_frame:start_frame+units_frame_len]
            
            batch['audio'][b] = batch['audio'][b][start_frame*self.block_size:(start_frame + units_frame_len)*self.block_size]
            
        for b in range(len(batch['audio'])):
            batch['units'] = torch.tensor(batch['units'])
            batch['f0'] = torch.tensor(batch['f0'])
            batch['volume'] = torch.tensor(batch['volume'])
            batch['audio'] = torch.tensor(batch['audio'])
            batch['spk_embed'] = torch.tensor(batch['spk_embed'])
            batch['spk_id'] = torch.tensor(batch['spk_id'])
        
        return batch


def get_data_loaders(args):
    loaders = {}
    
    ds_train = get_datasets(os.path.join(args.data.dataset_path, 'train.csv'))
    
    loaders['train'] = DataLoader(
        AudioDataset(
            root_path=args.data.dataset_path,
            metadatas=ds_train,
            crop_duration=args.data.duration,
            hop_size=args.data.block_size,
            sampling_rate=args.data.sampling_rate,
            whole_audio=False,
            device=args.train.cache_device,
            fp16=args.train.cache_fp16,
            use_aug=True,
            use_spk_embed=args.model.use_speaker_embed,
            use_mel="Diffusion" in args.model.type or "Reflow" in args.model.type,
            units_only=args.train.only_u2c_stack),
        batch_size=args.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.train.num_workers if args.train.cache_device=='cpu' else 0,
        persistent_workers=(args.train.num_workers > 0) if args.train.cache_device=='cpu' else False,
        pin_memory=True if args.train.cache_device=='cpu' else False
    )
    
    test_csv = os.path.join(args.data.dataset_path, 'test.csv')
    if os.path.isfile(test_csv):
        ds_test = get_datasets(test_csv)
        
        loaders['test'] = DataLoader(
            AudioDataset(
                root_path=args.data.dataset_path,
                metadatas=ds_test,
                crop_duration=args.data.duration,
                hop_size=args.data.block_size,
                sampling_rate=args.data.sampling_rate,
                whole_audio=True,
                device=args.train.cache_device,
                use_aug=False,
                use_spk_embed=args.model.use_speaker_embed,
                use_mel="Diffusion" in args.model.type or "Reflow" in args.model.type,
                units_only=args.train.only_u2c_stack),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True if args.train.cache_device=='cpu' else False
        )
    else:
        loaders['test'] = None
    
    return loaders


if __name__ == '__main__':
    import os, sys
    import numpy as np
    
    ds = get_datasets(sys.argv[1], name=os.path.basename(sys.argv[2]), data_dir=sys.argv[2], cache_dir=sys.argv[3])
    # for split in ds.keys():
    #     data_dir = os.path.join(sys.argv[2], 'data')
    #     units_dir = os.path.join(sys.argv[2], 'units')
    #     f0_dir = os.path.join(sys.argv[2], 'f0')
    #     volume_dir = os.path.join(sys.argv[2], 'volume')
    #     spk_info_file = os.path.join(sys.argv[2], f'spk_info_{split}.npz')
    #     spk_infos = np.load(spk_info_file, allow_pickle=True)
    #     ds[split] = Dataset.from_dict(
    #         {
    #             **ds[split].to_dict(),
    #             'audio': [x['array'] for x in ds[split]['audio']],
    #             'units': [
    #                 np.load(
    #                     f'{os.path.join(units_dir, os.path.relpath(p['audio']['path'], start=data_dir))}.npz')['units']
    #                 for p in ds[split]
    #             ],
    #             'f0': [
    #                 np.load(
    #                     f'{os.path.join(f0_dir, os.path.relpath(p['audio']['path'], start=data_dir))}.npz')['f0']
    #                 for p in ds[split]
    #             ],
    #             'volume': [
    #                 np.load(
    #                     f'{os.path.join(volume_dir, os.path.relpath(p['audio']['path'], start=data_dir))}.npz')['volume']
    #                 for p in ds[split]
    #             ],
    #             'spk_embed': [
    #                 spk_infos[str(p['spk_id'])].item()['spk_embed']
    #                 for p in ds[split]
    #             ]
    #         }
    #     )
        
    print(ds)
    print(ds['train'][0].keys())
    