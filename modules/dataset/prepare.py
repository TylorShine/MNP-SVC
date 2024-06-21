import os
import csv
import glob
import operator
import random
import shutil

import pathlib
import posixpath

import librosa
import soundfile as sf
from tqdm import tqdm

from collections.abc import Iterable


def sortup_audio(root_dir, extensions,
                 auto_indexing=False,
                 splits: dict[str, float] = {'train':0.99, 'test':0.01},
                 split_min: int = 1,
                 split_per_speaker: bool = True,
                 sampling_rate: int = None):
    root_basename = os.path.basename(root_dir)
    audio_dir = os.path.join(root_dir, 'audio')
    # gather dirs in top level
    dirs = [d.rstrip(os.sep) for d in glob.glob(f'{audio_dir}/*/')]
    
    file_metas: dict[int, list[Iterable]] = {}
    
    if len(dirs) <= 0:
        # if not containts any dirs in root_dir, process as single speaker
        spk_id = 1
        spk_name = "1"
        file_metas[spk_id] = []
        # gather datas
        for file in glob.iglob(f'{audio_dir}/*'):
            ext = os.path.splitext(file)[1]
            if len(ext) <= 0 or ext[1:] not in extensions:
                continue
            
            norm_path = os.path.normpath(os.path.relpath(file, start=root_dir))
            if os.sep == '\\':
                norm_path = pathlib.PureWindowsPath(norm_path).as_posix()
                    
            file_metas[spk_id].append((posixpath.join(root_basename, norm_path), spk_id, spk_name, file))
    else:
        # multi speakers
        # gather datas
        import re
        seps = re.compile(r'[-_.]')
        for id, dir in enumerate(dirs, start=1):
            if auto_indexing:
                spk_id = id
                spk_name = os.path.basename(dir)
            else:
                # get speaker id from dirname
                dir_basename = os.path.basename(dir)
                dir_split = seps.split(dir_basename, 1)
                if not dir_split[0].isdecimal():
                    print(f' [*] skip directory {dir_basename}: cannot parse spk_id from dirname')
                    continue
                spk_id = int(dir_split[0])
                if len(dir_split) > 1:
                    spk_name = dir_split[1]
                else:
                    spk_name = dir_split[0]
                
            for file in glob.iglob(f'{dir}/**', recursive=True):
                ext = os.path.splitext(file)[1]
                if len(ext) <= 0 or ext[1:] not in extensions:
                    continue
                
                norm_path = os.path.normpath(os.path.relpath(file, start=root_dir))
                if os.sep == '\\':
                    norm_path = pathlib.PureWindowsPath(norm_path).as_posix()
                    
                if spk_id not in file_metas.keys():
                    file_metas[spk_id] = []
                file_metas[spk_id].append((posixpath.join(root_basename, norm_path), spk_id, spk_name, file))
            
    # print(file_metas)
    
    file_meta_splits: dict[str, list] = {}
    
    # split
    if split_per_speaker:
        num_datas: dict[int, int] = {i: len(v) for i, v in file_metas.items()}
        split_nums: dict[int, dict[str, int]] = {
            i: {
                s: max(split_min, int(r*n))
                for s, r in splits.items()
            }
            for i, n in num_datas.items()
        }
        max_ratio_split = max(splits.keys(), key=splits.get)
        for i, n_all in num_datas.items():
            n_sum = sum(split_nums[i].values())
            if n_sum != n_all:
                split_nums[i][max_ratio_split] += n_all - n_sum
        # print(num_datas, split_nums)
        
        for i, v in split_nums.items():
            random.shuffle(file_metas[i])
            split_from = 0
            for s, n in v.items():
                split_to = split_from + n
                if not s in file_meta_splits:
                    file_meta_splits[s] = []
                file_meta_splits[s].extend(
                    file_metas[i][split_from:split_to]
                )
                split_from = split_to
    else:
        all_files_metas: list[Iterable] = [i for v in file_metas.values() for i in v]
        split_nums: dict[str, int] = {
                s: max(split_min, int(r*len(all_files_metas)))
                for s, r in splits.items()
        }
        max_ratio_split = max(splits.keys(), key=splits.get)
        n_sum = sum(split_nums.values())
        if n_sum != len(all_files_metas):
            split_nums[max_ratio_split] += len(all_files_metas) - n_sum
        # print(len(all_files_metas), split_nums)
        
        random.shuffle(all_files_metas)
        split_from = 0
        for s, n in split_nums.items():
            split_to = split_from + n
            if not s in file_meta_splits:
                file_meta_splits[s] = []
            file_meta_splits[s].extend(
                all_files_metas[split_from:split_to]
            )
            split_from = split_to
            
            
    if len([v for k in file_meta_splits.keys() for v in file_meta_splits[k]]) <= 0:
        print(" [*] none target file were found, nothing to do.")
        return
    
    
    # resample and sort up
    for s in file_meta_splits.keys():
        split_dir = os.path.join(root_dir, 'data', s)
        for i, m in tqdm(enumerate(file_meta_splits[s]), total=len(file_meta_splits[s]), desc=f'Prepare audio [{s}]'):
            _, spk_id, spk_name, orig_path = m
            orig_relpath = os.path.relpath(orig_path, start=audio_dir)
            id_name = f'{spk_id}_{spk_name}'
            dir_move_to = os.path.join(split_dir, id_name, *os.path.dirname(orig_relpath).split(os.sep)[1:])
            file_move_to = os.path.join(dir_move_to, os.path.basename(orig_relpath))
            os.makedirs(dir_move_to, exist_ok=True)
            
            
            if sampling_rate is None:
                # just copy
                shutil.copy2(orig_path, file_move_to)
            else:
                # resample and save
                resampled, _ = librosa.load(orig_path, sr=sampling_rate, mono=True)
                sf.write(file_move_to, resampled, sampling_rate)
            
            norm_path = os.path.normpath(os.path.relpath(file_move_to, start=root_dir))
            if os.sep == '\\':
                norm_path = pathlib.PureWindowsPath(norm_path).as_posix()
            file_meta_splits[s][i] = (norm_path, spk_id, spk_name)
            
            
def make_metadata(root_dir, extensions):
    root_basename = os.path.basename(root_dir)
    data_dir = os.path.join(root_dir, 'data')
    # gather dirs in top level
    dirs = [d.rstrip(os.sep) for d in glob.glob(f'{data_dir}/*/')]
    
    file_meta_splits: dict[str, list] = {}
    
    # gather datas
    import re
    seps = re.compile(r'[-_.]')
    for split_dir in dirs:
        split_name =os.path.basename(split_dir)
        
        file_meta_splits[split_name] = []
        
        for dir in glob.iglob(f'{split_dir}/*/'):
            # get speaker id from dirname
            dir_basename = os.path.basename(dir.rstrip(os.sep))
            dir_split = seps.split(dir_basename, 1)
            if not dir_split[0].isdecimal():
                print(f' [*] skip directory {dir_basename}: cannot parse spk_id from dirname')
                continue
            spk_id = int(dir_split[0])
            if len(dir_split) > 1:
                spk_name = dir_split[1]
            else:
                spk_name = dir_split[0]
                
            for file in glob.iglob(f'{dir}/**', recursive=True):
                ext = os.path.splitext(file)[1]
                if len(ext) <= 0 or ext[1:] not in extensions:
                    continue
                
                norm_path = os.path.normpath(os.path.relpath(file, start=root_dir))
                if os.sep == '\\':
                    norm_path = pathlib.PureWindowsPath(norm_path).as_posix()
                    
                # if spk_id not in file_metas.keys():
                #     file_metas[spk_id] = []
                # file_metas[spk_id].append((posixpath.join(root_basename, norm_path), spk_id, spk_name, file))
                # file_meta_splits[split_name].append((posixpath.join(root_basename, norm_path), spk_id, spk_name, file))
                file_meta_splits[split_name].append((norm_path, spk_id, spk_name))
                
                
    # write split metadatas
    for s in file_meta_splits.keys():
        metadata_csv = os.path.join(root_dir, f'{s}.csv')
        with open(metadata_csv, 'w', encoding='utf-8') as f:
            writer = csv.writer(f, lineterminator='\n')
            # write header
            writer.writerow(("file_name", "spk_id", "spk_name"))
            
            writer.writerows(sorted(file_meta_splits[s], key=operator.itemgetter(0)))
    
    
if __name__ == '__main__':
    import sys
    
    # sortup_audio(sys.argv[1], ['wav', 'flac', 'mp3'], sampling_rate=44100)
    make_metadata(sys.argv[1], ['wav', 'flac', 'mp3'])
    
    
