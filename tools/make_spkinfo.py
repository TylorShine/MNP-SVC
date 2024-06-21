import os
import argparse

import glob

import torch

import numpy as np

from modules.common import load_config

from modules.dataset.prepare import make_metadata
from modules.dataset.preprocess import PreprocessorParameters
from modules.dataset.preprocess import preprocess_spkinfo, preprocess_main
from modules.dataset.loader import get_datasets

from modules.extractors import SpeakerEmbedEncoder


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-c",
    #     "--config",
    #     type=str,
    #     required=True,
    #     help="path to the config file")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu or cuda, auto if not set")
    parser.add_argument(
        "-dir",
        "--data_dir",
        type=str,
        required=True,
        help="dataset directory")
    parser.add_argument(
        "-s",
        "--spk_embed_encoder",
        type=str,
        default="pyannote.audio"
    )
    parser.add_argument(
        "-sp",
        "--spk_embed_encoder_ckpt",
        type=str,
        default="./models/pretrained/pyannote.audio/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin"
    )
    parser.add_argument(
        "-sr",
        "--spk_embed_encoder_sample_rate",
        type=int,
        default=16000
    )
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()

    device = cmd.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load config
    # args = load_config(cmd.config)
    
    
    # # make metadatas
    # make_metadata(args.data.dataset_path, args.data.extensions)
    
    
    # preprocessor parameters
    params = PreprocessorParameters(
        cmd.data_dir,
        use_speaker_embed=True,
        speaker_embed_encoder=cmd.spk_embed_encoder,
        speaker_embed_encoder_path=cmd.spk_embed_encoder_ckpt,
        speaker_embed_encoder_sample_rate=cmd.spk_embed_encoder_sample_rate,
        per_file_speaker_embed=False,
        device=device)
    
    # # get dataset
    # ds_train = get_datasets(os.path.join(args.data.dataset_path, 'train.csv'))
    
    # test_csv = os.path.join(args.data.dataset_path, 'test.csv')
    # if os.path.isfile(test_csv):
    #     ds_test = get_datasets(test_csv)
    # else:
    #     ds_test = None
    
    speaker_embed_encoder = SpeakerEmbedEncoder(**params.speaker_embed_encoder)
    
    exts = [".wav", ".mp3", ".flac", ".m4a"]
    speaker_paths = [p for p in glob.iglob(f"{cmd.data_dir}/**/*.*", recursive=True) if any(p.endswith(ex) for ex in exts)]
    print(os.path.basename(cmd.data_dir))
    spk_id = 1
    speaker_embed_dict = {
        str(spk_id): {
            'name': os.path.basename(cmd.data_dir),
            'spk_embed': speaker_embed_encoder.encode_speaker(speaker_paths).numpy()
        }
    }
    
    spk_info_name = f'spk_info_{os.path.basename(cmd.data_dir)}.npz'
    
    np.savez_compressed(os.path.join(cmd.data_dir, "..", spk_info_name), **speaker_embed_dict)
    
    