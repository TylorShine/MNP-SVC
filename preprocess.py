import os
import argparse

import torch

from modules.common import load_config

from modules.dataset.prepare import make_metadata
from modules.dataset.preprocess import PreprocessorParameters
from modules.dataset.preprocess import preprocess_spkinfo, preprocess_main
from modules.dataset.loader import get_datasets


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu or cuda, auto if not set")
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()

    device = cmd.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load config
    args = load_config(cmd.config)
    
    
    # make metadatas
    make_metadata(args.data.dataset_path, args.data.extensions)
    
    
    # preprocessor parameters
    params = PreprocessorParameters(
        args.data.dataset_path,
        sample_rate=args.data.sampling_rate,
        block_size=args.data.block_size,
        use_f0=True,
        f0_extractor=args.data.f0_extractor,
        f0_min=args.data.f0_min,
        f0_max=args.data.f0_max,
        use_speaker_embed=args.model.use_speaker_embed,
        speaker_embed_encoder=args.data.spk_embed_encoder,
        speaker_embed_encoder_path=args.data.spk_embed_encoder_ckpt,
        speaker_embed_encoder_sample_rate=args.data.spk_embed_encoder_sample_rate,
        per_file_speaker_embed=args.data.per_file_speaker_embed,
        units_encoder=args.data.encoder,
        units_encoder_path=args.data.encoder_ckpt,
        units_encoder_sample_rate=args.data.encoder_sample_rate,
        units_encoder_hop_size=args.data.encoder_hop_size,
        units_encoder_extract_layers=args.model.units_layers,
        volume_extractor_window_size=args.data.volume_window_size,
        device=device)
    
    # get dataset
    ds_train = get_datasets(os.path.join(args.data.dataset_path, 'train.csv'))
    
    test_csv = os.path.join(args.data.dataset_path, 'test.csv')
    if os.path.isfile(test_csv):
        ds_test = get_datasets(test_csv)
    else:
        ds_test = None
    
    
    # process speaker embed
    if args.model.use_speaker_embed:
        preprocess_spkinfo(args.data.dataset_path, ds_train, params=params)
        
    # process units, f0 and volume
    preprocess_main(args.data.dataset_path, ds_train, params=params)
    if ds_test is not None:
        preprocess_main(args.data.dataset_path, ds_test, params=params)
    
    os.makedirs(args.env.expdir, exist_ok=True)
    