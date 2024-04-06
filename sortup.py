import os
import argparse

from modules.common import load_config

from modules.dataset.prepare import sortup_audio


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
        "-a",
        "--auto-indexing",
        action="store_true",
        help="auto prefix indexed number to output dirs")
    parser.add_argument(
        "-train",
        "--train-ratio",
        type=float,
        default=0.99,
        help="split ratio of the train set",
    )
    parser.add_argument(
        "-test",
        "--test-ratio",
        type=float,
        default=0.01,
        help="split ratio of the test set",
    )
    parser.add_argument(
        "-m",
        "--split-min",
        type=int,
        default=1,
        help="minimum num of split items",
    )
    parser.add_argument(
        "-s",
        "--split-overall",
        action="store_true",
        help="split through overall instead per speaker")
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()

    # load config
    args = load_config(cmd.config)
    
    # sort up
    sortup_audio(
        args.data.dataset_path,
        args.data.extensions,
        auto_indexing=cmd.auto_indexing,
        splits={'train': cmd.train_ratio, 'test': cmd.test_ratio},
        split_min=cmd.split_min,
        split_per_speaker=not cmd.split_overall,
        sampling_rate=args.data.sampling_rate)