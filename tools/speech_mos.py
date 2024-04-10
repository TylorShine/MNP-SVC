import argparse

import torch
import librosa


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu or cuda, auto if not set",
    )
    parser.add_argument(
        "input",
        nargs="+",
        type=str,
        help="path to the audio file(s)")
    return parser.parse_args(args=args, namespace=namespace)

if __name__ == '__main__':
    cmd = parse_args()
    
    device = cmd.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device)
    
    for input in cmd.input:
        wave, sr = librosa.load(input, sr=None, mono=True)
        score = predictor(torch.from_numpy(wave).unsqueeze(0).to(device), sr)
        print(f"{input}: {score.mean().item()}")