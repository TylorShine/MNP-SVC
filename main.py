import os
import torch
import librosa
import argparse
import numpy as np
import soundfile as sf
import hashlib
import torchaudio
from ast import literal_eval
from slicer import Slicer
from modules.vocoder import load_model, load_onnx_model
from modules.extractors import F0Extractor, VolumeExtractor, UnitsEncoder
from modules.extractors.common import upsample
from tqdm import tqdm

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="path to the model file",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu or cuda, auto if not set")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="path to the input audio file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="path to the output audio file",
    )
    parser.add_argument(
        "-id",
        "--spk_id",
        type=str,
        required=False,
        default=1,
        help="speaker id (for multi-speaker model) | default: 1",
    )
    parser.add_argument(
        "-semb",
        "--spk_embed",
        type=str,
        required=False,
        default="None",
        help="speaker embed .npz file (for multi-speaker with spk_embed_encoder model) | default: None",
    )
    parser.add_argument(
        "-mix",
        "--spk_mix_dict",
        type=str,
        required=False,
        default="None",
        help="mix-speaker dictionary (for multi-speaker model) | default: None",
    )
    parser.add_argument(
        "-intb",
        "--intonation_base",
        type=str,
        required=False,
        default=220.0,
        help="base freq of intonation changed | default: 220.0",
    )
    parser.add_argument(
        "-into",
        "--intonation",
        type=str,
        required=False,
        default=1.0,
        help="intonation changed (above 1.0 for exciter, below for calmer) | default: 1.0",
    )
    parser.add_argument(
        "-k",
        "--key",
        type=str,
        required=False,
        default=0,
        help="key changed (number of semitones) | default: 0",
    )
    parser.add_argument(
        "-pe",
        "--pitch_extractor",
        type=str,
        required=False,
        default='rmvpe',
        help="pitch extrator type: dio, harvest, crepe, fcpe, rmvpe (default)",
    )
    parser.add_argument(
        "-fmin",
        "--f0_min",
        type=str,
        required=False,
        default=50,
        help="min f0 (Hz) | default: 50",
    )
    parser.add_argument(
        "-fmax",
        "--f0_max",
        type=str,
        required=False,
        default=1200,
        help="max f0 (Hz) | default: 1200",
    )
    parser.add_argument(
        "-f0intp",
        "--interpolate_f0",
        type=str,
        required=False,
        default='true',
        help="true or false | default: true",
    )
    parser.add_argument(
        "-th",
        "--threhold",
        type=str,
        required=False,
        default=-60,
        help="response threhold (dB) | default: -60",
    )
    parser.add_argument(
        "-op",
        "--onnx_providers",
        type=str,
        required=False,
        default="CPUExecutionProvider",
        help="execution provider names of onnxruntime, separate by comma | default: CPUExecutionProvider"
    )
    return parser.parse_args(args=args, namespace=namespace)

    
def split(audio, sample_rate, hop_size, db_thresh = -40, min_len = 5000):
    slicer = Slicer(
                sr=sample_rate,
                threshold=db_thresh,
                min_length=min_len)       
    chunks = dict(slicer.slice(audio))
    result = []
    for k, v in chunks.items():
        tag = v["split_time"].split(",")
        if tag[0] != tag[1]:
            start_frame = int(int(tag[0]) // hop_size)
            end_frame = int(int(tag[1]) // hop_size)
            if end_frame > start_frame:
                result.append((
                        start_frame, 
                        audio[int(start_frame * hop_size) : int(end_frame * hop_size)]))
    return result


def cross_fade(a: np.ndarray, b: np.ndarray, idx: int):
    result = np.zeros(idx + b.shape[0])
    fade_len = a.shape[0] - idx
    np.copyto(dst=result[:idx], src=a[:idx])
    k = np.linspace(0, 1.0, num=fade_len, endpoint=True)
    result[idx: a.shape[0]] = (1 - k) * a[idx:] + k * b[: fade_len]
    np.copyto(dst=result[a.shape[0]:], src=b[fade_len:])
    return result


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()

    #device = 'cpu' 
    device = cmd.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load ddsp model
    model_path_splitext = os.path.splitext(cmd.model_path)
    if len(model_path_splitext) < 2:
        raise ValueError(f" [x] no extension found in filename, skip process: {cmd.model_path}")
    if model_path_splitext[1] == '.onnx':
        model, args, spk_info = load_onnx_model(cmd.model_path, providers=cmd.onnx_providers.split(','))
        device = 'cpu'  # TODO: change device by onnx session providers
    else:
        model, args, spk_info = load_model(cmd.model_path, device=device)
    
    # load input
    audio, sample_rate = librosa.load(cmd.input, sr=None)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    hop_size = args.data.block_size * sample_rate / args.data.sampling_rate
    
    # get MD5 hash from wav file
    md5_hash = ""
    with open(cmd.input, 'rb') as f:
        data = f.read()
        md5_hash = hashlib.md5(data).hexdigest()
        print("MD5: " + md5_hash)
        
    interpolate_f0 = cmd.interpolate_f0 == 'true'
    
    cache_dir_path = os.path.join(os.path.dirname(__file__), "cache")
    cache_file_path = os.path.join(cache_dir_path, f"{cmd.pitch_extractor}_{hop_size}_{cmd.f0_min}_{cmd.f0_max}_{md5_hash}_{interpolate_f0}.npy")
    
    is_cache_available = os.path.exists(cache_file_path)
    if is_cache_available:
        # f0 cache load
        print('Loading pitch curves for input audio from cache directory...')
        f0 = np.load(cache_file_path, allow_pickle=False)
    else:
        # extract f0
        print('Pitch extractor type: ' + cmd.pitch_extractor)
        pitch_extractor = F0Extractor(
                            cmd.pitch_extractor, 
                            sample_rate, 
                            hop_size, 
                            float(cmd.f0_min), 
                            float(cmd.f0_max))
        print('Extracting the pitch curve of the input audio...')
        f0 = pitch_extractor.extract(audio, uv_interp = interpolate_f0, device = device)
        
        del pitch_extractor
        pitch_extractor = None
        
        # f0 cache save
        os.makedirs(cache_dir_path, exist_ok=True)
        np.save(cache_file_path, f0, allow_pickle=False)
    
    f0 = torch.from_numpy(f0).float().to(device).unsqueeze(-1).unsqueeze(0)
    
    f0_uv = f0 == 0
    
    # key change
    f0 = f0 * 2 ** (float(cmd.key) / 12)
    
    # intonation curve
    if cmd.intonation != "1.0":
        f0[~f0_uv] = f0[~f0_uv] * float(cmd.intonation) ** (((f0[~f0_uv] - float(cmd.f0_min))/(float(cmd.f0_max) - float(cmd.f0_min)))*(float(cmd.f0_max) - float(cmd.intonation_base)) / float(cmd.f0_max))
        
    if not interpolate_f0:
        f0[f0_uv] = torch.rand_like(f0[f0_uv])*float(args.data.sampling_rate/args.data.block_size) + float(args.data.sampling_rate/args.data.block_size)
    
    # extract volume 
    print('Extracting the volume envelope of the input audio...')
    volume_extractor = VolumeExtractor(
        hop_size, 1 if args.data.volume_window_size is None else args.data.volume_window_size)
    volume = volume_extractor.extract(audio)
    mask = (volume > 10 ** (float(cmd.threhold) / 20)).astype('float')
    mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
    mask = np.array([np.max(mask[n : n + 9]) for n in range(len(mask) - 8)])
    mask = torch.from_numpy(mask).float().to(device).unsqueeze(-1).unsqueeze(0)
    mask = upsample(mask, args.data.block_size).squeeze(-1)
    volume = torch.from_numpy(volume).float().to(device).unsqueeze(-1).unsqueeze(0)
    
    # load units encoder
    if args.data.encoder == 'cnhubertsoftfish':
        cnhubertsoft_gate = args.data.cnhubertsoft_gate
    else:
        cnhubertsoft_gate = 10
    units_encoder = None
    if not (args.model.distilled_stack or args.model.spec_in_stack or args.model.audio_in_stack):
        units_encoder = UnitsEncoder(
                            args.data.encoder, 
                            args.data.encoder_ckpt, 
                            args.data.encoder_sample_rate, 
                            args.data.encoder_hop_size,
                            skip_frames=0 if args.data.units_skip_frames is None else args.data.units_skip_frames,
                            extract_layers=args.model.units_layers,
                            device = device)
        
    # load speaker embed
    use_spk_embed = False
    if args.model.use_speaker_embed is not None and cmd.spk_embed != "None":
        # speaker embed or mix-speaker dictionary
        spk_mix_dict = literal_eval(cmd.spk_mix_dict)
        if spk_mix_dict is not None:
            print('Mix-speaker mode')
            if cmd.spk_embed != "None" or spk_info is None:
                spk_id = torch.stack([
                    torch.from_numpy(np.load(cmd.spk_embed, allow_pickle=True)[str(k)].item()['spk_embed'][np.newaxis, :]).float().to(device)
                    for k in spk_mix_dict.keys()
                ])
            else:
                spk_id = torch.stack([
                    spk_info[str(k)].item()['spk_embed'][np.newaxis, :]
                    for k in spk_mix_dict.keys()
                ])
            spk_mix = torch.tensor([[[float(v) for v in spk_mix_dict.values()]]]).transpose(-1, 0)
        else:
            print('Speaker ID: '+ str(int(cmd.spk_id)))
            if cmd.spk_embed != "None" or spk_info is None:
                spk_id = torch.from_numpy(np.load(cmd.spk_embed, allow_pickle=True)[cmd.spk_id].item()['spk_embed'][np.newaxis, :]).float().to(device).unsqueeze(0)
            else:
                spk_id = spk_info[cmd.spk_id].item()['spk_embed'][np.newaxis, :].unsqueeze(0)
            spk_mix = torch.tensor([[[1.]]])
        use_spk_embed = True
    else:
        # speaker id or mix-speaker dictionary
        spk_mix_dict = literal_eval(cmd.spk_mix_dict)
        if spk_mix_dict is not None:
            print('Mix-speaker mode')
            spk_id = torch.LongTensor(np.array([[int(k) for k in spk_mix_dict.keys()]]))
            spk_mix = torch.tensor([[float(v) for v in spk_mix_dict.values()]])
        else:
            print('Speaker ID: '+ str(int(cmd.spk_id)))
            spk_id = torch.LongTensor(np.array([[int(cmd.spk_id)]]))
            spk_mix = torch.tensor([[1.]])
    spk_id = spk_id.to(device)
    spk_mix = spk_mix.to(device)
            
    units_ratio = (args.data.block_size / args.data.sampling_rate) / (args.data.encoder_hop_size / args.data.encoder_sample_rate)
    units_sample_ratio = args.data.sampling_rate / args.data.encoder_sample_rate
        
    downsamples = int((units_sample_ratio + 1 - 1e-7) // 1)    # ceil
    downsampled_units_sample_ratio = (args.data.sampling_rate//downsamples) / args.data.encoder_sample_rate
    
    # forward and save the output
    result = np.zeros(0)
    current_length = 0
    segments = split(audio, sample_rate, hop_size)
    print('Cut the input audio into ' + str(len(segments)) + ' slices')
    with torch.no_grad():
        for segment in tqdm(segments):
            start_frame = segment[0]
            seg_input = torch.from_numpy(segment[1]).float().unsqueeze(0).to(device)
            seg_units = units_encoder.encode(seg_input, sample_rate, hop_size)
                    
           
            seg_f0 = f0[:, start_frame : start_frame + seg_units.size(1), :]
            seg_volume = volume[:, start_frame : start_frame + seg_units.size(1), :]
            
            
            seg_output = model(seg_units, seg_f0, seg_volume, spk_id=spk_id, spk_mix=spk_mix)
            
            seg_output *= mask[:, start_frame * args.data.block_size : (start_frame + seg_units.size(1)) * args.data.block_size]
            
            output_sample_rate = args.data.sampling_rate
            
            
            seg_output = seg_output.squeeze().cpu().numpy()
            
            
            silent_length = round(start_frame * args.data.block_size * output_sample_rate / args.data.sampling_rate) - current_length
            if silent_length >= 0:
                result = np.append(result, np.zeros(silent_length))
                result = np.append(result, seg_output)
            else:
                result = cross_fade(result, seg_output, current_length + silent_length)
            current_length = current_length + silent_length + len(seg_output)
        sf.write(cmd.output, result, output_sample_rate)
    