import os
import torch
import librosa
import argparse
import numpy as np
import soundfile as sf
# import pyworld as pw
# import parselmouth
import hashlib
import torchaudio
from ast import literal_eval
from slicer import Slicer
from modules.vocoder import load_model
from modules.extractors import F0Extractor, VolumeExtractor, UnitsEncoder
from modules.extractors.common import upsample
# from enhancer import Enhancer
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
    # parser.add_argument(
    #     "-e",
    #     "--enhance",
    #     type=str,
    #     required=False,
    #     default='true',
    #     help="true or false | default: true",
    # )
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
    # parser.add_argument(
    #     "-eak",
    #     "--enhancer_adaptive_key",
    #     type=str,
    #     required=False,
    #     default=0,
    #     help="adapt the enhancer to a higher vocal range (number of semitones) | default: 0",
    # )
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
        # set unvoiced f0 to fixed samplerate/blocksize*2
        # f0[f0_uv] = sample_rate/hop_size*2.
        # f0[f0_uv] = args.data.f0_min*.3
        # f0[f0_uv] = sample_rate/hop_size
        # f0[f0_uv] = 1.
        # f0[f0_uv] = args.data.sampling_rate/args.model.win_length
        # f0[f0_uv] = args.data.sampling_rate/args.data.block_size
        f0[f0_uv] = torch.rand_like(f0[f0_uv])*float(args.data.sampling_rate/args.data.block_size) + float(args.data.sampling_rate/args.data.block_size)
        # f0[f0_uv] = torch.rand_like(f0[f0_uv])*float(args.data.f0_min) + float(args.data.sampling_rate/args.data.block_size)
        # f0[f0_uv] = torch.rand_like(f0[f0_uv])*float(args.data.f0_min)*.2 + float(args.data.f0_min)*.2
        # f0[f0_uv] = torch.rand_like(f0[f0_uv])*(sample_rate/hop_size*.5) + (sample_rate/hop_size)
        # f0[f0_uv] = torch.rand_like(f0[f0_uv])*(sample_rate/hop_size) + (sample_rate/hop_size*.5)
    
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
        if cmd.spk_embed != "None" or spk_info is None:
            spk_embed = np.load(cmd.spk_embed, allow_pickle=True)[cmd.spk_id].item()['spk_embed']
        else:
            spk_embed = spk_info[cmd.spk_id].item()['spk_embed']
        spk_embed = torch.from_numpy(spk_embed).float().to(device).unsqueeze(0)
        use_spk_embed = True
        spk_mix_dict = None # TODO: impl
    else:
        # speaker id or mix-speaker dictionary
        spk_mix_dict = literal_eval(cmd.spk_mix_dict)
        if spk_mix_dict is not None:
            print('Mix-speaker mode')
            spk_id = torch.LongTensor(np.array([int(k) for k in spk_mix_dict.keys()])).to(device)
        else:
            print('Speaker ID: '+ str(int(cmd.spk_id)))
            spk_id = torch.LongTensor(np.array([[int(cmd.spk_id)]])).to(device)
            
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
            if args.model.spec_in_stack:
                spec = torchaudio.functional.spectrogram(
                        seg_input,
                        pad=0,
                        window=torch.hann_window(args.data.block_size*2).to(seg_input),
                        n_fft=args.data.block_size*2,
                        hop_length=args.data.block_size,
                        win_length=args.data.block_size*2,
                        power=1,
                        normalized=False).transpose(2, 1)[:, 1:, :args.data.block_size//2]
                if args.model.distil_units_layers is None:
                    seg_units = spec
                else:
                    seg_units = model.unit2ctrl.units_distillator(spec.transpose(2, 1)).transpose(2, 1)
            # elif not args.model.distilled_stack:
            #     seg_units = units_encoder.encode(seg_input, sample_rate, hop_size)
            elif args.model.audio_in_stack:
                seg_audio = seg_input.view(seg_input.shape[0], -1, args.data.block_size)
                if args.model.distil_units_layers is None:
                    seg_units = seg_audio
                else:
                    # seg_units = model.unit2ctrl.units_distillator(seg_audio.transpose(2, 1)).transpose(2, 1)
                    downsampled_audio = seg_input[:, ::downsamples]
                    # alignment
                    fold_step = int(downsampled_units_sample_ratio*args.data.encoder_hop_size)
                    # audio = torch.nn.functional.pad(downsampled_audio, (fold_step//2, fold_step//2)).unfold(1, args.data.encoder_hop_size, fold_step-1).flatten(1).unsqueeze(-1)
                    # audio = torch.nn.functional.pad(downsampled_audio, (0, fold_step)).unfold(1, args.data.encoder_hop_size, fold_step-1).flatten(1).unsqueeze(-1)
                    audio = downsampled_audio.unfold(1, args.data.encoder_hop_size, fold_step).flatten(1).unsqueeze(-1)
                    units = model.unit2ctrl.units_distillator(audio.transpose(2, 1))
                    # n_frames = data['audio'].size(1) // args.data.block_size + 1
                    n_frames = seg_input.size(1) // args.data.block_size
                    index = torch.clamp(torch.round(units_ratio * torch.arange(n_frames).to(units)).long(), max = units.size(1) - 1)
                    repeats = [units.size(0), 1, units.size(-1)]
                    index = index.unsqueeze(0).unsqueeze(-1).repeat(repeats)
                    seg_units = torch.gather(units, 1, index)
            else:
                seg_units = units_encoder.encode(seg_input, sample_rate, hop_size)
                    
           
            seg_f0 = f0[:, start_frame : start_frame + seg_units.size(1), :]
            seg_volume = volume[:, start_frame : start_frame + seg_units.size(1), :]
            
            if not use_spk_embed:
                seg_output, _, (s_h, s_n) = model(seg_units, seg_f0, seg_volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict)
            else:
                seg_output, _, (s_h, s_n) = model(seg_units, seg_f0, seg_volume, spk_id=spk_embed, spk_mix_dict=spk_mix_dict)
            seg_output *= mask[:, start_frame * args.data.block_size : (start_frame + seg_units.size(1)) * args.data.block_size]
            
            # if cmd.enhance == 'true':
            #     seg_output, output_sample_rate = enhancer.enhance(
            #                                                 seg_output, 
            #                                                 args.data.sampling_rate, 
            #                                                 seg_f0, 
            #                                                 args.data.block_size, 
            #                                                 adaptive_key = cmd.enhancer_adaptive_key)
            # else:
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
    