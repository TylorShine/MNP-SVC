import os
import argparse

import torch

from modules.common import load_config, load_model
from modules.vocoder import CombSubMinimumNoisedPhase


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="path to the trained .pt file")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="path to the output onnx file. if not provided, use <input>-mnp.onnx")
    parser.add_argument(
        "-s",
        "--no-simplify-onnx",
        action="store_true",
        help="no simplify onnx by onnxsim",
    )
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    
    input_file, output_file = cmd.input, cmd.output
    
    if output_file == "":
        # build output filename
        input_file_basename = os.path.basename(input_file)
        output_file = os.path.join(
            os.path.dirname(input_file),
            f'{os.path.splitext(input_file_basename)[0]}.onnx')
        
    # load config
    config_file = os.path.join(os.path.dirname(input_file), 'config.yaml')
    args = load_config(config_file)
    print(' > config:', config_file)
        
    # load model
    model = None
    if args.model.type == 'CombSubMinimumNoisedPhase':
        model = CombSubMinimumNoisedPhase(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            win_length=args.model.win_length,
            n_unit=args.data.encoder_out_channels,
            n_hidden_channels=args.model.units_hidden_channels,
            n_spk=args.model.n_spk,
            use_speaker_embed=args.model.use_speaker_embed,
            use_embed_conv=not args.model.no_use_embed_conv,
            spk_embed_channels=args.data.spk_embed_channels,
            f0_input_variance=args.model.f0_input_variance,
            f0_offset_size_downsamples=args.model.f0_offset_size_downsamples,
            noise_env_size_downsamples=args.model.noise_env_size_downsamples,
            harmonic_env_size_downsamples=args.model.harmonic_env_size_downsamples,
            use_harmonic_env=args.model.use_harmonic_env,
            use_noise_env=args.model.use_noise_env,
            noise_to_harmonic_phase=args.model.noise_to_harmonic_phase,
            use_f0_offset=args.model.use_f0_offset,
            use_pitch_aug=args.model.use_pitch_aug,
            noise_seed=args.model.noise_seed,
            export_onnx=True)
            
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
    
    
    # load parameters
    ckpt = torch.load(input_file, map_location='cpu')
    if 'model' not in ckpt:
        raise ValueError(f" [x] Unsupported Model File: {input_file}")
    model.load_state_dict(ckpt['model'], strict=False)
    
    model.eval()
    
    # build dummy inputs
    dummy_inputs = {
        'units': torch.randn(1, 100, args.data.encoder_out_channels),
        'f0': torch.randn(1, 100, 1),
        'phase': torch.randn(1, 100, 1),
        'volume': torch.randn(1, 100, 1),
        'spk_id': torch.LongTensor([[i for i in args.model.n_spk]])
                    if not args.model.use_speaker_embed else
                    torch.randn(1, 1, args.data.spk_embed_channels),
        'spk_mix': torch.randn(args.model.n_spk)
                    if not args.model.use_speaker_embed else
                    torch.randn(1, 1, 1),        
    }
    
    # export onnx
    torch.onnx.export(
        model=model.unit2ctrl,
        args=tuple(dummy_inputs.values()),
        f=output_file,
        input_names=list(dummy_inputs.keys()),
        output_names=['signal'],
        dynamic_axes={
            'units': [1],
            'f0': [1],
            'phase': [1],
            'volume': [1],
            'spk_id': [0],
            'spk_mix': [0],
        },
        opset_version=10)
    
    if cmd.no_simplify_onnx:
        print(f'successful to export onnx: {output_file}')
    else:
        import onnx
        from onnxsim import simplify
        model = onnx.load(output_file)
        
        model, check = simplify(model)
        
        assert check, "Simplified ONNX model could not be validated"
        
        onnx.save(model, output_file)
        
        print()
        print(f'successful to export onnx(simplified): {output_file}')
        
        
    
    # onnx_program = torch.onnx.dynamo_export(
    #     model.unit2ctrl,
    #     **dummy_inputs,
    #     export_options=torch.onnx.ExportOptions(dynamic_shapes=True))
    
    # onnx_program.save(output_file)
    