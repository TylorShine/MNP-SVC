import os
import math
import argparse

import torch

from modules.common import load_config, load_model
# from modules.vocoder import CombSubMinimumNoisedPhase
from modules.vocoder import CombSubMinimumNoisedPhase_export


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
        "--simplify-onnx",
        action="store_true",
        help="simplify onnx by onnxsim",
    )
    parser.add_argument(
        "-q",
        "--quantization",
        action="store_true",
        help="enable onnxruntime quantization",
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
        # model = CombSubMinimumNoisedPhase(
        model = CombSubMinimumNoisedPhase_export(
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
            use_add_noise_env=args.model.use_add_noise_env,
            noise_to_harmonic_phase=args.model.noise_to_harmonic_phase,
            add_noise=args.model.add_noise,
            use_phase_offset=args.model.use_phase_offset,
            use_f0_offset=args.model.use_f0_offset,
            no_use_noise=args.model.no_use_noise,
            use_short_filter=args.model.use_short_filter,
            use_noise_short_filter=args.model.use_noise_short_filter,
            use_pitch_aug=args.model.use_pitch_aug,
            noise_seed=args.model.noise_seed,
            export_onnx=True)
        # )
            
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
    
    
    # load parameters
    ckpt = torch.load(input_file, map_location='cpu')
    # if 'model' not in ckpt:
    #     raise ValueError(f" [x] Unsupported Model File: {input_file}")
    # model.load_state_dict(ckpt['model'], strict=False)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    
    model.eval()
    
    # build dummy inputs
    # dummy_inputs = {
    #     'units_frames': torch.randn(1, 100, args.data.encoder_out_channels),
    #     'f0_frames': torch.randn(1, 100, 1),
    #     # 'phase': torch.randn(1, 100, 1),
    #     'volume_frames': torch.randn(1, 100, 1),
    #     'spk_id': torch.LongTensor([[i for i in args.model.n_spk]])
    #                 if not args.model.use_speaker_embed else
    #                 torch.randn(1, 1, args.data.spk_embed_channels),
    #     'spk_mix': torch.randn(args.model.n_spk)
    #                 if not args.model.use_speaker_embed else
    #                 torch.randn(1, 1, 1),        
    # }
    
    static_frames = math.ceil(args.data.duration*args.data.sampling_rate/args.data.block_size)
    # static_frames = torch.randint(low=1, high=512, size=(1,))[0]
    # static_frames = 130
    
    dummy_inputs = {
        'units_frames': torch.randn(1, static_frames, args.data.encoder_out_channels),
        'f0_frames': torch.randn(1, static_frames, 1),
        # 'phase': torch.randn(1, 100, 1),
        'volume_frames': torch.randn(1, static_frames, 1),
        'spk_id': torch.LongTensor([[i for i in args.model.n_spk]])
                    if not args.model.use_speaker_embed else
                    torch.randn(1, 1, args.data.spk_embed_channels),
        'spk_mix': torch.randn(args.model.n_spk)
                    if not args.model.use_speaker_embed else
                    torch.randn(1, 1, 1),        
    }
    
    # dummy_inputs = {
    #     'units_frames': torch.randn(1, 100, args.data.encoder_out_channels),
    #     'f0_frames': torch.randn(1, 100, 1),
    #     # 'phase': torch.randn(1, 100, 1),
    #     'volume_frames': torch.randn(1, 100, 1),
    #     'spk_id': torch.LongTensor([[i for i in args.model.n_spk]])
    #                 if not args.model.use_speaker_embed else
    #                 torch.randn(1, 1, args.data.spk_embed_channels),
    #     'spk_mix': torch.randn(args.model.n_spk)
    #                 if not args.model.use_speaker_embed else
    #                 torch.randn(1, 1, 1),        
    # }
    
    # # export onnx
    # torch.onnx.export(
    #     model=model.unit2ctrl,
    #     args=tuple(dummy_inputs.values()),
    #     f=output_file,
    #     input_names=list(dummy_inputs.keys()),
    #     output_names=['signal'],
    #     dynamic_axes={
    #         'units': [1],
    #         'f0': [1],
    #         'phase': [1],
    #         'volume': [1],
    #         'spk_id': [0],
    #         'spk_mix': [0],
    #     },
    #     opset_version=10)
    
    # if cmd.no_simplify_onnx:
    #     print(f'successful to export onnx: {output_file}')
    # else:
    #     import onnx
    #     from onnxsim import simplify
    #     model = onnx.load(output_file)
        
    #     model, check = simplify(model)
        
    #     assert check, "Simplified ONNX model could not be validated"
        
    #     onnx.save(model, output_file)
        
    #     print()
    #     print(f'successful to export onnx(simplified): {output_file}')
    
    # print(list(dummy_inputs.keys()))
    
    with torch.no_grad():
        torch.onnx.export(
            model=model,
            args=tuple(dummy_inputs.values()),
            # args=(dummy_inputs,),
            f=output_file,
            input_names=list(dummy_inputs.keys()),
            output_names=['signal'],
            dynamic_axes={
                # 'units_frames': [1],
                # 'f0_frames': [1],
                # 'volume_frames': [1],
                # 'spk_id': [0],
                # 'spk_mix': [0],
                'units_frames': {1: 'frames'},
                'f0_frames': {1: 'frames'},
                'volume_frames': {1: 'frames'},
                'spk_id': {0: 'spk_id_index'},
                'spk_mix': {0: 'spk_mix_index'},
            },
            opset_version=13)
        
    print(f'successful to export onnx: {output_file}')
        
    
    if cmd.simplify_onnx:
        import onnx
        from onnxsim import simplify
        model = onnx.load(output_file)
        
        model, check = simplify(model)
        
        assert check, "Simplified ONNX model could not be validated"
            
        onnx.save(model, output_file)
        
        print()
        print(f'successful to export onnx(simplified): {output_file}')
    else:
        print(f'successful to export onnx: {output_file}')
        
    if cmd.quantization:
        # quantize the model (TODO: now not to work. no use dynamic shapes needed but...)
        
        from onnxruntime import quantization
        
        class QuantizationDummyDataReader(quantization.CalibrationDataReader):
            def __init__(self, dummy_inputs, max_size=32):
                self.inputs = dummy_inputs
                
                self.max_size = max_size
                self.index = 0

            def to_numpy(self, pt_tensor):
                return pt_tensor.detach().cpu().numpy() if pt_tensor.requires_grad else pt_tensor.cpu().numpy()

            def get_next(self):
                self.index += 1
                if self.index >= self.max_size:
                    return None
                return {
                    k: self.to_numpy(torch.rand_like(v))
                    for k, v in self.inputs.items()
                }

            def rewind(self):
                self.index = 0

        qddr = QuantizationDummyDataReader(dummy_inputs=dummy_inputs)

        quantized_model_path = output_file.replace(".onnx", "_quantized.onnx")
        
        # Exception: Incomplete symbolic shape inference. Quantization requires complete shape information.
        # quantization.quant_pre_process(output_file, quantized_model_path) # we cannot use dynamic shape inputs...
        
        quantization.quantize_static(
            # quantized_model_path,
            output_file,
            quantized_model_path,
            calibration_data_reader=qddr,
            extra_options={
                "ActivationSymmetric": False,   # set True for GPU infer
                "WeightSymmetric": True})
        
        # torch.quantization.quantize(
        #     model,
        #     run_fn=model,
        #     run_args=tuple(dummy_inputs.values()),
        #     inplace=True)
        
        # print(f"Model quantized")
    
    # with torch.no_grad():
    #     # batch = torch.export.Dim('batch', min=1)
    #     # frames = torch.export.Dim('frames', min=1)
    #     # spk_id_index = torch.export.Dim('spk_id_index', min=1)
    #     # encoder_out_channels = torch.export.dy .Dim('encoder_out_channels', min=1)
    #     frames = torch.export.Dim('frames', min=1)
    #     spk_id_index = torch.export.Dim('spk_id_index', min=1)
        
    #     # constraints = [
    #     #     torch.export.dynamic_dim(dummy_inputs['units_frames'], 1) >= 1,
    #     #     torch.export.dynamic_dim(dummy_inputs['f0_frames'], 1) >= 1,
    #     #     torch.export.dynamic_dim(dummy_inputs['volume_frames'], 1) >= 1,
    #     #     torch.export.dynamic_dim(dummy_inputs['spk_id'], 1) >= 1,
    #     #     torch.export.dynamic_dim(dummy_inputs['spk_mix'], 1) >= 1,
    #     # ]
        
    #     exported_program = torch.export.export(
    #         model,
    #         args=(),
    #         kwargs=dummy_inputs,
    #         # args=tuple(dummy_inputs.values()),
    #         # dynamic_shapes={
    #         #     'units_frames': {1: frames},
    #         #     'f0_frames': {1: frames},
    #         #     'volume_frames': {1: frames},
    #         #     'spk_id': {0: spk_id_index},
    #         #     'spk_mix': {0: spk_id_index},
    #         # },
    #         # constraints=constraints,
    #         dynamic_shapes={
    #             'units_frames': {1: frames, 2: args.data.encoder_out_channels},
    #             'f0_frames': {1: frames, 2: 1},
    #             'volume_frames': {1: frames, 2: 1},
    #             'spk_id': {1:spk_id_index, 2: args.data.spk_embed_channels},
    #             'spk_mix': {1:spk_id_index, 2: 1},
    #         },
    #     )
        
    #     onnx_program = torch.onnx.dynamo_export(
    #         exported_program,
    #         export_options=torch.onnx.ExportOptions(dynamic_shapes=False),
    #     )
        
    #     print(onnx_program.model_proto.graph.input[0])
        
    #     onnx_program.save(output_file)
        
    
    # onnx_program = torch.onnx.dynamo_export(
    #     model,
    #     **dummy_inputs,
    #     export_options=torch.onnx.ExportOptions(dynamic_shapes=True))
    
    # print(onnx_program.model_proto.graph.input[0])
    
    # onnx_program.save(output_file)
    