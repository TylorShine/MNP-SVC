import os

import onnxruntime
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from modules.common import DotDict

from ...convnext_v2_like import GRN, ConvNeXtV2GLULikeEncoder


def load_model(
        model_path,
        device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    # load model
    model = None
    if args.model.type == 'melphen':
        model = Melphen(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            in_channels=args.model.in_channels,
            hidden_channels=args.model.hidden_channels,
            out_channels=args.model.out_channels)
            
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
    
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    return model, args


def load_onnx_model(
            model_path,
            providers=['CPUExecutionProvider'],
            device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    sess = onnxruntime.InferenceSession(
        model_path,
        providers=providers)
    
    # load model
    model = None
    if args.model.type == 'melphen':
        model = Melphen(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            in_channels=args.model.in_channels,
            hidden_channels=args.model.hidden_channels,
            out_channels=args.model.out_channels,
            device=device)
           
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
    
    return model, args


class Melphen(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            in_channels=128,
            hidden_channels=256,
            out_channels=128):
        super().__init__()
        
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        
        self.conv_in = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            torch.nn.InstanceNorm1d(hidden_channels),
            torch.nn.CELU(),
            torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        )
        self.encoder = ConvNeXtV2GLULikeEncoder(
            num_layers=3,
            dim_model=hidden_channels,
            kernel_size=7,
            bottoleneck_dilation=2,
        )
        self.norm = torch.nn.LayerNorm(hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, out_channels + 1)
        

    def forward(self, spec_frames, **kwargs):
        '''
            units_frames: B x n_frames x n_unit
            f0_frames: B x n_frames x 1
            volume_frames: B x n_frames x 1 
            spk_id: B x 1
        '''
        c = self.conv_in(spec_frames.transpose(2, 1)).transpose(2, 1)
        c = self.encoder(c)
        c = self.norm(c)
        o = self.out(c)
        
        o[:,:,-1] = 2. ** (o[:,:,-1] + 5.)
        
        return o
    