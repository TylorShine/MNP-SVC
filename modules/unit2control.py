import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from .convnext_v2_like import GRN, LayerNorm1d, ConvNeXtV2LikeEncoder


def split_to_dict(tensor, tensor_splits):
    """Split a tensor into a dictionary of multiple tensors."""
    labels = []
    sizes = []

    for k, v in tensor_splits.items():
        labels.append(k)
        sizes.append(v)

    tensors = torch.split(tensor, sizes, dim=-1)
    return dict(zip(labels, tensors))


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    
    def forward(self, x):
        return x.transpose(*self.dims)


class Unit2ControlGE2E_onnx:
    def __init__(self, onnx_sess, output_splits, device='cpu'):
        self.sess = onnx_sess
        self.output_splits = output_splits
        self.device = device
    
    def __call__(self, units, f0, phase, volume, spk_id = None, spk_mix = None, aug_shift = None):
        e = self.sess.run(['signal'],
                             {'units': units.cpu().numpy(),
                              'f0': f0.cpu().numpy(),
                              'phase': phase.cpu().numpy(),
                              'volume': volume.cpu().numpy(),
                              'spk_id': spk_id.cpu().numpy(),
                              'spk_mix': spk_mix.cpu().numpy()})[0]
        
        return split_to_dict(torch.from_numpy(e).to(self.device), self.output_splits), None
        
    

class Unit2ControlGE2E(nn.Module):
    def __init__(
            self,
            input_channel,
            spk_embed_channels,
            output_splits,
            n_hidden_channels=256,
            n_spk=1024,
            use_pitch_aug=False,
            use_spk_embed=True,
            use_embed_conv=True,
            embed_conv_channels=64,
            conv_stack_middle_size=32):
        super().__init__()
        
        self.output_splits = output_splits
        self.use_embed_conv = use_embed_conv
        self.f0_embed = nn.Linear(1, n_hidden_channels)
        self.phase_embed = nn.Linear(1, n_hidden_channels)
        self.volume_embed = nn.Linear(1, n_hidden_channels)
        self.n_hidden_channels = n_hidden_channels
        self.spk_embed_channels = spk_embed_channels
        self.use_spk_embed = use_spk_embed
        if use_spk_embed:
            # TODO: experiment with what happens if we don't use embed convs
            if use_embed_conv:
                self.spk_embed_conv = nn.Sequential(
                    nn.Conv1d(n_hidden_channels, embed_conv_channels, 3, 1, 1, bias=False),
                    nn.Sequential(
                        Transpose((2, 1)),
                        LayerNorm1d(embed_conv_channels, eps=1e-6),
                        nn.GELU(),
                        GRN(embed_conv_channels)),
                    nn.Linear(embed_conv_channels, n_hidden_channels))
                nn.init.kaiming_normal_(self.spk_embed_conv[0].weight)
                nn.init.normal_(self.spk_embed_conv[-1].weight, 0, 0.01)
                nn.init.constant_(self.spk_embed_conv[-1].bias, 0)
            self.spk_embed = nn.Linear(spk_embed_channels, n_hidden_channels)
        else:
            if use_embed_conv:
                self.spk_embed_conv = nn.Sequential(
                    nn.Conv1d(n_hidden_channels, embed_conv_channels, 3, 1, 1, bias=False),
                    nn.Sequential(
                        Transpose((2, 1)),
                        LayerNorm1d(embed_conv_channels, eps=1e-6),
                        nn.GELU(),
                        GRN(embed_conv_channels)),
                    nn.Linear(embed_conv_channels, n_hidden_channels))
                nn.init.kaiming_normal_(self.spk_embed_conv[0].weight)
                nn.init.normal_(self.spk_embed_conv[-1].weight, 0, 0.01)
                nn.init.constant_(self.spk_embed_conv[-1].bias, 0)
            self.spk_embed = nn.Embedding(n_spk, n_hidden_channels)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, n_hidden_channels, bias=False)
        else:
            self.aug_shift_embed = None
            
        # conv in stack
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, conv_stack_middle_size, 3, 1, 1),
                nn.Sequential(
                    Transpose((2, 1)),
                    LayerNorm1d(conv_stack_middle_size, eps=1e-6),
                    nn.GELU(),
                    GRN(conv_stack_middle_size)),
                nn.Linear(conv_stack_middle_size, n_hidden_channels),)
        nn.init.normal_(self.stack[-1].weight, 0, 0.01)
        nn.init.constant_(self.stack[-1].bias, 0)

        # transformer
        self.decoder = ConvNeXtV2LikeEncoder(
            num_layers=3,
            dim_model=n_hidden_channels,
            kernel_size=31,
            bottoleneck_dilation=2)
            
        self.norm = nn.LayerNorm(n_hidden_channels)

        # out
        output_splits: dict
        self.n_out = sum([v for v in output_splits.values()])
        self.dense_out = weight_norm(nn.Linear(n_hidden_channels, self.n_out), dim=0)

    def forward(self, units, f0, phase, volume, spk_id = None, spk_mix = None, aug_shift = None):
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        
        x = self.volume_embed(volume)
        
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
            
        # frequency => harmonically linear: log2
        f0_emb = self.f0_embed((1+ f0 / 700).log2())
        phase_emb = self.phase_embed(phase / torch.pi)
        
        if not self.use_embed_conv:
            x = x + f0_emb + phase_emb
                    
        if spk_mix is not None:
            if self.use_embed_conv:
                if self.use_spk_embed:
                    # for i, (k, v) in enumerate(spk_mix_dict.items()):
                    for i in range(spk_id.shape[0]):
                        x = x + spk_mix[i] * self.spk_embed_conv(
                            self.spk_embed(spk_id[i]).unsqueeze(1).expand(x.shape[0], x.shape[1], self.n_hidden_channels).transpose(2, 1) +
                            f0_emb.transpose(2, 1) + 
                            phase_emb.transpose(2, 1))
                else:
                    for i in range(spk_id.shape[0]):
                        x = x + spk_mix[i] * self.spk_embed_conv(
                            self.spk_embed(spk_id[i]).expand(x.shape[0], x.shape[1], self.n_hidden_channels).transpose(2, 1) +
                            f0_emb.transpose(2, 1) + 
                            phase_emb.transpose(2, 1))
            else:
                if self.use_spk_embed:
                    for i in range(spk_id.shape[0]):
                        x = x + spk_mix[i] * self.spk_embed(spk_id[i]).unsqueeze(1).expand(x.shape[0], x.shape[1], self.n_hidden_channels)
                else:
                    for i in range(spk_id.shape[0]):
                        x = x + spk_mix[i] * self.spk_embed(spk_id[i]).expand(x.shape[0], x.shape[1], self.n_hidden_channels)
        else:
            if self.use_embed_conv:
                if self.use_spk_embed:
                    x = x + self.spk_embed_conv(
                        self.spk_embed(spk_id).unsqueeze(1).expand(x.shape[0], x.shape[1], self.n_hidden_channels).transpose(2, 1) +
                        f0_emb.transpose(2, 1) +
                        phase_emb.transpose(2, 1))
                else:
                    x = x + self.spk_embed_conv(
                        self.spk_embed(spk_id).expand(x.shape[0], x.shape[1], self.n_hidden_channels).transpose(2, 1) +
                        f0_emb.transpose(2, 1) +
                        phase_emb.transpose(2, 1))
            else:
                if self.use_spk_embed:
                    x = x + self.spk_embed(spk_id).unsqueeze(1).expand(x.shape[0], x.shape[1], self.n_hidden_channels)
                else:
                    x = x + self.spk_embed(spk_id).expand(x.shape[0], x.shape[1], self.n_hidden_channels)
                
        x = x + self.stack(units.transpose(2, 1))
        
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        
        controls = split_to_dict(e, self.output_splits)
        
        return controls, x
    
    
class Unit2ControlGE2E_export(Unit2ControlGE2E):
    def __init__(self,
            input_channel,
            spk_embed_channels,
            output_splits,
            n_hidden_channels=256,
            n_spk=1024,
            use_pitch_aug=False,
            use_spk_embed=True,
            use_embed_conv=True,
            embed_conv_channels=64,
            conv_stack_middle_size=32):
        super().__init__(
            input_channel,
            spk_embed_channels,
            output_splits,
            n_hidden_channels=n_hidden_channels,
            n_spk=n_spk,
            use_pitch_aug=use_pitch_aug,
            use_spk_embed=use_spk_embed,
            use_embed_conv=use_embed_conv,
            embed_conv_channels=embed_conv_channels,
            conv_stack_middle_size=conv_stack_middle_size)
        
        
    def forward(self, units, f0, phase, volume, spk_id = None, spk_mix = None, aug_shift = None):
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        
        x = self.volume_embed(volume)
        
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
            
        # frequency => harmonically linear: log2
        f0_emb = self.f0_embed((1+ f0 / 700).log2())
        phase_emb = self.phase_embed(phase / torch.pi)
        
        if not self.use_embed_conv:
            x = x + f0_emb + phase_emb
            
        # NOTE: batching is not supported
        if self.use_embed_conv:
            # spk_ids = torch.stack(spk_id)
            # spk_mixes = torch.stack(spk_mix)
            if self.use_spk_embed:
                x = x + torch.sum(
                    self.spk_embed_conv(
                        self.spk_embed(spk_id).expand(spk_id.shape[0], x.shape[1], self.n_hidden_channels).transpose(-1, -2) +
                        f0_emb.repeat(spk_id.shape[0], 1, 1).transpose(2, 1) +
                        phase_emb.repeat(spk_id.shape[0], 1, 1).transpose(2, 1)) * spk_mix.repeat(1, x.shape[1], self.n_hidden_channels),
                    dim=0)
            else:
                x = x + torch.sum(
                    self.spk_embed_conv(
                        self.spk_embed(spk_id).expand(spk_id.shape[0], x.shape[0], self.n_hidden_channels).transpose(-1, -2) +
                        f0_emb.repeat(spk_id.shape[0], 1, 1).transpose(2, 1) +
                        phase_emb.repeat(spk_id.shape[0], 1, 1).transpose(2, 1)) * spk_mix.repeat(1, x.shape[1], self.n_hidden_channels),
                    dim=0)
                
        x = x + self.stack(units.transpose(2, 1))
        
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        
        return e
    
    
class Unit2ControlStackOnly(nn.Module):
    def __init__(
        self,
        input_channel,
        n_hidden_channels=256,
        conv_stack_middle_size=32,
        ):
        super().__init__()
        
        self.n_hidden_channels = n_hidden_channels
        
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, conv_stack_middle_size, 3, 1, 1),
                nn.Sequential(
                    Transpose((2, 1)),
                    LayerNorm1d(conv_stack_middle_size, eps=1e-6),
                    nn.GELU(),
                    GRN(conv_stack_middle_size)))
                # nn.Linear(conv_stack_middle_size, n_hidden_channels),)
        nn.init.kaiming_normal_(self.stack[0].weight)
        
    def forward(self, units):
        return self.stack(units.transpose(2, 1))