import os
import argparse
import torch

import shutil

from modules.common import load_config, load_model
from modules.dataset.loader import get_data_loaders
from modules.solver import train
from modules.vocoder import CombSubMinimumNoisedPhase, CombSubMinimumNoisedPhaseStackOnly
from modules.discriminator import MultiSpecDiscriminator
from modules.loss import RSSLoss, DSSLoss, DLFSSLoss


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    
    # load config
    args = load_config(cmd.config)
    print(' > config:', cmd.config)
    print(' >    exp:', args.env.expdir)

    # load model
    model = None
    if args.model.type == 'CombSubMinimumNoisedPhase':
        if args.train.only_u2c_stack:
            model = CombSubMinimumNoisedPhaseStackOnly(
                n_unit=args.data.encoder_out_channels,
                n_hidden_channels=args.model.units_hidden_channels
            )
        else:
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
                )
            if args.model.use_discriminator:
                model_d = MultiSpecDiscriminator()
            
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
    
    
    # load model parameters
    optimizer = torch.optim.AdamW(model.parameters())
    initial_global_step, model, optimizer, states = load_model(args.env.expdir, model, optimizer, device=args.device)
    
    lr = args.train.lr if states is None else states['last_lr'][0]
    
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.train.lr
        param_group['lr'] = lr
        param_group['weight_decay'] = args.train.weight_decay
        
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.train.sched_factor, patience=args.train.sched_patience,
                                                           threshold=args.train.sched_threshold, threshold_mode=args.train.sched_threshold_mode,
                                                           cooldown=args.train.sched_cooldown, min_lr=args.train.sched_min_lr)
    
    if states is not None:
        sched_states = states.get('scheduler')
        if sched_states is not None:
            scheduler.best = sched_states['best']
            scheduler.cooldown_counter = sched_states['cooldown_counter']
            scheduler.num_bad_epochs = sched_states['num_bad_epochs']
            scheduler._last_lr = sched_states['_last_lr']
    else:
        scheduler._last_lr = (lr,)
        
        
    # loss
    if args.loss.use_dual_scale:
        loss_func = DSSLoss(args.loss.fft_min, args.loss.fft_max,
                            beta=args.loss.beta, overlap=args.loss.overlap, device=args.device)
    elif args.loss.use_dual_scale_log_freq:
        loss_func = DLFSSLoss(args.loss.fft_min, args.loss.fft_max,
                            beta=args.loss.beta, overlap=args.loss.overlap, device=args.device)
    else:
        loss_func = RSSLoss(args.loss.fft_min, args.loss.fft_max, args.loss.n_scale, device=args.device)
        
        
    if args.model.use_discriminator:
        # load discriminator model parameters
        optimizer_d = torch.optim.AdamW(model_d.parameters())
        _, model_d, optimizer_d, states = load_model(args.env.expdir, model_d, optimizer_d, postfix="D_", device=args.device)
        lr = args.train.lr if states is None else states['last_lr'][0]
        
        for param_group in optimizer_d.param_groups:
            param_group['initial_lr'] = args.train.lr
            param_group['lr'] = lr
            param_group['weight_decay'] = args.train.weight_decay
            
        scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_d, mode='min', factor=args.train.sched_factor, patience=args.train.sched_patience,
                                                                 threshold=args.train.sched_threshold, threshold_mode=args.train.sched_threshold_mode,
                                                                 cooldown=args.train.sched_cooldown, min_lr=args.train.sched_min_lr)
    
        if states is not None:
            sched_states = states.get('scheduler')
            if sched_states is not None:
                scheduler_d.best = sched_states['best']
                scheduler_d.cooldown_counter = sched_states['cooldown_counter']
                scheduler_d.num_bad_epochs = sched_states['num_bad_epochs']
                scheduler_d._last_lr = sched_states['_last_lr']
        else:
            scheduler_d._last_lr = (lr,)
    else:
        model_d, optimizer_d, scheduler_d = None
        
    


    # device
    if args.device == 'cuda':
        torch.cuda.set_device(args.env.gpu_id)
    model.to(args.device)
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(args.device)
                    
    loss_func.to(args.device)


    # datas
    loaders = get_data_loaders(args)
    
    
    # copy spk_info
    if args.model.use_speaker_embed:
        shutil.copy2(os.path.join(args.data.dataset_path, 'spk_info.npz'), os.path.join(args.env.expdir, 'spk_info.npz'))
    
    
    # run
    train(args, initial_global_step, (model, optimizer, scheduler, loss_func), (model_d, optimizer_d, scheduler_d), loaders['train'], loaders['test'])
    
