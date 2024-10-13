import os
import argparse
import torch

import shutil

from modules.common import load_config
from modules.dataset.loader import get_data_loaders
from modules.solver import train
from modules.vocoder import CombSubMinimumNoisedPhase, CombSubMinimumNoisedPhaseStackOnly, NMPSFHiFi
from modules.discriminator import (
    MultiSpecDiscriminator,
    MultiPeriodSpecDiscriminator
)
from modules.loss import (
    RSSLoss, DSSLoss, DLFSSLoss,
    MLFSSLoss,
    DSVWSLoss, DLFSVWSLoss,
    DLFSSMPLoss,
    MelLoss, MultiScaleMelSpectrogramLoss
)


torch.backends.cudnn.benchmark = True


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
    
    vocoder = None
    
    if args.train.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

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
                )
            # model.unit2ctrl.compile(mode="reduce-overhead")
            if args.model.use_discriminator:
                model_d = MultiPeriodSpecDiscriminator()
            
    elif args.model.type == 'NMPSFHiFi':
        model = NMPSFHiFi(
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
            add_noise=args.model.add_noise,
            use_f0_offset=args.model.use_f0_offset,
            use_short_filter=args.model.use_short_filter,
            use_noise_short_filter=args.model.use_noise_short_filter,
            use_pitch_aug=args.model.use_pitch_aug,
            nsf_hifigan_in=args.model.nsf_hifigan.num_mels,
            nsf_hifigan_h=args.model.nsf_hifigan,
            noise_seed=args.model.noise_seed,
            )
        model.unit2ctrl.compile()
        model.generator.compile()
        # model.unit2ctrl.compile(mode="reduce-overhead")
        # model.generator.compile(mode="reduce-overhead")
        if args.model.use_discriminator:
            model_d = MultiSpecDiscriminator()
            
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay/args.train.lr)
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.train.sched_gamma)
    
        
    # loss
    if args.loss.use_dual_scale:
        loss_func = DSSLoss(args.loss.fft_min, args.loss.fft_max,
                            beta=args.loss.beta, overlap=args.loss.overlap, device=args.device)
    elif args.loss.use_multi_scale_log_freq:
        loss_func = MLFSSLoss(n_ffts=args.loss.n_ffts, beta=args.loss.beta,
                              overlap=args.loss.overlap, device=args.device)
    elif args.loss.use_dual_scale_variwindow:
        loss_func = DSVWSLoss(args.loss.fft_min, args.loss.fft_max,
                              n_fft=args.loss.n_fft, beta=args.loss.beta, overlap=args.loss.overlap, device=args.device)
    elif args.loss.use_dual_scale_log_freq:
        loss_func = DLFSSLoss(args.loss.fft_min, args.loss.fft_max,
                            beta=args.loss.beta, overlap=args.loss.overlap, device=args.device)
    elif args.loss.use_dual_scale_log_freq_variwindow:
        loss_func = DLFSVWSLoss(args.loss.fft_min, args.loss.fft_max,
                            n_fft=args.loss.n_fft, beta=args.loss.beta, overlap=args.loss.overlap, device=args.device)
    elif args.loss.use_dual_scale_log_freq_magphase:
        loss_func = DLFSSMPLoss(args.loss.fft_min, args.loss.fft_max,
                            beta=args.loss.beta, gamma=args.loss.gamma, overlap=args.loss.overlap, device=args.device)
    elif args.loss.use_mel:
        loss_func = MelLoss(n_fft=args.loss.n_fft, n_mels=args.loss.n_mels, 
                            sample_rate=args.data.sampling_rate, device=args.device)
    elif args.loss.use_multiscale_mel:
        loss_func = MultiScaleMelSpectrogramLoss(args.data.sampling_rate)
    else:
        loss_func = RSSLoss(args.loss.fft_min, args.loss.fft_max, args.loss.n_scale, device=args.device)
        
        
    if args.model.use_discriminator:
        # load discriminator model parameters
        optimizer_d = torch.optim.AdamW(model_d.parameters(), lr=args.train.lr*0.5, weight_decay=args.train.weight_decay/args.train.lr)
        
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=args.train.sched_gamma)
    else:
        model_d, optimizer_d, scheduler_d = None, None, None


    # datas
    loaders = get_data_loaders(args)
    
    
    # copy spk_info
    if args.model.use_speaker_embed and not args.train.only_u2c_stack:
        shutil.copy2(os.path.join(args.data.dataset_path, 'spk_info.npz'), os.path.join(args.env.expdir, 'spk_info.npz'))
    
    
    # run
    train(args, 0, (model, optimizer, scheduler, loss_func, vocoder), (model_d, optimizer_d, scheduler_d), loaders['train'], loaders['test'])
    
