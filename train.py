import os
import argparse
import torch

import shutil

from modules.common import load_config, load_model
from modules.dataset.loader import get_data_loaders
from modules.solver import train
from modules.vocoder import CombSubMinimumNoisedPhase, CombSubMinimumNoisedPhaseStackOnly, NMPSFHiFi
from modules.diffusion.vocoder import Unit2WavMinimumNoisedPhase
from modules.reflow.vocoder import Unit2WavMinimumNoisedPhase as Unit2WavMinimumNoisedPhaseReflow, Unit2WavMinimumNoisedPhaseDirect, Unit2WavMinimumNoisedPhaseHidden
from modules.discriminator import MultiSpecDiscriminator, MultiPeriodSignalDiscriminator
from modules.loss import (
    RSSLoss, DSSLoss, DLFSSLoss,
    DLFSSMPLoss, DLFSSMPMalLoss, DSMPMalLoss,
    DLFSSMPMalinblogsLoss, DLFSSMalinblogsLoss, MRLFSSMPMalinblogsLoss, MRSMPMalinblogsLoss,
    MRSMPL1Loss, MRLF4SMPMalinblogsLoss, MRRSMalinblogsLoss,
    MelLoss
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
            torch.set_float32_matmul_precision('high')
            # model.unit2ctrl.compile()
            model.unit2ctrl.compile(mode="reduce-overhead")
            if args.model.use_discriminator:
                model_d = MultiSpecDiscriminator()
                # model_d = MultiPeriodSignalDiscriminator()
                # model_d.compile(mode="reduce-overhead")
    elif args.model.type == 'DiffusionMinimumNoisedPhase':
        from modules.diffusion.vocoder import Vocoder
        # load vocoder
        vocoder = Vocoder(args.model.vocoder.type, args.model.vocoder.ckpt, device=args.device)
        # if args.train.only_u2c_stack:
        #     model = CombSubMinimumNoisedPhaseStackOnly(
        #         n_unit=args.data.encoder_out_channels,
        #         n_hidden_channels=args.model.units_hidden_channels
        #     )
        # else:
        model = Unit2WavMinimumNoisedPhase(
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
            use_short_filter=args.model.use_short_filter,
            use_noise_short_filter=args.model.use_noise_short_filter,
            use_pitch_aug=args.model.use_pitch_aug,
            noise_seed=args.model.noise_seed,
            out_dims=vocoder.dimension,
            n_layers=args.model.n_layers,
            n_chans=args.model.n_chans,
            )
        model.ddsp_model.unit2ctrl.compile(mode="reduce-overhead")
        model.diff_model.denoise_fn.compile(mode="reduce-overhead")
        if args.model.use_discriminator:
            model_d = MultiSpecDiscriminator()
            # model_d = MultiPeriodSignalDiscriminator()
            
    elif args.model.type == 'ReflowMinimumNoisedPhase':
        from modules.reflow.vocoder import Vocoder
        # load vocoder
        vocoder = Vocoder(args.model.vocoder.type, args.model.vocoder.ckpt, device=args.device)
        # if args.train.only_u2c_stack:
        #     model = CombSubMinimumNoisedPhaseStackOnly(
        #         n_unit=args.data.encoder_out_channels,
        #         n_hidden_channels=args.model.units_hidden_channels
        #     )
        # else:
        model = Unit2WavMinimumNoisedPhaseReflow(
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
            use_short_filter=args.model.use_short_filter,
            use_noise_short_filter=args.model.use_noise_short_filter,
            use_pitch_aug=args.model.use_pitch_aug,
            noise_seed=args.model.noise_seed,
            out_dims=vocoder.dimension,
            n_layers=args.model.n_layers,
            n_chans=args.model.n_chans,
            )
        model.ddsp_model.unit2ctrl.compile(mode="reduce-overhead")
        model.reflow_model.compile(mode="reduce-overhead")
        if args.model.use_discriminator:
            model_d = MultiSpecDiscriminator()
            # model_d = MultiPeriodSignalDiscriminator()
            
    elif args.model.type == 'ReflowDirectMinimumNoisedPhase':
        from modules.reflow.vocoder import Vocoder
        # load vocoder
        vocoder = Vocoder(args.model.vocoder.type, args.model.vocoder.ckpt, device=args.device)
        # if args.train.only_u2c_stack:
        #     model = CombSubMinimumNoisedPhaseStackOnly(
        #         n_unit=args.data.encoder_out_channels,
        #         n_hidden_channels=args.model.units_hidden_channels
        #     )
        # else:
        model = Unit2WavMinimumNoisedPhaseDirect(
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
            use_short_filter=args.model.use_short_filter,
            use_noise_short_filter=args.model.use_noise_short_filter,
            use_pitch_aug=args.model.use_pitch_aug,
            noise_seed=args.model.noise_seed,
            out_dims=vocoder.dimension,
            n_layers=args.model.n_layers,
            n_chans=args.model.n_chans,
            )
        model.ddsp_model.unit2ctrl.compile(mode="reduce-overhead")
        model.reflow_model.compile(mode="reduce-overhead")
        if args.model.use_discriminator:
            model_d = MultiSpecDiscriminator()
            # model_d = MultiPeriodSignalDiscriminator()
            
    elif args.model.type == 'ReflowDirectMinimumNoisedPhaseHidden':
        from modules.reflow.vocoder import Vocoder
        # load vocoder
        vocoder = Vocoder(args.model.vocoder.type, args.model.vocoder.ckpt, device=args.device)
        # if args.train.only_u2c_stack:
        #     model = CombSubMinimumNoisedPhaseStackOnly(
        #         n_unit=args.data.encoder_out_channels,
        #         n_hidden_channels=args.model.units_hidden_channels
        #     )
        # else:
        model = Unit2WavMinimumNoisedPhaseHidden(
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
            use_short_filter=args.model.use_short_filter,
            use_noise_short_filter=args.model.use_noise_short_filter,
            use_pitch_aug=args.model.use_pitch_aug,
            noise_seed=args.model.noise_seed,
            out_dims=vocoder.dimension,
            # out_dims=args.model.units_hidden_channels,
            n_layers=args.model.n_layers,
            n_chans=args.model.n_chans,
            )
        model.ddsp_model.unit2ctrl.compile(mode="reduce-overhead")
        model.reflow_model.compile(mode="reduce-overhead")
        if args.model.use_discriminator:
            model_d = MultiSpecDiscriminator()
            # model_d = MultiPeriodSignalDiscriminator()
            
    elif args.model.type == 'NSFHiFiGAN':
        # from modules.reflow.vocoder import Vocoder
        # # load vocoder
        # vocoder = Vocoder(args.model.vocoder.type, args.model.vocoder.ckpt, device=args.device)
        # if args.train.only_u2c_stack:
        #     model = CombSubMinimumNoisedPhaseStackOnly(
        #         n_unit=args.data.encoder_out_channels,
        #         n_hidden_channels=args.model.units_hidden_channels
        #     )
        # else:
        model = NSFHiFiGAN(
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
            use_short_filter=args.model.use_short_filter,
            use_noise_short_filter=args.model.use_noise_short_filter,
            use_pitch_aug=args.model.use_pitch_aug,
            noise_seed=args.model.noise_seed,
            nsf_hifigan_h=args.model.nsf_hifigan,
            )
        model.unit2ctrl.compile()
        model.generator.compile()
        # model.unit2ctrl.compile(mode="reduce-overhead")
        # model.generator.compile(mode="reduce-overhead")
        if args.model.use_discriminator:
            # model_d = MultiSpecDiscriminator()
            model_d = MultiPeriodSignalDiscriminator()
            
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
    
    
    # model.unit2ctrl.compile(backend="onnxrt")
    # load model parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay)
    # model.unit2ctrl = torch.compile(model.unit2ctrl, mode="reduce-overhead")
    # model.unit2ctrl.compile(mode="reduce-overhead")
    
    # initial_global_step, model, optimizer, states = load_model(args.env.expdir, model, optimizer, device=args.device)
    # # for param in model.parameters():
    # #     param.requires_grad = True
    
    # lr = args.train.lr if states is None else states['last_lr'][0]
    
    # for param_group in optimizer.param_groups:
    #     param_group['initial_lr'] = args.train.lr
    #     param_group['lr'] = lr
    #     param_group['weight_decay'] = args.train.weight_decay
        
    if args.train.only_u2c_stack:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.train.sched_gamma)
    elif not args.model.use_discriminator:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.train.sched_factor, patience=args.train.sched_patience,
                                                            threshold=args.train.sched_threshold, threshold_mode=args.train.sched_threshold_mode,
                                                            cooldown=args.train.sched_cooldown, min_lr=args.train.sched_min_lr)
        # if states is not None:
        #     sched_states = states.get('scheduler')
        #     if sched_states is not None:
        #         scheduler.best = sched_states['best']
        #         scheduler.cooldown_counter = sched_states['cooldown_counter']
        #         scheduler.num_bad_epochs = sched_states['num_bad_epochs']
        #         scheduler._last_lr = sched_states['_last_lr']
        # else:
        #     scheduler._last_lr = (lr,)
    else:
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.train.sched_gamma)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train.epochs, eta_min=args.train.sched_min_lr)
    
        
        
    # loss
    if args.loss.use_dual_scale:
        loss_func = DSSLoss(args.loss.fft_min, args.loss.fft_max,
                            beta=args.loss.beta, overlap=args.loss.overlap, device=args.device)
    elif args.loss.use_dual_scale_log_freq:
        loss_func = DLFSSLoss(args.loss.fft_min, args.loss.fft_max,
                            beta=args.loss.beta, overlap=args.loss.overlap, device=args.device)
    elif args.loss.use_dual_scale_log_freq_magphase:
        loss_func = DLFSSMPLoss(args.loss.fft_min, args.loss.fft_max,
                            beta=args.loss.beta, gamma=args.loss.gamma, overlap=args.loss.overlap, device=args.device)
    elif args.loss.use_dual_scale_log_freq_magphase_mall:
        loss_func = DLFSSMPMalLoss(args.loss.fft_min, args.loss.fft_max,
                            beta=args.loss.beta, gamma=args.loss.gamma, overlap=args.loss.overlap, device=args.device)
    elif args.loss.use_dual_scale_magphase_mall:
        loss_func = DSMPMalLoss(args.loss.fft_min, args.loss.fft_max,
                            beta=args.loss.beta, gamma=args.loss.gamma, overlap=args.loss.overlap, device=args.device)
    elif args.loss.use_dual_scale_log_freq_magphase_malinblogs:
        loss_func = DLFSSMPMalinblogsLoss(args.loss.fft_min, args.loss.fft_max,
                            beta=args.loss.beta, gamma=args.loss.gamma, overlap=args.loss.overlap, device=args.device)
    elif args.loss.use_dual_scale_log_freq_malinblogs:
        loss_func = DLFSSMalinblogsLoss(args.loss.fft_min, args.loss.fft_max,
                            beta=args.loss.beta, gamma=args.loss.gamma, overlap=args.loss.overlap, device=args.device)
    elif args.loss.use_multi_reso_log_freq_magphase_malinblogs:
        loss_func = MRLFSSMPMalinblogsLoss(args.loss.n_fft, args.loss.n_div,
                            beta=args.loss.beta, gamma=args.loss.gamma, overlap=args.loss.overlap, device=args.device)
    elif args.loss.use_multi_reso_log_freq_malinblogs:
        loss_func = MRSMPMalinblogsLoss(args.loss.n_fft, args.loss.n_div,
                            beta=args.loss.beta, gamma=args.loss.gamma, overlap=args.loss.overlap, device=args.device)
    elif args.loss.use_freq_win_log_freq_magphase_malinblogs:
        loss_func = MRLF4SMPMalinblogsLoss(args.loss.n_fft, args.loss.n_div,
                            beta=args.loss.beta, gamma=args.loss.gamma, overlap=args.loss.overlap, device=args.device)
    elif args.loss.use_multi_reso_magphase_l1:
        loss_func = MRSMPL1Loss(args.loss.n_fft, args.loss.n_div,
                            beta=args.loss.beta, gamma=args.loss.gamma, overlap=args.loss.overlap, device=args.device)
    elif args.loss.use_multi_reso_resample_malinblogs:
        loss_func = MRRSMalinblogsLoss(args.loss.n_fft, args.loss.n_div,
                            beta=args.loss.beta, gamma=args.loss.gamma, overlap=args.loss.overlap, device=args.device)
    elif args.loss.use_mel:
        loss_func = MelLoss(n_fft=args.loss.n_fft, n_mels=args.loss.n_mels, 
                            sample_rate=args.data.sampling_rate, device=args.device)
    else:
        loss_func = RSSLoss(args.loss.fft_min, args.loss.fft_max, args.loss.n_scale, device=args.device)
        
        
    if args.model.use_discriminator:
        # load discriminator model parameters
        optimizer_d = torch.optim.AdamW(model_d.parameters(), lr=args.train.lr*0.5, weight_decay=args.train.weight_decay)
        # model_d = torch.compile(model_d, mode="reduce-overhead")
        # model_d.compile(backend="onnxrt")
        
        # _, model_d, optimizer_d, states = load_model(args.env.expdir, model_d, optimizer_d, name='modelD', device=args.device)
        
        # # for param in model_d.parameters():
        # #     param.requires_grad = True
        # # lr = args.train.lr*2. if states is None else states['last_lr'][0]
        # lr = args.train.lr*0.5 if states is None else states['last_lr'][0]
        
        # for param_group in optimizer_d.param_groups:
        #     param_group['initial_lr'] = args.train.lr*0.5
        #     param_group['lr'] = lr
        #     param_group['weight_decay'] = args.train.weight_decay
            
        # scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_d, mode='min', factor=args.train.sched_factor, patience=args.train.sched_patience,
        #                                                          threshold=args.train.sched_threshold, threshold_mode=args.train.sched_threshold_mode,
        #                                                          cooldown=args.train.sched_cooldown, min_lr=args.train.sched_min_lr)
        scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=args.train.epochs, eta_min=args.train.sched_min_lr*0.5)
    
        # if states is not None:
        #     sched_states = states.get('scheduler')
        #     if sched_states is not None:
        #         # scheduler_d.best = sched_states['best']
        #         # scheduler_d.cooldown_counter = sched_states['cooldown_counter']
        #         # scheduler_d.num_bad_epochs = sched_states['num_bad_epochs']
        #         scheduler_d._last_lr = sched_states['_last_lr']
        # else:
        #     scheduler_d._last_lr = torch.tensor((lr,))
    else:
        model_d, optimizer_d, scheduler_d = None, None, None
        
    


    # # device
    # if args.device == 'cuda':
    #     torch.cuda.set_device(args.env.gpu_id)
    # model.to(args.device)
    # if model_d is not None:
    #     model_d.to(args.device)
    
    # for state in optimizer.state.values():
    #     for k, v in state.items():
    #         if torch.is_tensor(v):
    #             state[k] = v.to(args.device)
                    
    # loss_func.to(args.device)


    # datas
    loaders = get_data_loaders(args)
    
    
    # copy spk_info
    if args.model.use_speaker_embed and not args.train.only_u2c_stack:
        shutil.copy2(os.path.join(args.data.dataset_path, 'spk_info.npz'), os.path.join(args.env.expdir, 'spk_info.npz'))
    
    
    # run
    train(args, 0, (model, optimizer, scheduler, loss_func, vocoder), (model_d, optimizer_d, scheduler_d), loaders['train'], loaders['test'])
    
