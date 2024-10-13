import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from modules.loss import discriminator_loss, feature_loss, generator_loss

from modules.logger.saver import Saver
from modules.logger import utils

from accelerate import Accelerator

def test(args, model, loss_func, loader_test, saver, vocoder=None):
    model.eval()

    # losses
    test_loss = 0.
    
    # intialization
    num_batches = len(loader_test)
    rtf_all = []
    
    spk_id_key = 'spk_id'
    if args.model.use_speaker_embed:
        spk_id_key = 'spk_embed'
    
    # run
    with torch.no_grad():
        with tqdm(loader_test, desc="test") as pbar:
            for data in pbar:
                if data is None:
                    break
                fn = data['name'][0].lstrip("data/test/")

                # unpack data
                for k in data.keys():
                    if k != 'name':
                        data[k] = data[k].to(args.device)
                
                units = data['units']
                
                if "Diffusion" in args.model.type:
                    mel = data['mel']
                    # forward
                    st_time = time.time()
                    signal = model(units, data['f0'], data['volume'], data[spk_id_key], 
                                vocoder=vocoder, return_wav=True, gt_spec=mel.float(), k_step=args.model.k_step_max).squeeze(1)
                    ed_time = time.time()
                elif "ReflowDirect" in args.model.type:
                    # forward
                    # t_start = 0.0 if "Hidden" in args.model.type else args.model.t_start
                    t_start = args.model.t_start
                    gt_wav = None if "Hidden" in args.model.type else data['audio']
                    gt_spec = data['mel'] if "Hidden" in args.model.type else None
                    st_time = time.time()
                    signal = model(units, data['f0'], data['volume'], data[spk_id_key], 
                                return_wav=True, gt_wav=gt_wav, gt_spec=gt_spec,
                                vocoder=vocoder, infer_step=args.infer.infer_step, method=args.infer.method, t_start=t_start)
                    ed_time = time.time()
                elif "Reflow" in args.model.type:
                    mel = data['mel']
                    # forward
                    st_time = time.time()
                    signal = model(units, data['f0'], data['volume'], data[spk_id_key], 
                                vocoder=vocoder, return_wav=True, gt_spec=mel.float(),
                                infer_step=args.infer.infer_step, method=args.infer.method, t_start=args.model.t_start).squeeze(1)
                    ed_time = time.time()
                else:
                    # forward
                    st_time = time.time()
                    signal = model(units, data['f0'], data['volume'], data[spk_id_key])
                    ed_time = time.time()

                # crop
                min_len = np.min([signal.shape[1], data['audio'].shape[1]])
                signal        = signal[:,:min_len]
                data['audio'] = data['audio'][:,:min_len]

                # RTF
                run_time = ed_time - st_time
                song_time = data['audio'].shape[-1] / args.data.sampling_rate
                rtf = run_time / song_time
                rtf_all.append(rtf)
            
                # loss
                loss = loss_func(signal, data['audio'])

                test_loss += loss.item()

                # log
                saver.log_audio({fn+'/gt.wav': data['audio'], fn+'/pred.wav': signal})
                
                pbar.set_description(fn)
                pbar.set_postfix({'loss': loss.item(), 'RTF': rtf})
            
    # report
    if num_batches > 0:
        test_loss /= num_batches
    else:
        return None
    
    return test_loss


def train(args, initial_global_step, nets_g, nets_d, loader_train, loader_test):
    model, optimizer, scheduler, loss_func, vocoder = nets_g
    model_d, optimizer_d, scheduler_d = nets_d
    
    # saver
    saver = Saver(args, initial_global_step=initial_global_step)
    
    accelerator: Accelerator = saver.accelerator
    
    # accelerator = Accelerator(
    #     mixed_precision=args.train.accelerator.mixed_precision,
    #     log_with=args.train.accelerator.log,
    # )
    
    accelerator.init_trackers(
        project_name=args.env.project_name)
    
    # model, optimizer, scheduler, model_d, optimizer_d, scheduler_d, loss_func, loader_train, loader_test = accelerator.prepare(
    #     model, optimizer, scheduler,
    #     model_d, optimizer_d, scheduler_d,
    #     loss_func,
    #     loader_train, loader_test
    # )
    
    # # accelerator.register_for_checkpointing(scheduler)
    
    # # initial_global_step, model, optimizer, states = load_model(args.env.expdir, model, optimizer, device=args.device)
    
    # state_loaded = saver.load_state()
    # if not state_loaded:
    #     state_loaded = saver.load_from_pretrained(
    #         model, optimizer)
    #     if not state_loaded:
    #         print(f"No state found at {args.env.expdir} and no pretrained found at {args.env.pretrained}, starting from scratch")
    
    # device = args.device
    device = accelerator.device
    
    model.to(device)
    if model_d is not None:
        model_d.to(device)
    
    last_model_save_step = saver.global_step
    
    # unwrapped_model = accelerator.unwrap_model(model)
    
    # run
    num_batches = len(loader_train)
    model.train()
    if model_d is not None:
        model_d.train()
    # scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    
    ddsp_model_prefix = ""
    if "Diffusion" in args.model.type or "Reflow" in args.model.type:
        ddsp_model_prefix = "ddsp_model."
    expdir_dirname = os.path.split(args.env.expdir)[-1]
    
    spk_id_key = 'spk_id'
    if args.model.use_speaker_embed:
        spk_id_key = 'spk_embed'    
        
    if args.train.only_u2c_stack:
        # TODO: functionize
        for param_name, param in model.named_parameters():
            if not (param_name.startswith('unit2ctrl.stack.0.') or \
                param_name.startswith('unit2ctrl.stack.1.')):
                param.requires_grad = False
                
        # model size
        params_count = utils.get_network_params_amount({'model': model})
        saver.log_info('--- model size ---')
        saver.log_info(params_count)
            
        # TODO: too slow iteration, could be much faster
        saver.log_info('======= start training (stack only) =======')
        for epoch in range(args.train.epochs):
            for batch_idx, data in enumerate(loader_train):
                saver.global_step_increment()
                # optimizer.zero_grad()
                
                # unpack data
                for k in data.keys():
                    if k != 'name':
                        data[k] = data[k].to(device)
                        
                stack_input = data['units']
                units = stack_input
                        
                # forward
                if dtype == torch.float32:
                    signal = model(stack_input.float())
                else:
                    with autocast(device_type=device, dtype=dtype):
                        signal = model(stack_input.to(dtype))
                
                optimizer.zero_grad()
                
                losses = []
                
                # minibatch contrastive learning
                # Bring per-frame per-speaker's feature to centroid of top two of different speakers' similarities and itself,
                # keep away per-frame per-speaker's feature to centroid of worst two of different speakers' similarities and itself.
                # These are expecting to learn that different speakers' same utterance mapping to nearly vectors.
                B, frames, T = signal.shape
                for b in range(B):
                    # cosine similarity
                    opps = [o for o in range(B)
                            if data['spk_id'][b] != data['spk_id'][o]]
                    if len(opps) <= 0:
                        # TODO: should be quit backpropagation of epoch itself?
                        continue
                    
                    last_hop = 1
                    f = 0
                    while f < frames - 1:
                        ## brute-force
                        opps_sims = F.cosine_similarity(
                            units.float()[b, f:f+1].unsqueeze(0).repeat(len(opps), units.float().shape[1], 1),
                            units.float()[opps, 0:],
                            dim=2)
                        opps_sort_sim_frame = torch.argsort(opps_sims, dim=1)
                        
                        # sim_mini_opp_frames = [units.float()[b, f]]
                        sim_mini_opp_frames = []
                        sim_maxi_opp_frames = [units.float()[b, f]]
                        
                        for i in range(min(len(opps), 2)):  # TODO: parametrize?
                            opps_large_sims = opps_sims[:, opps_sort_sim_frame[-1-i]].diagonal()
                            opps_small_sims = opps_sims[:, opps_sort_sim_frame[i]].diagonal()
                            
                            sim_mini_opp = torch.argmin(opps_small_sims)
                            sim_mini_opp_frame = opps_sort_sim_frame[i][sim_mini_opp]
                            sim_maxi_opp = torch.argmax(opps_large_sims)
                            sim_maxi_opp_frame = opps_sort_sim_frame[-1-i][sim_maxi_opp]
                            
                            sim_maxi_opp_frames.append(units.float()[opps[sim_maxi_opp], sim_maxi_opp_frame])
                            sim_mini_opp_frames.append(units.float()[opps[sim_mini_opp], sim_mini_opp_frame])
                            
                        sim_mini_opps_centroid = torch.mean(torch.stack(sim_mini_opp_frames), dim=0)
                        sim_maxi_opps_centroid = torch.mean(torch.stack(sim_maxi_opp_frames), dim=0)
                        
                        # ## random pick
                        # rand_frame = torch.randint(0, frames, (len(opps),))
                        # opps_sims = F.cosine_similarity(
                        #     units.float()[b, f].repeat(len(opps), 1),
                        #     # units.float()[opps, 0:],
                        #     torch.stack([units[o, i] for o, i in zip(opps, rand_frame)]).float(),
                        #     dim=1)
                        # opps_sort_sim_batch = torch.argsort(opps_sims, dim=0)
                        
                        # # sim_mini_opp_frames = [units.float()[b, f]]
                        # sim_mini_opp_frames = []
                        # sim_maxi_opp_frames = [units.float()[b, f]]
                        
                        # for i in range(min(len(opps), 2)):  # TODO: parametrize?
                        #     sim_maxi_opp = opps_sort_sim_batch[-1-i]
                        #     sim_maxi_opp_frame = rand_frame[sim_maxi_opp]
                        #     sim_mini_opp = opps_sort_sim_batch[i]
                        #     sim_mini_opp_frame = rand_frame[sim_mini_opp]
                            
                        #     sim_maxi_opp_frames.append(units.float()[opps[sim_maxi_opp], sim_maxi_opp_frame])
                        #     sim_mini_opp_frames.append(units.float()[opps[sim_mini_opp], sim_mini_opp_frame])
                            
                        # sim_mini_opps_centroid = torch.mean(torch.stack(sim_mini_opp_frames), dim=0)
                        # sim_maxi_opps_centroid = torch.mean(torch.stack(sim_maxi_opp_frames), dim=0)
                        
                        if dtype == torch.float32:
                            losses.append(
                                F.l1_loss(
                                    1. - (F.cosine_similarity(signal[b, f], signal[opps[sim_maxi_opp], sim_maxi_opp_frame], dim=0)*0.5 + 0.5),
                                    (1. - (F.cosine_similarity(units.float()[b, f], sim_maxi_opps_centroid, dim=0)*0.5 + 0.5))*args.train.loss_variation)
                            )
                            
                            losses.append(
                                F.l1_loss(
                                    F.cosine_similarity(signal[b, f], signal[opps[sim_mini_opp], sim_mini_opp_frame], dim=0)*0.5 + 0.5,
                                    (F.cosine_similarity(units.float()[b, f], sim_mini_opps_centroid, dim=0)*0.5 + 0.5)*args.train.low_similar_loss_variation)
                            )
                        else:
                            with autocast(device_type=device, dtype=dtype):
                                losses.append(
                                    F.l1_loss(
                                        1. - (F.cosine_similarity(signal[b, f], signal[opps[sim_maxi_opp], sim_maxi_opp_frame], dim=0)*0.5 + 0.5),
                                        (1. - (F.cosine_similarity(units[b, f], sim_maxi_opps_centroid, dim=0)*0.5 + 0.5))*args.train.loss_variation)
                                )
                                
                                losses.append(
                                    F.l1_loss(
                                        F.cosine_similarity(signal[b, f], signal[opps[sim_mini_opp], sim_mini_opp_frame], dim=0)*0.5 + 0.5,
                                        (F.cosine_similarity(units[b, f], sim_mini_opps_centroid, dim=0)*0.5 + 0.5)*args.train.low_similar_loss_variation)
                                )
                                
                        last_hop = torch.randint(args.train.frame_hop_random_min, args.train.frame_hop_random_max, (1,))[0]
                        # last_hop = 1
                        f += last_hop
                        
                if len(losses) <= 0:
                    # TODO: should be quit backpropagation of epoch itself?
                    continue
                
                loss = torch.stack([l/(len(losses)/2) for l in losses]).sum()
                
                # handle nan loss
                if torch.isnan(loss):
                    raise ValueError(' [x] nan loss ')
                
                # backpropagate
                # if dtype == torch.float32:
                loss.backward()
                optimizer.step()
                # else:
                #     scaler.scale(loss).backward()
                #     scaler.step(optimizer)
                #     scaler.update()
                    
                # log loss
                if saver.global_step % args.train.interval_log == 0:
                    saver.log_info(
                        '\repoch: {} | {:3d}/{:3d} | {} | batch/s: {:2.2f} | loss: {:.7f} | lr: {:.6f} | time: {} | step: {}'.format(
                            epoch,
                            batch_idx,
                            num_batches,
                            expdir_dirname,
                            args.train.interval_log/saver.get_interval_time(),
                            loss.item(),
                            scheduler.get_last_lr()[0],
                            saver.get_total_time(),
                            saver.global_step
                        ),
                        end="",
                    )
                    
                    saver.log_value({
                        'train/loss': loss.item(),
                        'train/lr': scheduler.get_last_lr()[0],
                    })
                    
                # validation
                if saver.global_step % args.train.interval_val == 0:
                    optimizer_save = optimizer if args.train.save_opt else None
                    
                    states = {
                        'scheduler': scheduler.state_dict(),
                        'last_lr': scheduler.get_last_lr(),
                    }
                        
                    # save latest
                    saver.save_model(model, optimizer_save, postfix=f'_{saver.global_step}', states=states)
                    
            # scheduler.step(loss)
                scheduler.step()
        return
    else:   # args.train.only_u2c_stack
        if not args.data.encoder == "phrex":
            # freeze against last fc of stack
            for param_name, param in model.named_parameters():
                if param_name.startswith(f'{ddsp_model_prefix}unit2ctrl.stack.0.') or \
                    param_name.startswith(f'{ddsp_model_prefix}unit2ctrl.stack.1.'):
                    param.requires_grad = False
    
    freeze_model = False
    if freeze_model:
        for param_name, param in model.named_parameters():
            param.requires_grad = False
            
    train_params = []
            
    if args.train.ft_spk_embed:
        # train_params += ['unit2ctrl.spk_embed.', 'unit2ctrl.recon_spk_embed.']
        train_params += [f'{ddsp_model_prefix}unit2ctrl.spk_embed.']
        
    if args.train.ft_recon_spk_embed:
        train_params += [f'{ddsp_model_prefix}unit2ctrl.recon_spk_embed.']
                
    if args.train.ft_spk_embed_conv:
        train_params += [f'{ddsp_model_prefix}unit2ctrl.spk_embed_conv.']
                
    if args.train.ft_dense_out:
        train_params += [f'{ddsp_model_prefix}unit2ctrl.dense_out.']
        
    if args.train.ft_diffusion:
        train_params += [f'diff_model.']
        
    if args.train.ft_reflow:
        train_params += [f'reflow_model.']
        
    if args.train.ft_nsf:
        train_params += [f'generator.']
        
    if len(train_params) > 0:
        for param_name, param in model.named_parameters():
            if not any([param_name.startswith(p) for p in train_params]):
                param.requires_grad = False
                
    model, optimizer, scheduler, model_d, optimizer_d, scheduler_d, loss_func, loader_train, loader_test = accelerator.prepare(
        model, optimizer, scheduler,
        model_d, optimizer_d, scheduler_d,
        loss_func,
        loader_train, loader_test
    )
    
    # accelerator.register_for_checkpointing(scheduler)
    
    # initial_global_step, model, optimizer, states = load_model(args.env.expdir, model, optimizer, device=args.device)
    
    state_loaded = saver.load_state()
    if not state_loaded:
        state_loaded = saver.load_from_pretrained(
            model, optimizer)
        if not state_loaded:
            print(f"No state found at {args.env.expdir} and no pretrained found at {args.env.pretrained}, starting from scratch")
    
    # model size
    # model_dict = {'model': unwrapped_model}
    model_dict = {'model': model}
    if model_d is not None:
        model_dict['discriminator'] = model_d
    params_count = utils.get_network_params_amount(model_dict)
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    saver.log_info('======= start training =======')
    # model = torch.compile(model, mode="reduce-overhead")
    # model.unit2ctrl = torch.compile(model.unit2ctrl, mode="reduce-overhead")
    # model.unit2ctrl = torch.compile(model.unit2ctrl)
    if model_d is not None:
        # model_d.discriminators[0] = torch.compile(model_d.discriminators[0])
        # model_d = torch.compile(model_d, mode="reduce-overhead")
        # model_d = torch.compile(model_d)
        # model_d = torch.compile(model_d, mode="reduce-overhead")
        pass
    for epoch in range(args.train.epochs):
        saver.global_epoch_increment()
        for batch_idx, data in enumerate(loader_train):
            torch.compiler.cudagraph_mark_step_begin()
            saver.global_step_increment()
            # optimizer.zero_grad()
            # if optimizer_d is not None:
            #     optimizer_d.zero_grad()

            # unpack data
            for k in data.keys():
                if k != 'name':
                    data[k] = data[k].to(device)
            
            
            units = data['units']
            f0 = data['f0']
            volume = data['volume']
            audio = data['audio']
            
            # # TEMP
            # units = units.float()
            # f0 = f0.float()
            # volume = volume.float()
            # audio = audio.float()
            
            # torch.compiler.cudagraph_mark_step_begin()
            
            if "Diffusion" in args.model.type:
                mel = data['mel']
                # forward
                # if dtype == torch.float32:
                #     ddsp_loss, diff_loss = model(units.float(), f0, volume, data[spk_id_key], 
                #                 vocoder=vocoder, gt_spec=mel.float(), infer=False, k_step=args.model.k_step_max)
                #                                 # aug_shift=data['aug_shift'], infer=False)
                # else:
                #     # with autocast(device_type=args.device, dtype=dtype):
                with accelerator.autocast():
                    ddsp_loss, diff_loss=model(units.to(dtype), f0, volume, data[spk_id_key], 
                                vocoder=vocoder, gt_spec=mel.float(), infer=False, k_step=args.model.k_step_max)
                        
            elif "ReflowDirect" in args.model.type:
                mel = data['mel']
                # forward
                # if dtype == torch.float32:
                #     ddsp_loss, diff_loss = model(units.float(), f0, volume, data[spk_id_key], 
                #                 vocoder=vocoder, gt_wav=audio.float(), gt_spec=mel, infer=False, loss_func=loss_func, t_start=args.model.t_start)
                #                                 # aug_shift=data['aug_shift'], infer=False)
                # else:
                    # with autocast(device_type=args.device, dtype=dtype):
                with accelerator.autocast():
                    ddsp_loss, diff_loss=model(units.to(dtype), f0, volume, data[spk_id_key], 
                                vocoder=vocoder, gt_wav=audio.to(dtype), gt_spec=mel, infer=False, loss_func=loss_func, t_start=args.model.t_start)
                        
            elif "Reflow" in args.model.type:
                mel = data['mel']
                # forward
                # if dtype == torch.float32:
                #     ddsp_loss, diff_loss = model(units.float(), f0, volume, data[spk_id_key], 
                #                 vocoder=vocoder, gt_spec=mel.float(), infer=False, t_start=args.model.t_start)
                #                                 # aug_shift=data['aug_shift'], infer=False)
                # else:
                #     with autocast(device_type=args.device, dtype=dtype):
                with accelerator.autocast():
                    ddsp_loss, diff_loss=model(units.to(dtype), f0, volume, data[spk_id_key], 
                                vocoder=vocoder, gt_spec=mel.float(), infer=False, t_start=args.model.t_start)
            else:    
                # forward
                # if dtype == torch.float32:
                #     signal = model(units.float(), f0, volume, data[spk_id_key],
                #                                 infer=False)
                #                                 # aug_shift=data['aug_shift'], infer=False)
                # else:
                #     with autocast(device_type=args.device, dtype=dtype):
                with accelerator.autocast():
                    signal = model(units.to(dtype), f0, volume, data[spk_id_key])
                    #                             infer=False)
                    # signal = model(units, f0, volume, data[spk_id_key])
                                                # infer=False)
                                            # aug_shift=data['aug_shift'], infer=False)
                                            
            # optimizer_d.zero_grad()
            
            # loss
            # losses = []
               
            if model_d is not None:
                optimizer_d.zero_grad()
                # torch.compiler.cudagraph_mark_step_begin()
                # with autocast(device_type=args.device, dtype=dtype):
                with accelerator.autocast():
                    d_signal_real, d_signal_gen, _, _ = model_d(audio.to(dtype).unsqueeze(1), signal.detach().to(dtype).unsqueeze(1), flg_train=True)
                    
                # discriminator loss
                loss_d, losses_d_real, losses_d_gen = discriminator_loss(d_signal_real, d_signal_gen)
                
                # handle nan loss
                if torch.isnan(loss_d):
                    raise ValueError(' [x] discriminator: nan loss ')
                else:
                    # # backpropagate
                    # if dtype == torch.float32:
                    #     loss_d.backward()
                    #     torch.nn.utils.clip_grad_norm_(parameters=model_d.parameters(), max_norm=500)
                    #     optimizer_d.step()
                    # else:
                    #     scaler.scale(loss_d).backward()
                    #     scaler.unscale_(optimizer_d)
                    #     torch.nn.utils.clip_grad_norm_(parameters=model_d.parameters(), max_norm=500)
                    #     scaler.step(optimizer_d)
                    accelerator.backward(loss_d*0.5)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(parameters=model_d.parameters(), max_norm=500)
                    optimizer_d.step()
                optimizer.zero_grad()
                
                # generator loss
                # with autocast(device_type=args.device, dtype=dtype):
                with accelerator.autocast():
                    d_signal_real, d_signal_gen, fmap_real, fmap_gen = model_d(audio.to(dtype).unsqueeze(1), signal.to(dtype).unsqueeze(1))
                    
                loss_fm = feature_loss(fmap_real, fmap_gen)
                loss_gen, losses_gen = generator_loss(d_signal_gen)
                
                # losses = [loss_gen, loss_fm*0.02]    # TODO: parametrize
                # losses = [loss_gen*0.5, loss_fm*0.015]    # TODO: parametrize
                # losses = [loss_gen*0.25, loss_fm*0.01]    # TODO: parametrize
                # losses = [loss_gen*0.25, loss_fm*0.01]    # TODO: parametrize
                # losses = [loss_gen*0.25, loss_fm*0.1]    # TODO: parametrize
                # losses = [loss_gen*0.5, loss_fm*0.2]    # TODO: parametrize
                # losses = [loss_gen*0.5, loss_fm*0.02]    # TODO: parametrize
                # losses = [loss_gen, loss_fm*0.1]    # TODO: parametrize
                # losses = [loss_gen, loss_fm]    # TODO: parametrize
                losses = [loss_gen, loss_fm*2.]    # TODO: parametrize
                # losses = [loss_gen, loss_fm*1.5]    # TODO: parametrize
                # losses = [loss_gen, loss_fm*1.25]    # TODO: parametrize
                # losses = [loss_gen, loss_fm*1.3]    # TODO: parametrize
                # losses = [loss_gen, loss_fm*1.05]    # TODO: parametrize
            else:
                optimizer.zero_grad()
                losses = []
                
            if "Diffusion" in args.model.type or "Reflow" in args.model.type:
                losses.extend((args.train.lambda_ddsp*ddsp_loss, diff_loss))
            else:
                # if "NSFHiFiGAN" in args.model.type:
                #     signal = signal.squeeze(1)
                # crop
                # print(signal.shape, audio.shape)
                # min_len = np.min([signal.shape[1], audio.shape[1]])
                # signal = signal[:,:min_len]
                # audio = audio[:,:min_len]
                # with accelerator.autocast():
                # losses.append(loss_func(signal, audio)*2.)
                if model_d is not None:
                    with accelerator.autocast():
                        # losses.append(loss_func(signal.to(dtype), audio.to(dtype))*24.)
                        # losses.append(loss_func(signal.to(dtype), audio.to(dtype))*16.)
                        # losses.append(loss_func(signal.to(dtype), audio.to(dtype))*15.)
                        # losses.append(loss_func(signal.to(dtype), audio.to(dtype))*13.)
                        # losses.append(loss_func(signal.to(dtype), audio.to(dtype))*12.)
                        losses.append(loss_func(signal.to(dtype), audio.to(dtype))*11.)
                        # losses.append(loss_func(signal.to(dtype), audio.to(dtype))*8.)
                        # losses.append(loss_func(signal.to(dtype), audio.to(dtype))*5.)
                        # losses.append(loss_func(signal.to(dtype), audio.to(dtype))*10.)
                        # losses.append(loss_func(signal.to(dtype), audio.to(dtype))*3.)
                else:
                    with accelerator.autocast():
                        losses.append(loss_func(signal.to(dtype), audio.to(dtype)))
            
            loss = torch.stack(losses).sum()
            
            # accelerator.print(f"Step {saver.global_step} loss: {loss.item()}")
                
            if not freeze_model:
                # handle nan loss
                if torch.isnan(loss):
                    raise ValueError(' [x] nan loss ')
                else:
                    # # backpropagate
                    # if dtype == torch.float32:
                    #     loss.backward()
                    #     torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=500)
                    #     optimizer.step()
                    # else:
                    #     scaler.scale(loss).backward()
                    #     scaler.unscale_(optimizer)
                    #     torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=500)
                    #     scaler.step(optimizer)
                    #     scaler.update()
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(parameters=model.parameters(), max_norm=500)
                    optimizer.step()
            # elif model_d is not None and dtype != torch.float32:
            #     scaler.update()

            # log loss
            if saver.global_step % args.train.interval_log == 0:
                saver.log_info(
                    '\repoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | loss: {:.3f} | lr: {:.6f} | time: {} | step: {} '.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        loss.item(),
                        scheduler.get_last_lr()[0],
                        saver.get_total_time(),
                        saver.global_step
                    ),
                    end="",
                )
                
                # log_dict = {
                #     'train/g/loss': loss.item(),
                #     'train/g/lr': scheduler.get_last_lr()[0],
                # }
                
                log_dict = {
                    "train/generator/loss": loss.item(),
                    "train/generator/learning_rate": scheduler.get_last_lr()[0],
                    "train/batch_time": args.train.interval_log/(saver.get_interval_time()+1e-6),
                    "train/epoch": epoch,
                    "train/step": saver.global_step,
                }
                
                if model_d is not None:
                    log_dict.update({
                        'train/discriminator/loss': loss_d.item(),
                        'train/discriminator/learning_rate': scheduler_d.get_last_lr()[0],
                    })
                    log_dict.update({
                        f'train/discriminator_real/l{i}': v for i, v in enumerate(losses_d_real)
                    })
                    log_dict.update({
                        f'train/discriminator_gen/l{i}': v for i, v in enumerate(losses_d_gen)
                    })
                    log_dict.update({
                        f'train/generator/l{i}': v for i, v in enumerate(losses_gen)
                    })
                    log_dict.update({
                        f'train/generator/feature_matching': loss_fm.item()
                    })
                
                saver.log_value(log_dict)
            
            # validation
            if accelerator.is_main_process and saver.global_step % args.train.interval_val == 0:
                # optimizer_save = optimizer if args.train.save_opt else None
                
                # states = {
                #     'scheduler': scheduler.state_dict(),
                #     'last_lr': scheduler.get_last_lr(),
                # }
                
                # save model
                saver.save_model(model, postfix=f'_{saver.global_step}')
                
                # save states
                if args.train.save_states:
                    saver.save_state()
                    
                
                # saver.save_model(model, postfix=f'_{saver.global_step}', states=states)
                
                # if model_d is not None:
                #     # optimizer_d_save = optimizer_d if args.train.save_discriminator_opt else None
                #     # states_d = {
                #     #     'scheduler': scheduler_d.state_dict(),
                #     #     'last_lr': scheduler_d.get_last_lr(),
                #     # }

                #     # save latest discriminator
                #     saver.save_model(model_d, postfix=f'D_{saver.global_step}')
                #     # saver.save_model(model_d, optimizer_d_save, postfix=f'D_{saver.global_step}', states=states_d)
                    
                #     # if last_model_save_step > 0:
                #     #     saver.delete_model(postfix=f'D_{last_model_save_step}')
                        
                last_model_save_step = saver.global_step

                # run testing set
                if loader_test is not None:
                    test_loss = test(args, model, loss_func, loader_test, saver, vocoder=vocoder)
                    
                    if test_loss is not None:
                    
                        # log loss
                        saver.log_info(
                            ' --- <validation step:{:d}> loss: {:.3f}. '.format(
                                saver.global_step,
                                test_loss,
                            )
                        )
            
                        saver.log_value({
                            'validation/loss': test_loss
                        })
                
                model.train()
                
        if not args.model.use_discriminator:
            # NOTE: The scheduler associating the train loss instead of validation loss.
            #       This is not ideal. but we use long validation span which cause the lr
            #       doesn't get updated very often if use the validation one.
            scheduler.step(loss)
        else:
            scheduler.step()
            scheduler_d.step()
            
    # lastly, test, logging and save last states
    # optimizer_save = optimizer if args.train.save_opt else None
                
    # states = {
    #     'scheduler': scheduler.state_dict(),
    #     'last_lr': scheduler.get_last_lr(),
    # }
    
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        # save latest
        saver.save_model(model, postfix=f'_{saver.global_step}')
        
        if args.train.save_states:
            saver.save_state()
        
        # if model_d is not None:
        #     # optimizer_d_save = optimizer_d if args.train.save_discriminator_opt else None
        #     # states_d = {
        #     #     'scheduler': scheduler_d.state_dict(),
        #     #     'last_lr': scheduler_d.get_last_lr(),
        #     # }
        #     saver.save_model(model_d, postfix=f'D_{saver.global_step}')
            
        #     if last_model_save_step > 0 and last_model_save_step != saver.global_step:
        #         saver.delete_model(postfix=f'D_{last_model_save_step}')
                
        last_model_save_step = saver.global_step
        
        if loader_test is not None:
            test_loss = test(args, model, loss_func, loader_test, saver, vocoder=vocoder)

            if test_loss is not None:

                # log loss
                saver.log_info(
                    ' --- <validation> ep.{:d} loss: {:.3f}. '.format(
                        args.train.epochs,
                        test_loss,
                    )
                )

                saver.log_value({
                    'validation/loss_d': test_loss
                })
                
        # training done
        print(f'Training completed after {saver.global_step} steps, {args.train.epochs} epochs!')
        
        saver.sync()
        
    accelerator.end_training()
    

                          
