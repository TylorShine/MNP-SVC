'''
author: wayn391@mastertones

modified by: TylorShine
'''

import os
import json
import time
import yaml
import datetime
import torch
import matplotlib.pyplot as plt
from ..common import to_json
from torch.utils.tensorboard import SummaryWriter

class Saver(object):
    def __init__(
            self, 
            args,
            initial_global_step=-1):

        self.expdir = args.env.expdir
        self.sample_rate = args.data.sampling_rate
        
        # cold start
        self.global_step = initial_global_step
        self.init_time = time.time()
        self.last_time = time.time()

        # makedirs
        os.makedirs(self.expdir, exist_ok=True)       

        # path
        self.path_log_info = os.path.join(self.expdir, 'log_info.txt')

        # ckpt
        os.makedirs(self.expdir, exist_ok=True)       

        # writer
        self.writer = SummaryWriter(os.path.join(self.expdir, 'logs'))
        
        # save config
        path_config = os.path.join(self.expdir, 'config.yaml')
        with open(path_config, "w") as out_config:
            yaml.dump(dict(args), out_config)


    def log_info(self, msg, end="\n"):
        '''log method'''
        if isinstance(msg, dict):
            msg_list = []
            for k, v in msg.items():
                if isinstance(v, dict):
                    msg_list.append(f'{k}:')
                    for kk, vv in v.items():
                        if isinstance(v, int):
                            msg_list.append(f' {kk}: {vv:,}')
                        else:
                            msg_list.append(f' {kk}: {vv}')
                else:
                    if isinstance(v, int):
                        msg_list.append(f'{k}: {v:,}')
                    else:
                        msg_list.append(f'{k}: {v}')
            msg_str = '\n'.join(msg_list)
        else:
            msg_str = msg
        
        # display
        print(msg_str, end=end)
        
        msg_str = msg_str.lstrip('\r')

        # save
        with open(self.path_log_info, 'a') as fp:
            fp.write(msg_str+'\n')

    def log_value(self, dict):
        for k, v in dict.items():
            self.writer.add_scalar(k, v, self.global_step)
    
    def log_spec(self, name, spec, spec_out, vmin=-14, vmax=3.5):  
        spec_cat = torch.cat([(spec_out - spec).abs() + vmin, spec, spec_out], -1)
        spec = spec_cat[0]
        if isinstance(spec, torch.Tensor):
            spec = spec.cpu().numpy()
        fig = plt.figure(figsize=(12, 9))
        plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
        plt.tight_layout()
        self.writer.add_figure(name, fig, self.global_step)
    
    def log_audio(self, dict):
        for k, v in dict.items():
            self.writer.add_audio(k, v, global_step=self.global_step, sample_rate=self.sample_rate)
    
    def get_interval_time(self, update=True):
        cur_time = time.time()
        time_interval = cur_time - self.last_time
        if update:
            self.last_time = cur_time
        return time_interval

    def get_total_time(self, to_str=True):
        total_time = time.time() - self.init_time
        if to_str:
            total_time = str(datetime.timedelta(
                seconds=total_time))[:-5]
        return total_time

    def save_model(
            self,
            model, 
            optimizer,
            name='model',
            postfix='_',
            as_json=False,
            states=None):
        # path
        if postfix != '_':
            postfix = postfix
        path_pt = os.path.join(
            self.expdir , name+postfix+'.pt')
       
        # check
        print(' [*] model checkpoint saved: {}'.format(path_pt))

        # save
        save_model = {
            'global_step': self.global_step,
            'model': model.state_dict(),
            
        }
        if states is not None:
            save_model['states'] = states
        if optimizer is not None:
            save_model['optimizer'] = optimizer.state_dict()
        torch.save(save_model, path_pt)
            
        # to json
        if as_json:
            path_json = os.path.join(
                self.expdir , name+'.json')
            to_json(path_pt, path_json)
    
    def delete_model(self, name='model', postfix='_'):
        # path
        if postfix != '_':
            postfix = postfix
        path_pt = os.path.join(
            self.expdir , name+postfix+'.pt')
       
        # delete
        if os.path.exists(path_pt):
            os.remove(path_pt)
            print(' [*] model checkpoint deleted: {}'.format(path_pt))
        
    def global_step_increment(self):
        self.global_step += 1


