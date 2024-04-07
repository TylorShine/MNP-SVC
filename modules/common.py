import os
import glob
import json
import yaml
import torch

class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__
    
    
def load_config(path_config):
    with open(path_config, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    # print(args)
    return args


def to_json(path_params, path_json):
    params = torch.load(path_params, map_location=torch.device('cpu'))
    raw_state_dict = {}
    for k, v in params.items():
        val = v.flatten().numpy().tolist()
        raw_state_dict[k] = val

    with open(path_json, 'w') as outfile:
        json.dump(raw_state_dict, outfile,indent= "\t")
        
        
def load_model(
        expdir, 
        model,
        optimizer,
        name='model',
        postfix='_',
        device='cpu'):
    if postfix != '_':
        postfix = '_' + postfix
    path = os.path.join(expdir, name+postfix)
    path_pt = glob.glob(f'{expdir}/*.pt')
    global_step = 0
    states = None
    if len(path_pt) > 0:
        steps = [s.rstrip('.pt')[len(path):] for s in path_pt]
        maxstep = max([int(s) if s.isdigit() else 0 for s in steps])
        if maxstep >= 0:
            path_pt = path+str(maxstep)+'.pt'
        else:
            path_pt = path+'best.pt'
        print(' [*] restoring model from', path_pt)
        ckpt = torch.load(path_pt, map_location=torch.device(device))
        global_step = ckpt['global_step']
        states = ckpt.get('states')
        model.load_state_dict(ckpt['model'], strict=False)
        if ckpt.get('optimizer') != None:
            optimizer.load_state_dict(ckpt['optimizer'])
    return global_step, model, optimizer, states
