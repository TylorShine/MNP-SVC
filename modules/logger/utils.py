
def get_network_params_amount(model_dict):
    info = dict()
    for model_name, model in model_dict.items():
        all_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        info[model_name] = {'all': all_params, 'trainable': trainable_params}
    return info

