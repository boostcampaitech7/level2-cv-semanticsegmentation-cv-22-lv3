import torch.nn as nn

import torch.optim as optim

def loss_func_loader(criterion_name):
    if criterion_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()
    elif criterion_name == "":
        pass
    elif criterion_name == "":
        pass
    elif criterion_name == "":
        pass
    else:
        raise ValueError(f"Unsupported criterion: {criterion_name}")

def optimizer_loader(config, model_parameters):
    optimizer_name = config.optimizer.name
    if optimizer_name == "Adam":
        return optim.Adam(params=model_parameters, lr=config.optimizer.lr, 
                          weight_decay=config.optimizer.weight_decay)
    elif optimizer_name == "SGD":
        pass
        # return optim.SGD(model_parameters, lr=config['training']['learning_rate'], 
        #                  momentum=0.9, weight_decay=config['training']['weight_decay'])
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    

def lr_scheduler_loader(optimizer, scheduler_config):

    if scheduler_config['name'] == "":
        pass

    else:
        raise ValueError(f"Unsupported lr scheduler: {scheduler_config['name']}")