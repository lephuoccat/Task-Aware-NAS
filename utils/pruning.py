# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:42:29 2020

@author: catpl
"""
import os
import numpy as np
import torch
import torchvision
from torch import nn

device = 'cuda'

def weight_prune(model, pruning_perc):
    all_weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            all_weights += list(p.cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(all_weights), pruning_perc)

    # generate mask
    masks = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            pruned_inds = p.data.abs() > threshold
            masks.append(pruned_inds.float())
    return masks


def prune_rate(model, verbose=True):
    """
    Print out prune rate for each layer and the whole network
    """
    total_nb_param = 0
    nb_zero_param = 0

    layer_id = 0

    for parameter in model.parameters():

        param_this_layer = 1
        for dim in parameter.data.size():
            param_this_layer *= dim
        total_nb_param += param_this_layer

        # only pruning linear and conv layers
        if len(parameter.data.size()) != 1:
            layer_id += 1
            zero_param_this_layer = \
                np.count_nonzero(parameter.cpu().data.numpy()==0)
            nb_zero_param += zero_param_this_layer

            if verbose:
                print("Layer {} | {} layer | {:.2f}% parameters pruned" \
                    .format(
                        layer_id,
                        'Conv' if len(parameter.data.size()) == 4 \
                            else 'Linear',
                        100.*zero_param_this_layer/param_this_layer,
                        ))
    pruning_perc = 100.*nb_zero_param/total_nb_param
    if verbose:
        print("Final pruning rate: {:.2f}%".format(pruning_perc))
    return pruning_perc


def finetune(model, param, train_loader):
    optimizer = torch.optim.RMSprop(model.parameters(), lr=param['learning_rate'], 
                                weight_decay=param['weight_decay'])
    error = nn.MSELoss()
    EPOCHS = param['epoch']
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = error(output, targets)
            loss.backward()
            optimizer.step()
            
            # Total loss
            total_loss += loss.item()
        # print("epoch : {}/{}, MSE loss = {:.6f}".format(epoch + 1, EPOCHS, total_loss/len(train_loader)))
        