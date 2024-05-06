import os
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.utils import remove_self_loops

import net as net
import layers
from args import parser_loader
import utils
from sklearn.metrics import f1_score
import pdb
import pruning
import copy
from scipy.sparse import coo_matrix
import warnings
warnings.filterwarnings('ignore')

def run_get_mask(args):
    rewind_weight=None
    device = args['device']
    
    adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type = \
        utils.load_citation(args['dataset'], task_type=args['task_type'])  # adj: csr_matrix

    adj = adj.to_dense().to(device).to(torch.float32)
    adj = adj.nonzero().t().contiguous()
    features = features.to(device).to(torch.float32)
    labels = labels.to(device)
    loss_func = nn.CrossEntropyLoss()

    net_gcn = net.net_gcn_dense(embedding_dim=args['embedding_dim'], edge_index=adj, device=device, 
                                spar_wei=args['spar_wei'], spar_adj=args['spar_adj'], num_nodes=features.shape[0],
                                use_bn=args['use_bn'], use_res=args['use_res'],coef=args['coef'])
    net_gcn = net_gcn.to(device)

    optimizer = torch.optim.Adam(net_gcn.parameters(
    ), lr=args['lr'], weight_decay=args['weight_decay'])

    acc_test = 0.0
    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0, "adj_spar": 0, "wei_spar": 0}
    best_target = {'val_acc': 0, 'epoch': 0, 'test_acc': 0, "mask": None}
    best_mask = None
    adj_mask_ls, wei_mask_ls = [], []
    
    rewind_weight = copy.deepcopy(net_gcn.state_dict())
    for epoch in range(args['total_epoch']):        
        net_gcn.train()

        optimizer.zero_grad()
        output = net_gcn(features, adj, pretrain=(epoch < args['pretrain_epoch']))

        loss = loss_func(output[idx_train], labels[idx_train])

        adj_loss = 0
        if args['spar_adj']:
            for thres in net_gcn.adj_thresholds:
                adj_loss += torch.exp(-thres).sum() / args['num_layers']
            adj_loss = (adj_loss * args['e1'])
            loss = loss + adj_loss

        wei_loss = 0
        if args['spar_wei']:
            for layer in net_gcn.modules():
                if isinstance(layer, layers.MaskedLinear):
                    wei_loss += args['e2'] * \
                        torch.sum(torch.exp(-layer.threshold))
        # print(wei_loss)
        loss += wei_loss

        loss.backward()
        
        optimizer.step()
        with torch.no_grad():
            net_gcn.eval()
            output = net_gcn(features, adj, val_test=True, pretrain=(epoch < args['pretrain_epoch']))
            acc_val = f1_score(labels[idx_val].cpu().numpy(
            ), output[idx_val].cpu().numpy().argmax(axis=1), average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(
            ), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')
            acc_train = f1_score(labels[idx_train].cpu().numpy(
            ), output[idx_train].cpu().numpy().argmax(axis=1), average='micro')

            aspar_here = utils.calcu_sparsity(net_gcn.edge_mask_archive, adj.shape[1])
            wspar_here = utils.net_weight_sparsity(net_gcn)

            # continuous setting for adj/wei
            in_interval = (args['spar_adj'] and utils.judge_spar(aspar_here, args['target_adj_spar']) or not args['spar_adj']) and \
                (args['spar_wei'] and utils.judge_spar(wspar_here, args['target_wei_spar']) or not args['spar_wei'])
            if in_interval and args['continuous']:
                # print(net_gcn.edge_mask_archive)
                adj_mask_ls.append(copy.deepcopy(net_gcn.edge_mask_archive))
                wei_mask_ls.append(copy.deepcopy(net_gcn.generate_wei_mask()))
        
            # for report
            meet = ((args['spar_adj'] and aspar_here > args['target_adj_spar']) or not args['spar_adj']) and \
                    ((args['spar_wei'] and wspar_here > args['target_wei_spar']) or not args['spar_wei']) 
            if acc_val > best_val_acc['val_acc'] and meet:
                best_val_acc['test_acc'] = acc_test
                best_val_acc['val_acc'] = acc_val
                best_val_acc['epoch'] = epoch
                best_mask = [copy.deepcopy(net_gcn.edge_mask_archive), copy.deepcopy(net_gcn.generate_wei_mask())]
                best_val_acc['wei_spar'] = wspar_here
                best_val_acc['adj_spar'] = aspar_here

            # sole setting for adj/wei
            if in_interval and acc_val > best_target['val_acc']:
                best_target['test_acc'] = acc_test
                best_target['val_acc'] = acc_val
                best_target['epoch'] = epoch
                best_target['mask'] = [copy.deepcopy(net_gcn.edge_mask_archive), copy.deepcopy(net_gcn.generate_wei_mask())]

            print("Epoch:[{}] L:[{:.3f}] AL:[{:.2f}] Train:[{:.2f}] Val:[{:.2f}] Test:[{:.2f}] WS:[{:.2f}%] AS:[{:.2f}%] |"
                  .format(epoch, loss.item(), (adj_loss), acc_train * 100, acc_val * 100, acc_test * 100, wspar_here, aspar_here), end=" ")
            if meet:
                print("Best Val:[{:.2f}] Test:[{:.2f}] AS:[{:.2f}%] WS:[{:.2f}%] at Epoch:[{}]"
                      .format(
                          best_val_acc['val_acc'] * 100,
                          best_val_acc['test_acc'] * 100,
                          best_val_acc['adj_spar'],
                          best_val_acc['wei_spar'],
                          best_val_acc['epoch']))
            else:
                print("")

    if args['continuous']:
        return adj_mask_ls, wei_mask_ls
    else:
        return best_target['mask']

def run_fix_mask(args, edge_masks, wei_masks, rewind_weight=None):
    device = args['device']

    edge_masks = [mask.to(device) for mask in edge_masks]
    wei_masks = [mask.to(device) for mask in wei_masks]
    
    adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type = \
        utils.load_citation(args['dataset'])  # adj: csr_matrix

    adj = adj.to_dense().to(device).to(torch.float32)
    adj = adj.nonzero().t().contiguous()
    features = features.to(device).to(torch.float32)
    labels = labels.to(device)
    loss_func = nn.CrossEntropyLoss()
    
    net_gcn = net.net_gcn_dense(embedding_dim=args['embedding_dim'], edge_index=adj, device=device,
                                spar_wei=args['spar_wei'], spar_adj=False, num_nodes=features.shape[0],
                                use_bn=args['use_bn'], use_res=args['use_res'], mode="retain")
    net_gcn = net_gcn.to(device)

    # net_gcn.load_state_dict(dict(filter(lambda x:"threshold" not in x[0], rewind_weight.items())))

    optimizer = torch.optim.Adam(net_gcn.parameters(
    ), lr=args['lr'], weight_decay=args['weight_decay'])

    acc_test = 0.0
    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0}
    
    for epoch in range(args['retain_epoch']):
        net_gcn.train()
        optimizer.zero_grad()
        output = net_gcn(features, adj, edge_masks=edge_masks,wei_masks=wei_masks)

        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        
        optimizer.step()
        with torch.no_grad():
            net_gcn.eval()
            output = net_gcn(features, adj, val_test=True,
                             edge_masks=edge_masks,wei_masks=wei_masks)
            acc_val = f1_score(labels[idx_val].cpu().numpy(
            ), output[idx_val].cpu().numpy().argmax(axis=1), average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(
            ), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')
            acc_train = f1_score(labels[idx_train].cpu().numpy(
            ), output[idx_train].cpu().numpy().argmax(axis=1), average='micro')
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['test_acc'] = acc_test
                best_val_acc['val_acc'] = acc_val
                best_val_acc['epoch'] = epoch

            print("Epoch:[{}] Train:[{:.2f}] Val:[{:.2f}] Test:[{:.2f}] | Best Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
                  .format(epoch, acc_train * 100 ,acc_val * 100, acc_test * 100,
                          best_val_acc['val_acc'] * 100,
                          best_val_acc['test_acc'] * 100,
                          best_val_acc['epoch']))
    # return best_val_acc['test_acc']

if __name__ == "__main__":

    args = parser_loader()
    print(args)
    # torch.autograd.set_detect_anomaly#(True)
    utils.fix_seed(args['seed'])
    
    edge_masks, wei_masks = run_get_mask(args)

    if not args['continuous']:
        run_fix_mask(args, edge_masks, wei_masks, rewind_weight=None)
    else:
        print(len(edge_masks))
        for emask, wmask in zip(edge_masks, wei_masks):
            run_fix_mask(args, emask, wmask, rewind_weight=None)