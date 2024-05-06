import torch
import torch.nn as nn
import pdb
import copy
import math
import utils
import scipy.optimize as optimize
# from torch_geometric.nn import GCNConv
from layers import BinaryStep, MaskedLinear
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_adj, to_undirected

import matplotlib.pyplot as plt



class net_gcn_dense(nn.Module):
    def __init__(self, embedding_dim, edge_index, device, spar_wei, spar_adj, num_nodes, use_res, use_bn, coef=None, mode="prune"):
        super().__init__()

        self.mode = mode
        self.adj_binary = to_dense_adj(edge_index)[0]
        self.layer_num = len(embedding_dim) - 1
        
        self.spar_wei = spar_wei
        self.spar_adj = spar_adj
        self.edge_mask_archive = []
        self.coef = coef
        
        self.use_bn = use_bn
        self.use_res = use_res
        
        if self.use_bn:
            self.norms = nn.ModuleList()
            for i in range(self.layer_num):
                self.norms.append(nn.BatchNorm1d(embedding_dim[i]))

        if self.spar_wei:
            self.net_layer = nn.ModuleList([ MaskedLinear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
        else:
            self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.normalize = utils.torch_normalize_adj
        self.device = device
        
        if self.spar_adj:
            self.adj_thresholds = nn.ParameterList(
                [nn.Parameter(torch.ones(size=(num_nodes,)) * utils.initalize_thres(coef)) for i in range(self.layer_num)])
            self.edge_learner = nn.Sequential(
                nn.Linear(embedding_dim[0] * 2, 2048),
                nn.ReLU(),
                # nn.Linear(1024, 256),
                # nn.ReLU(),
                nn.Linear(2048,1)
            )
            self.sim_learner_src = nn.Linear(embedding_dim[0], 512)
            self.sim_learner_tgt = nn.Linear(embedding_dim[0], 512)
    
    def __create_learner_input(self, edge_index, embeds):
        row, col = edge_index[0], edge_index[1]
        row_embs, col_embs = embeds[row], embeds[col]
        edge_learner_input = torch.cat([row_embs, col_embs], 1)
        return edge_learner_input

    def forward_retain(self, x, edge_index, val_test, edge_masks, wei_masks):
        adj_ori = self.adj_binary
        for ln in range(self.layer_num):
            adj = adj_ori * edge_masks[ln] if len(edge_masks) != 0 else adj_ori
            adj = self.normalize(adj, device=self.device)
            if ln and self.use_bn: x = self.norms[ln](x)
            x = torch.mm(adj, x)
            if  len(wei_masks):
                x = self.net_layer[ln](x, wei_masks[ln])
            else:
                x = self.net_layer[ln](x)
            if not ln: h = x
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if not val_test:
                x = self.dropout(x)
            if ln and self.use_res: x += h
        return x

    def forward(self, x, edge_index, val_test=False, **kwargs):

        if self.mode == "retain":
            return self.forward_retain(x, edge_index, val_test, kwargs['edge_masks'], kwargs['wei_masks'])

        edge_weight = None
        if self.spar_adj:
            edge_weight = self.learn_soft_edge(x, edge_index)
            adj_mask = self.adj_binary
            self.edge_mask_archive = []

        adj_ori = to_dense_adj(edge_index, edge_attr=edge_weight)[0]
        # adj_ori = self.adj_binary        
        for ln in range(self.layer_num):
            
            adj = adj_ori
            if self.spar_adj and not kwargs['pretrain']:
                adj_mask = self.adj_pruning2(adj_ori, self.adj_thresholds[ln], adj_mask)
                # if val_test: 
                #     print(f"l{ln}: [{(1 - (adj_mask.nonzero().shape[0] / edge_index.shape[1]))*100 :.3f}%]", end=" | ")
                    # print(f"mean: {((self.adj_thresholds[0])).mean().item()}")
                self.edge_mask_archive.append(copy.deepcopy(adj_mask.detach().cpu()))
                adj = adj_mask * adj_ori

            adj = self.normalize(adj, self.device) if not kwargs['pretrain'] else adj_ori
            if ln and self.use_bn: x = self.norms[ln](x)
            x = torch.mm(adj, x)
            x = self.net_layer[ln](x)
            if not ln: h = x
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if not val_test:
                x = self.dropout(x)
            if ln and self.use_res: x += h
        return x
    
    def learn_soft_edge(self, x, edge_index, ln=0):
        input = self.__create_learner_input(edge_index, x)
        if ln == 0:
            edge_weight =  self.edge_learner(input).squeeze(-1)
        elif ln == 1:
            edge_weight =  self.edge_learner2(input).squeeze(-1)
        else:
            raise NotImplementedError
        mean = edge_weight.mean()
        variance = edge_weight.var()
        edge_weight = ((edge_weight - mean) * torch.sqrt(0.0001 / variance)) + 1
        return edge_weight
    
    def learn_soft_edge2(self, x, edge_index):
        row, col = edge_index
        # print(x.device)
        row_embs, col_embs = self.sim_learner_src(x[row]), self.sim_learner_tgt(x[col])
        # left = torch.einsum("ik,kk->ik",row_embs,self.mid_learner)
        edge_weight =  torch.einsum("ik,ik->i",row_embs, col_embs)
        deg = torch.zeros(edge_index.max().item() + 1,
                          dtype=torch.float, device=self.device)

        exp_wei = torch.exp(edge_weight / 3)
        deg.scatter_add_(0, edge_index[0], exp_wei)
        edge_weight = exp_wei / (deg[edge_index[0]] ) 

        return edge_weight


    def adj_pruning(self, adj, thres, prev_mask):
        mask = BinaryStep.apply(adj - utils.log_custom(thres).view(-1,1))
        return mask * prev_mask if prev_mask is not None else mask
    
    def adj_pruning2(self, adj, thres, prev_mask, tau=0.1, val_test=False):
        edge_weight = adj[adj.nonzero(as_tuple=True)]
        edge_index = adj.nonzero().t().contiguous()
        mean = edge_weight.mean()
        variance = edge_weight.var()
        edge_weight = ((edge_weight - mean) * torch.sqrt(0.01 / variance))
        adj = to_dense_adj(edge_index, edge_attr=edge_weight)[0]
        B = adj.size(0)
        thres_trans = lambda x: self.coef*(torch.pow(x,3) + 20*x)     
        y_soft = torch.sigmoid((adj - thres_trans(thres)) / tau)
        # y_hrad = (y_soft > 0.5).float()
        y_hrad = ((y_soft + torch.eye(adj.shape[0]).to(self.device))  > 0.5).float()
        ret = y_hrad - y_soft.detach() + y_soft
        return ret * prev_mask # if prev_mask is not None else ret


    def generate_wei_mask(self):
        if not self.spar_wei:
            return []
        with torch.no_grad():
            wei_masks = []
            for layer in self.net_layer:
                wei_masks.append(layer.mask.detach().cpu())
        return wei_masks
