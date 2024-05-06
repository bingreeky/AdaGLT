import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import utils
from functools import partial

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl.nn import EdgeWeightNorm

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""

from layer_gin import GINLayer, ApplyNodeFunc, MLP, MaskedMLP, MaskedLinear, BinaryStep
import pdb


class GINNet(nn.Module):
    
    def __init__(self, embedding_dim, graph, device, spar_wei, spar_adj,  coef=None, mode="prune"):
        super().__init__()

        self.mode = mode
        self.device = device
        self.layer_num = len(embedding_dim) - 1
        
        self.spar_wei = spar_wei
        self.spar_adj = spar_adj
        self.edge_mask_archive = []
        self.coef = coef

        self.norm =  partial(EdgeWeightNorm(norm='right'), graph)
        in_dim = embedding_dim[0]
        hidden_dim = embedding_dim[1]
        n_classes = embedding_dim[-1]
        dropout = 0.5
        # self.n_layers = 2
        self.edge_num = graph.all_edges()[0].numel()
        n_mlp_layers = 1               # GIN
        learn_eps = True              # GIN
        neighbor_aggr_type = 'mean' # GIN
        graph_norm = False      
        batch_norm = False
        residual = False
        self.n_classes = n_classes
        
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        
        for layer in range(self.layer_num):
            if layer == 0:
                if self.spar_wei:
                    mlp = MaskedMLP(n_mlp_layers, in_dim, hidden_dim, hidden_dim)
                else:
                    mlp = MLP(n_mlp_layers, in_dim, hidden_dim, hidden_dim)
            elif layer == self.layer_num - 1:
                if self.spar_wei:
                    mlp = MaskedMLP(n_mlp_layers, hidden_dim, hidden_dim, n_classes)
                else:
                    mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, n_classes)
            else:
                if self.spar_wei:
                    mlp = MaskedMLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
                else:
                    mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, graph_norm, batch_norm, residual, 0, learn_eps))

        self.linears_prediction = nn.Linear(hidden_dim, n_classes, bias=False)

        if self.spar_adj: # long-tail
            self.adj_thresholds = nn.ParameterList(
                [nn.Parameter(torch.ones(size=(graph.number_of_nodes(),)) * utils.initalize_thres(coef)) for i in range(self.layer_num)])
            self.edge_learner = nn.Sequential(
                nn.Linear(embedding_dim[0] * 2, 2048),
                nn.ReLU(),
                nn.Linear(2048,1)
            )
            self.sim_learner_src = nn.Linear(embedding_dim[0], 512)
            self.sim_learner_tgt = nn.Linear(embedding_dim[0], 512)
    
    def __create_learner_input(self, row, col, embeds):
        row_embs, col_embs = embeds[row], embeds[col]
        # node_emb = embeds[node_id].repeat(row.size(0), 1)
        edge_learner_input = torch.cat([row_embs, col_embs], 1)
        return edge_learner_input
    
    
    def learn_soft_edge(self, x, edge_index, ln=0):
        row, col = edge_index
        input = self.__create_learner_input(row, col, x)
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
        row_embs, col_embs = self.sim_learner_src(x[row]), self.sim_learner_tgt(x[col])
        # left = torch.einsum("ik,kk->ik",row_embs,self.mid_learner)
        edge_weight =  torch.einsum("ik,ik->i",row_embs, col_embs)
        mean = edge_weight.mean()
        variance = edge_weight.var()
        edge_weight = ((edge_weight - mean) * torch.sqrt(0.0001 / variance)) + 1
        return edge_weight

    def adj_pruning(self, edge_weight, thres, row, col, prev_mask):
        mean = edge_weight.mean()
        variance = edge_weight.var()
        edge_weight = ((edge_weight - mean) * torch.sqrt(0.1 / variance))

        self.edge_weight = edge_weight
        mask = BinaryStep.apply(edge_weight - (thres[row]))
        
    def adj_pruning2(self, edge_weight, thres, row, col, prev_mask, tau=0.1):
        mean = edge_weight.mean()
        variance = edge_weight.var()
        edge_weight = ((edge_weight - mean) * torch.sqrt(1 / variance))
        thres_trans = lambda x: self.coef*(torch.pow(x,3) + 5*x)
        y_soft = torch.sigmoid((edge_weight - thres_trans(thres[row])) / tau)
        y_hrad = ((y_soft  > 0.5)).float()
        ret = y_hrad - y_soft.detach() + y_soft
        return ret * prev_mask if prev_mask is not None else ret

    def forward_retain(self, g, h, snorm_n, snorm_e, edge_masks, wei_masks):
        hidden_rep = []
        g.edata['mask'] = torch.ones(g.edges()[0].shape).to(h.device)
        for i in range(self.layer_num):
            if edge_masks:
                g.edata['mask'] = edge_masks[i]
            
            if len(wei_masks):
                h = self.ginlayers[i](g, h, snorm_n, wei_mask=wei_masks[i])
            else:
                h = self.ginlayers[i](g, h, snorm_n)
            hidden_rep.append(h)
        # score_over_layer = (self.linears_prediction(hidden_rep[0]) + hidden_rep[1]) / 2
        score_over_layer = (self.linears_prediction(hidden_rep[0]) + hidden_rep[1]) / 2
        # TODO implementation is only for 2 layers!
        return score_over_layer

    def forward(self, g, h, snorm_n, snorm_e, **kwargs):

        if self.mode == "retain":
            return self.forward_retain(g, h, snorm_n, snorm_e, kwargs['edge_masks'], kwargs['wei_masks'])

        edge_weight = None
        if self.spar_adj:
            self.edge_weight = edge_weight = self.learn_soft_edge(h, g.edges())
            edge_mask = None
            self.edge_mask_archive = []        
        
        hidden_rep = []
        g.edata['mask'] = torch.ones(size=(g.number_of_edges(),)).to(h.device)
        for i in range(self.layer_num):
            if self.spar_adj and not kwargs['pretrain']:
                edge_mask = self.adj_pruning2(edge_weight, self.adj_thresholds[i], g.edges()[0], g.edges()[1], edge_mask)
                if not self.training: print(f"l{i}: [{(1 - edge_mask.sum() / edge_mask.shape[0])*100 : .3f}%]", end=" | ")
                self.edge_mask_archive.append(copy.deepcopy(edge_mask.detach()))
                g.edata['mask'] = (edge_mask * edge_weight)
            elif (not self.spar_adj) and (not self.spar_wei):
                g.edata['mask'] = (edge_weight)
                pass

            h = self.ginlayers[i](g, h, snorm_n)
            hidden_rep.append(h)

        # score_over_layer = (self.linears_prediction(hidden_rep[0]) + hidden_rep[1]) / 2
        score_over_layer = (self.linears_prediction(hidden_rep[0]) + hidden_rep[1]) / 2
        # TODO acutual implementation is only for 2 layers!
        return score_over_layer
    
    def generate_wei_mask(self):
        if not self.spar_wei:
            return []
        with torch.no_grad():
            wei_masks = []
            for layer in self.ginlayers:
                wei_masks.append(layer.apply_func.mlp.linear.mask.detach())
        return wei_masks
      
