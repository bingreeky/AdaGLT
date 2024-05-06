import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import copy
"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
from layer_gat import GATLayer, MaskedGATLayer
# from gnns.mlp_readout_layer import MLPReadout
import pdb

class GATNet(nn.Module):

    def __init__(self, embedding_dim, graph, device, spar_wei, spar_adj,coef=None, mode="prune"):
        super().__init__()

        self.mode = mode
        self.device = device
        self.layer_num = len(embedding_dim) - 1
        
        self.spar_wei = spar_wei
        self.spar_adj = spar_adj
        self.edge_mask_archive = []
        self.coef = coef
        
        in_dim_node = embedding_dim[0] # node_dim (feat is an integer)
        hidden_dim = embedding_dim[1]
        out_dim = embedding_dim[-1]
        n_classes = embedding_dim[-1]
        num_heads = 4
        dropout = 0.6
        n_layers = 1
        self.edge_num = graph.number_of_edges() + graph.number_of_nodes()
        self.graph_norm = False
        self.batch_norm = False
        self.residual = True
        self.dropout = dropout
        self.n_classes = n_classes
        
        self.layers = nn.ModuleList()
        if not self.spar_wei:

            for ln in range(self.layer_num):
                if ln == 0:
                    self.layers.append(GATLayer(in_dim_node, hidden_dim, num_heads,
                                                dropout, self.graph_norm, self.batch_norm, self.residual))
                elif ln == self.layer_num - 1:
                    self.layers.append(GATLayer(hidden_dim * num_heads, out_dim, 1, 0, self.graph_norm, self.batch_norm, self.residual))
                else:
                    self.layers.append(GATLayer(hidden_dim * num_heads, hidden_dim, num_heads,
                                                dropout, self.graph_norm, self.batch_norm, self.residual))
        else:
            self.layers = nn.ModuleList([MaskedGATLayer(in_dim_node, hidden_dim, num_heads,
                                                dropout, self.graph_norm, self.batch_norm, self.residual) for _ in range(n_layers)])
            self.layers.append(MaskedGATLayer(hidden_dim * num_heads, out_dim, 1, 0, self.graph_norm, self.batch_norm, self.residual))

        if self.spar_adj: 
            self.adj_thresholds = nn.ParameterList(
                [nn.Parameter(torch.ones(size=(graph.number_of_nodes(),)) * 0.1) for i in range(self.layer_num)])
            self.sim_learner_src = nn.Linear(embedding_dim[0], 512)
            self.sim_learner_tgt = nn.Linear(embedding_dim[0], 512)
    
    def learn_soft_edge(self, x, edge_index):
        row, col = edge_index
        # print(x.device)
        row_embs, col_embs = self.sim_learner_src(x[row]), self.sim_learner_tgt(x[col])
        # left = torch.einsum("ik,kk->ik",row_embs,self.mid_learner)
        edge_weight =  torch.einsum("ik,ik->i",row_embs, col_embs)

        mean = edge_weight.mean()
        variance = edge_weight.var()
        edge_weight = ((edge_weight - mean) * torch.sqrt(0.0001 / variance)) + 1

        return edge_weight

    def adj_pruning2(self, edge_weight, thres, row, col, prev_mask, tau=0.1):
        mean = edge_weight.mean()
        variance = edge_weight.var()
        edge_weight = ((edge_weight - mean) * torch.sqrt(0.01 / variance))
    
        thres_trans = lambda x: self.coef*(torch.pow(x,3) + 2*x)
        y_soft = torch.sigmoid((edge_weight - thres_trans(thres[row])) / tau)
        y_hrad = ((y_soft  > 0.5)).float()

        ret = y_hrad - y_soft.detach() + y_soft
        return ret * prev_mask if prev_mask is not None else ret

    def forward_retain(self, g, h, snorm_n, snorm_e, edge_masks, wei_masks):
        for ln, conv in enumerate(self.layers):
            if len(edge_masks):
                h = conv(g, h, snorm_n, edge_masks[ln])
            else:
                h = conv(g, h, snorm_n, None)                
        return h
    
    
    def forward(self, g, h, snorm_n, snorm_e, **kwargs):
        
        if self.mode == "retain":
            return self.forward_retain(g, h, snorm_n, snorm_e, kwargs['edge_masks'], kwargs['wei_masks'])


        edge_weight = None
        if self.spar_adj:
            self.edge_weight = edge_weight = self.learn_soft_edge(h, g.edges())
            edge_mask = None
            self.edge_mask_archive = []           
        
        
        # GAT
        for ln, conv in enumerate(self.layers):
            if self.spar_adj and not kwargs['pretrain']:
                edge_mask = self.adj_pruning2(edge_weight, self.adj_thresholds[ln], g.edges()[0], g.edges()[1], edge_mask)
                # if not self.training: print(f"l{ln}: [{(1 - edge_mask.sum() / edge_mask.shape[0])*100 : .3f}%]", end=" | ")
                self.edge_mask_archive.append(copy.deepcopy(edge_mask.detach()))
            elif (not self.spar_adj) :
                edge_mask = None
                pass
            h = conv(g, h, snorm_n, edge_mask)
            
        return h
    
    def generate_wei_mask(self,):
        if not self.spar_wei:
            return []
        return {key:item for key, item in self.state_dict().items() if "mask_archive" in key }
