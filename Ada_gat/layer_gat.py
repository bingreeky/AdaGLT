import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import pdb

from typing import Optional
import math


from torch import Tensor
from torch.nn import Parameter


"""
code partially from https://github.com/junjieliu2910/DynamicSparseTraining
"""
class BinaryStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero_index = torch.abs(input) > 1
        middle_index = (torch.abs(input) <= 1) * (torch.abs(input) > 0.4)
        additional = 2-4*torch.abs(input)
        additional[zero_index] = 0.
        additional[middle_index] = 0.4
        return grad_input*additional


class MaskedLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.threshold = Parameter(torch.empty(out_features))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.register_parameter("mask_archive", None)
        self.mask = None
        self.step = BinaryStep.apply
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        with torch.no_grad():
            # std = self.weight.std()
            self.threshold.data.fill_(0.)

    def forward(self, input: Tensor, mask=None) -> Tensor:
        if not mask is None:
            return F.linear(input, self.weight * mask, self.bias)
        if not self.mask_archive is None:
            return F.linear(input, self.weight * self.mask_archive, self.bias)
        abs_weight = torch.abs(self.weight)
        threshold = self.threshold.view(abs_weight.shape[0], -1)
        abs_weight = abs_weight-threshold
        self.mask = self.step(abs_weight)
        self.mask_archive = nn.Parameter(self.mask)
        masked_weight = self.weight * self.mask
        return F.linear(input, masked_weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class GATHeadLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, graph_norm, batch_norm, heads):
        super().__init__()
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.batchnorm_h = nn.BatchNorm1d(out_dim)

        self.heads = heads

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        
        ### edges.src['z']: edge src node feature
        ### edges.data['e']: edge feature
        return {'z': edges.src['z'], 'e': edges.data['e']} # this dict all save in message box

    def reduce_func(self, nodes):
       
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h, snorm_n, edge_mask):

        z = self.fc(h)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention) # The function to generate new edge features
        ### pruning edges 
        # print(g.edata['e'].shape, edge_mask.shape)
        if not edge_mask is None:
            g.edata['e'] = g.edata['e'] * edge_mask.view(-1,1)
        g.update_all(self.message_func, self.reduce_func)

        h = g.ndata['h']
        if self.graph_norm:
            h = h * snorm_n
        if self.batch_norm:
            h = self.batchnorm_h(h)

        if not self.heads == 1:
            h = F.elu(h)
            h = F.dropout(h, self.dropout, training=self.training)
        return h

class GATLayer(nn.Module):
    """
        Param: [in_dim, out_dim, n_heads]
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout, graph_norm, batch_norm, residual=False):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.residual = residual
        
        if in_dim != (out_dim*num_heads):
            self.residual = False
        
        self.heads = nn.ModuleList()
        for i in range(num_heads): # 8
            self.heads.append(GATHeadLayer(in_dim, out_dim, dropout, graph_norm, batch_norm, num_heads))
        self.merge = 'cat' 

    def forward(self, g, h, snorm_n, edge_mask):
        h_in = h # for residual connection
        head_outs = [attn_head(g, h, snorm_n, edge_mask) for attn_head in self.heads]
        
        if self.merge == 'cat':
            h = torch.cat(head_outs, dim=1)
        else:
            h = torch.mean(torch.stack(head_outs))
        
        if self.residual:
            h = h_in + h # residual connection
        return h
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)


class MaskedGATHeadLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, graph_norm, batch_norm, heads):
        super().__init__()
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        
        self.fc = MaskedLinear(in_dim, out_dim, bias=False)
        self.attn_fc = MaskedLinear(2 * out_dim, 1, bias=False)
        self.batchnorm_h = nn.BatchNorm1d(out_dim)

        self.heads = heads

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        
        ### edges.src['z']: edge src node feature
        ### edges.data['e']: edge feature
        return {'z': edges.src['z'], 'e': edges.data['e']} # this dict all save in message box

    def reduce_func(self, nodes):
       
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h, snorm_n, edge_mask):

        z = self.fc(h)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention) # The function to generate new edge features
        ### pruning edges 
        g.edata['e'] = g.edata['e'] * edge_mask if not edge_mask is None else g.edata['e']
        g.update_all(self.message_func, self.reduce_func)

        h = g.ndata['h']
        if self.graph_norm:
            h = h * snorm_n
        if self.batch_norm:
            h = self.batchnorm_h(h)

        if not self.heads == 1:
            h = F.elu(h)
            h = F.dropout(h, self.dropout, training=self.training)
        return h

class MaskedGATLayer(nn.Module):
    """
        Param: [in_dim, out_dim, n_heads]
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout, graph_norm, batch_norm, residual=False):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.residual = residual
        
        if in_dim != (out_dim*num_heads):
            self.residual = False
        
        self.heads = nn.ModuleList()
        for i in range(num_heads): # 8
            self.heads.append(MaskedGATHeadLayer(in_dim, out_dim, dropout, graph_norm, batch_norm, num_heads))
        self.merge = 'cat' 

    def forward(self, g, h, snorm_n, edge_mask):
        h_in = h # for residual connection
        head_outs = [attn_head(g, h, snorm_n, edge_mask) for attn_head in self.heads]
        
        if self.merge == 'cat':
            h = torch.cat(head_outs, dim=1)
        else:
            h = torch.mean(torch.stack(head_outs))
        
        if self.residual:
            h = h_in + h # residual connection
        return h
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):

        super().__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.input_dim = input_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim, bias=False)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.linears[i](h))
            return self.linears[-1](h)

class MaskedMLP(nn.Module):
    """MLP with linear output -> merely 1 layer!"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):

        super().__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.input_dim = input_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = MaskedLinear(input_dim, output_dim, bias=False)
        else:
            raise NotImplementedError

    def forward(self, x, **kwargs):
        if self.linear_or_not:
            # If linear model
            mask = kwargs.get("wei_mask")
            return self.linear(x, mask)
        else:
            raise NotImplementedError