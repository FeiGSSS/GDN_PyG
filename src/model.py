# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2023/12/03 10:27:41
@Author  :   Fei Gao
Beijing, China
'''
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing, Linear, MLP, knn_graph
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_geometric.typing import OptTensor

class EmbeddingGAT(MessagePassing):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 heads: int = 1,
                 negative_slope: float = 0.2,
                 dropout: float = 0,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        self.lin = Linear(in_channels, heads * out_channels, bias=True)
        
        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.empty(1, heads, 2*out_channels))
        self.att_dst = Parameter(torch.empty(1, heads, 2*out_channels))
        self.bias = Parameter(torch.empty(out_channels))
        
        self.batch_norm = nn.BatchNorm1d(out_channels)
        
        self.reset_parameters()
        
        
    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)
        
    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                embedding: Tensor,
                return_attention_weights: bool = True) -> Tensor:
        """_summary_

        Args:
            x (Tensor): node features with sie [number_nodes * batch_size, in_channels]
            edge_index (Tensor): edge index with size [2, edge_num*batch_size]
            embedding (Tensor): embedding matrix with size [number_nodes * batch_size, out_channels]
            return_attention_weights (_type_, optional): _description_. Defaults to None.
        """
        H, C = self.heads, self.out_channels
        assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
        x =  self.lin(x).view(-1, H, C) # [number_nodes * batch_size, heads, out_channels])
        embedding = embedding.unsqueeze(1).repeat(1, H, 1) # [number_nodes * batch_size, heads, out_channels]
        xAndemb_src = xAndemb_dst = torch.concat((x, embedding), dim=-1)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (xAndemb_src * self.att_src).sum(dim=-1)
        alpha_dst = (xAndemb_dst * self.att_dst).sum(dim=-1)
        alpha = (alpha_src, alpha_dst)
        
        # We only want to add self-loops for nodes that appear both as
        # source and target nodes:
        assert x.size(0) == x.size(0)
        num_nodes = x.size(0)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            
        alpha = self.edge_updater(edge_index, alpha=alpha)
        out = self.propagate(edge_index, x=x, alpha=alpha)
        out = out.mean(dim=1)
        out = out + self.bias
        out = self.batch_norm(out)
        out = F.relu(out)
        
        if return_attention_weights:
            return out, (edge_index, alpha)
        else:
            return out
        
        
    def edge_update(self, alpha_j: Tensor,
                          alpha_i: OptTensor,
                          index: Tensor, 
                          ptr: OptTensor,
                          size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha
    
    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j


class GDN(nn.Module):
    def __init__(self,
                 number_nodes: int,
                 in_dim: int=10,
                 hid_dim: int=64,
                 out_layer_hid_dim: int=256,
                 out_layer_num: int=1,
                 topk: int=20,
                 heads: int=1):
        super(GDN, self).__init__()
        self.number_nodes = number_nodes
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.topk = topk
        self.embedding = nn.Embedding(number_nodes, hid_dim)
        self.batch_norm_out = nn.BatchNorm1d(hid_dim)
        self.dropout = nn.Dropout(0.2)
        
        self.gnn = EmbeddingGAT(in_dim, hid_dim, heads=heads)
        
        self.out_layer = MLP(in_channels=hid_dim,
                             out_channels=1,
                             hidden_channels=out_layer_hid_dim,
                             num_layers=out_layer_num,
                             act="relu",
                             norm="batch_norm")
        
        self.loss_fun = nn.MSELoss(reduction='mean')
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.embedding.weight,
                                 a = torch.sqrt(torch.Tensor([5])).item())
        self.batch_norm_out.reset_parameters()
    
    @staticmethod
    def get_batch_edge_index(edge_index: Tensor, number_batch: int, number_nodes: int):
        number_edges = edge_index.size(1)
        edge_index = edge_index.repeat(1, number_batch)
        for i in range(1, number_batch):
            s = i * number_edges
            e = (i+1) * number_edges
            edge_index[:, s:e] += i * number_nodes
        return edge_index
        
        
        
    def forward(self, batch_x: Tensor) -> Tensor:
        """GDN forward

        Args:
            batch_x (Tensor): node features with size [batch_size, number_nodes, number_node_features]

        Returns:
            Tensor: node prediction with size [batch_size, number_nodes]
        """
        B, N, C = batch_x.shape
        assert C == self.in_dim, "the input dim of x not match!"
        
        # get node features for GNN
        # --> size [number_nodes * batch_size, number_node_features]
        x = batch_x.reshape(-1, C)
        
        # get batched KNN edge_index for GNN
        embeddings = self.embedding.weight
        knn_edge_index = knn_graph(embeddings, self.topk)
        edge_index_repeat = self.get_batch_edge_index(knn_edge_index, B, N)
        
        # get batched embeddings for GNN
        embeddings_repeat = embeddings.unsqueeze(0).repeat(B, 1, 1).view(-1, self.hid_dim)
        
        # now, input all above into GNN layer
        out, (edge_index, alpha) = self.gnn(x,
                                            edge_index_repeat,
                                            embeddings_repeat,
                                            return_attention_weights=True)
        
        # output
        out = torch.mul(out, embeddings_repeat)
        out = self.dropout(self.batch_norm_out(out))
        out = self.out_layer(out) # [number_nodes * batch_size, 1]
        return out.view(B, N)
    
    def loss(self, pred: Tensor, label: Tensor):
        return self.loss_fun(pred, label)