import torch
import pandas as pd
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import global_add_pool, global_mean_pool

# The PyG built-in GCNConv
from torch_geometric.nn import GCNConv

import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):
        super(GCN, self).__init__()

        
        self.convs = None # A list of GCNConv layers
        self.bns = None # A list of 1D batch normalization layers
        self.softmax = None # The log softmax layer

        self.num_layers = num_layers
        
        self.convs = torch.nn.ModuleList([GCNConv(in_channels=input_dim, out_channels= hidden_dim)])
        self.convs.extend([GCNConv(in_channels=hidden_dim, out_channels= hidden_dim) for i in range(num_layers -2)])
        self.convs.extend([GCNConv(in_channels=hidden_dim, out_channels= output_dim)])
    
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features = hidden_dim) for i in range(num_layers -1)])
        
        self.softmax = torch.nn.LogSoftmax(dim = 1)

        # Probability of an element getting zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i in range(self.num_layers):
            if i == (self.num_layers -1):
                x=self.convs[i](x,adj_t)
                if self.return_embeds == True:
                    return x
                else:
                    out = self.softmax(x)
            else:
                x = self.convs[i](x, adj_t)
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p = self.dropout, training = self.training)  
        return out


### GCN to predict graph property
class GCN_Graph(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout):
        super(GCN_Graph, self).__init__()

        # Load encoders for Atoms in molecule graphs
        self.node_encoder = AtomEncoder(hidden_dim)

        # Node embedding model
        # Note that the input_dim and output_dim are set to hidden_dim
        self.gnn_node = GCN(hidden_dim, hidden_dim,
            hidden_dim, num_layers, dropout, return_embeds=True)

        self.pool = global_mean_pool

        # Output layer
        self.linear = torch.nn.Linear(hidden_dim, output_dim)


    def reset_parameters(self):
      self.gnn_node.reset_parameters()
      self.linear.reset_parameters()

    def forward(self, batched_data):
        # Extract important attributes of our mini-batch
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        embed = self.node_encoder(x)
        embed = self.gnn_node(embed, edge_index)
        features = self.pool(embed,batch)
        out = self.linear(features)

        return out