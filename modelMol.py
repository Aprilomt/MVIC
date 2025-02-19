import torch.nn as nn
from layer import GINConv
import dgl


class modelMol(nn.Module):
    def __init__(self, in_features=55, hidden_dim=128):
        super(modelMol, self).__init__()

        self.node_fc = nn.Linear(in_features, hidden_dim)
        self.gnn = nn.ModuleList([GINConv(hidden_dim, 5) for i in range(3)])

    def do_gnn(self, g, x):
        for gnn in self.gnn:
            x = gnn(g, x)
        
        return x

    def forward(self, drugs, x):
        x = self.node_fc(x)

        x = self.do_gnn(drugs, x)
        drugs.ndata['h'] = x
        out = dgl.sum_nodes(drugs, 'h')
        return out
