import torch.nn as nn
import torch
import torch.nn.functional as F
import dgl
import dgl.function as fn



class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features // 2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features // 2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features // 2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))

    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        values = receiver

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        attentions = e_scores

        return attentions


class Aggregation(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features

    def forward(self, heads, tails, alpha_scores):
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)

        scores = heads @ tails.transpose(-2, -1)

        if alpha_scores is not None:
            scores = alpha_scores * scores
        
        score_nosum = scores
        scores = scores.sum(dim=(-2, -1))
        return scores, score_nosum


class MLP(nn.Sequential):
    def __init__(self, hidden_dim, num_layers, dropout=0.5):
        def build_block(input_dim, output_dim):
            return [
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]

        m = build_block(hidden_dim, 2 * hidden_dim)
        for i in range(1, num_layers - 1):
            m += build_block(2 * hidden_dim, 2 * hidden_dim)
        m.append(nn.Linear(2 * hidden_dim, hidden_dim))

        super().__init__(*m)


class GINConv(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(GINConv, self).__init__()
        self.mlp = MLP(hidden_dim, num_layers, 0.0)

    def forward(self, g, x):
        g.ndata['h'] = x

        g.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.sum('m', 'h_neigh'))
        h_neigh = g.ndata.pop('h_neigh')

        out = x + self.mlp(h_neigh)
        return out

