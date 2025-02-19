import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import math

from modelMol import modelMol
from layer import CoAttentionLayer, Aggregation


class CoAttentionModel(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.kge_dim = emb_dim
        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.agg = Aggregation(self.kge_dim)

    def forward(self, repr_dg, repr_pt):
        attentions = self.co_attention(repr_dg, repr_pt)
        scores, score9 = self.agg(repr_dg, repr_pt, attentions)
        return scores, score9, attentions


class Transformer(nn.Module):
    def __init__(self, n_channels, num_heads=1, att_drop=0, act='none'):
        super().__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        assert self.n_channels % (self.num_heads * 4) == 0

        self.query = nn.Linear(self.n_channels, self.n_channels // 4)
        self.key = nn.Linear(self.n_channels, self.n_channels // 4)
        self.value = nn.Linear(self.n_channels, self.n_channels)

        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif act == 'none':
            self.act = lambda x: x
        else:
            assert 0, f'Unrecognized activation function {act} for class Transformer'
        
        self.reset_parameters()

    def reset_parameters(self):
        for k, v in self._modules.items():
            if hasattr(v, 'reset_parameters'):
                v.reset_parameters()
        nn.init.zeros_(self.gamma)
    
    def forward(self, x, y, mask=None):
        B, M, C = x.size() 
        M2 = y.size(-2)   
        H = self.num_heads

        f = self.query(x).view(B, M, H, -1).permute(0, 2, 1, 3)
        g = self.key(y).view(B, M2, H, -1).permute(0, 2, 3, 1)
        h = self.value(y).view(B, M2, H, -1).permute(0, 2, 1, 3)

        beta = F.softmax(self.act(f @ g / math.sqrt(f.size(-1))), dim=-1)
        beta = self.att_drop(beta)

        o = self.gamma * (beta @ h)
        return o.permute(0, 2, 1, 3).reshape((B, M, C)) + x



class NetView(nn.Module):
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(NetView, self).__init__()

        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads, dropout, dropout, activation=F.elu, allow_zero_in_degree=True,))

        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

        self.sefusion = Transformer(out_size * layer_num_heads)


    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]

            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))


        direct_emb = semantic_embeddings[0]
        semantic_embeddings = semantic_embeddings[1:]   
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)    # (n, num_mp, dim)
        semanticLevel = self.sefusion(direct_emb.unsqueeze(1), semantic_embeddings).squeeze(1)
        return direct_emb, semanticLevel


class FeatsExtractor(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, num_heads, dropout):
        super(FeatsExtractor, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(NetView(meta_paths, in_size, hidden_size, num_heads[0], dropout))

        for l in range(1, len(num_heads)):
            self.layers.append(NetView(meta_paths, hidden_size * num_heads[l - 1], hidden_size, num_heads[l], dropout))

        self.mol = modelMol(hidden_dim = hidden_size * num_heads[0])

    def forward(self, g, h, drugs):
        fts = drugs.ndata['feat']
        outputs_mol = self.mol(drugs, fts)

        for gnn in self.layers:
            dir_emb, indir_emb = gnn(g, h)

        repr = torch.stack((dir_emb, indir_emb, outputs_mol), dim=1)
        return repr, dir_emb, indir_emb, outputs_mol


class DDModel(nn.Module):
    def __init__(self, d_meta_paths, in_size, hidden_size, num_heads, dropout):
        super(DDModel, self).__init__()

        self.hetero1 = FeatsExtractor(d_meta_paths, in_size, hidden_size, num_heads, dropout)

        self.co_attention = CoAttentionModel(emb_dim=hidden_size * num_heads[0])

        self.mlp = nn.ModuleList([nn.Linear(768, 256),
                                  nn.ReLU(),
                                  nn.Dropout(0.5), 
                                  nn.Linear(256, 128), 
                                  nn.ReLU(),
                                  nn.Dropout(0.5),
                                  nn.Linear(128, 1)])
        
    def do_mlp(self, vectors, layernum):
        for i in range(layernum):
            vectors = self.mlp[i](vectors)
        
        return vectors

    def forward(self, pos_pair, neg_pair, hg_d, feats_d, drugs):

        repr_dg, dirEmbs, indirEmbs, molEmbs = self.hetero1(hg_d, feats_d, drugs)
        repr_dg = F.normalize(repr_dg, p=2, dim=2)

        repr_head_pos = repr_dg[pos_pair[:, 0]] # (n, 3, d)
        repr_tail_pos = repr_dg[pos_pair[:, 1]]
        repr_head_neg = repr_dg[neg_pair[:, 0]]
        repr_tail_neg = repr_dg[neg_pair[:, 1]]


        p_score, score9, atte = self.co_attention(repr_head_pos, repr_tail_pos)
        n_score, _, _ = self.co_attention(repr_head_neg, repr_tail_neg)

        return p_score, n_score
    
        # repr_dg = repr_dg.view(1704, -1)
        # pn_pair = np.concatenate((pos_pair, neg_pair), axis=0)
        # repr_dg_head = repr_dg[pn_pair[:, 0]]
        # repr_dg_tail = repr_dg[pn_pair[:, 1]]
        # out = torch.cat([repr_dg_head, repr_dg_tail], dim=-1)
        # score = self.mlp(out).squeeze(-1)

        # return score

