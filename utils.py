import numpy as np
import torch.cuda
import torch.nn.functional as F
import random
import scipy.sparse as sp
import dgl
import copy
import math
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from custom_loss import SigmoidLoss
from ogb.utils import smiles2graph
from torch_geometric.data import Data



default_configure = {
    "lr": 0.015,  # Learning rate 0.001
    "num_heads": [4],  # Number of attention heads for node-level attention
    "hidden_units": 32,
    "dropout": 0.3,  # default 0.6
    "weight_decay": 0.0001,  # d: 0.001
    "num_epochs":1000,   # before: 50
    "patience": 100,
}


def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup(args):
    args.update(default_configure)
    set_random_seed(args["seed"], deterministic=False)
    args["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    return args


def getFeats(in_dims):

    feats_dict = {}
    for node, dim in in_dims.items():
        indices = np.vstack((np.arange(dim), np.arange(dim)))  # (2, n)
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(np.ones(dim))
        feats_dict[node] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim]))

    return feats_dict


def create_hg(adj):

    in_dims = {}

    drug_disease = np.load('dataZhang/drug_disease.npy')
    drug_disease = sp.coo_matrix(drug_disease)
    in_dims["drug"] = drug_disease.shape[0]
    in_dims["disease"] = drug_disease.shape[1]

    drug_se = np.load('dataZhang/drug_se.npy')
    drug_se = sp.coo_matrix(drug_se)
    in_dims["se"] = drug_se.shape[1]

    drug_protein = np.load('dataZhang/drug_protein.npy')
    drug_protein = sp.coo_matrix(drug_protein)
    in_dims["protein"] = drug_protein.shape[1]

    adj = sp.coo_matrix(adj)

    hg_drug = dgl.heterograph(
        {
            ("drug", "dd", "drug"): (adj.row, adj.col),
            ("drug", "dp", "protein"): (drug_protein.row, drug_protein.col),
            ("protein", "pd", "drug"): (drug_protein.col, drug_protein.row),
            ("drug", "dse", "se"): (drug_se.row, drug_se.col),
            ("se", "sed", "drug"): (drug_se.col, drug_se.row),
            ("drug", "ddis", "disease"): (drug_disease.row, drug_disease.col),
            ("disease", "disd", "drug"): (drug_disease.col, drug_disease.row)
        }
    )

    feats_dict = getFeats(in_dims)

    return hg_drug, feats_dict["drug"]



def compute_loss_metrics(pos_score, neg_score):
    loss_fn = SigmoidLoss()
    loss, _, _ = loss_fn(pos_score, neg_score)

    probas_pred, ground_truth = [], []
    probas_pred.append(torch.sigmoid(pos_score.detach()).cpu())
    ground_truth.append(np.ones(len(pos_score)))

    probas_pred.append(torch.sigmoid(neg_score.detach()).cpu())
    ground_truth.append(np.zeros(len(neg_score)))

    probas_pred = np.concatenate(probas_pred)
    ground_truth = np.concatenate(ground_truth)
    pred = (probas_pred >= 0.5).astype(np.int64)

    auc_roc = roc_auc_score(ground_truth, probas_pred)
    ap_score = average_precision_score(ground_truth, probas_pred)
    p, r, t = precision_recall_curve(ground_truth, probas_pred)
    auc_prc = auc(r, p)
    acc = accuracy_score(ground_truth, pred)
    f1 = f1_score(ground_truth, pred)
    recall = recall_score(ground_truth, pred)

    return loss, auc_roc, auc_prc, acc, f1, recall, ap_score
