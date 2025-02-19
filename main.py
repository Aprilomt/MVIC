import os

import torch
import argparse
import numpy as np
import pandas as pd
import copy
import torch.nn as nn

from utils import setup, create_hg, compute_loss_metrics
from model_hetero import DDModel
from data_preprocess import DrugDataset, DrugDataLoader


def load_vocab(filepath: str):
    df = pd.read_csv(filepath, index_col=False)
    id2index = {id: idx for id, idx in zip(df['drugbank_id'], range(len(df)))}
    return id2index

vocab_path = 'dataZhang/drug_list_zhang.csv'
smiles2idx = load_vocab(vocab_path) if vocab_path is not None else None


train_path = 'dataZhang/ZhangDDI_train.csv'
val_path = 'dataZhang/ZhangDDI_valid.csv'
test_path = 'dataZhang/ZhangDDI_test.csv'


def load_csv_data(filepath: str, smiles2id: dict,):
    df = pd.read_csv(filepath, index_col=False)

    edges = []
    edges_false = []
    for row_id, row in df.iterrows():
        row_dict = dict(row)
        smiles_1 = row_dict['drugbank_id_1']
        smiles_2 = row_dict['drugbank_id_2']
        if smiles_1 in smiles2id.keys() and smiles_2 in smiles2id.keys():
            idx_1 = smiles2id[smiles_1]
            idx_2 = smiles2id[smiles_2]
            label = int(row_dict['label'])
        else:
            continue
        if label > 0:
            edges.append((idx_1, idx_2))
        else:
            edges_false.append((idx_1, idx_2))
    

    edges = np.array(edges, dtype=np.int64)
    edges_false = np.array(edges_false, dtype=np.int64)
    return edges, edges_false



def load_structure_data():
    dgID = []
    with open('dataZhang/drug.txt') as dr:
        for s in dr:
            dgID.append(s.strip())

    drugid = dgID
    drug_data = DrugDataset(drugid)
    drug_data_loader = DrugDataLoader(drug_data, batch_size=len(drugid), shuffle=False)

    return drug_data_loader


def main(args):
    for i in range(1):  

        max_acc = 0
        max_auc = 0
        max_aupr = 0
        max_epoch = 0


        adj_train = np.load('dataZhang/drug_drug_trainAdj.npy')

        hg_drug, feats_drug = create_hg(adj_train)

        feats_drug = feats_drug.to_dense()
        feats_drug = feats_drug.to(args["device"])
        hg_drug = hg_drug.to(args["device"])


        model = DDModel(d_meta_paths=[["dd"], ["dp", "pd"], ["dse", "sed"], ["ddis", "disd"], ['ddis', 'disd', 'ddis', 'disd'], ['dp', 'pd', 'dp', 'pd'], ['dse', 'sed', 'dse', 'sed']], in_size=feats_drug.shape[1],
                        hidden_size=args["hidden_units"], num_heads=args["num_heads"], dropout=args["dropout"]).to(args["device"])
        optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])

        model_max = copy.deepcopy(model)

        drug_loader = load_structure_data()


        p_pair_train, n_pair_train = load_csv_data(train_path, smiles2idx)
        p_pair_valid, n_pair_valid = load_csv_data(val_path, smiles2idx)
        p_pair_test, n_pair_test = load_csv_data(test_path, smiles2idx)

        for iter in range(args["num_epochs"]):
            for drugs in drug_loader:
                drugs = drugs.to(args["device"])

                model.train()

                p_score, n_score, = model(p_pair_train, n_pair_train, hg_drug, feats_drug, drugs)

                loss1, auc_epoch, aupr_epoch, acc_epoch, f1_epoch, recall_epoch, ap_epoch = compute_loss_metrics(p_score, n_score)

                loss = loss1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                
                # model.eval()
                with torch.no_grad():
                    model.eval()
                    p_score, n_score, = model(p_pair_valid, n_pair_valid, hg_drug, feats_drug, drugs)

                    _, auc_epoch_val, aupr_epoch_val, acc_epoch_val, f1_epoch_val, recall_epoch_val, ap_epoch_val = compute_loss_metrics(p_score, n_score)
                    if auc_epoch_val > max_auc and acc_epoch_val > max_acc:
                        model_max = copy.deepcopy(model)
                        max_auc = auc_epoch_val
                        max_acc = acc_epoch_val
                        max_epoch = iter


                print("In epoch {}, loss: {:.4f}, auc_train: {:.4f}, ap_train: {:.4f}, auc_val: {:.4f}, ap_val: {:.4f}".format(iter, loss, auc_epoch, ap_epoch, auc_epoch_val, ap_epoch_val))


        p_score, n_score, = model_max(p_pair_test, n_pair_test, hg_drug, feats_drug, drugs)

        _, auc_test, aupr_test, acc_test, f1_test, recall_test, ap_test = compute_loss_metrics(p_score, n_score)

        print("max epoch: {}, test auc: {:.4f}, test aupr: {:.4f}, test acc: {:.4f}, test f1: {:.4f}, test recall: {:.4f}, test ap: {:.4f}".format(max_epoch, auc_test, aupr_test, acc_test, f1_test, recall_test, ap_test))
        with open(args['out_file'], 'a') as f:
            f.write(str(args['seed']) + '  ' + str(args['lr']) + ' ' + str(args['dropout']) + ' ' + str(args['weight_decay']) + ' ' + str(max_epoch) + '  ' + f"{auc_test:.4f}" + '  ' + f"{ap_test:.4f}" + '  ' + f"{acc_test:.4f}" + '  ' + f"{f1_test:.4f}" + '\n') 

    return auc_test, ap_test, acc_test, f1_test


if __name__ == "__main__":
    # run with seed 0 1 2 3 4
    parser = argparse.ArgumentParser('DDI')
    parser.add_argument("-s", "--seed", type=int, default=0, help="Random seed")
    parser.add_argument("-i", "--index", default=1, type=int, help="index-th fold")
    parser.add_argument('--kfold', default=4, type=int)
    parser.add_argument('--save_dir', type=str, default='./savept',
                        help='Directory where model checkpoints will be saved')
    
    parser.add_argument('--out_file', default='res/result_zhangddi.txt')  # store the running results.

    args = parser.parse_args().__dict__
    args = setup(args)   

    main(args)

