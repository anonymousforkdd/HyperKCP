import time
from copy import deepcopy
import networkx as nx
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from dhg import Graph, Hypergraph
from dhg.data import Cora, Pubmed, Citeseer
from dhg.models import HGNN, HGNNP, HNHN
from dhg.random import set_seed
# from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from sklearn.metrics import roc_auc_score, average_precision_score,f1_score,precision_score,recall_score, accuracy_score
import numpy as np
import csv
import pandas as pd
import json
import scipy.sparse as sparse
from scipy.sparse import csr_matrix,coo_matrix
import scipy.sparse as sp
import itertools
from sklearn.metrics import roc_auc_score, average_precision_score,f1_score,precision_score,recall_score
from torch_geometric.utils import negative_sampling
from utils import threshold, normalization, npz_pre, get_new, get_edge, get_link_labels,hier_edge 
from models import HyperKCP


set_seed(2022)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
hidden_dim = 128
out_dim = 16
data_train = 'group'
print('This result is based on', data_train,' data')

def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(net, X, HG, train_edge, optimizer, epoch):
    net.train()
    st = time.time()
    optimizer.zero_grad()
    logit, logit_new, a = net(X, HG, train_edge, train_edge)
    #logit = logit[0]
    link_label = get_link_labels(train_pos_edge1,train_neg_edge,device)
    loss = F.binary_cross_entropy_with_logits(logit, link_label).to(device)
    # loss = F.binary_cross_entropy(logit, link_label).to(device)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item(), a
    

@torch.no_grad()
def infer(net, X, HG, edge , edge_new, test=False):
    net.eval()
    logit, logit_new, a = net(X, HG, edge, edge_new)
    #logit = logit[0]
    logit = logit.cpu().detach().numpy()
    # logit_new = logit_new[0]
    logit_new = logit_new.cpu().detach().numpy()
    link_newlabel = np.ones(len(logit_new))
    num = np.percentile(logit,90)
    
    logit = threshold(logit,num = num)
    logit_new = threshold(logit_new,num = num)
    if not test:
        link_label = get_link_labels(valid_pos_edge, valid_neg_edge, device)
        link_label = link_label.cpu().detach().numpy()
        #res = np.sum(logit)/np.sum(link_label)
        res_new = np.sum(logit_new)/np.sum(link_newlabel)
        auc = roc_auc_score(link_label, logit)
        acc = accuracy_score(link_label, logit)
        ap = average_precision_score(link_label, logit)
        f1 = f1_score(link_label, logit)
        precision = precision_score(link_label, logit)
        recall = recall_score(link_label, logit)
        #res_new = roc_auc_score(logit_new, link_newlabel)
    else:
        link_label = get_link_labels(test_pos_edge, test_neg_edge, device)
        link_label = link_label.cpu().detach().numpy()
        #res = np.sum(logit)/np.sum(link_label)
        auc = roc_auc_score(link_label, logit)
        acc = accuracy_score(link_label, logit)
        res_new = np.sum(logit_new)/np.sum(link_newlabel)
        ap = average_precision_score(link_label, logit)
        f1 = f1_score(link_label, logit)
        precision = precision_score(link_label, logit)
        recall = recall_score(link_label, logit)
        #res_new = roc_auc_score(logit_new, link_newlabel)
    return auc,acc, ap, f1, precision, recall, res_new



if data_train == 'subclass':
    train_npz_co = sparse.load_npz('chulihou_subclass2000-2018.npz')
    valid_npz = sparse.load_npz('chulihou_subclass2019-2020.npz')
    test_npz = sparse.load_npz('chulihou_subclass2021-2022.npz')
    train_npz_ci = sparse.load_npz('subclass_2000-2018.npz')
    filename3 = 'subclass_feature.json'
    filename = 'hier_subclass.json'
    with open(filename,'r',encoding='utf-8') as f:
        hg_subclass = json.load(f)


else:
    train_npz_co = sparse.load_npz('group1_2000-2018.npz')
    train_npz_ci = sparse.load_npz('group1_2000-2018.npz')
    valid_npz = sparse.load_npz('group1_2019-2020.npz')
    test_npz = sparse.load_npz('group1_2019-2020.npz')
    filename3 = 'group_feature1.json'
    filename = 'hier_group1.json'
    with open(filename,'r',encoding='utf-8') as f:
        hg_subclass = json.load(f)

train_npz_co = npz_pre(train_npz_co)
valid_npz = npz_pre(valid_npz)
test_npz = npz_pre(test_npz)
train_npz_ = train_npz_co.power(0) 

valid_npz_ = valid_npz.power(0)
test_npz_ = test_npz.power(0)
valid_newnpz = valid_npz_ - train_npz_
valid_newnpz = get_new(valid_newnpz)
test_newnpz = test_npz_ - train_npz_
test_newnpz = get_new(test_newnpz)


train_pos_edge1= get_edge(train_npz_co)
train_pos_edge2 = get_edge(train_npz_ci)
train_pos_edge3 = hier_edge(hg_subclass)

train_pos_edge_all = torch.cat([train_pos_edge1, train_pos_edge2, train_pos_edge3],dim =1)

################
train_neg_edge = negative_sampling(edge_index=train_pos_edge_all, num_nodes= train_npz_co.shape[0],
    num_neg_samples=train_pos_edge1.size(1)*9,force_undirected=True)
train_edge = torch.cat([train_pos_edge1, train_neg_edge],dim=-1)
valid_pos_edge = get_edge(valid_npz)
valid_neg_edge = negative_sampling(edge_index=valid_pos_edge, num_nodes= valid_npz.shape[0],
    num_neg_samples=valid_pos_edge.size(1)*9,force_undirected=True)
valid_edge = torch.cat([valid_pos_edge, valid_neg_edge],dim=-1)
valid_newedge = get_edge(valid_newnpz)
test_pos_edge = get_edge(test_npz)
test_neg_edge = negative_sampling(edge_index=test_pos_edge, num_nodes= test_npz.shape[0],
    num_neg_samples=test_pos_edge.size(1)*9,force_undirected=True)
test_edge = torch.cat([test_pos_edge, test_neg_edge],dim=-1)
test_newedge = get_edge(test_newnpz)


with open(filename3,'r',encoding='utf-8') as f:
    feature_subclass = json.load(f)

X = torch.Tensor(feature_subclass)
# X = torch.randn((674,404))
feature_dim = X.shape[1]

# list_temp = [train_npz.row,train_npz.col] 
num_vertices = train_npz_co.get_shape()[0]
# edge_list = list(map(tuple,zip(*list_temp)))
G1 = Graph(num_vertices, list(map(tuple,zip(*train_pos_edge1.cpu().tolist()))))
G2 = Graph(num_vertices, list(map(tuple,zip(*train_pos_edge2.cpu().tolist()))))
# G3 = Graph(num_vertices, list(map(tuple,zip(*train_pos_edge3.cpu().tolist()))))
# G = Graph(num_vertices, list(map(tuple,zip(*train_pos_edge_all.cpu().tolist()))))

HG = Hypergraph(num_v= num_vertices)
HG.add_hyperedges(e_list = hg_subclass, group_name = 'hier') 
HG.add_hyperedges_from_graph_kHop(G1, k=1, group_name = 'co-occurence')
HG.add_hyperedges_from_graph_kHop(G2, k=1, group_name= 'citation')
X = X.to(device)
HG = HG.to(X.device)

out_dim = 32
########### 
def run(X, G, HG, train_edge, valid_edge , valid_newedge,   test_edge, test_newedge, model_name = 'HyperKCP'):

    net = HyperKCP(feature_dim, 32, device)
    Hyper= True
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    net = net.to(device)
    init_network(net)
    best_state = None
    best_epoch, best_val = -1, -1
    a_list = []
    for epoch in range(500):
        # train
        if Hyper == True:
            temp, a = train(net, X, HG, train_edge, optimizer, epoch)
        else:
            temp, a = train(net, X, G, train_edge, optimizer, epoch)
        # validation
        a_list.append(a)
        if epoch % 1 == 0:
            with torch.no_grad():
                if Hyper == True:
                    val_auc, acc, ap, f1, precision, recall, res_new = infer(net, X, HG, valid_edge , valid_newedge, test=False)
                else:
                    val_auc, acc, ap, f1, precision, recall, res_new = infer(net, X, G, valid_edge , valid_newedge, test=False)
                
            if val_auc > best_val:
                print(f"update best: {val_auc:.5f}")
                best_epoch = epoch
                best_val = val_auc
                best_state = deepcopy(net.state_dict())
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    # test
    print("test...")
    net.load_state_dict(best_state)
    if Hyper == True:
        res = infer(net, X, HG, test_edge, test_newedge, test=True)
    else:
        res = infer(net, X, G, test_edge, test_newedge, test=True)
    print(f'model_name:', model_name)
    print(f"final result: epoch: {best_epoch}")
    print(res)
    print(a_list)
    return

run(X, HG, HG, train_edge, valid_edge , valid_newedge,   test_edge, test_newedge, model_name = 'HyperKCP')