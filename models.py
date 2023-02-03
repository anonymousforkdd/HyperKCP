import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
from copy import deepcopy
import networkx as nx
import torch
import torch.optim as optim
import torch.nn.functional as F
from dhg import Graph, Hypergraph
from dhg.data import Cora, Pubmed, Citeseer
from dhg.models import HGNN, HGNNP, HNHN
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

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
from utils import threshold, normalization, npz_pre, get_new, get_edge, get_link_labels

import torch
import torch.nn as nn

import dhg
from dhg.nn import HGNNConv, HGNNPConv, GCNConv, HyperGCNConv, MLP, GINConv, GraphSAGEConv, GATConv, MultiHeadWrapper, UniGATConv
from dhg.structure.graphs import Graph

class GATconv(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = True,
        drop_rate: float = 0.5,
        atten_neg_slope: float = 0.2,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.atten_dropout = nn.Dropout(drop_rate)
        self.atten_act = nn.LeakyReLU(atten_neg_slope)
        self.act = nn.LeakyReLU(inplace=True)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)
        self.atten_v = nn.Linear(out_channels, 1, bias=False)
        self.atten_e = nn.Linear(out_channels, 1, bias=False)
        self.atten_dst = nn.Linear(out_channels, 1, bias=False)
        self.a1 = nn.Parameter(torch.ones(1,1), requires_grad=True)   
        self.a2 = nn.Parameter(torch.ones(1,1), requires_grad=True)   
        self.a3 = nn.Parameter(torch.ones(1,1), requires_grad=True)   
        self.drop_layer = nn.Dropout(drop_rate)

    def forward(self,X,hg):
        X = self.theta(X)
        if self.bn is not None:
            X = self.bn(X)
        Y1 = hg.v2e_of_group(group_name = 'hier' ,X = X,aggr = "mean")
        Y2 = hg.v2e_of_group(group_name = 'co-occurence' ,X = X,aggr = "mean")
        Y3 = hg.v2e_of_group(group_name = 'citation' ,X = X,aggr = "mean")

        alpha_e1 = self.atten_e(Y1)
        e_atten_score1 = alpha_e1[hg.e2v_src_of_group('hier')]
        e_atten_score1 = self.atten_dropout(self.atten_act(e_atten_score1).squeeze())
        e_atten_score1 = torch.clamp(e_atten_score1, min=0, max=5)
        X1 = hg.e2v_of_group('hier', Y1, aggr="sum", e2v_weight=e_atten_score1)

        alpha_e2 = self.atten_e(Y2)
        e_atten_score2 = alpha_e2[hg.e2v_src_of_group('co-occurence')]
        e_atten_score2 = self.atten_dropout(self.atten_act(e_atten_score2).squeeze())
        e_atten_score2 = torch.clamp(e_atten_score2, min=0, max=5)
        X2 = hg.e2v_of_group('co-occurence', Y2, aggr="sum", e2v_weight=e_atten_score2)

        alpha_e3 = self.atten_e(Y3)
        e_atten_score3 = alpha_e3[hg.e2v_src_of_group('citation')]
        e_atten_score3 = self.atten_dropout(self.atten_act(e_atten_score3).squeeze())
        e_atten_score3 = torch.clamp(e_atten_score3, min=0, max=5)
        X3 = hg.e2v_of_group('citation', Y3, aggr="sum", e2v_weight=e_atten_score3)

        X = self.a1 * X1 + self.a2 * X2 + self.a3 * X3
        if not self.is_last:
            X = self.act(X)
        # print('a1,a2,a3:',self.a1, self.a2, self.a3)
        a = [self.a1.item(), self.a2.item(), self.a3.item()]
        return X, a


class HyperKCP(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            device,
            bias: bool = True,
            use_bn: bool = False,
            drop_rate: float = 0.5,
            atten_neg_slope: float = 0.2,
    ):
        super().__init__()
        self.drop_layer = nn.Dropout(drop_rate)
        self.GATconv = GATconv(in_channels, out_channels)

    def forward(self, X: torch.Tensor, hg: Hypergraph, edge, new_edge) -> torch.Tensor:
        X = self.drop_layer(X)
        X, a = self.GATconv(X, hg)

        sim = (X[edge[0]] * X[edge[1]]).sum(dim=-1)
        sim_new = (X[new_edge[0]] * X[new_edge[1]]).sum(dim=-1)
        return sim, sim_new, a, X


