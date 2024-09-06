import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import (
    degree,
    remove_self_loops,
    add_self_loops,
    add_remaining_self_loops,
)
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import ChebConv
import config

class Velocity(nn.Module):
    def __init__(self, init_v=7):
        super(Velocity, self).__init__()
        self.ve = nn.Parameter(torch.FloatTensor([init_v]))

    def forward(self, LA):
        return torch.pow(LA, self.ve)



class TEDGCN(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, softmax=True):
        super(TEDGCN, self).__init__()
        self.ve = Velocity(7)
        self.W = nn.Linear(in_channels, hidden_dim)
        self.softmax = softmax
        self.bn = nn.BatchNorm1d(hidden_dim)
        if config.dataset in ["squirrel"]:
            self.p = 0.1
        else:
            self.p = 0.2
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(self.p)
        self.MLP = nn.Linear(hidden_dim, out_channels)

    def forward(self, X, La, U):
        V_La = self.ve(La)
        out_A = torch.mm(torch.mm(U, torch.diag(V_La)), torch.transpose(U, 0, 1))
        out = torch.mm(out_A, X)
        out = self.W(out)
        hidden_emd = out
        if not config.small_graph and config.dataset not in []:
            out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.MLP(out)
        if self.softmax:
            return F.log_softmax(out, dim=1), hidden_emd
        return torch.nn.Sigmoid()(out), hidden_emd


class SLOG_B(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, softmax=True):
        super(SLOG_B, self).__init__()
        self.ve = Velocity(7)
        self.ve2 = Velocity(0)
        self.W = nn.Linear(in_channels, hidden_dim)
        self.softmax = softmax
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.MLP = nn.Linear(hidden_dim, out_channels)

    def forward(self, X, La, U):
        V_La = self.ve(La)
        out_A = torch.mm(torch.mm(U, torch.diag(V_La)), torch.transpose(U, 0, 1))
        V_La2 = self.ve2((2 * (La - 0.00000001) - 1) ** 2 + 1)
        out_A2 = torch.mm(torch.mm(U, torch.diag(V_La2)), torch.transpose(U, 0, 1))
        out = torch.mm(out_A, X)
        out = torch.mm(out_A2, out)
        out = self.W(out)
        hidden_emd = out
        if not config.small_graph and config.dataset not in []:
            out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.MLP(out)
        if self.softmax:
            return F.log_softmax(out, dim=1), hidden_emd
        return torch.nn.Sigmoid()(out), hidden_emd


class SLOG_wo_S1(nn.Module): # for ablation study
    def __init__(self, in_channels, hidden_dim, out_channels, softmax=True):
        super(SLOG_wo_S1, self).__init__()
        self.ve = Velocity(7)
        self.ve2 = Velocity(0)
        self.W = nn.Linear(in_channels, hidden_dim)
        self.softmax = softmax
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.MLP = nn.Linear(hidden_dim, out_channels)

    def forward(self, X, La, U):
        V_La2 = self.ve2((2 * (La - 0.00000001) - 1) ** 2 + 1)
        out_A2 = torch.mm(torch.mm(U, torch.diag(V_La2)), torch.transpose(U, 0, 1))
        out = torch.mm(out_A2, X)
        out = self.W(out)
        hidden_emd = out
        if not config.small_graph and config.dataset not in []:
            out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.MLP(out)
        if self.softmax:
            return F.log_softmax(out, dim=1), hidden_emd
        return torch.nn.Sigmoid()(out), hidden_emd

class SLOG_wo_S2(nn.Module): # for ablation study
    def __init__(self, in_channels, hidden_dim, out_channels, softmax=True):
        super(SLOG_wo_S2, self).__init__()
        self.ve = Velocity(7)
        self.ve2 = Velocity(0)
        self.W = nn.Linear(in_channels, hidden_dim)
        self.softmax = softmax
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.MLP = nn.Linear(hidden_dim, out_channels)

    def forward(self, X, La, U):
        V_La = self.ve(La)
        out_A = torch.mm(torch.mm(U, torch.diag(V_La)), torch.transpose(U, 0, 1))
        out = torch.mm(out_A, X)
        out = self.W(out)
        hidden_emd = out
        if not config.small_graph and config.dataset not in []:
            out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.MLP(out)
        if self.softmax:
            return F.log_softmax(out, dim=1), hidden_emd
        return torch.nn.Sigmoid()(out), hidden_emd

class SLOG_N(nn.Module):
    def __init__(
        self, in_channels, hidden_dim, out_channels, layer_num=3, softmax=True, res=True
    ):
        super(SLOG_N, self).__init__()

        self.dropout = nn.Dropout(0.2)
        self.bns1 = nn.ModuleList()
        self.bns2 = nn.ModuleList()
        self.tedgcns = nn.ModuleList()
        self.lines = nn.ModuleList()
        self.tedgcns.append(SLOG_B(in_channels, hidden_dim, hidden_dim, False))
        if layer_num == 1:
            self.lines.append(nn.Linear(in_channels, out_channels))
        else:
            self.lines.append(nn.Linear(in_channels, hidden_dim))
        for i in range(layer_num - 2):
            self.lines.append(nn.Linear(hidden_dim, hidden_dim))
            self.tedgcns.append(SLOG_B(hidden_dim, hidden_dim, hidden_dim, False))
        for i in range(layer_num - 1):
                self.bns1.append(nn.BatchNorm1d(hidden_dim))
                self.bns2.append(nn.BatchNorm1d(hidden_dim))
        self.bns1.append(nn.BatchNorm1d(hidden_dim))
        self.tedgcns.append(SLOG_B(hidden_dim, hidden_dim, out_channels, False))
        self.lines.append(nn.Linear(hidden_dim, out_channels))
        self.act = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.MLP = nn.Linear(hidden_dim, out_channels)
        self.softmax = softmax
        self.layer_num = layer_num
        self.res = res

    def forward(self, X, La, U, edge_weight=None):
        for i in range(self.layer_num - 1):
            out, hidden_emd = self.tedgcns[i](X, La, U)
            hidden_emd = self.bns1[i](hidden_emd)
            hidden_emd = self.act(hidden_emd)
            hidden_emd = self.dropout(hidden_emd)
            if self.res:
                X = self.act(hidden_emd + self.lines[i](X))
                X = self.bns2[i](self.dropout(X))
            else:
                X = hidden_emd
            
        out, hidden_emd = self.tedgcns[self.layer_num - 1](X, La, U)
        hidden_emd = self.bns1[self.layer_num - 1](hidden_emd)
        hidden_emd = self.act(hidden_emd)
        hidden_emd = self.dropout(hidden_emd)
        if self.res:
            X = self.act2(self.MLP(hidden_emd)) + self.lines[self.layer_num - 1](X)
        else:
            X = self.act2(self.MLP(hidden_emd))

        if self.softmax:
            return F.log_softmax(X, dim=1), None
        return X, None

class SLOG_B_gp(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, softmax=True):
        super(SLOG_B_gp, self).__init__()
        self.W = nn.Linear(in_channels, hidden_dim)
        self.MLP = nn.Linear(hidden_dim, out_channels)
        self.softmax = softmax
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.p = 0.1

    def forward(self, X, La, U, d1, d2):
        # V_La = self.ve(La)
        V_La = torch.pow(La, d1)
        out_A = torch.mm(torch.mm(U, torch.diag(V_La)), torch.transpose(U, 0, 1))
        
        V_La2 = torch.pow((2 * (La - 0.00000001) - 1) ** 2 + 1, d2)


        out_A2 = torch.mm(torch.mm(U, torch.diag(V_La2)), torch.transpose(U, 0, 1))
        out = torch.mm(out_A, X)
        out = torch.mm(out_A2, out)
        out = self.W(out)
        out = torch.relu(out)
        out = F.dropout(out, training=self.training, p=self.p)
        hidden_emd = out
        out = self.MLP(out)
        if self.softmax:
            return F.log_softmax(out, dim=1), hidden_emd
        return torch.nn.Sigmoid()(out), hidden_emd