import copy
import os
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn
from torch.nn import Embedding
from torch.nn import Transformer, TransformerEncoderLayer
from torch_geometric.nn import Sequential, GCNConv
from torch_geometric.utils import *

import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity


def MLP(dimensions, dropout=False, batch_norm=False, batch_norm_momentum=1e-3):
    return nn.Sequential(*[
        nn.Sequential(
            nn.Dropout(p=0.5) if dropout else nn.Identity(),
            nn.Linear(dimensions[i - 1], dimensions[i]),
            nn.ReLU(dimensions[i]),
            nn.BatchNorm1d(dimensions[i], affine=True, momentum=batch_norm_momentum) if batch_norm else nn.Identity())
        for i in range(1, len(dimensions))])


class MLPModel(nn.Module):
    def __init__(self, dim):
        super(MLPModel, self).__init__()
        self.dim = dim
        self.layers = nn.Sequential(MLP(self.dim[0:-1]),
                                    nn.Linear(self.dim[-2], self.dim[-1]))

    def forward(self, x):
        return self.layers(x)


def act_function(dim, func: str = "ReLU"):
    if func == 'ReLU':
        return nn.ReLU()
    elif func == "PReLU":
        return nn.PReLU()
    elif func == "LeakyReLU":
        return nn.LeakyReLU()
    elif func == "Hardswish":
        return nn.Hardswish()
    elif func == "Mish":
        return nn.Mish()
    return


class ResBlock(nn.Module):
    def __init__(self, dim: int, act: str = "ReLU", batch_norm: bool = False, dropout: bool = False):
        super(ResBlock, self).__init__()
        self.dim = dim
        self.act = act
        self.layers = nn.Sequential(nn.Dropout(p=0.5) if dropout else nn.Identity(),
                                    nn.Linear(dim, dim),

                                    nn.BatchNorm1d(dim, affine=True, momentum=1e-3) if batch_norm else nn.Identity(),
                                    act_function(dim, func=self.act))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class SkeletonLinear(nn.Module):
    def __init__(self, in_channels, out_channels, node_num, edges):
        super(SkeletonLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edges = edges
        self.node_num = node_num

        self.weight = torch.zeros(out_channels * node_num, in_channels * node_num)
        self.mask = torch.zeros(out_channels * node_num, in_channels * node_num)
        self.bias = nn.Parameter(torch.Tensor(out_channels * node_num))

        for edge_idx in range(self.edges.shape[1]):
            edge = self.edges[:, edge_idx]
            tmp = torch.zeros((out_channels, in_channels))
            self.mask[out_channels * edge[1]:out_channels * (edge[1] + 1),
            in_channels * edge[0]:in_channels * (edge[0] + 1)] = 1
            nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
            self.weight[out_channels * edge[1]:out_channels * (edge[1] + 1),
            in_channels * edge[0]:in_channels * (edge[0] + 1)] = tmp

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.weight = nn.Parameter(self.weight)
        self.mask = nn.Parameter(self.mask, requires_grad=False)

    def forward(self, x):
        weight_masked = self.weight * self.mask
        res = F.linear(x, weight_masked, self.bias)
        return res


class Marker2SkeletonLinear(nn.Module):
    def __init__(self, in_channels, out_channels, marker_num, joint_num, edges):
        super(Marker2SkeletonLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edges = edges
        self.joint_num = joint_num
        self.marker_num = marker_num

        self.weight = torch.zeros(out_channels * joint_num, in_channels * marker_num)
        self.mask = torch.zeros(out_channels * joint_num, in_channels * marker_num)
        self.bias = nn.Parameter(torch.Tensor(out_channels * joint_num))

        for edge_idx in range(self.edges.shape[1]):
            edge = self.edges[:, edge_idx]
            tmp = torch.zeros((out_channels, in_channels))
            self.mask[out_channels * edge[1]:out_channels * (edge[1] + 1),
            in_channels * edge[0]:in_channels * (edge[0] + 1)] = 1
            nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
            self.weight[out_channels * edge[1]:out_channels * (edge[1] + 1),
            in_channels * edge[0]:in_channels * (edge[0] + 1)] = tmp

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.weight = nn.Parameter(self.weight)
        self.mask = nn.Parameter(self.mask, requires_grad=False)

    def forward(self, x):
        weight_masked = self.weight * self.mask
        res = F.linear(x, weight_masked, self.bias)
        return res


class GNNBiLSTM(nn.Module):
    def __init__(self, args, marker_num: int, input_num: int, hidden_num: int, output_num: int, connect_mat_path: str,
                 act: str = "ReLU", layer_num: int = 1):
        super(GNNBiLSTM, self).__init__()
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.layer_num = layer_num
        self.act = act
        self.marker_num = marker_num
        self.marker_gcn_act = args.marker_gcn_act

        self.marker_connect_mat = torch.from_numpy(np.load(os.path.join(args.root_dir, connect_mat_path)))
        self.marker_edges = dense_to_sparse(self.marker_connect_mat)[0]
        self.marker_edges = add_self_loops(self.marker_edges)[0]

        self.marker_gcn_list = nn.ModuleList()
        for i in range(len(args.marker_gcn_dim) - 1):
            self.marker_gcn_list.append(SkeletonLinear(args.marker_gcn_dim[i], args.marker_gcn_dim[i + 1],
                                                       self.marker_num, self.marker_edges))
            self.marker_gcn_list.append(act_function(-1, self.marker_gcn_act[i]))

        self.lstm = nn.LSTM(input_size=args.marker_gcn_dim[-1] * self.marker_num, hidden_size=hidden_num,
                            num_layers=layer_num,
                            bidirectional=True, batch_first=True)
        # since it is a BI-LSTM, the hidden dim has to be multiplied 2
        self.out_linear = nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(hidden_num * 2, output_num),
                                        act_function(output_num, self.act),
                                        # nn.BatchNorm1d(output_num),
                                        nn.Linear(output_num, output_num))

    def forward(self, x, hidden=None, c=None):
        if hidden is None:
            device = self.lstm.all_weights[0][0].device
            hidden = torch.zeros(self.layer_num * 2, x.shape[0], self.hidden_num).to(device)
            c = torch.zeros(self.layer_num * 2, x.shape[0], self.hidden_num).to(device)
        batch_size, window_size = x.shape[0], x.shape[1]
        # x = torch.transpose(x, 0, 1)
        for layer in self.marker_gcn_list:
            x = layer(x.view(batch_size * window_size, -1))

        x = x.view((batch_size, window_size, -1))
        y, (hidden, c) = self.lstm(x, (hidden, c))

        out = self.out_linear(y.reshape((batch_size * window_size, -1)))
        out = out.reshape((batch_size, window_size, -1))
        # out = torch.transpose(out, 0, 1)
        return out, (hidden, c)


class GraphResNetModel(nn.Module):
    def __init__(self, args, connect_mat_path, weight_path, batch_norm: bool = False, dropout: bool = False):
        super(GraphResNetModel, self).__init__()
        self.dim = args.network_dim
        self.marker_gcn_dim = args.marker_gcn_dim
        self.marker_gcn_act = args.marker_gcn_act
        self.skeleton_gcn_dim = args.skeleton_gcn_dim
        self.skeleton_gcn_act = args.skeleton_gcn_act
        self.marker_skeleton_conv_gcn_dim = args.marker_skeleton_conv_gcn_dim
        self.marker_skeleton_conv_gcn_act = args.marker_skeleton_conv_gcn_act

        self.batch_norm = batch_norm
        self.act = args.act_fun
        self.dropout = dropout
        self.marker_num = args.marker_num
        self.joint_num = args.joint_num
        self.marker_connect_mat = torch.from_numpy(np.load(os.path.join(args.root_dir, connect_mat_path)))
        self.marker_edges = dense_to_sparse(self.marker_connect_mat)[0]
        self.marker_edges = add_self_loops(self.marker_edges)[0]

        self.skin_weight = torch.from_numpy(np.load(os.path.join(args.root_dir, weight_path))["mean"].T)
        if self.skin_weight.shape[0] == self.joint_num:
            self.skin_weight = self.skin_weight.T
        skin_weight_expand = torch.zeros((self.skin_weight.shape[0], self.skin_weight.shape[1] + 1))
        skin_weight_expand[:, :-1] = self.skin_weight
        skin_weight_expand[:, -1] = 1

        self.marker_skeleton_conv_gcn_list = nn.ModuleList()
        for i in range(len(self.marker_skeleton_conv_gcn_dim)):
            if i % 2 == 0:
                self.marker_skeleton_conv_gcn_list.append(Marker2SkeletonLinear(self.marker_skeleton_conv_gcn_dim[i][0],
                                                                                self.marker_skeleton_conv_gcn_dim[i][1],
                                                                                self.marker_num, self.joint_num + 1,
                                                                                np.vstack(
                                                                                    np.where(skin_weight_expand))))
                self.marker_skeleton_conv_gcn_list.append(act_function(-1, self.marker_skeleton_conv_gcn_act[i]))
            else:
                self.marker_skeleton_conv_gcn_list.append(Marker2SkeletonLinear(self.marker_skeleton_conv_gcn_dim[i][0],
                                                                                self.marker_skeleton_conv_gcn_dim[i][1],
                                                                                self.joint_num + 1, self.marker_num,
                                                                                np.vstack(
                                                                                    np.where(skin_weight_expand.T))))
                self.marker_skeleton_conv_gcn_list.append(act_function(-1, self.marker_skeleton_conv_gcn_act[i]))

        self.skeleton_parent_list = args.parent_list
        self.skeleton_connect_mat = np.zeros((len(self.skeleton_parent_list) + 1,
                                              len(self.skeleton_parent_list) + 1))
        self.skeleton_connect_mat = torch.from_numpy(self.skeleton_connect_mat)

        for i in range(len(self.skeleton_parent_list)):
            self.skeleton_connect_mat[i, i] = 1
            self.skeleton_connect_mat[i, self.skeleton_parent_list[i]] = 1
            self.skeleton_connect_mat[self.skeleton_parent_list[i], i] = 1
        self.skeleton_connect_mat[-1] = 1
        self.skeleton_connect_mat[:, -1] = 1

        self.skeleton_edges = dense_to_sparse(self.skeleton_connect_mat)[0]

        self.marker_gcn_list = nn.ModuleList()
        for i in range(len(args.marker_gcn_dim) - 1):
            self.marker_gcn_list.append(SkeletonLinear(args.marker_gcn_dim[i], args.marker_gcn_dim[i + 1],
                                                       self.marker_num, self.marker_edges))
            self.marker_gcn_list.append(act_function(-1, self.marker_gcn_act[i]))

        self.skeleton_gcn_list = nn.ModuleList()
        for i in range(len(args.skeleton_gcn_dim) - 1):
            if i == 0:
                self.skeleton_gcn_list.append(
                    SkeletonLinear(args.skeleton_gcn_dim[i] + self.marker_skeleton_conv_gcn_dim[-1][-1],
                                   args.skeleton_gcn_dim[i + 1], self.joint_num + 1,
                                   self.skeleton_edges))
            else:
                self.skeleton_gcn_list.append(
                    SkeletonLinear(args.skeleton_gcn_dim[i], args.skeleton_gcn_dim[i + 1], self.joint_num + 1,
                                   self.skeleton_edges))
            if i == len(args.skeleton_gcn_dim) - 2:
                pass
            else:
                self.skeleton_gcn_list.append(act_function(-1, self.skeleton_gcn_act[i]))

        self.layers = nn.ModuleList()
        for i in range(len(self.dim) - 1):
            if self.dim[i] == self.dim[i + 1]:
                self.layers.append(
                    ResBlock(self.dim[i], act=self.act[i], batch_norm=self.batch_norm, dropout=self.dropout))
                # self.layers.append(ResBlock(self.dim[i], batch_norm=self.batch_norm, dropout=self.dropout))
            # elif i == len(self.dim) - 2:
            #     self.layers.append(nn.Sequential(nn.Linear(self.dim[-2], self.dim[-1])))
            else:
                self.layers.append(nn.Sequential(nn.Linear(self.dim[i], self.dim[i + 1]),
                                                 act_function(self.dim[i + 1], self.act[i]),
                                                 nn.BatchNorm1d(self.dim[i + 1], affine=True,
                                                                momentum=1e-3) if batch_norm else nn.Identity()))

        # for i in range(len(self.dim) - 1):
        #     if self.dim[i] == self.dim[i + 1]:
        #         self.layers.append(
        #             ResBlock(self.dim[i], act=self.act[i], batch_norm=self.batch_norm, dropout=self.dropout))
        #     else:
        #         self.layers.append(nn.Sequential(nn.Linear(self.dim[i], self.dim[i + 1]),
        #                                          act_function(self.dim[i + 1], self.act[i]),
        #                                          nn.BatchNorm1d(self.dim[i + 1], affine=True,
        #                                                         momentum=1e-3) if batch_norm else nn.Identity()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, window_size, feature_num = x.shape[0], x.shape[1], x.shape[2]
        self.marker_edges = self.marker_edges.to(x.device)
        for i in range(len(self.marker_gcn_dim) - 1):
            x = self.marker_gcn_list[i * 2](x.view((batch_size * window_size, -1)))
            x = self.marker_gcn_list[i * 2 + 1](x)
        x = x.view((batch_size * window_size, -1))

        joint_feature = None
        for i in range(len(self.marker_skeleton_conv_gcn_list)):
            if i == 0:
                joint_feature = self.marker_skeleton_conv_gcn_list[i](x)
            else:
                joint_feature = self.marker_skeleton_conv_gcn_list[i](joint_feature)

        for module in self.layers:
            x = module(x)

        x = torch.cat([x, joint_feature], dim=-1)
        for i in range(len(self.skeleton_gcn_list)):
            x = self.skeleton_gcn_list[i](x)

        x = x[:, :-(self.skeleton_gcn_dim[-1] - 3)]
        return x
