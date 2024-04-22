import torch.nn as nn
import torch.nn.functional as F
# from position_encoding import DeepWalk
# import networkx as nx
import numpy as np
from student_model.layers import (
    MLP
)
from mlp_mixer_pytorch import MLPMixer,MLPMixer4
import torch

class Readout(nn.Module):
    def __init__(
            self,

    ):
        super(Readout, self).__init__()
        self.out = nn.Sequential()
        self.out.append(nn.Conv2d(8, 1, 1))

    def forward(self, h):
        for l, layer in enumerate(self.out):
            h = layer(h)
        return h
class MLP_Student(nn.Module):
    def __init__(
        self,
        depth,
        dropout2,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        norm_type="none",
        hidden_list=[],
        output=9
    ):

        super(MLP_Student, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.model_name = 'MLP'
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.Sequential()
        self.norms = nn.Sequential()
        self.mlp1 = MLP(128, 1)


        self.MLPMixer= MLPMixer(
            image_size = (12, 207),
            channels = 3,
            patch_size1 = 12,
            patch_size2 = 1,
            dim = 12*64,
            depth = depth,
            num_classes = 1000,
            dropout=dropout2,
            output=12
        )



        self.MLPMixer4 = MLPMixer4(
            image_size=(12, 207),
            channels=64,
            patch_size1=12,
            patch_size2=1,
            dim=output*128,
            depth=depth,
            num_classes=1000,
            dropout=dropout2,
            output=output
        )

        if num_layers == 1:
            self.layers.append(nn.Conv2d(input_dim, output_dim, 1))
        else:
            if hidden_list==[]:
                self.layers.add_module('linear1',nn.Conv2d(input_dim, hidden_dim, 1))
            else:
                self.layers.add_module('linear1',nn.Conv2d(input_dim, hidden_list[0],1))
            if self.norm_type == "batch":
                self.norms.add_module(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.add_module(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.add_module(str('linear'+str(i+2)),nn.Conv2d(hidden_list[i], hidden_list[i+1],1))
                # self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.add_module(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.add_module(nn.LayerNorm(hidden_dim))
            if num_layers==2:
                self.layers.add_module(nn.Conv2d(hidden_dim, output_dim,1))
            else:
                self.layers.add_module(str('linear'+str(num_layers)),nn.Conv2d(hidden_list[-1], output_dim,1))

            # self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, feats):
        h = feats.permute(0, 3, 1, 2)
        h = h.to(torch.float32)
        h = self.MLPMixer(h)
        h = h.permute(0, 3, 2, 1)
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        h = self.MLPMixer4(h)
        h = h.permute(0, 3, 1, 2)
        pred = self.mlp1(h.cuda()).permute(0, 3, 2, 1).squeeze(dim=3)
        return pred




