import torch
import torch.nn as nn
import torch.optim as optim
from dgl.utils import expand_as_pair
from dgl import function as fn
from dgl.base import DGLError
from dgl.nn.functional import edge_softmax
import numpy as np
import pandas as pd
from math import sqrt
from dgl.nn.pytorch import GraphConv  # 导入 GraphConv

class PosEncoding(nn.Module):

    def __init__(self, dim, device, base=10000, bias=0):

        super(PosEncoding, self).__init__()
        p = []
        sft = []
        for i in range(dim):
            b = (i - i % 2) / dim
            p.append(base ** -b)
            if i % 2:
                sft.append(np.pi / 2.0 + bias)
            else:
                sft.append(bias)
        self.device = device
        self.sft = torch.tensor(
            sft, dtype=torch.float32).view(1, -1).to(device)
        self.base = torch.tensor(p, dtype=torch.float32).view(1, -1).to(device)

    def forward(self, pos):
        with torch.no_grad():
            if isinstance(pos, list):
                pos = torch.tensor(pos, dtype=torch.float32).to(self.device)
            pos = pos.view(-1, 1)
            x = pos / self.base + self.sft
            return torch.sin(x)


class Tabular1DCNN2(nn.Module):
    def __init__(
            self,
            input_dim: int,
            embed_dim: int,
            K: int = 4,
            dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hid_dim = input_dim * embed_dim * 2
        self.cha_input = self.cha_output = input_dim
        self.cha_hidden = (input_dim * K) // 2
        self.sign_size1 = 2 * embed_dim
        self.sign_size2 = embed_dim
        self.K = K

        self.bn1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(input_dim, self.hid_dim)

        self.bn_cv1 = nn.BatchNorm1d(self.cha_input)
        self.conv1 = nn.Conv1d(
            in_channels=self.cha_input,
            out_channels=self.cha_input * self.K,
            kernel_size=5,
            padding=2,
            groups=self.cha_input,
            bias=False
        )

        self.ave_pool1 = nn.AdaptiveAvgPool1d(self.sign_size2)

        self.bn_cv2 = nn.BatchNorm1d(self.cha_input * self.K)
        self.dropout2 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            in_channels=self.cha_input * self.K,
            out_channels=self.cha_input * (self.K),
            kernel_size=3,
            padding=1,
            bias=True
        )

        self.bn_cv3 = nn.BatchNorm1d(self.cha_input * self.K)
        self.conv3 = nn.Conv1d(
            in_channels=self.cha_input * (self.K),
            out_channels=self.cha_input * (self.K // 2),
            kernel_size=3,
            padding=1,
            bias=True
        )

        self.bn_cvs = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(6):
            self.bn_cvs.append(nn.BatchNorm1d(self.cha_input * (self.K // 2)))
            self.convs.append(nn.Conv1d(
                in_channels=self.cha_input * (self.K // 2),
                out_channels=self.cha_input * (self.K // 2),
                kernel_size=3,
                padding=1,
                bias=True
            ))

        self.bn_cv10 = nn.BatchNorm1d(self.cha_input * (self.K // 2))
        self.conv10 = nn.Conv1d(
            in_channels=self.cha_input * (self.K // 2),
            out_channels=self.cha_output,
            kernel_size=3,
            padding=1,
            bias=True
        )

    def forward(self, x):
        x = self.dropout1(self.bn1(x))
        x = nn.functional.celu(self.dense1(x))
        x = x.reshape(x.shape[0], self.cha_input,
                      self.sign_size1)

        x = self.bn_cv1(x)
        x = nn.functional.relu(self.conv1(x))
        x = self.ave_pool1(x)

        x_input = x
        x = self.dropout2(self.bn_cv2(x))
        x = nn.functional.relu(self.conv2(x))
        x = x + x_input

        x = self.bn_cv3(x)
        x = nn.functional.relu(self.conv3(x))

        for i in range(6):
            x_input = x
            x = self.bn_cvs[i](x)
            x = nn.functional.relu(self.convs[i](x))
            x = x + x_input

        x = self.bn_cv10(x)
        x = nn.functional.relu(self.conv10(x))

        return x


class TransEmbedding(nn.Module):

    def __init__(
            self,
            df=None,
            device='cpu',
            dropout=0.2,
            in_feats_dim=82,
            cat_features=None,
            neigh_features: dict = None,
            att_head_num: int = 4,
            neighstat_uni_dim=64
    ):
        super(TransEmbedding, self).__init__()
        self.time_pe = PosEncoding(dim=in_feats_dim, device=device, base=100)

        self.cat_table = nn.ModuleDict({col: nn.Embedding(max(df[col].unique(
        )) + 1, in_feats_dim).to(device) for col in cat_features if col not in {"Labels", "Time"}})

        if isinstance(neigh_features, dict):
            self.nei_table = Tabular1DCNN2(input_dim=len(
                neigh_features), embed_dim=in_feats_dim)

        self.att_head_num = att_head_num
        self.att_head_size = int(in_feats_dim / att_head_num)
        self.total_head_size = in_feats_dim
        self.lin_q = nn.Linear(in_feats_dim, self.total_head_size)
        self.lin_k = nn.Linear(in_feats_dim, self.total_head_size)
        self.lin_v = nn.Linear(in_feats_dim, self.total_head_size)

        self.lin_final = nn.Linear(in_feats_dim, in_feats_dim)
        self.layer_norm = nn.LayerNorm(in_feats_dim, eps=1e-8)

        self.neigh_mlp = nn.Linear(in_feats_dim, 1)

        self.neigh_add_mlp = nn.ModuleList([nn.Linear(in_feats_dim, in_feats_dim) for i in range(
            len(neigh_features.columns))]) if isinstance(neigh_features, pd.DataFrame) else None

        self.label_table = nn.Embedding(
            3, in_feats_dim, padding_idx=2).to(device)
        self.time_emb = None
        self.emb_dict = None
        self.label_emb = None
        self.cat_features = cat_features
        self.neigh_features = neigh_features
        self.forward_mlp = nn.ModuleList(
            [nn.Linear(in_feats_dim, in_feats_dim) for i in range(len(cat_features))])
        self.dropout = nn.Dropout(dropout)

    def forward_emb(self, cat_feat):
        if self.emb_dict is None:
            self.emb_dict = self.cat_table
        support = {col: self.emb_dict[col](
            cat_feat[col]) for col in self.cat_features if col not in {"Labels", "Time"}}
        return support

    def transpose_for_scores(self, input_tensor):
        new_x_shape = input_tensor.size(
        )[:-1] + (self.att_head_num, self.att_head_size)
        input_tensor = input_tensor.view(*new_x_shape)
        return input_tensor.permute(0, 2, 1, 3)

    def forward_neigh_emb(self, neighstat_feat):
        cols = neighstat_feat.keys()
        tensor_list = []
        for col in cols:
            tensor_list.append(neighstat_feat[col])
        neis = torch.stack(tensor_list).T
        input_tensor = self.nei_table(neis)

        mixed_q_layer = self.lin_q(input_tensor)
        mixed_k_layer = self.lin_k(input_tensor)
        mixed_v_layer = self.lin_v(input_tensor)

        q_layer = self.transpose_for_scores(mixed_q_layer)
        k_layer = self.transpose_for_scores(mixed_k_layer)
        v_layer = self.transpose_for_scores(mixed_v_layer)

        att_scores = torch.matmul(q_layer, k_layer.transpose(-1, -2))
        att_scores = att_scores / sqrt(self.att_head_size)

        att_probs = nn.Softmax(dim=-1)(att_scores)
        context_layer = torch.matmul(att_probs, v_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.total_head_size,)
        context_layer = context_layer.view(*new_context_shape)
        hidden_states = self.lin_final(context_layer)
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states, cols

    def forward(self, cat_feat: dict, neighstat_feat: dict):
        support = self.forward_emb(cat_feat)
        cat_output = 0
        nei_output = 0
        for i, k in enumerate(support.keys()):
            support[k] = self.dropout(support[k])
            support[k] = self.forward_mlp[i](support[k])
            cat_output = cat_output + support[k]

        if neighstat_feat is not None:
            nei_embs, cols_list = self.forward_neigh_emb(neighstat_feat)
            nei_output = self.neigh_mlp(nei_embs).squeeze(-1)

        return cat_output, nei_output


class RGTAN(nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 heads,
                 activation,
                 skip_feat=True,
                 gated=True,
                 layer_norm=True,
                 post_proc=True,
                 n2v_feat=True,
                 drop=None,
                 ref_df=None,
                 cat_features=None,
                 neigh_features=None,
                 nei_att_head=4,
                 device='cpu'):

        super(RGTAN, self).__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.heads = heads
        self.activation = activation
        self.input_drop = nn.Dropout(drop[0])
        self.drop = drop[1]
        self.output_drop = nn.Dropout(self.drop)

        if n2v_feat:
            self.n2v_mlp = TransEmbedding(
                ref_df, device=device, in_feats_dim=in_feats, cat_features=cat_features, neigh_features=neigh_features,
                att_head_num=nei_att_head)
            self.nei_feat_dim = len(neigh_features.keys()) if isinstance(
                neigh_features, dict) else 0
        else:
            self.n2v_mlp = lambda x: x
            self.nei_feat_dim = 0

        self.layers = nn.ModuleList()
        # Label embedding layers are the same as original
        self.layers.append(nn.Embedding(
            n_classes + 1, in_feats + self.nei_feat_dim, padding_idx=n_classes))
        self.layers.append(
            nn.Linear(self.in_feats + self.nei_feat_dim, self.hidden_dim * self.heads[0]))
        self.layers.append(
            nn.Linear(self.in_feats + self.nei_feat_dim, self.hidden_dim * self.heads[0]))
        self.layers.append(nn.Sequential(nn.BatchNorm1d(self.hidden_dim * self.heads[0]),
                                         nn.PReLU(),
                                         nn.Dropout(self.drop),
                                         nn.Linear(self.hidden_dim *
                                                   self.heads[0], in_feats + self.nei_feat_dim)
                                         ))

        # 使用 GraphConv 构建GNN层 ---
        self.layers.append(GraphConv(in_feats=self.in_feats + self.nei_feat_dim,
                                     out_feats=self.hidden_dim * self.heads[0],
                                     activation=self.activation,
                                     allow_zero_in_degree=True))

        for l in range(0, (self.n_layers - 1)):
            self.layers.append(GraphConv(in_feats=self.hidden_dim * self.heads[l - 1],
                                         out_feats=self.hidden_dim * self.heads[l],
                                         activation=self.activation,
                                         allow_zero_in_degree=True))

        if post_proc:
            self.layers.append(
                nn.Sequential(nn.Linear(self.hidden_dim * self.heads[-1], self.hidden_dim * self.heads[-1]),
                              nn.BatchNorm1d(
                                  self.hidden_dim * self.heads[-1]),
                              nn.PReLU(),
                              nn.Dropout(self.drop),
                              nn.Linear(self.hidden_dim * self.heads[-1], self.n_classes)))
        else:
            self.layers.append(nn.Linear(self.hidden_dim *
                                         self.heads[-1], self.n_classes))

    # forward方法保持不变
    def forward(self, blocks, features, labels, n2v_feat=None, neighstat_feat=None):
        if n2v_feat is None and neighstat_feat is None:
            h = features
        else:
            cat_h, nei_h = self.n2v_mlp(n2v_feat, neighstat_feat)
            h = features + cat_h
            if isinstance(nei_h, torch.Tensor):
                h = torch.cat([h, nei_h], dim=-1)

        label_embed = self.input_drop(self.layers[0](labels))
        label_embed = self.layers[1](
            h) + self.layers[2](label_embed)
        label_embed = self.layers[3](label_embed)
        h = h + label_embed

        for l in range(self.n_layers):
            h = self.output_drop(self.layers[l + 4](blocks[l], h))

        logits = self.layers[-1](h)

        return logits