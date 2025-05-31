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

# 位置编码模块
class PosEncoding(nn.Module):

    def __init__(self, dim, device, base=10000, bias=0):

        super(PosEncoding, self).__init__()
        """
        初始化位置编码组件
        :param dim: 编码维度
        :param device: 模型训练设备
        :param base: 编码基数
        :param bias: 编码偏差
        """
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

# 图注意力卷积模块
class TransformerConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 bias=True,
                 allow_zero_in_degree=False,
                 # feat_drop=0.6,
                 # attn_drop=0.6,
                 skip_feat=True,
                 gated=True,
                 layer_norm=True,
                 activation=nn.PReLU()):
        """
        初始化Transformer层。
        注意力权重与图神经网络和欺诈检测网络通过端到端机制共同优化。
        :param in_feat: 输入特征的形状
        :param out_feats: 输出特征的形状
        :param num_heads: 多头注意力的头数
        :param bias: 是否使用偏置
        :param allow_zero_in_degree: 是否允许零入度节点
        :param skip_feat: 是否跳过某些特征
        :param gated: 是否使用门控机制
        :param layer_norm: 是否使用层归一化
        :param activation: 激活函数的类型
        """

        super(TransformerConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_heads = num_heads

        self.lin_query = nn.Linear(
            self._in_src_feats, self._out_feats*self._num_heads, bias=bias)
        self.lin_key = nn.Linear(
            self._in_src_feats, self._out_feats*self._num_heads, bias=bias)
        self.lin_value = nn.Linear(
            self._in_src_feats, self._out_feats*self._num_heads, bias=bias)

        # self.feat_dropout = nn.Dropout(p=feat_drop)
        # self.attn_dropout = nn.Dropout(p=attn_drop)
        if skip_feat:
            self.skip_feat = nn.Linear(
                self._in_src_feats, self._out_feats*self._num_heads, bias=bias)
        else:
            self.skip_feat = None
        if gated:
            self.gate = nn.Linear(
                3*self._out_feats*self._num_heads, 1, bias=bias)
        else:
            self.gate = None
        if layer_norm:
            self.layer_norm = nn.LayerNorm(self._out_feats*self._num_heads)
        else:
            self.layer_norm = None
        self.activation = activation

    def forward(self, graph, feat, get_attention=False):
        """
        描述：Transformer图卷积
        :param graph: 输入图
        :param feat: 输入特征
        :param get_attention: 是否获取注意力权重
        """

        graph = graph.local_var()

        if not self._allow_zero_in_degree:
            if (graph.in_degrees() == 0).any():
                raise DGLError('图中存在入度为0的节点，'
                               '这些节点的输出将无效。'
                               '这对某些应用有害，'
                               '会导致性能无声下降。'
                               '通过调用 `g = dgl.add_self_loop(g)` 在输入图上添加自循环将解决此问题。'
                               '在构造此模块时将 ``allow_zero_in_degree`` 设置为 `True` 将抑制检查并允许代码运行。')

        # 检查 feat 是否为元组
        if isinstance(feat, tuple):
            h_src = feat[0]
            h_dst = feat[1]
        else:
            h_src = feat
            h_dst = h_src[:graph.number_of_dst_nodes()]

        # 步骤 0. q, k, v
        q_src = self.lin_query(
            h_src).view(-1, self._num_heads, self._out_feats)
        k_dst = self.lin_key(h_dst).view(-1, self._num_heads, self._out_feats)
        v_src = self.lin_value(
            h_src).view(-1, self._num_heads, self._out_feats)
        # 将特征分配给节点
        graph.srcdata.update({'ft': q_src, 'ft_v': v_src})
        graph.dstdata.update({'ft': k_dst})
        # 点积
        graph.apply_edges(fn.u_dot_v('ft', 'ft', 'a'))

        # 边softmax计算注意力分数
        graph.edata['sa'] = edge_softmax(
            graph, graph.edata['a'] / self._out_feats**0.5)

        # 将softmax值广播到每条边，并聚合目标节点
        graph.update_all(fn.u_mul_e('ft_v', 'sa', 'attn'),
                         fn.sum('attn', 'agg_u'))

        # 将结果输出到目标节点
        rst = graph.dstdata['agg_u'].reshape(-1,
                                             self._out_feats*self._num_heads)

        if self.skip_feat is not None:
            skip_feat = self.skip_feat(feat[:graph.number_of_dst_nodes()])
            if self.gate is not None:
                gate = torch.sigmoid(
                    self.gate(
                        torch.concat([skip_feat, rst, skip_feat - rst], dim=-1)))
                rst = gate * skip_feat + (1 - gate) * rst
            else:
                rst = skip_feat + rst

        if self.layer_norm is not None:
            rst = self.layer_norm(rst)

        if self.activation is not None:
            rst = self.activation(rst)

        if get_attention:
            return rst, graph.edata['sa']
        else:
            return rst

# 时间卷积模块
class Tabular1DCNN2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        K: int = 4,  # K*input_dim -> 隐藏维度
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hid_dim = input_dim * embed_dim * 2
        self.cha_input = self.cha_output = input_dim
        self.cha_hidden = (input_dim*K) // 2
        self.sign_size1 = 2 * embed_dim
        self.sign_size2 = embed_dim
        self.K = K

        self.bn1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(input_dim, self.hid_dim)

        self.bn_cv1 = nn.BatchNorm1d(self.cha_input)
        self.conv1 = nn.Conv1d(
            in_channels=self.cha_input,
            out_channels=self.cha_input*self.K,
            kernel_size=5,
            padding=2,
            groups=self.cha_input,
            bias=False
        )

        self.ave_pool1 = nn.AdaptiveAvgPool1d(self.sign_size2)

        self.bn_cv2 = nn.BatchNorm1d(self.cha_input*self.K)
        self.dropout2 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            in_channels=self.cha_input*self.K,
            out_channels=self.cha_input*(self.K),
            kernel_size=3,
            padding=1,
            bias=True
        )

        self.bn_cv3 = nn.BatchNorm1d(self.cha_input*self.K)
        self.conv3 = nn.Conv1d(
            in_channels=self.cha_input*(self.K),
            out_channels=self.cha_input*(self.K//2),
            kernel_size=3,
            padding=1,
            # groups=self.cha_hidden,
            bias=True
        )

        self.bn_cvs = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(6):
            self.bn_cvs.append(nn.BatchNorm1d(self.cha_input*(self.K//2)))
            self.convs.append(nn.Conv1d(
                in_channels=self.cha_input*(self.K//2),
                out_channels=self.cha_input*(self.K//2),
                kernel_size=3,
                padding=1,
                # groups=self.cha_hidden,
                bias=True
            ))

        self.bn_cv10 = nn.BatchNorm1d(self.cha_input*(self.K//2))
        self.conv10 = nn.Conv1d(
            in_channels=self.cha_input*(self.K//2),
            out_channels=self.cha_output,
            kernel_size=3,
            padding=1,
            # groups=self.cha_hidden,
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
        x = nn.functional.relu(self.conv2(x))  # -> (|b|,24,32)
        x = x + x_input

        x = self.bn_cv3(x)
        x = nn.functional.relu(self.conv3(x))  # -> (|b|,6,32)

        for i in range(6):
            x_input = x
            x = self.bn_cvs[i](x)
            x = nn.functional.relu(self.convs[i](x))
            x = x + x_input

        x = self.bn_cv10(x)
        x = nn.functional.relu(self.conv10(x))

        return x

# 属性嵌入模块
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

        # 创新点: 定义一个用于计算类别特征注意力的网络
        # 这是一个简单的MLP，输入是特征嵌入，输出是一个标量(注意力分数)
        self.feature_attention_net = nn.Sequential(
            nn.Linear(in_feats_dim, in_feats_dim // 2),
            nn.Tanh(),
            nn.Linear(in_feats_dim // 2, 1)
        )


    def forward_emb(self, cat_feat):
        if self.emb_dict is None:
            self.emb_dict = self.cat_table
        support = {col: self.emb_dict[col](
            cat_feat[col]) for col in self.cat_features if col not in {"Labels", "Time"}}
        return support

    def transpose_for_scores(self, input_tensor):
        new_x_shape = input_tensor.size(
        )[:-1] + (self.att_head_num, self.att_head_size)
        # (|batch|, feat_num, dim) -> (|batch|, feta_num, head_num, head_size)
        input_tensor = input_tensor.view(*new_x_shape)
        return input_tensor.permute(0, 2, 1, 3)

    def forward_neigh_emb(self, neighstat_feat):
        if not isinstance(neighstat_feat, dict) or len(neighstat_feat) == 0:
            batch_size = next(iter(self.cat_table.values())).weight.shape[0]  # 粗略推测一个batch size
            dummy_tensor = torch.zeros((batch_size, self.total_head_size), device=self.layer_norm.weight.device)
            return dummy_tensor, []

        cols = list(neighstat_feat.keys())
        tensor_list = []
        for col in cols:
            tensor_list.append(neighstat_feat[col])

        try:
            neis = torch.stack(tensor_list).T  # shape: (batch_size, num_features)
        except Exception as e:
            print("neighstat_feat tensor stack error:", e)
            dummy_tensor = torch.zeros((tensor_list[0].shape[0], self.total_head_size),
                                       device=self.layer_norm.weight.device)
            return dummy_tensor, cols

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
        # 创新点: 使用注意力机制融合类别特征，替代简单的相加

        processed_embeddings = []
        for i, k in enumerate(support.keys()):
            # 每个类别特征依然先经过自己的MLP层
            processed_emb = self.dropout(support[k])
            processed_emb = self.forward_mlp[i](processed_emb)
            processed_embeddings.append(processed_emb)

        stacked_embeddings = torch.stack(processed_embeddings, dim=1)

        attention_scores = self.feature_attention_net(stacked_embeddings)
        attention_weights = torch.softmax(attention_scores, dim=1)

        cat_output = torch.sum(stacked_embeddings * attention_weights, dim=1)

        try:
            nei_embs, cols_list = self.forward_neigh_emb(neighstat_feat)
            nei_output = self.neigh_mlp(nei_embs).squeeze(-1)
        except Exception as e:
            print(f"[WARNING] Neighbor embedding failed: {e}")
            nei_output = torch.zeros(cat_output.shape[0], device=cat_output.device)

        return cat_output, nei_output

# 主模块
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
        """
        初始化 RGTAN-GNN 模型
        :param in_feats: 输入特征的形状
        :param hidden_dim: 模型隐藏层维度
        :param n_layers: GTAN 层的数量
        :param n_classes: 分类的类别数量
        :param heads: 多头注意力的头数
        :param activation: 激活函数的类型
        :param skip_feat: 是否跳过某些特征
        :param gated: 是否使用门控机制
        :param layer_norm: 是否使用层归一化
        :param post_proc: 是否使用后处理
        :param n2v_feat: 是否使用 n2v 特征
        :param drop: 是否使用 dropout
        :param ref_df: 是否引用其他节点特征
        :param cat_features: 类别特征
        :param neigh_features: 邻居统计特征
        :param nei_att_head: 邻居风险统计特征的多头注意力头数
        :param device: 模型训练设备
        """

        super(RGTAN, self).__init__()
        self.in_feats = in_feats  # 特征维度
        self.hidden_dim = hidden_dim  # 64
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.heads = heads  # [4,4,4]
        self.activation = activation  # PRelu
        # self.input_drop = lambda x: x
        self.input_drop = nn.Dropout(drop[0])
        self.drop = drop[1]
        self.output_drop = nn.Dropout(self.drop)
        # self.pn = PairNorm(mode=pairnorm)
        if n2v_feat:
            self.n2v_mlp = TransEmbedding(
                ref_df, device=device, in_feats_dim=in_feats, cat_features=cat_features, neigh_features=neigh_features, att_head_num=nei_att_head)
            self.nei_feat_dim = len(neigh_features.keys()) if isinstance(
                neigh_features, dict) else 0
        else:
            self.n2v_mlp = lambda x: x
            self.nei_feat_dim = 0
        self.layers = nn.ModuleList()
        self.layers.append(nn.Embedding(
            n_classes+1, in_feats + self.nei_feat_dim, padding_idx=n_classes))
        self.layers.append(
            nn.Linear(self.in_feats + self.nei_feat_dim, self.hidden_dim*self.heads[0]))
        self.layers.append(
            nn.Linear(self.in_feats + self.nei_feat_dim, self.hidden_dim*self.heads[0]))
        self.layers.append(nn.Sequential(nn.BatchNorm1d(self.hidden_dim*self.heads[0]),
                                         nn.PReLU(),
                                         nn.Dropout(self.drop),
                                         nn.Linear(self.hidden_dim *
                                                   self.heads[0], in_feats + self.nei_feat_dim)
                                         ))

        # 构建多层
        self.layers.append(TransformerConv(in_feats=self.in_feats + self.nei_feat_dim,
                                           out_feats=self.hidden_dim,
                                           num_heads=self.heads[0],
                                           skip_feat=skip_feat,
                                           gated=gated,
                                           layer_norm=layer_norm,
                                           activation=self.activation))

        for l in range(0, (self.n_layers - 1)):
            # 由于是多头，in_dim = num_hidden * num_heads
            self.layers.append(TransformerConv(in_feats=self.hidden_dim * self.heads[l - 1],
                                               out_feats=self.hidden_dim,
                                               num_heads=self.heads[l],
                                               skip_feat=skip_feat,
                                               gated=gated,
                                               layer_norm=layer_norm,
                                               activation=self.activation))
        if post_proc:
            self.layers.append(nn.Sequential(nn.Linear(self.hidden_dim * self.heads[-1], self.hidden_dim * self.heads[-1]),
                                             nn.BatchNorm1d(
                                                 self.hidden_dim * self.heads[-1]),
                                             nn.PReLU(),
                                             nn.Dropout(self.drop),
                                             nn.Linear(self.hidden_dim * self.heads[-1], self.n_classes)))
        else:
            self.layers.append(nn.Linear(self.hidden_dim *
                               self.heads[-1], self.n_classes))

    def forward(self, blocks, features, labels, n2v_feat=None, neighstat_feat=None):
        """
        :param blocks: 训练块
        :param features: 训练特征
        :param labels: 训练标签
        :param n2v_feat: 是否使用 n2v 特征
        :param neighstat_feat: 邻居风险统计特征
        """
        if n2v_feat is None and neighstat_feat is None:
            h = features
        else:
            cat_h, nei_h = self.n2v_mlp(n2v_feat, neighstat_feat)
            h = features + cat_h
            if isinstance(nei_h, torch.Tensor):
                h = torch.cat([h, nei_h], dim=-1)

        label_embed = self.input_drop(self.layers[0](labels))
        label_embed = self.layers[1](h) + self.layers[2](label_embed)
        label_embed = self.layers[3](label_embed)
        h = h + label_embed

        for l in range(self.n_layers):
            h = self.output_drop(self.layers[l+4](blocks[l], h))

        logits = self.layers[-1](h)

        return logits