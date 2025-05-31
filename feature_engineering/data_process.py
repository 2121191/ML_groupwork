# %%
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.io import loadmat
import torch
import dgl
import random
import os
import time
import argparse
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
# from . import *
DATADIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "data/")


def featmap_gen(tmp_df=None):

    time_span = [2, 3, 5, 15, 20, 50, 100, 150,
                 200, 300, 864, 2590, 5100, 10000, 24000] # “感知”它在多个过去时间窗口中的行为
    time_name = [str(i) for i in time_span]
    time_list = tmp_df['Time']
    post_fe = []
    for trans_idx, trans_feat in tqdm(tmp_df.iterrows()):
        new_df = pd.Series(trans_feat)
        temp_time = new_df.Time
        temp_amt = new_df.Amount
        # 在这批 window_df 上算各种统计量
        for length, tname in zip(time_span, time_name):
            lowbound = (time_list >= temp_time - length)
            upbound = (time_list <= temp_time)
            correct_data = tmp_df[lowbound & upbound]
            new_df['trans_at_avg_{}'.format(
                tname)] = correct_data['Amount'].mean()
            new_df['trans_at_totl_{}'.format(
                tname)] = correct_data['Amount'].sum()
            new_df['trans_at_std_{}'.format(
                tname)] = correct_data['Amount'].std()
            new_df['trans_at_bias_{}'.format(
                tname)] = temp_amt - correct_data['Amount'].mean()
            new_df['trans_at_num_{}'.format(tname)] = len(correct_data)
            new_df['trans_target_num_{}'.format(tname)] = len(
                correct_data.Target.unique())
            new_df['trans_location_num_{}'.format(tname)] = len(
                correct_data.Location.unique())
            new_df['trans_type_num_{}'.format(tname)] = len(
                correct_data.Type.unique())
        post_fe.append(new_df)
    return pd.DataFrame(post_fe)


def sparse_to_adjlist(sp_matrix, filename):
    """
    将稀疏矩阵转换为邻接表
    :param sp_matrix: 稀疏矩阵
    :param filename: 邻接表的文件名
    """
    # 添加自环
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # 创建邻接表
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    with open(filename, 'wb') as file:
        pickle.dump(adj_lists, file)
    file.close()


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def MinMaxScaling(data):
    mind, maxd = data.min(), data.max()
    # return mind + (data - mind) / (maxd - mind)
    return (data - mind) / (maxd - mind)


def k_neighs(
    graph: dgl.DGLGraph,
    center_idx: int,
    k: int,
    where: str,
    choose_risk: bool = False,
    risk_label: int = 1
) -> torch.Tensor:
    """return indices of risk k-hop neighbors 返回风险 k-hop邻居的索引参数:

    参数:
        graph (dgl.DGLGraph): dgl 图数据集
        center_idx (int): 中心节点的索引
        k (int): k 跳邻居
        where (str): {"predecessor", "successor"} ，表示查找的邻居类型是前驱还是后继
        risk_label (int, 可选): 欺诈标签的值。默认为 1。
    """
    target_idxs: torch.Tensor
    if k == 1:
        if where == "in":
            neigh_idxs = graph.predecessors(center_idx)
        elif where == "out":
            neigh_idxs = graph.successors(center_idx)

    elif k == 2:
        if where == "in":
            subg_in = dgl.khop_in_subgraph(
                graph, center_idx, 2, store_ids=True)[0]
            neigh_idxs = subg_in.ndata[dgl.NID][subg_in.ndata[dgl.NID] != center_idx]
            # delete center node itself
            neigh1s = graph.predecessors(center_idx)
            neigh_idxs = neigh_idxs[~torch.isin(neigh_idxs, neigh1s)]
        elif where == "out":
            subg_out = dgl.khop_out_subgraph(
                graph, center_idx, 2, store_ids=True)[0]
            neigh_idxs = subg_out.ndata[dgl.NID][subg_out.ndata[dgl.NID] != center_idx]
            neigh1s = graph.successors(center_idx)
            neigh_idxs = neigh_idxs[~torch.isin(neigh_idxs, neigh1s)]

    neigh_labels = graph.ndata['label'][neigh_idxs]
    if choose_risk:
        target_idxs = neigh_idxs[neigh_labels == risk_label]
    else:
        target_idxs = neigh_idxs

    return target_idxs


def count_risk_neighs(
    graph: dgl.DGLGraph,
    risk_label: int = 1
) -> torch.Tensor:

    ret = []
    for center_idx in graph.nodes():
        neigh_idxs = graph.successors(center_idx)
        neigh_labels = graph.ndata['label'][neigh_idxs]
        risk_neigh_num = (neigh_labels == risk_label).sum()
        ret.append(risk_neigh_num)

    return torch.Tensor(ret)


def feat_map():
    tensor_list = []
    feat_names = []
    for idx in tqdm(range(graph.num_nodes())):
        neighs_1_of_center = k_neighs(graph, idx, 1, "in")
        neighs_2_of_center = k_neighs(graph, idx, 2, "in")

        tensor = torch.FloatTensor([
            edge_feat[neighs_1_of_center, 0].sum().item(),
            # edge_feat[neighs_1_of_center, 0].std().item(),
            edge_feat[neighs_2_of_center, 0].sum().item(),
            # edge_feat[neighs_2_of_center, 0].std().item(),
            edge_feat[neighs_1_of_center, 1].sum().item(),
            # edge_feat[neighs_1_of_center, 1].std().item(),
            edge_feat[neighs_2_of_center, 1].sum().item(),
            # edge_feat[neighs_2_of_center, 1].std().item(),
        ])
        tensor_list.append(tensor)

    feat_names = ["1hop_degree", "2hop_degree",
                  "1hop_riskstat", "2hop_riskstat"]

    tensor_list = torch.stack(tensor_list)
    return tensor_list, feat_names


if __name__ == "__main__":

    set_seed(42)

    # %%
    """
        For Yelpchi dataset
        Code partially from https://github.com/YingtongDou/CARE-GNN
    """
    print(f"processing YELP data...")
    yelp = loadmat(os.path.join(DATADIR, 'YelpChi.mat'))
    #包含四种矩阵（对应表格中的R - U - R、R - S - R、R - T - R和合并的homo）
    net_rur = yelp['net_rur']
    net_rtr = yelp['net_rtr']
    net_rsr = yelp['net_rsr']
    yelp_homo = yelp['homo']


    '''
        拆分完的四个矩阵转化为邻接表
        原矩阵太过稀疏，直接用空间时间复杂度太高
        转换后的邻接表大致为：
        {
            0: {0, 3, 5},   # 用户0的自环和邻居3、5
            1: {1, 2, 4},   # 用户1的自环和邻居2、4
            ...
        }
    '''
    sparse_to_adjlist(net_rur, os.path.join(
        DATADIR, "yelp_rur_adjlists.pickle"))
    sparse_to_adjlist(net_rtr, os.path.join(
        DATADIR, "yelp_rtr_adjlists.pickle"))
    sparse_to_adjlist(net_rsr, os.path.join(
        DATADIR, "yelp_rsr_adjlists.pickle"))
    sparse_to_adjlist(yelp_homo, os.path.join(
        DATADIR, "yelp_homo_adjlists.pickle"))

    data_file = yelp
    labels = pd.DataFrame(data_file['label'].flatten())[0] #欺诈标记，用于之后计算交叉熵损失
    feat_data = pd.DataFrame(data_file['features'].todense().A) #节点原始特征矩阵X（n*d)

    # 加载预处理后的邻接表，基于邻接表构建DGL图（论文中原始图嵌入的时候就可以用）
    with open(os.path.join(DATADIR, "yelp_homo_adjlists.pickle"), 'rb') as file:
        homo = pickle.load(file) #这里原始图的构建只是依赖了homo
                                 # 没有那几个异构的多关系子图，这里也许可以改进）
    file.close()
    src = []
    tgt = []
    for i in homo:
        for j in homo[i]:
            src.append(i) # 源节点列表
            tgt.append(j) # 目标节点列表
    src = np.array(src)
    tgt = np.array(tgt)
    g = dgl.graph((src, tgt)) # 通过边列表构建DGL图
    g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
    g.ndata['feat'] = torch.from_numpy(
        feat_data.to_numpy()).to(torch.float32)
    dgl.data.utils.save_graphs(DATADIR + "graph-yelp.bin", [g])

    # %%
    """
        For Amazon dataset
    """
    print(f"processing AMAZON data...")
    amz = loadmat(os.path.join(DATADIR, 'Amazon.mat'))
    net_upu = amz['net_upu']
    net_usu = amz['net_usu']
    net_uvu = amz['net_uvu']
    amz_homo = amz['homo']

    sparse_to_adjlist(net_upu, os.path.join(
        DATADIR, "amz_upu_adjlists.pickle"))
    sparse_to_adjlist(net_usu, os.path.join(
        DATADIR, "amz_usu_adjlists.pickle"))
    sparse_to_adjlist(net_uvu, os.path.join(
        DATADIR, "amz_uvu_adjlists.pickle"))
    sparse_to_adjlist(amz_homo, os.path.join(
        DATADIR, "amz_homo_adjlists.pickle"))

    data_file = amz
    labels = pd.DataFrame(data_file['label'].flatten())[0]
    feat_data = pd.DataFrame(data_file['features'].todense().A)
    # load the preprocessed adj_lists
    with open(DATADIR + 'amz_homo_adjlists.pickle', 'rb') as file:
        homo = pickle.load(file)
    file.close()
    src = []
    tgt = []
    for i in homo:
        for j in homo[i]:
            src.append(i)
            tgt.append(j)
    src = np.array(src)
    tgt = np.array(tgt)
    g = dgl.graph((src, tgt))
    g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
    g.ndata['feat'] = torch.from_numpy(
        feat_data.to_numpy()).to(torch.float32)
    dgl.data.utils.save_graphs(DATADIR + "graph-amazon.bin", [g])



    # # %%
    # """
    #     For S-FFSD dataset
    # """
    print(f"processing S-FFSD data...")
    data = pd.read_csv(os.path.join(DATADIR, 'S-FFSD.csv'))
    data = featmap_gen(data.reset_index(drop=True)) #加入了时间信息
    data.replace(np.nan, 0, inplace=True)
    data.to_csv(os.path.join(DATADIR, 'S-FFSDneofull.csv'), index=None)
    data = pd.read_csv(os.path.join(DATADIR, 'S-FFSDneofull.csv'))

    data = data.reset_index(drop=True)
    out = []
    alls = []
    allt = []
    pair = ["Source", "Target", "Location", "Type"]
    for column in pair:
        src, tgt = [], [] # 本属性下临时存放的边
        edge_per_trans = 3
        # 按当前属性(column)分组，每组内的交易共享同一个属性值
        for c_id, c_df in tqdm(data.groupby(column), desc=column):
            # c_df 是只含同一属性值（如同一 Source=S10000）的子表
            # 将该子表按时间排序，保证只连“后续”交易
            c_df = c_df.sort_values(by="Time")  # 该组交易数
            df_len = len(c_df)# 排序后在原 DataFrame 中的索引列表
            # 对组内每个位置 i，向它后面 j=1..edge_per_trans 的交易连边
            sorted_idxs = c_df.index
            src.extend([sorted_idxs[i]
                        for i in range(df_len)
                        for j in range(edge_per_trans)
                        if i + j < df_len])
            tgt.extend([sorted_idxs[i+j]
                        for i in range(df_len)
                        for j in range(edge_per_trans)
                        if i + j < df_len])
        alls.extend(src)
        allt.extend(tgt)
    alls = np.array(alls)
    allt = np.array(allt)
    g = dgl.graph((alls, allt))
    cal_list = ["Source", "Target", "Location", "Type"]
    for col in cal_list:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].apply(str).values)
    feat_data = data.drop("Labels", axis=1)
    labels = data["Labels"]
    g.ndata['label'] = torch.from_numpy(
        labels.to_numpy()).to(torch.long)
    g.ndata['feat'] = torch.from_numpy(
        feat_data.to_numpy()).to(torch.float32)
    dgl.data.utils.save_graphs(DATADIR + "graph-S-FFSD.bin", [g])

    # generate neighbor riskstat features
    for file_name in ['S-FFSD', 'yelp', 'amazon']:
        print(
            f"Generating neighbor risk-aware features for {file_name} dataset...")
        graph = dgl.load_graphs(DATADIR + "graph-" + file_name + ".bin")[0][0]
        graph: dgl.DGLGraph
        print(f"graph info: {graph}")

        edge_feat: torch.Tensor
        degree_feat = graph.in_degrees().unsqueeze_(1).float()
        risk_feat = count_risk_neighs(graph).unsqueeze_(1).float()

        origin_feat_name = []
        edge_feat = torch.cat([degree_feat, risk_feat], dim=1)
        origin_feat_name = ['degree', 'riskstat']

        features_neigh, feat_names = feat_map()
        # print(f"feature neigh: {features_neigh.shape}")

        features_neigh = torch.cat(
            (edge_feat, features_neigh), dim=1
        ).numpy()
        feat_names = origin_feat_name + feat_names
        features_neigh[np.isnan(features_neigh)] = 0.

        output_path = DATADIR + file_name + "_neigh_feat.csv"
        features_neigh = pd.DataFrame(features_neigh, columns=feat_names)
        scaler = StandardScaler()
        # features_neigh = np.log(features_neigh + 1)
        features_neigh = pd.DataFrame(scaler.fit_transform(
            features_neigh), columns=features_neigh.columns)

        features_neigh.to_csv(output_path, index=False)
