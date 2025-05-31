import os
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.dataloading import NodeDataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, roc_curve, precision_recall_curve
from scipy.io import loadmat
from tqdm import tqdm
from . import * # 确保 early_stopper 类被正确导入
from .rgtan_aff_lpa import load_lpa_subtensor
from .rgtan_aff_model import RGTAN
import matplotlib.pyplot as plt  # 导入 matplotlib
import seaborn as sns  # 导入 seaborn
import matplotlib.font_manager as fm # 导入 matplotlib.font_manager

try:
    # 优先尝试微软雅黑或苹方，因为它们通常更美观
    if os.name == 'nt': # Windows
        font_paths = ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf']
    elif os.uname().sysname == 'Darwin': # macOS
        font_paths = ['/System/Library/Fonts/PingFang.ttc', '/Library/Fonts/Arial Unicode.ttf']
    else: # Linux
        font_paths = ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc']

    selected_font_path = None
    for path in font_paths:
        if os.path.exists(path):
            selected_font_path = path
            break

    if selected_font_path:
        myfont = fm.FontProperties(fname=selected_font_path, size=12)
        plt.rcParams['font.sans-serif'] = [myfont.get_name()]
    else:
        print("Warning: No common Chinese font found. Falling back to default font. Chinese characters might not display correctly.")
        # 如果找不到常见字体，可以尝试设置一个通用的字体家族，或者让matplotlib自己选择
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # matplotlib默认字体，可能不支持中文
        myfont = None # 不再使用自定义的fontproperties
except Exception as e:
    print(f"Error setting Chinese font: {e}. Falling back to default font.")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    myfont = None

plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
# --- 中文字体配置 END ---


def rgtan_aff_main(feat_df, graph, train_idx, test_idx, labels, args, cat_features, neigh_features: pd.DataFrame,
               nei_att_head):
    device = args['device']
    graph = graph.to(device)

    # 用于绘图的指标列表，每个折叠独立存储
    train_losses_per_fold = []
    val_losses_per_fold = []
    val_aps_per_fold = []
    val_aucs_per_fold = []

    oof_predictions = torch.from_numpy(
        np.zeros([len(feat_df), 2])).float().to(device)
    test_predictions = torch.from_numpy(
        np.zeros([len(feat_df), 2])).float().to(device)
    kfold = StratifiedKFold(
        n_splits=args['n_fold'], shuffle=True, random_state=args['seed'])

    y_target_for_kfold = labels.iloc[train_idx].values # 用于KFold分割训练集
    num_feat = torch.from_numpy(feat_df.values).float().to(device)
    cat_feat = {col: torch.from_numpy(feat_df[col].values).long().to(
        device) for col in cat_features}

    neigh_padding_dict = {}
    nei_feat = {}
    if isinstance(neigh_features, pd.DataFrame) and not neigh_features.empty:
        nei_feat = {col: torch.from_numpy(neigh_features[col].values).to(torch.float32).to(
            device) for col in neigh_features.columns}

    y_all_labels = labels
    labels = torch.from_numpy(y_all_labels.values).long().to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)

    for fold, (trn_idx, val_idx) in enumerate(kfold.split(feat_df.iloc[train_idx], y_target_for_kfold)):
        print(f'Training fold {fold + 1}')
        trn_ind = torch.from_numpy(np.array(train_idx)[trn_idx]).long().to(device)
        val_ind = torch.from_numpy(np.array(train_idx)[val_idx]).long().to(device)

        train_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        train_dataloader = NodeDataLoader(graph,
                                          trn_ind,
                                          train_sampler,
                                          device=device,
                                          use_ddp=False,
                                          batch_size=args['batch_size'],
                                          shuffle=True,
                                          drop_last=False,
                                          num_workers=0
                                          )
        val_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        val_dataloader = NodeDataLoader(graph,
                                        val_ind,
                                        val_sampler,
                                        use_ddp=False,
                                        device=device,
                                        batch_size=args['batch_size'],
                                        shuffle=False,
                                        drop_last=False,
                                        num_workers=0,
                                        )
        model = RGTAN(in_feats=feat_df.shape[1],
                      hidden_dim=args['hid_dim'] // 4,
                      n_classes=2,
                      heads=[4] * args['n_layers'],
                      activation=nn.PReLU(),
                      n_layers=args['n_layers'],
                      drop=args['dropout'],
                      device=device,
                      gated=args['gated'],
                      ref_df=feat_df,
                      cat_features=cat_feat,
                      neigh_features=nei_feat,
                      nei_att_head=nei_att_head).to(device)
        lr = args['lr'] * np.sqrt(args['batch_size'] / 1024)
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=args['wd'])
        lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[
            4000, 12000], gamma=0.3)

        earlystoper = early_stopper(
            patience=args['early_stopping'], verbose=True)

        fold_train_losses = []
        fold_val_losses = []
        fold_val_aps = []
        fold_val_aucs = []

        for epoch in range(args['max_epochs']):
            train_loss_list = []
            model.train()
            for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
                    num_feat, cat_feat, nei_feat, neigh_padding_dict, labels,
                    seeds, input_nodes, device, blocks)

                blocks = [block.to(device) for block in blocks]
                train_batch_logits = model(
                    blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)
                mask = batch_labels == 2
                train_batch_logits = train_batch_logits[~mask]
                batch_labels = batch_labels[~mask]

                train_loss = loss_fn(train_batch_logits, batch_labels)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                train_loss_list.append(train_loss.cpu().detach().numpy())

                if step % 10 == 0:
                    tr_batch_pred = torch.sum(torch.argmax(train_batch_logits.clone(
                    ).detach(), dim=1) == batch_labels) / batch_labels.shape[0]
                    score = torch.softmax(train_batch_logits.clone().detach(), dim=1)[
                            :, 1].cpu().numpy()
                    try:
                        print('In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, '
                              'train_ap:{:.4f}, train_acc:{:.4f}, train_auc:{:.4f}'.format(epoch, step,
                                                                                           np.mean(
                                                                                               train_loss_list),
                                                                                           average_precision_score(
                                                                                               batch_labels.cpu().numpy(),
                                                                                               score),
                                                                                           tr_batch_pred.detach(),
                                                                                           roc_auc_score(
                                                                                               batch_labels.cpu().numpy(),
                                                                                               score)))
                    except Exception as e:
                        print(f"Error calculating train metrics in batch: {e}")

            val_loss_sum = 0
            val_correct_predictions = 0
            val_total_samples = 0
            val_all_labels = []
            val_all_scores = []

            model.eval()
            with torch.no_grad():
                for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
                    batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
                        num_feat, cat_feat, nei_feat, neigh_padding_dict, labels,
                        seeds, input_nodes, device, blocks)

                    blocks = [block.to(device) for block in blocks]
                    val_batch_logits = model(
                        blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)

                    oof_predictions[seeds] = val_batch_logits

                    mask = batch_labels == 2
                    val_batch_logits = val_batch_logits[~mask]
                    batch_labels = batch_labels[~mask]

                    if batch_labels.shape[0] > 0:
                        val_loss_sum += loss_fn(val_batch_logits, batch_labels).item() * batch_labels.shape[0]
                        val_correct_predictions += torch.sum(torch.argmax(val_batch_logits, dim=1) == batch_labels).item()
                        val_total_samples += batch_labels.shape[0]

                        score = torch.softmax(val_batch_logits, dim=1)[:, 1].cpu().numpy()
                        val_all_labels.extend(batch_labels.cpu().numpy())
                        val_all_scores.extend(score)

                    if step % 10 == 0:
                        try:
                            current_val_loss = val_loss_sum / val_total_samples if val_total_samples > 0 else 0
                            current_val_ap = average_precision_score(val_all_labels, val_all_scores) if len(val_all_labels) > 0 else 0
                            current_val_auc = roc_auc_score(val_all_labels, val_all_scores) if len(val_all_labels) > 0 else 0
                            current_val_acc = val_correct_predictions / val_total_samples if val_total_samples > 0 else 0

                            print('In epoch:{:03d}|batch:{:04d}, val_loss:{:4f}, val_ap:{:.4f}, '
                                  'val_acc:{:.4f}, val_auc:{:.4f}'.format(epoch,
                                                                          step,
                                                                          current_val_loss,
                                                                          current_val_ap,
                                                                          current_val_acc,
                                                                          current_val_auc))
                        except Exception as e:
                            print(f"Error calculating val metrics in batch: {e}")

            avg_train_loss = np.mean(train_loss_list) if train_loss_list else 0
            avg_val_loss = val_loss_sum / val_total_samples if val_total_samples > 0 else 0
            avg_val_ap = average_precision_score(val_all_labels, val_all_scores) if len(val_all_labels) > 0 else 0
            avg_val_auc = roc_auc_score(val_all_labels, val_all_scores) if len(val_all_labels) > 0 else 0

            fold_train_losses.append(avg_train_loss)
            fold_val_losses.append(avg_val_loss)
            fold_val_aps.append(avg_val_ap)
            fold_val_aucs.append(avg_val_auc)

            earlystoper.earlystop(avg_val_loss, model)
            if earlystoper.is_earlystop:
                print("Early Stopping!")
                break
        print("Best val_loss for fold {} is: {:.7f}".format(fold + 1, earlystoper.best_cv))

        train_losses_per_fold.append(fold_train_losses)
        val_losses_per_fold.append(fold_val_losses)
        val_aps_per_fold.append(fold_val_aps)
        val_aucs_per_fold.append(fold_val_aucs)

        model.eval() # 确保模型处于评估模式
        with torch.no_grad():
            test_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
            test_dataloader = NodeDataLoader(graph,
                                             torch.from_numpy(np.array(test_idx)).long().to(device), # 修复后的代码
                                             test_sampler,
                                             use_ddp=False,
                                             device=device,
                                             batch_size=args['batch_size'],
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=0,
                                             )
            for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
                batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
                    num_feat, cat_feat, nei_feat, neigh_padding_dict, labels,
                    seeds, input_nodes, device, blocks)

                blocks = [block.to(device) for block in blocks]
                test_batch_logits = model(
                    blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)
                test_predictions[seeds] = test_batch_logits


    # 绘图部分 - 在所有折叠训练完成后进行
    title_suffix = " -- 翁振昊"

    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    for i, (train_losses, val_losses) in enumerate(zip(train_losses_per_fold, val_losses_per_fold)):
        plt.plot(train_losses, label=f'Fold {i+1} Train Loss')
        plt.plot(val_losses, label=f'Fold {i+1} Validation Loss', linestyle='--')
    plt.title(f'训练和验证损失曲线{title_suffix}', fontproperties=myfont, fontsize=16)
    plt.xlabel('Epoch', fontproperties=myfont)
    plt.ylabel('损失', fontproperties=myfont)
    plt.legend(prop=myfont) # 确保图例也支持中文
    plt.grid(True)
    plt.show()

    # 绘制验证AP曲线
    plt.figure(figsize=(12, 6))
    for i, val_aps in enumerate(val_aps_per_fold):
        plt.plot(val_aps, label=f'Fold {i+1} Validation AP')
    plt.title(f'验证平均精度（AP）曲线{title_suffix}', fontproperties=myfont, fontsize=16)
    plt.xlabel('Epoch', fontproperties=myfont)
    plt.ylabel('平均精度', fontproperties=myfont)
    plt.legend(prop=myfont)
    plt.grid(True)
    plt.show()

    # 绘制验证AUC曲线
    plt.figure(figsize=(12, 6))
    for i, val_aucs in enumerate(val_aucs_per_fold):
        plt.plot(val_aucs, label=f'Fold {i+1} Validation AUC')
    plt.title(f'验证ROC曲线下面积（AUC）曲线{title_suffix}', fontproperties=myfont, fontsize=16)
    plt.xlabel('Epoch', fontproperties=myfont)
    plt.ylabel('ROC AUC', fontproperties=myfont)
    plt.legend(prop=myfont)
    plt.grid(True)
    plt.show()

    # 绘制最终测试集的ROC曲线和PR曲线
    final_test_labels = y_all_labels.iloc[test_idx].values
    final_test_scores = torch.softmax(test_predictions, dim=1)[test_idx, 1].cpu().numpy()

    mask_final = final_test_labels != 2
    final_test_labels_filtered = final_test_labels[mask_final]
    final_test_scores_filtered = final_test_scores[mask_final]

    if len(np.unique(final_test_labels_filtered)) > 1:
        # ROC曲线
        fpr, tpr, _ = roc_curve(final_test_labels_filtered, final_test_scores_filtered)
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_score(final_test_labels_filtered, final_test_scores_filtered):.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正例率', fontproperties=myfont)
        plt.ylabel('真正例率', fontproperties=myfont)
        plt.title(f'测试集接收者操作特征（ROC）曲线{title_suffix}', fontproperties=myfont, fontsize=16)
        plt.legend(loc="lower right", prop=myfont)
        plt.grid(True)
        plt.show()

        # Precision-Recall曲线
        precision, recall, _ = precision_recall_curve(final_test_labels_filtered, final_test_scores_filtered)
        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {average_precision_score(final_test_labels_filtered, final_test_scores_filtered):.2f})')
        plt.xlabel('召回率', fontproperties=myfont)
        plt.ylabel('精确率', fontproperties=myfont)
        plt.title(f'测试集精确率-召回率曲线{title_suffix}', fontproperties=myfont, fontsize=16)
        plt.legend(loc="lower left", prop=myfont)
        plt.grid(True)
        plt.show()
    else:
        print("Skipping ROC and PR curve plotting for test set: Only one class present in filtered test labels.")


    # 最终的OOF和测试集评估
    y_target_oof = y_all_labels.iloc[train_idx].values
    oof_scores = torch.softmax(oof_predictions[train_idx], dim=1)[:, 1].cpu().numpy()
    mask_oof = y_target_oof != 2
    y_target_oof_filtered = y_target_oof[mask_oof]
    oof_scores_filtered = oof_scores[mask_oof]

    if len(np.unique(y_target_oof_filtered)) > 1:
        my_ap = average_precision_score(y_target_oof_filtered, oof_scores_filtered)
        print("NN out of fold AP is:", my_ap)
    else:
        print("Skipping OOF AP calculation: Only one class present in filtered OOF labels.")

    test_score_prob = torch.softmax(test_predictions, dim=1)[test_idx, 1].cpu().numpy()
    test_score_class = torch.argmax(test_predictions, dim=1)[test_idx].cpu().numpy()
    y_true_test = y_all_labels.iloc[test_idx].values

    mask_test_eval = y_true_test != 2
    y_true_test_filtered = y_true_test[mask_test_eval]
    test_score_prob_filtered = test_score_prob[mask_test_eval]
    test_score_class_filtered = test_score_class[mask_test_eval]

    if len(np.unique(y_true_test_filtered)) > 1:
        print("test AUC:", roc_auc_score(y_true_test_filtered, test_score_prob_filtered))
        print("test AP:", average_precision_score(y_true_test_filtered, test_score_prob_filtered))
    else:
        print("Skipping test AUC and AP calculation: Only one class present in filtered test labels.")

    print("test f1 (macro):", f1_score(y_true_test_filtered, test_score_class_filtered, average="macro"))

def loda_rgtan_data(dataset: str, test_size: float):
    # prefix = "./antifraud/data/"
    prefix = "data/"
    if dataset == 'S-FFSD':
        cat_features = ["Target", "Location", "Type"]

        df = pd.read_csv(prefix + "S-FFSDneofull.csv")
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        #####
        neigh_features = []
        #####
        data = df[df["Labels"] <= 2]
        data = data.reset_index(drop=True)
        out = []
        alls = []
        allt = []
        pair = ["Source", "Target", "Location", "Type"]
        for column in pair:
            src, tgt = [], []
            edge_per_trans = 3
            for c_id, c_df in tqdm(data.groupby(column), desc=column):
                c_df = c_df.sort_values(by="Time")
                df_len = len(c_df)
                sorted_idxs = c_df.index
                src.extend([sorted_idxs[i] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
                tgt.extend([sorted_idxs[i + j] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
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

        #######
        g.ndata['label'] = torch.from_numpy(
            labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        #######

        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])
        index = list(range(len(labels)))

        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=0.6,
                                                                random_state=2, shuffle=True)
        feat_neigh = pd.read_csv(
            prefix + "S-FFSD_neigh_feat.csv")
        print("neighborhood feature loaded for nn input.")
        neigh_features = feat_neigh

    elif dataset == 'yelp':
        cat_features = []
        neigh_features = []
        data_file = loadmat(prefix + 'YelpChi.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        # load the preprocessed adj_lists
        with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        index = list(range(len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=test_size,
                                                                random_state=2, shuffle=True)
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
        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])

        try:
            feat_neigh = pd.read_csv(
                prefix + "yelp_neigh_feat.csv")
            print("neighborhood feature loaded for nn input.")
            neigh_features = feat_neigh
        except:
            print("no neighbohood feature used.")

    elif dataset == 'amazon':
        cat_features = []
        neigh_features = []
        data_file = loadmat(prefix + 'Amazon.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        # load the preprocessed adj_lists
        with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        index = list(range(3305, len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:],
                                                                test_size=test_size, random_state=2, shuffle=True)
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
        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])
        try:
            feat_neigh = pd.read_csv(
                prefix + "amazon_neigh_feat.csv")
            print("neighborhood feature loaded for nn input.")
            neigh_features = feat_neigh
        except:
            print("no neighbohood feature used.")

    return feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features
