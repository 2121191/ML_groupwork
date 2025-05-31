import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from config import Config
from feature_engineering.data_engineering import data_engineer_benchmark, span_data_2d, span_data_3d
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

logger = logging.getLogger(__name__)
def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--method", default='rgtan', type=str)
    method = vars(parser.parse_args())['method']

    if method in ['gtan']:
        yaml_file = "config/gtan_cfg.yaml"
    elif method in ['rgtan']:
        yaml_file = "config/rgtan_cfg.yaml"

    elif method in ['rgtan-n']:
        yaml_file = "config/rgtan_n_cfg.yaml"
    elif method in ['rgtan-r']:
        yaml_file = "config/rgtan_r_cfg.yaml"
    elif method in ['rgtan-a']:
        yaml_file = "config/rgtan_a_cfg.yaml"
    elif method in ['rgtan-n']:
        yaml_file = "config/rgtan_n_cfg.yaml"
    # 小创新
    elif method in ['rgtan-aff']:
        yaml_file = "config/rgtan_aff_cfg.yaml"
    # ---------------------
    else:
        raise NotImplementedError(f"Unsupported method: {method}")

    with open(yaml_file) as file:
        args = yaml.safe_load(file)
    args['method'] = method
    return args


def base_load_data(args: dict):
    # 供普通的非图结构的模型使用的S-SSFD
    data_path = "data/S-FFSD.csv"
    feat_df = pd.read_csv(data_path)
    train_size = 1 - args['test_size']
    method = args['method']
    if os.path.exists("data/tel_2d.npy"):
        return
    features, labels = span_data_2d(feat_df)
    num_trans = len(feat_df)
    trf, tef, trl, tel = train_test_split(
        features, labels, train_size=train_size, stratify=labels, shuffle=True)
    trf_file, tef_file, trl_file, tel_file = args['trainfeature'], args[
        'testfeature'], args['trainlabel'], args['testlabel']
    np.save(trf_file, trf)
    np.save(tef_file, tef)
    np.save(trl_file, trl)
    np.save(tel_file, tel)
    return


def main(args):
    # 2025的论文模型
    if args['method'] == 'rgtan':
        from methods.rgtan.rgtan_main import rgtan_main, loda_rgtan_data
        feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features = loda_rgtan_data(
            args['dataset'], args['test_size'])
        rgtan_main(feat_data, g, train_idx, test_idx, labels, args,
                   cat_features, neigh_features, nei_att_head=args['nei_att_heads'][args['dataset']])
    # 2023的论文模型
    elif args['method'] == 'gtan':
        from methods.gtan.gtan_main import gtan_main, load_gtan_data
        feat_data, labels, train_idx, test_idx, g, cat_features = load_gtan_data(
            args['dataset'], args['test_size'])
        gtan_main(
            feat_data, g, train_idx, test_idx, labels, args, cat_features)
    # --- RGTAN-N (移除邻居风险) ---
    elif args['method'] == 'rgtan-n':
        from methods.rgtan_n.rgtan_n_main import rgtan_n_main, loda_rgtan_data
        feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features = loda_rgtan_data(
            args['dataset'], args['test_size'])
        rgtan_n_main(feat_data, g, train_idx, test_idx, labels, args,
                     cat_features, neigh_features, nei_att_head=args['nei_att_heads'][args['dataset']])
    # --- RGTAN-R (移除风险嵌入) ---
    elif args['method'] == 'rgtan-r':
        from methods.rgtan_r.rgtan_r_main import rgtan_r_main, loda_rgtan_data
        feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features = loda_rgtan_data(
            args['dataset'], args['test_size'])
        rgtan_r_main(feat_data, g, train_idx, test_idx, labels, args,
                     cat_features, neigh_features, nei_att_head=args['nei_att_heads'][args['dataset']])
    # --- RGTAN-A (移除注意力机制) ---
    elif args['method'] == 'rgtan-a':
        from methods.rgtan_a.rgtan_a_main import rgtan_a_main, loda_rgtan_data
        feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features = loda_rgtan_data(
            args['dataset'], args['test_size'])
        rgtan_a_main(feat_data, g, train_idx, test_idx, labels, args,
                     cat_features, neigh_features, nei_att_head=args['nei_att_heads'][args['dataset']])
    # --- RGTAN-AFF 如创 ---
    elif args['method'] == 'rgtan-aff':
        from methods.rgtan_aff.rgtan_aff_main import rgtan_aff_main, loda_rgtan_data
        feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features = loda_rgtan_data(
            args['dataset'], args['test_size'])
        rgtan_aff_main(feat_data, g, train_idx, test_idx, labels, args,
                       cat_features, neigh_features, nei_att_head=args['nei_att_heads'][args['dataset']])
    else:
        raise NotImplementedError("Unsupported method. ")

if __name__ == "__main__":
    main(parse_args())
