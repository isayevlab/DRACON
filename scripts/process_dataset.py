import pandas as pd
import pickle
import argparse
import os
import init_path
import numpy as np

from glob import glob
from lib.dataset.build_dataset import build_dataset, get_meta


def normalize_datasets(datasets, meta):
    for p_name in datasets:
        for idx in datasets[p_name]:
            data = datasets[p_name][idx]
            data['reactants']['features'] -= np.array(list(meta['features_min'].values())).reshape(-1, 1)
            data['product']['features'] -= np.array(list(meta['features_min'].values())).reshape(-1, 1)
            datasets[p_name][idx] = data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dataset",
                        help="Path to directory with raw data files which is df with 'smarts' column in pickle")
    parser.add_argument("--save_path", help="Path to folder where processed dataset will be saved")
    parser.add_argument("--drop_hs",  type=bool, help="Drop hydrogenous from molecules")
    parser.add_argument("--label_atoms", type=bool)
    args = parser.parse_args()
    files = glob(f'{args.raw_dataset}/*.pkl')
    datasets = {}
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    for file in files:
        file_name = file.split('/')[-1]
        df = pd.read_pickle(file)
        df = df[df["smarts"].str.contains(r'>*>', na=False)]
        dataset = build_dataset(df, drop_hs=args.drop_hs, atom_labels=args.label_atoms)
        datasets[file_name] = dataset

    meta = get_meta(datasets.values())
    normalize_datasets(datasets, meta)

    for file_name in datasets:
        with open(f'{args.save_path}/{file_name}', 'wb') as f:
            pickle.dump(datasets[file_name], f)

    with open(f'{args.save_path}/meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
