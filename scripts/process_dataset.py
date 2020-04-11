import pandas as pd
import pickle
import argparse
import os
import init_path

from glob import glob
from lib.dataset.build_dataset import build_dataset, get_meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dataset",
                        help="Path to directory with raw data files which is df with 'smarts' column in pickle")
    parser.add_argument("--save_path", help="Path to folder where processed dataset will be saved")
    args = parser.parse_args()
    files = glob(f'{args.raw_dataset}/*.pkl')
    datasets = []
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    for file in files:
        file_name = file.split('/')[-1]
        df = pd.read_pickle(file)
        df = df[df["smarts"].str.contains(r'>*>', na=False)]
        dataset = build_dataset(df)
        with open(f'{args.save_path}/{file_name}', 'wb') as f:
            pickle.dump(dataset, f)
        datasets.append(dataset)

    with open(f'{args.save_path}/meta.pkl', 'wb') as f:
        pickle.dump(get_meta(datasets), f)
