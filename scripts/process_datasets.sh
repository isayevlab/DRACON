#!/usr/bin/env bash
python -u process_dataset.py --raw_dataset ../data/raw_datasets/USPTO_50k --save_path ../data/processed_datasets/USPTO_50k
python -u process_dataset.py --raw_dataset ../data/raw_datasets/USPTO_MIT --save_path ../data/processed_datasets/USPTO_MIT
python -u process_dataset.py --raw_dataset ../data/raw_datasets/USPTO_STEREO --save_path ../data/processed_datasets/USPTO_STEREO