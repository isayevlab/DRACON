#!/usr/bin/env bash
python -u train.py --config ./graph_models/C_BASE.yml --device $1
python -u train.py --config ./graph_models/C_T.yml --device $1
python -u train.py --config ./graph_models/C_EG.yml --device $1

python -u train.py --config ./graph_models/C_EGT.yml --device $1
python -u train.py --config ./graph_models/C_EGTB.yml --device $1
python -u train.py --config ./graph_models/C_EGTBF.yml --device $1
