#!/usr/bin/env bash
python -u train.py --config ./graph_models/MT_EGBTF.yml --device $1
python -u train.py --config ./graph_models/C_TBF_sm_vis.yml --device $1
python -u train.py --config ./graph_models/MT_TBF_sm_vis.yml --device $1
python -u train.py --config ./graph_models/C_EGBF_sm_vis.yml --device $1
python -u train.py --config ./graph_models/MT_EGBF_sm_vis.yml --device $1
