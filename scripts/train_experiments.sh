#!/usr/bin/env bash
source activate chem
python -u experiment.py --config ../experiments/MT_EGTBF_150.yml --device $1
#python -u train.py --config ./graph_models/C_T.yml --device $1
#python -u train.py --config ./graph_models/C_EG.yml --device $1
#
#python -u train.py --config ./graph_models/C_EGT.yml --device $1
#python -u train.py --config ./graph_models/C_EGTB.yml --device $1
#python -u train.py --config ./graph_models/C_EGTBF.yml --device $1
