#!/usr/bin/env bash
python -u train.py --config ./graph_models/MP_BASE.yml --device cuda:2
python -u train.py --config ./graph_models/MP_T.yml --device cuda:2
python -u train.py --config ./graph_models/MP_EG.yml --device cuda:2

python -u train.py --config ./graph_models/MP_EGT.yml --device cuda:2
python -u train.py --config ./graph_models/MP_EGTB.yml --device cuda:2
python -u train.py --config ./graph_models/MP_EGTBF.yml --device cuda:2
