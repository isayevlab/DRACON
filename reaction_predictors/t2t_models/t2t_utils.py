import tensorflow as tf
import numpy as np

from tensorflow.contrib import eager as tfe
from tensorflow.estimator import ModeKeys as Modes
from tensor2tensor import problems
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import registry


from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import metrics
from reaction_predictors.t2t_models import aug_smiles


def encode(input_str, encoders,  output_str=None):
    inputs = encoders["inputs"].encode(input_str) + [1]
    batch_inputs = tf.reshape(inputs, [1, -1, 1])
    return {"inputs": batch_inputs}


def decode(integers, encoders):
    integers = list(np.squeeze(integers))
    if 1 in integers:
        integers = integers[:integers.index(1)]
    return encoders["inputs"].decode(np.squeeze(integers))


def translate(inputs, translate_model, ckpt_path, encoders):
    encoded_inputs = encode(inputs, encoders)
    with tfe.restore_variables_on_create(ckpt_path):
        model_output = translate_model.infer(encoded_inputs)["outputs"]
    return decode(model_output, encoders)


def get_predictions(problem_name, data_dir, ckpt_path, hparams_set, model_name):
    tfe.enable_eager_execution()
    forward_reaction = problems.problem(problem_name)
    encoders = forward_reaction.feature_encoders(data_dir)
    hparams = trainer_lib.create_hparams(hparams_set, data_dir=data_dir, problem_name=problem_name)
    translate_model = registry.model(model_name)(hparams, Modes.EVAL)
    return lambda inputs: translate(inputs, translate_model, ckpt_path, encoders)
