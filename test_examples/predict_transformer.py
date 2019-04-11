import tensorflow as tf

from reaction_predictors.t2t_models.t2t_utils import get_predictions


if __name__ == '__main__':
    data_dir = '/home/phillnik/Science/t2t/aug_data2x'
    model_name = "transformer"
    hparams_set = "transformer_base"
    ckpt_path = tf.train.latest_checkpoint('/data/phillnik/out_aug')
    problem_name = "aug_smiles"
    translate = get_predictions(problem_name, data_dir, ckpt_path, hparams_set, model_name)
    print(translate('C C S ( = O ) ( = O ) Cl . O C C Br > A_CCN(CC)CC A_CCOCC '))
