import tensorflow as tf
import re

from django.shortcuts import render
from django.views.generic import TemplateView
from django.http import HttpResponse
from rdkit import Chem

from reaction_predictors.t2t_models.t2t_utils import get_predictions
from utils.smiles_utils import get_reaction, parse_smile


data_dir = '/home/phillnik/Science/t2t/aug_data2x'
model_name = "transformer"
hparams_set = "transformer_base"
ckpt_path = tf.train.latest_checkpoint('/data/phillnik/out_aug')
problem_name = "aug_smiles"
predict_target = get_predictions(problem_name, data_dir, ckpt_path, hparams_set, model_name)


def home(request):
    context = {
        'source_smiles': "",
        'target_smiles': "",
        'reaction': "",
        'mol_file': ""
    }
    return render(request, 'index.html', context)


class HomePageView(TemplateView):
    def get(self, request, **kwargs):
        context = {
            'source_smiles': "",
            'target_smiles': "",
            'reaction': "",
            'mol_file': ""
        }
        return render(request, 'index.html', context)

    def post(self, request, **kwargs):
        target_smiles = None
        if 'src_smi' in request.POST:
            # import pdb
            # pdb.set_trace()
            source_smiles = parse_smile(request.POST["src_smi"])
            target_smiles = predict_target(source_smiles)
            print('Source smiles: {}'.format(source_smiles))
            print('Target smiles: {}'.format(target_smiles))
            mol = Chem.MolToMolBlock(Chem.MolFromSmiles(target_smiles))
            reaction = get_reaction(source_smiles, target_smiles)
            context = {
                'source_smiles': request.POST["src_smi"],
                'target_smiles': target_smiles.replace(" ", ""),
                'reaction': reaction,
                'mol_file': '\n' + mol
            }
            return render(request, 'index.html', context)
        return HttpResponse("Alert")
