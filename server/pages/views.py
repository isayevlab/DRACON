import tensorflow as tf
import re

from django.shortcuts import render
from django.views.generic import TemplateView
from django.http import HttpResponse
from rdkit import Chem

from reaction_predictors.t2t_models.t2t_utils import get_predictions


data_dir = '/home/phillnik/Science/t2t/aug_data2x'
model_name = "transformer"
hparams_set = "transformer_base"
ckpt_path = tf.train.latest_checkpoint('/data/phillnik/out_aug')
problem_name = "aug_smiles"
predict_target = get_predictions(problem_name, data_dir, ckpt_path, hparams_set, model_name)
regexp = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\|=|#|-|\+|\\\\\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

def home(request):
    return render(request, 'index.html', {'what': 'DWT File Upload with Django'})


class HomePageView(TemplateView):
    def get(self, request, **kwargs):
        return render(request, 'index.html', context=None)

    def post(self, request, **kwargs):
        target_smiles = None
        if 'smi' in request.POST:
            # import pdb
            # pdb.set_trace()
            source_smiles = ' '.join(re.findall(regexp, request.POST["smi"]))
            print('source smiles: {}'.format(source_smiles))
            target_smiles = predict_target(source_smiles).replace(' ', '')
            mol = Chem.MolToMolBlock(Chem.MolFromSmiles(target_smiles))
            print("mol {}".format(mol))
            context = {
                'smiles': target_smiles,
                'mol_file': '\n' + mol
            }
            return render(request, 'index.html', context)
        return HttpResponse("Alert")
