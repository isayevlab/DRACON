import re
import random

from rdkit import Chem, RDConfig
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry


def augment_reaction(reaction):
    molecules = reaction.split('.')
    molecules_smiles = [mol.replace(" ", "") for mol in molecules]
    molecules_rdkit = [Chem.MolFromSmiles(mol) for mol in molecules_smiles]
    random_molecules = [randomSmiles(mol) for mol in molecules_rdkit]
    random.shuffle(random_molecules)
    random_molecules = [" ".join(list(mol)) for mol in random_molecules]
    result = ' . '.join(random_molecules)
    return result


def randomSmiles(m1):
    m1.SetProp("_canonicalRankingNumbers", "True")
    idxs = list(range(0, m1.GetNumAtoms()))
    random.shuffle(idxs)
    for i, v in enumerate(idxs):
        m1.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
    return Chem.MolToSmiles(m1)


@registry.register_problem
class InvSmiles(text_problems.Text2TextProblem):
    """Predict next line of poetry from the last line. From Gutenberg texts."""

    @property
    def approx_vocab_size(self):
        return 2 ** 10

    @property
    def is_generate_per_split(self):
        # generate_data will shard the data into TRAIN and EVAL for us.
        return False

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        # 10% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        with open("/home/phillnik/Science/ReactionPrediction/data/USP_src-train.txt") as f:
            source_smiles = f.read().split('\n')

        with open("/home/phillnik/Science/ReactionPrediction/data/USP_tgt-train.txt") as f:
            target_smiles = f.read().split('\n')

        for source, target in zip(source_smiles, target_smiles):
            try:
                for _ in range(2):
                    target = augment_reaction(target)
                    yield {
                        "inputs": target,
                        "targets": source.split('>')[0],
                    }
            except Exception:
                pass
