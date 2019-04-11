import re

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry


@registry.register_problem
class Smiles(text_problems.Text2TextProblem):
    """Predict next line of poetry from the last line. From Gutenberg texts."""

    @property
    def approx_vocab_size(self):
        return 2 ** 13  # ~8k

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
            yield {
                "inputs": source,
                "targets": target,
            }
