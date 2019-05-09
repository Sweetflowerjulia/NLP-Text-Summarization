from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import csv

@registry.register_problem
class JuliaProject2019(text_problems.Text2TextProblem):

    @property
    def approx_vocab_size(self):
        return 2**14  #15 ~32768  #13 ~8k

    @property
    def is_generate_per_split(self):
        return False

    @property
    def dataset_splits(self):
    # 10% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        del dataset_split

#         # pylint: disable=g-import-not-at-top
#         from gutenberg import acquire
#         from gutenberg import cleanup
#         # pylint: enable=g-import-not-at-top

        # load dataset
        with open('../../../data/food_review.csv') as f:
            reader = csv.reader(f)
            header = True
            for line in reader:
                if header == True:
                    header = False
                    continue
                else:
                    text, summary = line
                    yield {
                        "inputs": text,
                        "targets": summary,
                    }
                