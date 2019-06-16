from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
# from tensor2tensor.utils import metrics

import tensorflow as tf
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import generator_utils


import gzip # import library for reading datasets

@registry.register_problem
class onlinereview(text_problems.Text2TextProblem):

    @property
    def approx_vocab_size(self):
        return 2**13  #15  #15 ~32768 # 13~8k

    @property
    def is_generate_per_split(self):
        return True

    @property
    def dataset_splits(self):
    # 1% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 10,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.TEST,
            "shards": 1,
        }]   
    
    @registry.register_hparams
    def onlinereviewhp():
        hparams = transformer.transformer_base()
        hparams.num_hidden_layers = 2
        hparams.hidden_size = 128
        hparams.filter_size = 512
        hparams.num_heads = 4
        hparams.attention_dropout = 0.6
        hparams.layer_prepostprocess_dropout = 0.6
        hparams.learning_rate = 0.05
        return hparams
    
    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        del dataset_split

        # load dataset
        dataset = '../../../data/reviews.json.gz'
        nn=0
        with gzip.open(dataset, 'rb') as f:
            
            for line in f:
                data = eval(line)
                review = data['review']
                summary = data['summary']
                if summary in review:
                    continue 
                    
                if ''.join(review.split()[:3]) == ''.join(summary.split()[:3]):
                    continue                                        
                
                nn+=1
                if nn<100:
                
                    yield {
                        "inputs": data['review'],
                        "targets": data['summary'],
                    }
                
                if nn>100:
                    break
                