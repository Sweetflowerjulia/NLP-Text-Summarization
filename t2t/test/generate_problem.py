from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import os
import re

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.layers import modalities
from tensor2tensor.utils import metrics
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import text_problems

import tensorflow as tf

import gzip # import library for reading datasets

@registry.register_problem
class OnlineRevewProjectUSYD(text_problems.Text2TextProblem):

    @property
    def approx_vocab_size(self):
        return 2**15  #15 ~32768 # 13~8k

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
        },{
            "split": problem.DatasetSplit.TEST,
            "shards": 0.0001,
        }]   
    
    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        del dataset_split

        # load dataset
        dataset = '../../../data/reviews.json.gz'
        
        with gzip.open(dataset, 'rb') as f:
            
            for line in f:
                data = eval(line)
                review = data['review']
                summary = data['summary']
                if summary in review:
                    continue 
                
                yield {
                    "inputs": data['review'],
                    "targets": data['summary'],
                }

                