from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import re
import string
import tempfile

import six
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import tokenizer
from tensor2tensor.data_generators.wikisum import utils as cc_utils
from tensor2tensor.layers import modalities
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry
import tensorflow as tf

from tensor2tensor.utils import registry
from tensor2tensor.data_generators import problem
# from tensor2tensor.data_generators import text_problems
# from tensor2tensor.utils import metrics

import gzip # import library for reading datasets

@registry.register_problem
class OnlineRevew(problem.Problem):
    def example_reading_spec(self):
        data_fields = {
            "inputs": tf.VarLenFeature(tf.int64),
            "targets": tf.VarLenFeature(tf.int64),
            "section_boundaries": tf.VarLenFeature(tf.int64),
        }
        data_items_to_decoders = None
        return (data_fields, data_items_to_decoders)

    @property
    def target_vocab_size(self):
        return 2**15

    @property
    def vocab_filename(self):
        return "vocab.%s.%d" % (self.dataset_filename(), self.target_vocab_size)

    def feature_encoders(self, data_dir):
        vocab_filename = os.path.join(data_dir, self.vocab_filename)
        encoder = text_encoder.SubwordTextEncoder(vocab_filename)
        # Shared encoder for inputs and targets
        return {"inputs": encoder, "targets": encoder}
    
    def hparams(self, defaults, unused_model_hparams):
        p = defaults
        p.stop_at_eos = True

        p.vocab_size = {
            "inputs": self._encoders["inputs"].vocab_size,
            "targets": self._encoders["targets"].vocab_size,
        }
        p.modality = {
            "inputs": modalities.ModalityType.SYMBOL,
            "targets": modalities.ModalityType.SYMBOL,
        }    
    
    def eval_metrics(self):
        return super(WikisumBase, self).eval_metrics() + [
            metrics.Metrics.ROUGE_2_F, metrics.Metrics.ROUGE_L_F
        ]

#     def generate_lines_for_vocab(self, wikis_dir, refs_dir, max_chars=10**7):
#         total_chars = 0
#         ref_files_by_shard = _references_files_by_shard(refs_dir)
#         for shard_id in range(cc_utils.NUM_SHARDS):
#           # Wikipedia articles
#             for wiki in _wiki_articles(shard_id, wikis_dir):
#                 yield _normalize_text(wiki.title) + EOT
#                 for section in wiki.sections:
#                     yield _format_title(_normalize_text(section.title))
#                     yield _normalize_text(section.text)
#                     total_chars += len(section.title)
#                     total_chars += len(section.text)

#             # References
#             for i, content in enumerate(
#               six.itervalues(_references_content(ref_files_by_shard[shard_id]))):
#                 for line in content.split("\n"):
#                     if line:
#                         yield _normalize_text(line)
#                         total_chars += len(line)

#                 # Make sure we use at least 1k references
#                 if i >= 1000 and total_chars >= max_chars:
#                     break

#             if total_chars >= max_chars:
#                 tf.logging.info("Seen enough chars: %d; finished.", max_chars)
#                 break
#         tf.logging.info("Built vocabulary using %d chars", total_chars)    
    
#     def generate_vocab(self, data_dir, wikis_dir, refs_dir):
#     # Produce a SubwordTextEncoder from a subset of the data
#         return generator_utils.get_or_generate_vocab_inner(
#             data_dir, self.vocab_filename, self.target_vocab_size,
#             self.generate_lines_for_vocab(wikis_dir, refs_dir))

    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        tf.logging.warn("See wikisum/README.md for instructions to generate data.")
    
    def out_filepaths(self, data_dir):
        train_shards = 8
        dev_shards = 1
        test_shards = 1
        train_filepaths = self.training_filepaths(
            data_dir, train_shards, shuffled=True)
        dev_filepaths = self.dev_filepaths(data_dir, dev_shards, shuffled=True)
        test_filepaths = self.test_filepaths(data_dir, test_shards, shuffled=True)
        out_filepaths = train_filepaths + dev_filepaths + test_filepaths
        out_filepaths.sort()
        assert len(out_filepaths) == cc_utils.NUM_SHARDS
        return out_filepaths    
    
    
    
#     @property
#     def approx_vocab_size(self):
#         return 2**15  #15 ~32768 # 13~8k

#     @property
#     def is_generate_per_split(self):
#         return False

#     @property
#     def dataset_splits(self):
#     # 1% evaluation data
#         return [{
#             "split": problem.DatasetSplit.TRAIN,
#             "shards": 100,
#         }, {
#             "split": problem.DatasetSplit.EVAL,
#             "shards": 1,
#         }, {
#             "split": problem.DatasetSplit.TEST,
#             "shards": 1,
#         }]   
    
    
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
                    
                if ''.join(review.split()[:3]) == ''.join(summary.split()[:3]):
                    continue                                        
                
                yield {
                    "inputs": data['review'],
                    "targets": data['summary'],
                }

                