#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:16:48 2019

@author: hua
"""

from tensor2tensor.utils.trainer_lib import create_hparams, registry
from tensor2tensor import problems

INPUT_TEXT_TO_TRANSLATE = 'Translate this sentence into French'

# Set Tensor2Tensor Arguments
MODEL_DIR_PATH = '~/Desktop/COMP5709_CapstoneProject/'
MODEL = 'transformer'
#HPARAMS = 'transformer_big_single_gpu'
T2T_PROBLEM = 'summarize_cnn_dailymail32k'

hparams = create_hparams(HPARAMS, data_dir=model_dir, problem_name=T2T_PROBLEM)

# Make any changes to default Hparams for model architechture used during training
hparams.batch_size = 1024
hparams.hidden_size = 7*80
hparams.filter_size = 7*80*4
hparams.num_heads = 8

# Load model into Memory
T2T_MODEL = registry.model(MODEL)(hparams, tf.estimator.ModeKeys.PREDICT)

# Init T2T Token Encoder/ Decoders
DATA_ENCODERS = problems.problem(T2T_PROBLEM).feature_encoders(model_dir)

### START USING MODELS
encoded_inputs= encode(INPUT_TEXT_TO_TRANSLATE, DATA_ENCODERS)
model_output = T2T_MODEL.infer(encoded_inputs, beam_size=2)["outputs"]
translated_text_in_french =  decode(model_output, DATA_ENCODERS)

print(translated_text_in_french)