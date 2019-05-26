import math
import numpy as np
# from keras_pos_embd import PositionEmbedding
# from keras_layer_normalization import LayerNormalization
# from keras_transformer import get_encoders
# from keras_transformer import get_custom_objects as get_encoder_custom_objects
# from .backend import keras
# from .backend import backend as K
# from .layers import get_inputs, get_embedding, TokenEmbedding, EmbeddingSimilarity, Masked, Extract
# from .optimizers import AdamWarmup


# __all__ = [
#     'TOKEN_PAD', 'TOKEN_UNK', 'TOKEN_CLS', 'TOKEN_SEP', 'TOKEN_MASK',
#     'gelu', 'get_model', 'get_custom_objects', 'get_base_dict', 'gen_batch_inputs',
# ]
from keras_bert import get_base_dict

TOKEN_PAD = ''  # Token for padding
TOKEN_UNK = '[UNK]'  # Token for unknown words
TOKEN_CLS = '[CLS]'  # Token for classification
TOKEN_SEP = '[SEP]'  # Token for separation
TOKEN_MASK = '[MASK]'  # Token for masking


def batch_inputs(sentence_pairs,
                     token_dict,
                     token_list,
                     seq_len=512):
#                      mask_rate=0.15,
#                      mask_mask_rate=0.8,
#                      mask_random_rate=0.1,
#                      swap_sentence_rate=0.5,
#                      force_mask=True):
    """Generate a batch of inputs and outputs for training.
    :param sentence_pairs: A list of pairs containing lists of tokens.
    :param token_dict: The dictionary containing special tokens.
    :param token_list: A list containing all tokens.
    :param seq_len: Length of the sequence.
    :param mask_rate: The rate of choosing a token for prediction.
    :param mask_mask_rate: The rate of replacing the token to `TOKEN_MASK`.
    :param mask_random_rate: The rate of replacing the token to a random word.
    :param swap_sentence_rate: The rate of swapping the second sentences.
    :param force_mask: At least one position will be masked.
    :return: All the inputs and outputs.
    """
    batch_size = len(sentence_pairs)
    base_dict = get_base_dict()
    unknown_index = token_dict[TOKEN_UNK]
    # Generate sentence swapping mapping
    nsp_outputs = np.zeros((batch_size,))
    mapping = {}
#     if swap_sentence_rate > 0.0:
#         indices = [index for index in range(batch_size) if np.random.random() < swap_sentence_rate]
#         mapped = indices[:]
#         np.random.shuffle(mapped)
#         for i in range(len(mapped)):
#             if indices[i] != mapped[i]:
#                 nsp_outputs[indices[i]] = 1.0
#         mapping = {indices[i]: mapped[i] for i in range(len(indices))}
    # Generate MLM
    token_inputs, segment_inputs, masked_inputs = [], [], []
    mlm_outputs = []
    for i in range(batch_size):
        first, second = sentence_pairs[i][0], sentence_pairs[mapping.get(i, i)][1]
        segment_inputs.append(([0] * (len(first) + 2) + [1] * (seq_len - (len(first) + 2)))[:seq_len])
        tokens = [TOKEN_CLS] + first + [TOKEN_SEP] + second + [TOKEN_SEP]
        tokens = tokens[:seq_len]
        tokens += [TOKEN_PAD] * (seq_len - len(tokens))
        
        token_input, masked_input, mlm_output = [], [], []
        has_mask = False
        
        for token in tokens:
            mlm_output.append(token_dict.get(token, unknown_index))
            
            if has_mask: # mask after 'SEP'
                masked_input.append(1)
                token_input.append(token_dict[TOKEN_MASK])
            
            else: 
                if token == token_dict['SEP']:
                    has_mask = True

                masked_input.append(0)    
                token_input.append(token_dict.get(token, unknown_index))    
                
                
#             if token not in base_dict and np.random.random() < mask_rate:
#                 has_mask = True
#                 masked_input.append(1)
#                 r = np.random.random()
#                 if r < mask_mask_rate:
#                     token_input.append(token_dict[TOKEN_MASK])
#                 elif r < mask_mask_rate + mask_random_rate:
#                     while True:
#                         token = np.random.choice(token_list)
#                         if token not in base_dict:
#                             token_input.append(token_dict[token])
#                             break
#                 else:
#                     token_input.append(token_dict.get(token, unknown_index))            
#             else:
#                 masked_input.append(0)
#                 token_input.append(token_dict.get(token, unknown_index))

#         if force_mask and not has_mask:
        if not has_mask: #
            masked_input[1] = 1
        token_inputs.append(token_input)
        masked_inputs.append(masked_input)
        mlm_outputs.append(mlm_output)
    inputs = [np.asarray(x) for x in [token_inputs, segment_inputs, masked_inputs]]
    outputs = [np.asarray(np.expand_dims(x, axis=-1)) for x in [mlm_outputs, nsp_outputs]]
    return inputs, outputs