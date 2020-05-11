
from textgenrnn import textgenrnn
from textgenrnn.model import textgenrnn_model
from textgenrnn.model_training import *
from textgenrnn.utils import *
import numpy as np
import pickle
from time import sleep
import os
import re
import glob

class Rnn:
    def __init__(self):
        self.textgen = textgenrnn(weights_path='second_iteration_weights.hdf5',
                             vocab_path='second_iteration_vocab.json',
                             config_path='second_iteration_config.json')

    def predict_letter_prob(self, prefix):
        if not prefix:
            max_len = 0
        else:
            max_len = (len(prefix) + 1)
        probs, indice_to_char = textgenrnn_prob_generate(self.textgen.model,
                                           self.textgen.vocab,
                                           self.textgen.indices_char,
                                           [0.5],
                                           self.textgen.config['max_length'],
                                           self.textgen.META_TOKEN,
                                           self.textgen.config['word_level'],
                                           self.textgen.config.get(
                                               'single_text', False),
                                           max_len,
                                           interactive=False,
                                           top_n=3,
                                           prefix=prefix)
        prob = {indice_to_char[i+1]: probs for i, probs in enumerate(probs.tolist()[1:])}
        return prob
