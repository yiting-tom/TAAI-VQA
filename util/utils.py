import os
import time
import random

import numpy as np
import torch


def get_vocab_list(vocab_path):
    with open(vocab_path, encoding='utf-8') as f:
        vocab_list = f.read().split('\n')
    return vocab_list

def random_seed(seed=10):
    """
    set random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if using multi-GPU
    torch.backends.cudnn.deterministic = True

def set_device():
    """
    set device as 'cuda' if available, otherwise 'cpu'
    """
    # Use cuda if available
    if torch.cuda.is_available():
        device = 'cuda'
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    else:
        device = 'cpu'
    return device

class Logger():
    def __init__(self, exp_name, log_name='log.txt'):
        save_path = os.path.join('checkpoint', exp_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        t = time.strftime("%y%m%d-%H-%M-%S_", time.localtime()) # timestamp
        self.log_file = open(os.path.join(save_path, t + log_name), 'w+')
        self.exp_name = exp_name

    def write(self, msg):
        self.log_file.write(time.strftime("%y%m%d-%H:%M:%S ", time.localtime())) # timestamp
        self.log_file.write(msg+'\n')
        self.log_file.flush()

    def show(self, msg):
        print(msg)
        self.write(msg)


def get_tokens(sentence, vocab_list, is_cap=False):
    sentence = sentence.lower()
    for c in [' \'', '\' ', ' \"', '\" ', '\n']:
        sentence = sentence.replace(c, ' ')
    for c in '.,?':
        sentence = sentence.replace(c, '')
    sentence = sentence.replace('\'s', ' \'s')
    words = [i for i in sentence.split() if len(i) > 0]
    
    if is_cap:
        words.insert(0, '<start>')
        words.append('<end>')

    tokens = []
    for word in words:
        if word in vocab_list: token = vocab_list.index(word)
        else: token = vocab_list.index('<oov>')
        tokens.append(token)
    return tokens, words


def padding(tokens, max_l, vocab_list):
        l = min(len(tokens), max_l)
        if l < max_l:
            tokens.extend([vocab_list.index('<pad>')] * (max_l - l))
        else:
            tokens = tokens[:l]
        return tokens, l