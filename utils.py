import csv
import logging
import os
import time

import numpy as np
import pandas as pd
import torch

class TaskTimer:
    def __init__(self):
        self.active=False
        self.msg=None
        self.start_millis = 0

    def start(self, msg, suffix=' ... '):
        if self.active:
            self.stop()
        self.active = True
        if msg is not None:
            print(msg + suffix, end='')
        self.msg = msg
        self.start_millis = time.time()*1000.0

    def stop(self):
        if not self.active:
            raise Exception(f'No Task in progress')
        ellapsed_millis = time.time()*1000.0 - self.start_millis
        self.active = False
        print(f'done! ({ellapsed_millis:.1f} ms)')
base_timer = TaskTimer()

def mean_squared_error(pred, ground_truth):
    return np.mean((pred - ground_truth) ** 2.)

def accuracy(pred, ground_truth):
    return (np.round(pred) == ground_truth).sum() / float(len(pred))

def get_glove_embeddings(idx_to_vocab, dim=100):
    glove_fn = f'data/glove.6B.{dim}d.txt'
    N = len(idx_to_vocab)
    words = dict()
    with open(glove_fn, 'r') as f:
        for line in f:
            tokens = line.strip().split(' ')
            words[tokens[0]] = np.array(list(map(float, tokens[1:])))
    embeds = []
    num_random = 0
    for i in range(N):
        if idx_to_vocab[i] in words:
            embeds.append(words[idx_to_vocab[i]])
        else:
            num_random += 1
            embeds.append(np.random.normal(0, 0.7, dim))
    embeds = np.array(embeds)
    return embeds, num_random
