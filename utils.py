import csv
import os
import time

import pandas as pd

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

def load_glove(glove_data_file):
    words = pd.read_table(glove_data_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
    return words

def convert_word_to_vec(model, w, embed_size, verbose=True):
    if w in model.index:
        if verbose:
            print("found")
        return model.loc[w].as_matrix()
    else:
        if verbose:
            print("not found")
        return np.random.normal(0, 0.7, embed_size)

def create_vsm(corpus, embed_size=50, glove_file_path='glove.6B'):
    vsm_list = []
    vocab = set(corpus)
    assert(embed_size == 50 or embed_size==100)
    glove_file = os.path.join(glove_file_path, 'glove.6B.%dd.txt' % embed_size)
    model = None
    model = load_glove(glove_file)
    if model is not None:
        for word in vocab:
            word_as_vec = convert_word_to_vec(model, word, embed_size)
            vsm_list.append(word_as_vec)
    return np.array(vsm_list)
