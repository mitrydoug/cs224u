import numpy as np
import pandas as pd

class RecommenderModel:

    def __init__(self):
        pass

    def fit(self, data):
        raise NotImplementedError

    def predict(self, users_products):
        raise NotImplementedError


class RandomModel(RecommenderModel):

    def fit(self, data):
        pass

    def predict(self, users_products):
        return pd.Series(np.random.choice(8, len(users_products)+1)/2.)
