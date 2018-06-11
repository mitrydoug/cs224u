import numpy as np 
import pandas as pd

from data import RandomData, AmazonBooks, ToyData, MovieLensData
from clustering import ClusteringModel

ds = MovieLensData(min_user_ratings=5).get_dataset(verbose=True)
train = ds['train']
val = ds['val']

c_model = ClusteringModel(num_iterations=70)