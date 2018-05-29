import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import scipy    
import queue as Q
from collections import defaultdict


class RecommenderModel:

    def __init__(self):
        pass

    def fit(self, data):
        raise NotImplementedError

    def predict(self, users_products):
        raise NotImplementedError


class RandomModel(RecommenderModel):

    def fit(self, data):
        self.ratings = list(set(data['user_product_ratings'].rating))

    def predict(self, users_products):
        return pd.Series(
                np.random.choice(self.ratings, len(users_products)),
                index = users_products.index)


class ItemItemCollaborationModel(RecommenderModel):

    def __init__(self):
        RecommenderModel.__init__(self)
        self.num_neighbors = 3
        self.num_sampled_users = 10

    def cosine_sim(self, product_i, product_j):
        product_i_ratings = self.user_product_ratings[self.user_product_ratings.product_id == product_i]
        product_j_ratings = self.user_product_ratings[self.user_product_ratings.product_id == product_j]
        
        product_i_ratings = product_i_ratings[['user_id', 'rating']]
        product_j_ratings = product_j_ratings[['user_id', 'rating']]

        product_i_ratings.rating -= product_i_ratings.rating.mean()
        product_j_ratings.rating -= product_j_ratings.rating.mean()

        cos_num = (product_i_ratings.rating * product_j_ratings.rating).fillna(0).sum()
        cos_denom = np.sqrt((product_i_ratings.rating * product_i_ratings.rating).sum()) * np.sqrt((product_j_ratings.rating * product_j_ratings.rating).sum())
        return cos_num / float(cos_denom + 1e-9)

    def pearson_corr(self, u1, u2):
        a = self.user_product_ratings[(self.user_product_ratings.user_id == u1)]
        u = self.user_product_ratings[(self.user_product_ratings.user_id == u2)]
        
        s1 = pd.merge(a, u, how='inner', on=['product_id'])
        ra_bar = a['rating'].mean()
        ru_bar = u['rating'].mean()
        numerator = np.sum( (s1['rating_x'] - ra_bar) *  (s1['rating_y'] - ru_bar))
        denom = np.sqrt(np.sum((s1['rating_x'] - ra_bar)**2) * np.sum((s1['rating_y'] - ru_bar)**2))
        if denom == 0:
            return 0
        Pau = numerator / denom
        return Pau
     
    def fit(self, data):
        self.user_product_ratings = data['user_product_ratings']

        # step 1: map from movie --> all users that rate that movie
        self.product_user_dict = defaultdict(list)
        for index, row in self.user_product_ratings.iterrows():
            user_id = row['user_id']
            product_id = row['product_id']
            self.product_user_dict[product_id].append(user_id)

        # step 2: for every user, collect mean rating
        self.user_mean_rating = self.user_product_ratings.groupby(['user_id'])['rating'].mean()
        self.product_mean_rating = self.user_product_ratings.groupby(['product_id'])['rating'].mean()
        self.global_mean_rating = self.user_product_ratings['rating'].mean()
     
    def collab_user_product(self, user, product, num_neighbors=5): 
        user_present = user in self.user_mean_rating.index
        product_present = product in self.product_mean_rating.index
        if not user_present and product_present:
            return self.product_mean_rating[product]
        if not product_present and user_present:
            return self.user_mean_rating[user]
        if not user_present and not product_present:
            return self.global_mean_rating

        users = self.product_user_dict[product]
        user_pqueue = Q.PriorityQueue()
        for u in users:
            user_pqueue.put((self.pearson_corr(user, u), u))
        user_avg = self.user_mean_rating[user]
        
        sum_pearson = 0
        weighted_score = 0
        for i in range(min(num_neighbors, user_pqueue.qsize())): # TODO: tune this hyperparameter
            neighbor = user_pqueue.get()
            pearson = neighbor[0]
            neighbor_usr = neighbor[1]
            sum_pearson += pearson
            neighbor_rating = (self.user_product_ratings[(self.user_product_ratings.user_id == neighbor_usr)]
                                                        [(self.user_product_ratings.product_id == product)].get('rating'))
            weighted_score += (neighbor_rating.iloc[0] - self.user_mean_rating[neighbor_usr]) * (pearson)
        if sum_pearson == 0:
            new_score = user_avg
        else:
            new_score = user_avg + (weighted_score) / sum_pearson
        return new_score   

    def predict(self, users_products, item_item=True):
        results = []
        for index, row in users_products.iterrows():
            user_id = row['user_id']
            product_id = row['product_id']
            
            collab = self.collab_user_product(user_id, product_id)
            results.append(collab)
        return pd.Series(results, index=users_products.index)

