from queue import PriorityQueue
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

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

class SimpleMeanModel(RecommenderModel):

    def __init__(self, **kwargs):
        RecommenderModel.__init__(self, **kwargs)

    def fit(self, data):
        self.global_mean = data['user_product_ratings'].rating.mean()

    def predict(self, users_products):
        return pd.Series(self.global_mean, index=users_products.index)

class UserMeanModel(RecommenderModel):

    def __init__(self, **kwargs):
        RecommenderModel.__init__(self, **kwargs)

    def fit(self, data):
        upr = data['user_product_ratings']
        self.user_params = (upr.groupby('user_id')[['rating']].mean()
                               .rename(columns={'rating': 'mean_rating'}))
        self.global_mean = upr.rating.mean()

    def predict(self, users_products):
        up = pd.merge(users_products, self.user_params, how='left',
                      left_on='user_id', right_index=True)
        return up.mean_rating.fillna(self.global_mean)

class ProductMeanModel(RecommenderModel):

    def __init__(self, **kwargs):
        RecommenderModel.__init__(self, **kwargs)

    def fit(self, data):
        upr = data['user_product_ratings']
        self.product_params = (upr.groupby('product_id')[['rating']].mean()
                                  .rename(columns={'rating': 'mean_rating'}))
        self.global_mean = upr.rating.mean()

    def predict(self, users_products):
        up = pd.merge(users_products, self.product_params, how='left',
                      left_on='product_id', right_index=True)
        return up.mean_rating.fillna(self.global_mean)

class CombinedMeanModel(RecommenderModel):

    def __init__(self, **kwargs):
        RecommenderModel.__init__(self, **kwargs)

    def fit(self, data):
        upr = data['user_product_ratings']
        self.global_mean = upr.rating.mean()
        self.global_std = upr.rating.std()
        self.user_params = (upr.groupby('user_id')['rating']
                               .agg(['mean', 'std']))
        upr = pd.merge(upr, self.user_params, how='left',
                       left_on='user_id', right_index=True)
        upr['normed'] = ((upr.rating - upr['mean']) / 
                         upr['std'].where(upr['std'] > 0., 1.))
        self.film_params = upr.groupby('product_id')[['normed']].mean()

    def predict(self, users_products):
        up = users_products.copy()
        up = pd.merge(up, self.user_params, how='left',
                      left_on='user_id', right_index=True)
        up = pd.merge(up, self.film_params, how='left',
                      left_on='product_id', right_index=True)
        up['mean'].fillna(self.global_mean, inplace=True)
        up['std'].fillna(self.global_std, inplace=True)
        up['normed'].fillna(0., inplace=True)
        return up['mean'] + up['normed'] * up['std']

class ItemItemCollaborationModel(RecommenderModel):

    def __init__(self):
        RecommenderModel.__init__(self)
        self.num_neighbors = 3
        self.num_sampled_users = 10

    def pearson_corr(self, u1, u2):
        return 1
        if u1 == u2:
            return 1
        a = self.__user_ratings.loc[[u1]] # [(self.user_product_ratings.user_id == u1)]
        u = self.__user_ratings.loc[[u2]] # [(self.user_product_ratings.user_id == u2)]

        s1 = pd.merge(a, u, how='inner', on=['product_id'])
        #print(s1)
        ra_bar = self.user_mean_rating.loc[u1]
        ru_bar = self.user_mean_rating.loc[u2]
        numerator = ((s1.rating_x - ra_bar) *  (s1.rating_y - ru_bar)).sum()
        denom = np.sqrt(((s1.rating_x - ra_bar)**2).sum() * ((s1.rating_y - ru_bar)**2).sum())
        if denom == 0:
            return 0
        Pau = numerator / denom
        return Pau

    def fit(self, data):
        self.user_product_ratings = data['user_product_ratings'].drop_duplicates(subset=['user_id','product_id'])
        self.__user_product_ratings = self.user_product_ratings.set_index(['user_id', 'product_id'])

        self.__user_ratings = self.user_product_ratings.set_index('user_id')

        # step 1: map from movie --> all users that rate that movie
        self.product_user_dict = defaultdict(list)
        for (user_id, product_id), _ in self.__user_product_ratings.iterrows():
            self.product_user_dict[product_id].append(user_id)

        # step 1: for every user, collect mean rating
        self.user_mean_rating = self.user_product_ratings.groupby(['user_id'])['rating'].mean()
        self.product_mean_rating = self.user_product_ratings.groupby(['product_id'])['rating'].mean()
        self.global_mean_rating = self.user_product_ratings['rating'].mean()

    def collab_user_product(self, user, product, num_neighbors=5):
        user_present = user in self.user_mean_rating.index
        product_present = product in self.product_mean_rating.index
        if not user_present and product_present:
            return self.product_mean_rating.loc[product]
        if not product_present and user_present:
            return self.user_mean_ratingi.loc[user]
        if not user_present and not product_present:
            return self.global_mean_rating

        users = self.product_user_dict[product]
        user_pqueue = PriorityQueue()
        for u in users:
            user_pqueue.put((-self.pearson_corr(user, u), u))
        user_avg = self.user_mean_rating.loc[user]

        sum_pearson = 0
        weighted_score = 0
        for i in range(min(num_neighbors, user_pqueue.qsize())):
            neighbor = user_pqueue.get()
            pearson = -neighbor[0]
            neighbor_usr = neighbor[1]
            sum_pearson += pearson
            neighbor_rating = self.__user_product_ratings.loc[(neighbor_usr, product)]
            neighbor_mean = self.user_mean_rating.loc[neighbor_usr]
            #print(pearson)
            #print(neighbor_mean)
            #print(neighbor_rating)
            weighted_score += (neighbor_rating - neighbor_mean) * pearson
        if sum_pearson == 0:
            new_score = user_avg
        else:
            new_score = user_avg + (weighted_score) / sum_pearson
        return new_score

    def predict(self, users_products, item_item=True):
        print(len(users_products))
        results = []
        i = 0
        for _, row in users_products.iterrows():
            if i % 100 == 0:
                print(i)
            i += 1
            user_id = row['user_id']
            product_id = row['product_id']
            collab = self.collab_user_product(user_id, product_id)
            results.append(collab)
        return pd.Series(results, index=users_products.index)
