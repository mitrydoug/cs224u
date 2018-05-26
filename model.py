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


    def fit(self, data):
        self.user_product_ratings = data['user_product_ratings']


        # Old implementation: item item cf with nearest neighbors
        # def subtract_mean(Df):
        #     Df.rating -= Df.rating.mean()
        #     return Df
        # self.user_product_ratings = self.user_product_ratings.groupby('user_id').apply(subtract_mean)
        # rating_counts = self.user_product_ratings.groupby('user_id', as_index=False).count()
        # self.sampled_users = list(rating_counts[rating_counts.rating > 1].user_id.sample(n=self.num_sampled_users))
        # users_map = dict(zip(self.sampled_users, np.arange(self.num_sampled_users)))
        # sparse_table = self.user_product_ratings[self.user_product_ratings.user_id.isin(self.sampled_users)]
        # mapped_user_id_list = [users_map[k] for k in list(sparse_table.user_id)]
        # self.sampled_user_product_matrix = csr_matrix(
        #     (sparse_table.rating, (sparse_table.product_id, mapped_user_id_list))).todense()
        # self.neighbors = NearestNeighbors(n_neighbors=self.num_neighbors, metric='cosine').fit(self.sampled_user_product_matrix) 


    def predict(self, users_products):
        # user_id, product_id: predict rating for each row
        # self.neighbors.kneighbors()

        # Given user_id, product_id
        # Collect all users for a product rating
        results = []
        for index, row in users_products.iterrows():
            user_id = row['user_id']
            products_rated = self.user_product_ratings[self.user_product_ratings.user_id == user_id]
            product_id = row['product_id']
            products_rated['product_sim'] = products_rated.product_id.apply(lambda pid: self.cosine_sim(product_id, pid))
            sim_sum = products_rated.product_sim.sum()
            if sim_sum == 0:
                prediction = products_rated.rating.mean()
            else:
                prediction = (products_rated.product_sim * products_rated.rating).sum() / sim_sum
            results.append(prediction)
        return pd.Series(results, index=users_products.index)

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

