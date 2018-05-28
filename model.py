import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import scipy    
import queue as Q


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
        if numerator == 0:
            return 0
        Pau = numerator / denom
        return Pau
     
    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))   

    def fit(self, data):
        self.user_product_ratings = data['user_product_ratings']

        # step 1: map from movie --> all users that rate that movie
        self.product_user_dict = {}
        for x in range(len(self.user_product_ratings)):
            currentid = self.user_product_ratings.iloc[x,0]
            currentvalue = self.user_product_ratings.iloc[x,2]
            self.product_user_dict.setdefault(currentid, [])
            self.product_user_dict[currentid].append(currentvalue)


        # step 2: for every user, collect mean rating
        self.mean_rating = self.user_product_ratings.groupby(['user_id'])['rating'].mean()


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


    def predict_user_product(self, user, product):
        users = self.product_user_dict[product]
        user_pqueue = Q.PriorityQueue()
        for u in users:
            user_pqueue.put((pearson_corr(user, u), u))
        
        sum_pearson = 0
        for i in range(5): #TODO: tune this hyperparameter
            sum_pearson += user_pqueue.get()

     
    def test_collab(self, user, product, upr): 
        users = mydict[product]
        user_pqueue = Q.PriorityQueue()
        for u in users:
            user_pqueue.put((pearson_corr(user, u, upr), u))
        user_avg = test[(test.user_id == user)]['rating'].mean()
        sum_pearson = 0
        weighted_score = 0
        for i in range(min(n, user_pqueue.qsize())): # TODO: tune this hyperparameter
            neighbor = user_pqueue.get()
            pearson = neighbor[0]
            neighbor_usr = neighbor[1]
            sum_pearson += pearson
            neighbor_rating = upr[(upr.user_id == neighbor_usr)][(upr.product_id == product)].get('rating')
            if not neighbor_rating.empty:
                weighted_score += (neighbor_rating.iloc[0] - user_avg) * (pearson)
        if sum_pearson == 0:
            new_score = user_avg
        else:
            new_score = user_avg + (weighted_score) / sum_pearson
        return new_score   

    def predict(self, users_products, item_item=True):
        results = []

        error = []
        accuracy = 0
        for index, row in user_products.iterrows():
            user_id = row['user_id']
            product_id = row['product_id']
            
            collab = test_collab(user_id, product_id, train)
            actual = user_products[(user_products.user_id == user_id)][(user_products.product_id == product_id)].get('rating').iloc[0]
            error.append(collab - actual)
            results.append(collab)
            if(np.round(collab) == np.round(actual)):
                accuracy += 1


        # Given user_id, product_id
        # Collect all users for a product rating
        # results = []
        # for index, row in users_products.iterrows():
        #     user_id = row['user_id']
        #     products_rated = self.user_product_ratings[self.user_product_ratings.user_id == user_id]
        #     product_id = row['product_id']
        #     products_rated['product_sim'] = products_rated.product_id.apply(lambda pid: self.cosine_sim(product_id, pid))
        #     sim_sum = products_rated.product_sim.sum()
        #     if sim_sum == 0:
        #         prediction = products_rated.rating.mean()
        #     else:
        #         prediction = (products_rated.product_sim * products_rated.rating).sum() / sim_sum
        #     results.append(prediction)
        return pd.Series(results, index=users_products.index)



