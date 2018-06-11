from queue import PriorityQueue
from collections import defaultdict
import re

import numpy as np
import pandas as pd
# from sklearn.neighbors import NearestNeighbors
# from scipy.sparse import csr_matrix
# import scipy
import queue as Q

import torch

from torch_data import *
import utils


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
        #self.user_product_ratings.user_id = self.user_product_ratings.user_id.astype('int')
        #self.user_product_ratings.product_id = self.user_product_ratings.product_id.astype('int')
        self.__user_product_ratings = self.user_product_ratings.set_index(['user_id', 'product_id'])

        self.__user_ratings = self.user_product_ratings.set_index('user_id')

        # step 1: map from movie --> all users that rate that movie
        self.product_user_dict = defaultdict(list)
        #self.__ratings = dict()
        for (user_id, product_id), _ in self.__user_product_ratings.iterrows():
            self.product_user_dict[product_id].append(user_id)
            #self.__ratings[(row.user_id, row.product_id)] = row.rating
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
            return self.user_mean_rating.loc[user]
        if not product_present and not user_present:
            return self.global_mean_rating

        user_pqueue = PriorityQueue()
        for u in self.product_user_dict[product]:
            user_pqueue.put((-self.pearson_corr(user, u), u))
        user_avg = self.user_mean_rating.loc[user]

        sum_pearson = 0
        weighted_score = 0
        for i in range(min(num_neighbors, user_pqueue.qsize())):
            neighbor = user_pqueue.get()
            pearson = -neighbor[0]
            neighbor_usr = neighbor[1]
            sum_pearson += pearson
            neighbor_rating = self.__user_product_ratings.loc[(neighbor_usr, product)].rating
            neighbor_mean = self.user_mean_rating.loc[neighbor_usr]
            weighted_score += (neighbor_rating - neighbor_mean) * pearson
        if sum_pearson == 0.:
            return 0.
        return user_avg + weighted_score / sum_pearson

    def predict(self, users_products, item_item=True):
        results = []
        i = 0
        N = len(users_products)
        for index, row in users_products.iterrows():
            if i % 1000 == 0:
                print(f'{i}/{len(users_products)}')
            user_id = row['user_id']
            product_id = row['product_id']
            
            collab = self.collab_user_product(user_id, product_id)
            results.append(collab)
            i += 1
        return pd.Series(results, index=users_products.index)


class RNNModel(RecommenderModel):

    def __init__(self, 
            desc_vocab_size=5000,
            revw_vocab_size=5000,
            desc_embed_size=100,
            revw_embed_size=100,
            desc_sem_size=20,
            revw_sem_size=20,
            train_epochs=10,
            train_batch_size=256,
            learning_rate=1e-3,
            param_l2_norm=15e-5,
            user_l2_norm=75e-3,
        ):
        RecommenderModel.__init__(self)
        self.desc_vocab_size = desc_vocab_size
        self.revw_vocab_size = revw_vocab_size
        self.desc_embed_size = desc_embed_size
        self.revw_embed_size = revw_embed_size
        self.desc_sem_size = desc_sem_size
        self.revw_sem_size = revw_sem_size
        self.train_epochs = train_epochs
        self.train_batch_size = train_batch_size
        self.lr = learning_rate
        self.param_l2_norm = param_l2_norm
        self.user_l2_norm = user_l2_norm

        # we add one for the bias term of a linear classifier
        self.user_embed_size = self.desc_sem_size + self.revw_sem_size + 1

    # def _init_vocab_embeddings(self):
    #     std = 1e-4
    #     self.desc_embed = torch.randn(self.desc_vocab_size, self.desc_embed_size, requires_grad=True) * std
    #     self.revw_embed = torch.randn(self.revw_vocab_size, self.revw_embed_size, requires_grad=True) * std

    def fit(self, train, val):

        dataset = UserProductRatingsDataset(train, self.desc_vocab_size, self.revw_vocab_size,
                transform=ReviewSampler())
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.train_batch_size,
                                                  shuffle=True, num_workers=16,
                                                  pin_memory=torch.cuda.is_available(),
                                                  collate_fn=CombineSequences(),
                                                  drop_last=False)

        num_users = int(train['user_product_ratings'].user_id.max()+1)
        model = NeuralModule(
            self.desc_embed_size, self.desc_vocab_size, self.desc_sem_size,
            self.revw_embed_size, self.revw_vocab_size, self.revw_sem_size,
            self.user_embed_size, num_users)
        if torch.cuda.is_available():
            model = model.cuda()

        utils.base_timer.start(f'using Adam optimizer with lr={self.lr}')
        optimizer = torch.optim.Adam(
            list(param for name, param in model.named_parameters() if 'embedding' not in name),
            weight_decay=self.param_l2_norm, lr=self.lr)
        utils.base_timer.start(f'using SparseAdam optimizer with lr={self.lr}')
        embed_optimizer = torch.optim.SparseAdam(
            list(param for name, param in model.named_parameters() if 'embedding' in name),
            lr=self.lr)
        utils.base_timer.stop()

        for epoch in range(self.train_epochs):
            print(f'epoch {epoch}')
            for i_batch, (user_ids, ratings, product_desc, product_desc_lens, product_desc_idxs,
                                             product_revw, product_revw_lens, product_revw_idxs) \
                    in enumerate(data_loader):

                if torch.cuda.is_available():
                    user_ids = user_ids.cuda(non_blocking=True)
                    ratings = ratings.cuda(non_blocking=True)
                    product_desc = product_desc.cuda(non_blocking=True)
                    product_desc_lens = product_desc_lens.cuda(non_blocking=True)
                    product_desc_idxs = product_desc_idxs.cuda(non_blocking=True)
                    product_revw = product_revw.cuda(non_blocking=True)
                    product_revw_lens = product_revw_lens.cuda(non_blocking=True)
                    product_revw_idxs = product_revw_idxs.cuda(non_blocking=True)

                preds, user_embeds = model(
                    user_ids, product_desc, product_desc_lens, product_desc_idxs,
                              product_revw, product_revw_lens, product_revw_idxs)
                loss = torch.nn.functional.mse_loss(preds, ratings, size_average=False)
                embeds_norm = (user_embeds * user_embeds).sum()
                loss += self.user_l2_norm * embeds_norm
                loss.backward()
                optimizer.step()
                embed_optimizer.step()
                optimizer.zero_grad()
                embed_optimizer.zero_grad()
                with torch.no_grad():
                    param_norm = 0
                    for name, param in model.named_parameters():
                        if 'embedding' not in name:
                            param_norm += (param * param).sum()
                print(f'train, batch = {i_batch:04}, loss = {loss:02.2f}, '
                      f'user_norm = {embeds_norm:02.2f}, param_norm = {param_norm:02.2f}')

    def predict(self, users_products):
        pass

class NeuralModule(torch.nn.Module):

    def __init__(self,
            desc_embed_size, desc_vocab_size, desc_sem_size,
            revw_embed_size, revw_vocab_size, revw_sem_size,
            user_embed_size, num_users
        ):
        super(NeuralModule, self).__init__()
        utils.base_timer.start('initializing description reader')
        self.desc_reader = LSTMReader(desc_embed_size, desc_vocab_size, desc_sem_size)

        utils.base_timer.start('initializing review reader')
        self.revw_reader = LSTMReader(revw_embed_size, revw_vocab_size, revw_sem_size)

        utils.base_timer.start('initializing user embeddings')
        self.user_embeddings = torch.nn.Embedding(num_users, user_embed_size, sparse=True)
        utils.base_timer.stop()

    def forward(self, user_ids, product_desc, product_desc_lens, product_desc_idxs,
                                product_revw, product_revw_lens, product_revw_idxs):

        product_desc_sem = self.desc_reader(product_desc, product_desc_lens).index_select(
                dim=0, index=product_desc_idxs)
        product_revw_sem = self.revw_reader(product_revw, product_revw_lens).index_select(
                dim=0, index=product_revw_idxs)
        product_sem = torch.cat((product_desc_sem, product_revw_sem,
                                 torch.ones(user_ids.shape[0], 1,
                                            device='cuda' if torch.cuda.is_available() else 'cpu')),
                                dim=1)
        user_embeddings = self.user_embeddings(user_ids)
        preds = (user_embeddings * product_sem).sum(dim=1)
        return preds, user_embeddings


class LSTMReader(torch.nn.Module):

    def __init__(self, embedding_dim, vocab_size, output_dim, hidden_dim=100, dropout=0.5):
        super(LSTMReader, self).__init__()
        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.semantic_transform = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.Sigmoid(),
        )
        self.num_directions = 1
        self.num_layers = 1
        self.h0 = torch.nn.Parameter(
                torch.zeros(self.num_directions * self.num_layers,
                            1, hidden_dim, requires_grad=True))
        self.c0 = torch.nn.Parameter(
            torch.zeros(self.num_directions * self.num_layers,
                        1, hidden_dim, requires_grad=True))

    def forward(self, sequences, lengths):
        """
        :param batch_sentences: token indices, have dimension (sequence_length)
        :return:
        """
        #print(f'batch_sentences: {sequences.shape}')
        embeds = self.word_embeddings(sequences)
        #print(f'embeds: {embeds.shape}')
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True)
        h0 = self.h0.repeat(1, sequences.shape[0], 1)
        c0 = self.c0.repeat(1, sequences.shape[0], 1)
        lstm_out, _ = self.lstm(packed, (h0, c0))
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True,
                                                             padding_value=1e-8)
        mask, _ = torch.gt(lstm_out, 1e-7).max(dim=2, keepdim=True)
        #print(f'lstm_out: {lstm_out.shape}')
        sem_out = self.semantic_transform(lstm_out) * mask.float()
        #print(f'sem_out: {sem_out.shape}')
        sem_max, _ = torch.max(sem_out, dim=1)
        #print(f'sem_max: {sem_max.shape}')
        return sem_max

class ClusteringModel(RecommenderModel):
    def __init__(self, **kwargs):
        RecommenderModel.__init__(self, **kwargs)
        self.num_clusters = 100
        self.user_pow = 1.0
        self.cluster_pow = 0.5

    def k_means(self):
        counts = self.user_product_ratings.groupby('user_id')['rating'].count()
        # counts = counts[counts.rating > ]

    def fit(self, data):

        def apply_mean_std(Df):
            Df['rating_mean'] = Df.rating.mean()
            Df['rating_std'] = np.std(list(Df.rating) + [self.global_mean])
            return Df

        self.user_product_ratings = data['user_product_ratings'].copy()
        self.global_mean = self.user_product_ratings.rating.mean()
        self.user_mean = self.user_product_ratings.groupby('user_id')['rating'].mean()
        self.user_std = self.user_product_ratings.groupby('user_id')['rating'].apply(
            lambda S: np.std(list(S) + [self.global_mean])
        )
        self.user_product_ratings = self.user_product_ratings.groupby('user_id').apply(apply_mean_std)
        self.user_product_ratings.rating = (
                (self.user_product_ratings.rating - self.user_product_ratings.rating_mean) /
                 self.user_product_ratingss.rating_std)
        user_clusters = self.k_means()
