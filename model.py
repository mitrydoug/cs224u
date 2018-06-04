from queue import PriorityQueue
from collections import defaultdict
import re
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import scipy    
import queue as Q
from collections import defaultdict

import torch

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
=======
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
        return new_score   

    def predict(self, users_products, item_item=True):
        results = []
        for index, row in users_products.iterrows():
            user_id = row['user_id']
            product_id = row['product_id']
            
            collab = self.collab_user_product(user_id, product_id)
            results.append(collab)
        return pd.Series(results, index=users_products.index)


class RNNModel(RecommenderModel):

    def __init__(self, 
            desc_vocab_size=5000,
            revw_vocab_size=5000,
            desc_embed_size=100,
            revw_embed_size=100,
            desc_sem_size=20,
            revw_sem_size=20,
            max_read_revw=10,
            train_epochs=10,
            train_batch_size=32,
            learning_rate=1e-2,
            init_std=1e-3
        ):
        RecommenderModel.__init__(self)
        self.desc_vocab_size = desc_vocab_size
        self.revw_vocab_size = revw_vocab_size
        self.desc_embed_size = desc_embed_size
        self.revw_embed_size = revw_embed_size
        self.desc_sem_size = desc_sem_size
        self.revw_sem_size = revw_sem_size
        self.max_read_revw = max_read_revw
        self.train_epochs = train_epochs
        self.train_batch_size = train_batch_size
        self.lr = learning_rate
        self.init_std = init_std

        # we add one for the bias term of a linear classifier
        self.user_embed_size = self.desc_sem_size + self.revw_sem_size + 1

    def _clean_descs_and_reviews(self):
        
        def clean_text(text):
            text = text.lower()
            text = re.sub('[^a-z0-9., ]', '', text)
            text = re.sub('\.', ' . ', text)
            text = re.sub(',', ' , ', text)
            text = ' ' + text + ' '
            return text

        self.product_descriptions.description = (
                self.product_descriptions.description.apply(clean_text))
        self.product_reviews.review = (
                self.product_reviews.review.apply(clean_text))

    def _vocab_for_series(self, texts, top_n, special_tokens):
        counts = defaultdict(int)
        for text in texts:
            for match in re.findall(r' *([a-z0-9.,]+) +', text):
                counts[match] += 1
        vocab = (
            list(map(lambda t: t[1],
                sorted(((counts[w], w) for w in counts),
                       reverse=True)[:top_n]))
            + special_tokens)
        vocab_to_idx = {w: i for i, w in enumerate(vocab)}
        idx_to_vocab = {i: w for i, w in enumerate(vocab)}
        return vocab_to_idx, idx_to_vocab

    def _build_vocab(self):
        (self.desc_vocab_to_idx,
         self.desc_idx_to_vocab) = self._vocab_for_series(
                self.product_descriptions.description,
                self.desc_vocab_size, ['<UNK>'])

        (self.revw_vocab_to_idx,
         self.revw_idx_to_vocab) = self._vocab_for_series(
                self.product_reviews.review,
                self.revw_vocab_size, ['<UNK>'])

    def _build_desc_revw_sequences(self):
        self.product_to_desc_seq = defaultdict(list)
        for _, row in self.product_descriptions.iterrows():
            pid = row['product_id']
            desc = []
            for match in re.findall(r' *([a-z0-9.,]+) +', row['description']):
                if match in self.desc_vocab_to_idx:
                    desc.append(self.desc_vocab_to_idx[match])
                else:
                    desc.append(self.desc_vocab_to_idx['<UNK>'])
            self.product_to_desc_seq[pid] = torch.tensor(desc, dtype=torch.long)
        self.product_to_revw_seq = defaultdict(list)
        for _, row in self.product_reviews.iterrows():
            pid = row['product_id']
            revw = []
            for match in re.findall(r' *([a-z0-9.,]+) +', row['review']):
                if match in self.revw_vocab_to_idx:
                    revw.append(self.revw_vocab_to_idx[match])
                else:
                    revw.append(self.revw_vocab_to_idx['<UNK>'])
            self.product_to_revw_seq[pid].append(torch.tensor(revw, dtype=torch.long))

    # def _init_vocab_embeddings(self):
    #     std = 1e-4
    #     self.desc_embed = torch.randn(self.desc_vocab_size, self.desc_embed_size, requires_grad=True) * std
    #     self.revw_embed = torch.randn(self.revw_vocab_size, self.revw_embed_size, requires_grad=True) * std

    def fit(self, data):

        timer = utils.TaskTimer()

        # store data
        timer.start('copying required data')
        self.raw_product_descriptions = data['product_descriptions']
        self.raw_product_reviews = data['product_reviews']
        self.user_product_ratings = data['user_product_ratings']
        self.product_descriptions = data['product_descriptions'].copy()
        self.product_reviews = data['product_reviews'].copy()

        timer.start('cleaning product descriptions and reviews')
        self._clean_descs_and_reviews()
        timer.start('building product descriptions and review vocabs')
        self._build_vocab()
        timer.start('building product description and review index sequence')
        self._build_desc_revw_sequences()

        timer.start('initializing description reader')
        self.desc_reader = LSTMReader(self.desc_embed_size,
                                      len(self.desc_vocab_to_idx),
                                      self.desc_sem_size)

        timer.start('initializing review reader')
        self.revw_reader = LSTMReader(self.revw_embed_size,
                                      len(self.revw_vocab_to_idx),
                                      self.revw_sem_size)

        timer.start('initializing user embeddings')
        num_users = int(self.user_product_ratings.user_id.max())+1
        self.user_embeds = torch.nn.Embedding(num_users, self.user_embed_size, sparse=True)

        timer.start(f'using Adam optimizer with lr={self.lr}')
        optimizer = torch.optim.Adam(
            list(param for name, param in self.desc_reader.named_parameters() if 'word_embeddings' not in name) +
            list(param for name, param in self.revw_reader.named_parameters() if 'word_embeddings' not in name),
            lr=self.lr)
        timer.start(f'using SparseAdam optimizer with lr={self.lr}')
        embed_optimizer = torch.optim.SparseAdam(
            list(self.desc_reader.word_embeddings.parameters()) +
            list(self.revw_reader.word_embeddings.parameters()) +
            list(self.user_embeds.parameters()),
            lr=self.lr)
        timer.stop()

        for epoch in range(self.train_epochs):
            self.user_product_ratings = self.user_product_ratings.sample(frac=1.0)
            print(f'beginning epoch: {epoch}')
            for idx in range(0, len(self.user_product_ratings), self.train_batch_size):
                batch = self.user_product_ratings.iloc[idx:idx+self.train_batch_size]
                #with torch.autograd.profiler.profile() as prof:
                loss = 0
                # print('forward')
                for _, row in batch.iterrows():
                    # print(len(self.product_to_desc_seq[row.product_id]))
                    # print(len(self.product_to_revw_seq[row.product_id]))
                    desc_seq = self.product_to_desc_seq[row.product_id]
                    #print(f'desc_seq: {desc_seq.shape}')
                    desc_sem = self.desc_reader(desc_seq)
                    # print(f'desc_sem: {desc_sem.shape}')
                    revw_idxs = np.random.choice(len(self.product_to_revw_seq[row.product_id]),
                            size=min(self.max_read_revw, len(self.product_to_revw_seq[row.product_id])),
                            replace=False)
                    revw_sems = torch.stack(
                            tuple(self.revw_reader(self.product_to_revw_seq[row.product_id][idx])
                                  for idx in revw_idxs), dim=1)
                    # print(f'revw_sems: {revw_sems.shape}')
                    revw_sem, _ = torch.max(revw_sems, dim=1)
                    # print(f'revw_sem: {revw_sem.shape}')

                    sem_total = torch.cat((desc_sem, revw_sem, torch.ones(1)))

                    user_embed = self.user_embeds(torch.tensor(int(row.user_id), dtype=torch.long))
                    # print(f'user_embed: {user_embed}')
                    pred = (sem_total * user_embed).sum()
                    # print(f'pred: {pred}, rating: {row.rating}')
                    loss += (pred - row.rating) ** 2. / self.train_batch_size
                print(f'batch loss: {loss}')
                loss.backward()
                # print('step')
                optimizer.step()
                embed_optimizer.step()
                optimizer.zero_grad()
                embed_optimizer.zero_grad()

                #print(prof.key_averages())

    def predict(self, users_products):
        pass


class SemanticTransform(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(SemanticTransform, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, xs):
        return self.linear(xs)


class LSTMReader(torch.nn.Module):

    def __init__(self, embedding_dim, vocab_size, output_dim, hidden_dim=100):
        super(LSTMReader, self).__init__()
        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)
        self.semantic_transform = SemanticTransform(hidden_dim, output_dim)
        num_directions = 1
        num_layers = 1
        self.h0 = torch.nn.Parameter(
                torch.zeros(num_directions * num_layers, 1, hidden_dim,
                            requires_grad=True))
        self.c0 = torch.nn.Parameter(
            torch.zeros(num_directions * num_layers, 1, hidden_dim,
                        requires_grad=True))

    def forward(self, sentence):
        """
        :param batch_sentences: token indices, have dimension (sequence_length)
        :return:
        """
        # print(f'batch_sentences: {sentence.shape}')
        embeds = self.word_embeddings(sentence)
        # print(f'embeds: {embeds.shape}')
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1), (self.h0, self.c0))
        # print(f'lstm_out: {lstm_out.shape}')
        sem_out = self.semantic_transform(lstm_out.view(len(sentence), -1))
        # print(f'sem_out: {sem_out.shape}')
        sem_max, _ = torch.max(sem_out, dim=0)
        # print(f'sem_max: {sem_max.shape}')
        return sem_max
