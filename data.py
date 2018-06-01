import hashlib
import itertools

import numpy as np
import pandas as pd
import json
import os
import re

class DataSource:

    def __init__(self, min_user_ratings=5,
                       max_user_ratings=200,
                       min_product_reviews=1,
                       require_product_description=True,
                       min_desc_len=10,
                       max_desc_len=1000,
                       min_rev_len=3,
                       max_rev_len=1000,
                       rnd_state=None):
        # 
        #       Products
        #      -----------
        #     |       |   |
        #   U | cell  |   |
        #   s | hold- |   |
        #   e |  out  |   |
        #   r |       |   |
        #   s |-------|---|
        #     |       |   | <- user_holdout
        #      -----------
        #               ^ product_holdout
        #              
        ### SENSITIVE PARAMETERS, DO NOT CHANGE ###
        self.test_cell_holdout = 0.1
        self.test_user_holdout = 0.1
        self.test_product_holdout = 0.1
        self.val_cell_holdout = 0.1
        self.val_user_holdout = 0.1
        self.val_product_holdout = 0.1
        ###########################################
        self.min_user_ratings = min_user_ratings
        self.max_user_ratings = max_user_ratings
        self.min_product_reviews = min_product_reviews
        self.require_product_description = require_product_description
        self.min_desc_len = min_desc_len
        self.max_desc_len = max_desc_len
        self.min_rev_len = min_rev_len
        self.max_rev_len = max_rev_len
        self.rnd_state = rnd_state
        self.data_name = (f'{self.__class__.__name__}'
                          f'__mn_ur.{self.min_user_ratings}'
                          f'__mx_ur.{self.max_user_ratings}'
                          f'__mn_pv.{self.min_product_reviews}'
                          f'__require_pd.{self.require_product_description}')

    def get_test(self, load_cache=True, save_cache=True):
        return self.get_dataset(load_cache=load_cache, save_cache=save_cache)['test']

    def get_val(self, load_cache=True, save_cache=True):
        return self.get_dataset(load_cache=load_cache, save_cache=save_cache)['val']

    def get_train(self, load_cache=True, save_cache=True):
        return self.get_dataset(load_cache=load_cache, save_cache=save_cache)['train']

    @staticmethod
    def bundle_dataset(test_up_rat,  test_product_desc,  test_product_rev,
                        val_up_rat,   val_product_desc,   val_product_rev,
                      train_up_rat, train_product_desc, train_product_rev):
        return {
                'test': {'user_product_ratings': test_up_rat,
                         'product_descriptions': test_product_desc,
                         'product_reviews': test_product_rev},
                'val': {'user_product_ratings': val_up_rat,
                        'product_descriptions': val_product_desc,
                        'product_reviews': val_product_rev},
                'train': {'user_product_ratings': train_up_rat,
                          'product_descriptions': train_product_desc,
                          'product_reviews': train_product_rev}
        }

    @staticmethod
    def hash_fn(row):
        # return 8 bytes of row hash as integer
        hd = hashlib.sha256(bytes(str(tuple(row)), 'utf8')).hexdigest()
        return int(hd[:16], 16) / float(2 ** 64)

    def get_dataset(self, load_cache=True, save_cache=True, verbose=False):

        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'preprocessed_datasets', self.data_name)

        test_up_rat_path = os.path.join(data_path, 'test.user_product_ratings.feather')
        test_product_desc_path = os.path.join(data_path, 'test.product_descriptions.feather')
        test_product_rev_path = os.path.join(data_path, 'test.product_reviews.feather')
        val_up_rat_path = os.path.join(data_path, 'val.user_product_ratings.feather')
        val_product_desc_path = os.path.join(data_path, 'val.product_descriptions.feather')
        val_product_rev_path = os.path.join(data_path, 'val.product_reviews.feather')
        train_up_rat_path = os.path.join(data_path, 'train.user_product_ratings.feather')
        train_product_desc_path = os.path.join(data_path, 'train.product_descriptions.feather')
        train_product_rev_path = os.path.join(data_path, 'train.product_reviews.feather')

        if load_cache and os.path.isdir(data_path):
            if verbose:
                print('loading preprocessed dataset from disk')
            test_up_rat = pd.read_feather(test_up_rat_path)
            test_product_desc = pd.read_feather(test_product_desc_path)
            test_product_rev = pd.read_feather(test_product_rev_path)
            val_up_rat = pd.read_feather(val_up_rat_path)
            val_product_desc = pd.read_feather(val_product_desc_path)
            val_product_rev = pd.read_feather(val_product_rev_path)
            train_up_rat = pd.read_feather(train_up_rat_path)
            train_product_desc = pd.read_feather(train_product_desc_path)
            train_product_rev = pd.read_feather(train_product_rev_path)
            return DataSource.bundle_dataset(test_up_rat,  test_product_desc,  test_product_rev,
                                              val_up_rat,   val_product_desc,   val_product_rev,
                                            train_up_rat, train_product_desc, train_product_rev)

        # load data from sub-class implementations
        if verbose:
            print('loading raw data')
        up_rat = self._raw_user_product_ratings()
        product_desc = self._raw_product_descriptions()
        product_rev = self._raw_product_reviews()

        assert set(up_rat.columns) == {'user_id', 'product_id', 'rating'}
        assert set(product_desc.columns) == {'product_id', 'description'}
        assert set(product_rev.columns) == {'product_id', 'review'}

        
        # a bit of cleaning of description/review string lengths
        if verbose:
            print('cleaning review lenghts')
        pd_lens = product_desc.description.apply(len)
        product_desc = product_desc[(pd_lens >= self.min_desc_len) &
                                    (pd_lens <= self.max_desc_len)]
        rv_lens = product_rev.review.apply(len)
        product_rev = product_rev[(rv_lens >= self.min_rev_len) &
                                  (rv_lens <= self.max_rev_len)]

        # clean out products with no description
        if self.require_product_description:
            if verbose:
                print('removing products with no description')
            pids = set(product_desc.product_id)
            up_rat = up_rat[up_rat.product_id.isin(pids)]
            product_rev = product_rev[product_rev.product_id.isin(pids)]

        # clean out products with too few reviews
        if verbose:
            print('removing poducts with too few reviews')
        rev_count = (product_rev.groupby('product_id', as_index=False)['review']
                .count().rename(columns={'review': 'review_count'}))
        arr = rev_count.review_count >= self.min_product_reviews
        good_products = set(rev_count[arr].product_id)
        product_desc = product_desc[product_desc.product_id.isin(good_products)]
        up_rat = up_rat[up_rat.product_id.isin(good_products)]
        product_rev = product_rev[product_rev.product_id.isin(good_products)]

        # clean out users with too few ratings
        if verbose:
            print('removing users with too few (or too many) ratings')
        rat_count = (up_rat.groupby('user_id', as_index=False)['rating']
                           .count().rename(columns={'rating': 'rating_count'}))
        arr = ((rat_count.rating_count >= self.min_user_ratings) &
               (rat_count.rating_count <= self.max_user_ratings))
        good_users = set(rat_count[arr].user_id)
        up_rat = up_rat[up_rat.user_id.isin(good_users)]
        
        # compute the raw product ids and user ids from underlying data
        raw_products = (set(up_rat.product_id) | 
                        set(product_desc.product_id) |
                        set(product_rev.product_id))

        raw_users = set(up_rat.user_id)

        if verbose:
            print('compute test/validation user/product/cell holdout')

        # compute held-out users and products for test set
        test_heldout_products = set(rp for rp in raw_products 
                if DataSource.hash_fn((rp,)) < self.test_product_holdout)
        test_heldout_users = set(ru for ru in raw_users
                if DataSource.hash_fn((ru,)) < self.test_user_holdout)

        # compute held-out users and products for val set
        val_heldout_products = set(rp for rp in raw_products
                if self.test_product_holdout \
                        <= DataSource.hash_fn((rp,)) \
                        <  self.test_product_holdout + \
                           self.val_product_holdout)

        val_heldout_users = set(ru for ru in raw_users
                if self.test_user_holdout \
                        <= DataSource.hash_fn((ru,)) \
                        <  self.test_user_holdout + \
                           self.val_user_holdout)

        # compute random hash of each row of user-product rating matrix
        random_hash = up_rat.apply(DataSource.hash_fn, axis=1)
        
        if verbose:
            print('constructing dataset split')

        # construct test set
        test_up_rat = up_rat[up_rat.product_id.isin(test_heldout_products) |
                             up_rat.user_id.isin(test_heldout_users) |
                             (random_hash < self.test_cell_holdout)]
        test_product_desc = product_desc[
                product_desc.product_id.isin(test_up_rat.product_id)]
        test_product_rev = product_rev[
                product_rev.product_id.isin(test_up_rat.product_id)]

        # construct validation set
        val_up_rat = up_rat[up_rat.product_id.isin(val_heldout_products) |
                            up_rat.user_id.isin(val_heldout_users) |
                            ((random_hash >= self.test_cell_holdout) & 
                             (random_hash <  (self.test_cell_holdout +
                                              self.val_cell_holdout)))]
        val_product_desc = product_desc[
                product_desc.product_id.isin(val_up_rat.product_id)]
        val_product_rev = product_rev[
                product_rev.product_id.isin(val_up_rat.product_id)]

        # construct train set
        train_up_rat = up_rat[
                ~up_rat.product_id.isin(test_heldout_products) &
                ~up_rat.product_id.isin(val_heldout_products) &
                ~up_rat.user_id.isin(test_heldout_users) &
                ~up_rat.user_id.isin(val_heldout_users) &
                (random_hash >= (self.test_cell_holdout +
                                 self.val_cell_holdout))]
        train_product_desc = product_desc[
                product_desc.product_id.isin(train_up_rat.product_id)]
        train_product_rev = product_rev[
                product_rev.product_id.isin(train_up_rat.product_id)]

        # relabel user and product ids to be integers, thus easy to work with
        if verbose:
            print('relabelling raw users/products')
        if self.rnd_state is not None:
            np.random.seed(234 + self.rnd_state)
        user_id_map = dict(zip(np.random.permutation(list(raw_users)),
                               range(len(raw_users))))
        product_id_map = dict(zip(np.random.permutation(list(raw_products)),
                                  range(len(raw_products))))
        
        pl = (lambda pid: product_id_map[pid])
        ul = (lambda uid: user_id_map[uid])
        pd.options.mode.chained_assignment = None
        test_up_rat.product_id = test_up_rat.product_id.apply(pl)
        test_up_rat.user_id = test_up_rat.user_id.apply(ul)
        test_product_desc.product_id = test_product_desc.product_id.apply(pl)
        test_product_rev.product_id = test_product_rev.product_id.apply(pl)
        val_up_rat.product_id = val_up_rat.product_id.apply(pl)
        val_up_rat.user_id = val_up_rat.user_id.apply(ul)
        val_product_desc.product_id = val_product_desc.product_id.apply(pl)
        val_product_rev.product_id = val_product_rev.product_id.apply(pl)
        train_up_rat.product_id = train_up_rat.product_id.apply(pl)
        train_up_rat.user_id = train_up_rat.user_id.apply(ul)
        train_product_desc.product_id = train_product_desc.product_id.apply(pl)
        train_product_rev.product_id = train_product_rev.product_id.apply(pl)
        pd.options.mode.chained_assignment = 'warn'

        test_up_rat.reset_index(drop=True, inplace=True)
        test_product_desc.reset_index(drop=True, inplace=True)
        test_product_rev.reset_index(drop=True, inplace=True)
        val_up_rat.reset_index(drop=True, inplace=True)
        val_product_desc.reset_index(drop=True, inplace=True)
        val_product_rev.reset_index(drop=True, inplace=True)
        train_up_rat.reset_index(drop=True, inplace=True)
        train_product_desc.reset_index(drop=True, inplace=True)
        train_product_rev.reset_index(drop=True, inplace=True)

        if save_cache:
            if verbose:
                print('saving dataset to disk')
            if not os.path.isdir(data_path):
                os.makedirs(data_path)
            if os.path.isfile(test_up_rat_path):
                os.remove(test_up_rat_path)
            if os.path.isfile(test_product_desc_path):
                os.remove(test_product_desc_path)
            if os.path.isfile(test_product_rev_path):
                os.remove(test_product_rev_path)
            if os.path.isfile(val_up_rat_path):
                os.remove(val_up_rat_path)
            if os.path.isfile(val_product_desc_path):
                os.remove(val_product_desc_path)
            if os.path.isfile(val_product_rev_path):
                os.remove(val_product_rev_path)
            if os.path.isfile(train_up_rat_path):
                os.remove(train_up_rat_path)
            if os.path.isfile(train_product_desc_path):
                os.remove(train_product_desc_path)
            if os.path.isfile(train_product_rev_path):
                os.remove(train_product_rev_path)

            test_up_rat.to_feather(test_up_rat_path)
            test_product_desc.to_feather(test_product_desc_path)
            test_product_rev.to_feather(test_product_rev_path)
            val_up_rat.to_feather(val_up_rat_path)
            val_product_desc.to_feather(val_product_desc_path)
            val_product_rev.to_feather(val_product_rev_path)
            train_up_rat.to_feather(train_up_rat_path)
            train_product_desc.to_feather(train_product_desc_path)
            train_product_rev.to_feather(train_product_rev_path)
            
        return DataSource.bundle_dataset(test_up_rat,  test_product_desc,  test_product_rev,
                                          val_up_rat,   val_product_desc,   val_product_rev,
                                        train_up_rat, train_product_desc, train_product_rev)


    def _raw_user_product_ratings(self):
        raise NotImplementedError

    def _raw_product_descriptions(self):
        raise NotImplementedError

    def _raw_product_reviews(self):
        raise NotImplementedError

class RandomData(DataSource):

    def __init__(self, num_users=10, num_products=1000,
                 prob_rate=0.1, prob_review=0.5,
                 **kwargs):
        DataSource.__init__(self, **kwargs)
        self.num_users = num_users
        self.num_products = num_products
        self.prob_rate = prob_rate
        self.prob_review = prob_review

    def _raw_user_product_ratings(self):
        if self.rnd_state is not None:
            np.random.seed(self.rnd_state)
        data = np.random.choice(
                a=[np.nan, 1.0, 2.0, 3.0, 4.0, 5.0],
                p=[(1.-self.prob_rate)] + [self.prob_rate/5.] * 5,
                size=self.num_users * self.num_products)
        users, products = zip(*itertools.product(range(self.num_users),
                                                 range(self.num_products)))
        res = pd.DataFrame(
                list(zip(users, products, data)),
                columns=['user_id', 'product_id', 'rating'])
        res.dropna(subset=['rating'], inplace=True)
        return res

    def _raw_product_descriptions(self):
        vocab = ['love', 'funny', 'tragic', 'western', 'horror', 'riveting']
        data = {'product_id': [], 'description': []}
        for product_id in range(self.num_products):
            data['product_id'].append(product_id)
            data['description'].append(' ')
            while np.random.uniform() < 0.8:
                data['description'][-1] += np.random.choice(vocab) + ' '
            data['description'][-1] = data['description'][-1][:-1]
        return pd.DataFrame(data)

    def _raw_product_reviews(self):
        vocab = ['wow', 'how', 'such', 'much', 'awesome', 'terrible']
        data = {'product_id': [], 'review': []}
        for user_id in range(self.num_users):
            for product_id in range(self.num_products):
                if np.random.uniform() >= self.prob_review:
                    break
                data['product_id'].append(product_id)
                data['review'].append(' ')
                while np.random.uniform() < 0.8:
                    data['review'][-1] += np.random.choice(vocab) + ' '
                data['review'][-1] = data['review'][-1][:-1]
        return pd.DataFrame(data)


class ToyData(DataSource):

    def __init__(self, **kwargs):
        DataSource.__init__(self, **kwargs)

    def _raw_user_product_ratings(self):
        return pd.DataFrame(
                [['A', 'V', 1.0],##
                 ['A', 'W', 1.0],##
                 ['A', 'Y', 1.0],##
                 ['B', 'X', 2.0],##
                 ['B', 'Z', 2.0],##
                 ['C', 'V', 3.0],
                 ['C', 'W', 3.0],#
                 ['C', 'X', 3.0],
                 ['C', 'Y', 3.0],#
                 ['D', 'W', 4.0],##
                 ['E', 'V', 5.0],
                 ['E', 'W', 5.0],#
                 ['E', 'X', 5.0],
                 ['E', 'Z', 5.0]],#
                columns = ['user_id', 'product_id', 'rating'])

    def _raw_product_descriptions(self):
        # drops Y and Z from products,
        # this should drop A as a user
        return pd.DataFrame(
                [['V', 'A car that drives really fast.'],
                 ['W', 'A book that is very boring.'],
                 ['X', 'A food item that tastes spicy.'],
                 ['Y', '']], # missing Z
                columns = ['product_id', 'description'])

    def _raw_product_reviews(self):
        return pd.DataFrame(
                [['V', 'really good'],
                 ['V', 'okay'],
                 ['X', 'a'],
                 ['X', 'wow, very hot!'],
                 ['Y', 'not too shabby'],
                 ['Z', 'ruined my life']],
                columns = ['product_id', 'review'])


class MovieLensData(DataSource):

    user_ratings_fn = 'data/movielens/ratings.csv'
    user_ratings_columns = [
            ('user_id', 'str'),
            ('movie_id', 'str'),
            ('rating', 'float'),
            ('timestamp', 'int')
    ]

    movies_meta_fn = 'data/movielens/movies_metadata.csv'
    movies_meta_columns = [
        ('adult', 'bool'),
        ('belongs_to_collection', 'str'),
        ('budget', 'int'),
        ('genres', 'str'),
        ('homepage', 'str'),
        ('tmdb_id', 'str'),
        ('imdb_id', 'str'),
        ('original_language', 'str'),
        ('original_title', 'str'),
        ('overview', 'str'),
        ('popularity', 'float'),
        ('poster_path', 'str'),
        ('production_companies', 'str'),
        ('production_countries', 'str'),
        ('release_date', 'str'),
        ('revenue', 'int'),
        ('runtime', 'float'),
        ('spoken_languages', 'str'),
        ('status', 'str'),
        ('tagline', 'str'),
        ('title', 'str'),
        ('video', 'bool'),
        ('vote_average', 'float'),
        ('vote_count', 'int')
    ]
    
    links_fn = 'data/movielens/links.csv'
    links_columns = [
        ('movie_id', 'str'),
        ('imdb_id', 'str'),
        ('tmdb_id', 'str')
    ]

    reviews_fn = 'data/movielens/reviews.csv'

    def __init__(self, **kwargs):
        DataSource.__init__(self, **kwargs)
        self.links = pd.read_csv(
            MovieLensData.links_fn,
            header=None, skiprows=[0],
            names=[t[0] for t in MovieLensData.links_columns],
            dtype=dict(MovieLensData.links_columns)
        )
        self.links.imdb_id = self.links.imdb_id.apply(lambda s: f'tt{s}')

    def _raw_user_product_ratings(self):
        ratings = pd.read_csv(
            MovieLensData.user_ratings_fn,
            header=None, skiprows=[0],
            names=[t[0] for t in MovieLensData.user_ratings_columns],
            dtype=dict(MovieLensData.user_ratings_columns))
        ratings.drop('timestamp', axis=1, inplace=True)
        ratings.columns = ['user_id', 'product_id', 'rating']
        ratings.rating *= 2
        return ratings

    def _raw_product_descriptions(self):
        movies_meta = pd.read_csv(
            MovieLensData.movies_meta_fn,
            header=None, skiprows=[0],
            names=[t[0] for t in MovieLensData.movies_meta_columns],
            dtype=dict(MovieLensData.movies_meta_columns)
        ).drop_duplicates(subset='tmdb_id')
        movies_meta.overview.fillna('', inplace=True)
        merged = movies_meta.merge(self.links, how='left', on='imdb_id')
        return (merged[['movie_id', 'overview']]
                .rename(columns={'movie_id': 'product_id',
                                 'overview': 'description'}))

    def _raw_product_reviews(self):
        return pd.read_csv(MovieLensData.reviews_fn,
                dtype={'product_id': 'str', 'review': 'str'})

class AmazonBooks(DataSource):

    data_path = 'data/amazonbooks/'
    product_desc_path = os.path.join(data_path, 'descriptions.feather')

    def __init__(self, **kwargs):
        DataSource.__init__(self, **kwargs)
        self.loaded = False

    def _load_data(self):
        self.product_descriptions = pd.read_feather(AmazonBooks.product_desc_path)
        rating_review_arr = []
        for i in range(0, int(1e10), 1000000):
            fn = f'{AmazonBooks.data_path}/ratings_reviews{i}.feather'
            if not os.path.exists(fn):
                break
            rating_review_arr.append(pd.read_feather(fn))
        Df = pd.concat(rating_review_arr, axis=0)
        self.user_product_ratings = Df[['user_id', 'product_id', 'rating']]
        self.product_reviews = Df[['product_id', 'review']]        
        self.loaded = True

    def _raw_user_product_ratings(self):
        if not self.loaded:
            self._load_data()
        return self.user_product_ratings

    def _raw_product_descriptions(self):
        if not self.loaded:
            self._load_data()
        return self.product_descriptions

    def _raw_product_reviews(self):
        if not self.loaded:
            self._load_data()
        return self.product_reviews
