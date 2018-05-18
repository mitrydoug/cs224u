
import hashlib
import itertools

import numpy as np
import pandas as pd

# 1. user-product ratings
# 2. product descriptions
# 3. user-product reviews


class DataSource:

    def __init__(self):
        # SENSITIVE PARAMETER, DO NOT CHANGE
        self.test_frac = 0.2

    def get_dataset(self):

        # load data from sub-class implementations
        up_rat = self._raw_user_product_ratings()
        p_desc = self._raw_product_descriptions()
        up_rev = self._raw_user_product_reviews()

        products = (set(up_rat.product_id) | 
                    set(p_desc.product_id) |
                    set(up_rev.product_id))
        users = set(up_rat.user_id) | set(up_rev.user_id)

        ######################################
        #### Process user-product ratings ####
        ######################################
        assert set(up_rat.columns) == {'user_id', 'product_id', 'rating'}
        up_rat.drop_duplicates(inplace=True)

        # IMPORTANT: the TRAIN/TEST split is determined by the hash value
        # of each (user_id, product_id, rating). This ensures that 
        def hash_fn(row):
            # return 8 bytes of row hash as integer
            hd = hashlib.sha256(bytes(str(tuple(row)), 'utf8')).hexdigest()
            return int(hd[:16], 16)
        up_rat['hash'] = up_rat.apply(hash_fn, axis=1)
        up_rat.sort_values('hash', inplace=True)

        # convert raw user(s) and product(s) to a contiguous range
        # of integers
        user_ids = dict(zip(sorted(users), range(len(users))))
        product_ids = dict(zip(sorted(products), range(len(products))))
        # update up_rat columns
        up_rat.user_id = up_rat.user_id.apply(lambda uid: user_ids[uid])
        up_rat.product_id = up_rat.product_id.apply(lambda pid: product_ids[pid])

        up_rat_test  = (up_rat[up_rat.hash <= self.test_frac * (2 ** 64)]
                        .drop('hash', axis=1))
        up_rat_train = (up_rat[up_rat.hash > self.test_frac * (2 ** 64)]
                        .drop('hash', axis=1))

        ######################################
        #### Process product descriptions ####
        ######################################
        assert set(p_desc.columns) == {'product_id', 'description'}
        p_desc = (p_desc.groupby('product_id').apply(
                  lambda df: max(df.description, key=lambda s: len(s)))
                  .to_frame().reset_index().rename(columns={0: 'description'}))
        p_desc.product_id = p_desc.product_id.apply(lambda pid: product_ids[pid])

        ######################################
        #### Process user-product reviews ####
        ######################################
        assert set(up_rev.columns) == {'user_id', 'product_id', 'review'}
        up_rev = (up_rev.groupby(['user_id', 'product_id']).apply(
                  lambda df: max(df.review, key=lambda s: len(s)))
                  .to_frame().reset_index().rename(columns={0:'review'}))
        up_rev.user_id = up_rev.user_id.apply(lambda uid: user_ids[uid])
        up_rev.product_id = up_rev.product_id.apply(lambda pid: product_ids[pid])

        merged = pd.merge(up_rev, up_rat_test, how='left',
                          on=['user_id', 'product_id'])
        up_rev_test = merged[merged.rating.notna()].drop('rating', axis=1)
        up_rev_train = merged[merged.rating.isna()].drop('rating', axis=1)

        return {
            'train': {'user_product_ratings': up_rat_train,
                      'product_descriptions': p_desc,
                      'user_product_reviews': up_rev_train},
            'test' : {'user_product_ratings': up_rat_test,
                      'product_descriptions': p_desc,
                      'user_product_reviews': up_rev_test}}

    def _raw_user_product_ratings(self):
        raise NotImplementedError

    def _raw_product_descriptions(self):
        raise NotImplementedError

    def _raw_user_product_reviews(self):
        raise NotImplementedError

class RandomData(DataSource):

    def __init__(self, num_users=10, num_products=100,
                 prob_rate=0.1, prob_review=0.1,
                 rnd_state=None):
        DataSource.__init__(self)
        self.rnd_state=rnd_state
        self.num_users = num_users
        self.num_products = num_products
        self.prob_rate = prob_rate
        self.prob_review = prob_review

    def _raw_user_product_ratings(self):
        if self.rnd_state is not None:
            np.random.seed(self.rnd_state)
        data = np.random.choice(
                a=[np.nan, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                p=[(1.-self.prob_rate)] + [self.prob_rate/8.] * 8,
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

    def _raw_user_product_reviews(self):
        vocab = ['wow', 'how', 'such', 'much', 'awesome', 'terrible']
        data = {'user_id': [], 'product_id': [], 'review': []}
        for user_id in range(self.num_users):
            for product_id in range(self.num_products):
                if np.random.uniform() >= self.prob_review:
                    break
                data['user_id'].append(user_id)
                data['product_id'].append(product_id)
                data['review'].append(' ')
                while np.random.uniform() < 0.8:
                    data['review'][-1] += np.random.choice(vocab) + ' '
                data['review'][-1] = data['review'][-1][:-1]
        return pd.DataFrame(data)

class MovieLensData(DataSource):

    user_ratings_fn = 'data/movielens/ratings.csv'
    user_ratings_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    user_ratings_dtypes  = ['int', 'int', 'float', 'int']

    movies_meta_fn = 'data/movielens/movies_meta.csv'
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
        ('overview', 'str')
        ('popularity', 'float'),
        ('poster_path', 'str'),
        ('production_companies', 'str'),
        'production_countries', 'release_date', 'revenue', 'runtime',
        'spoken_languages', 'status', 'tagline', 'title', 'video',
        'vote_average', 'vote_count'
    ]
    movies_meta_dtypes =  [
        'bool',
        'str',
        'int',
        'str',
        'str',
        'int', 'str', 'str', 'str',
        'str', 'float', 'str', 'str', 'str', 'str', 'int', 'float',
        'str', 'str', 'str', 'str', 'bool', 'float', 'int'
    ]
    
    links_fn = 'data/movielens/links.csv'
    links_columns = ['movie_id', 'imdb_id', 'tmdb_id']
    links_dtypes = ['str', 'str', 'str']

    def __init__(self, frac=1.0):
        """
          frac: the fraction of the dataset to sample.
        """
        DataSource.__init__(self)
        self.frac = frac
    
    def _raw_user_product_ratings(self):
        dtypes = dict(zip(
            MovieLensData.user_ratings_columns,
            MovieLensData.user_ratings_dtypes))
        ratings = pd.read_csv(
            MovieLensData.user_ratings,
            header=None, skiprows=[0],
            names=ratings_columns,
            dtype=dict(zip(ratings_columns, ratings_dtypes)))
        ratings.drop('timestamp', axis=1, inplace=True)
        ratings.columns = ['user_id', 'product_id', 'rating']
        return ratings.sample(frac=self.frac)

    def _raw_product_descriptions(self):
        movies_meta = pd.read_csv(
            'movies_metadata.csv', header=None, skiprows=[0],
            names=movie_meta_columns,
            dtype=dict(zip(movie_meta_columns, movie_meta_dtypes))
        ).drop_duplicates(subset='tmdb_id')
        
        

    def _raw_user_product_reviews(self):
        pass
