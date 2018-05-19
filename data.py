
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
        if not up_rev.empty:
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

    def __init__(self, frac=1.0):
        """
          frac: the fraction of the dataset to sample.
        """
        DataSource.__init__(self)
        self.frac = frac
        self.links = pd.read_csv(
            MovieLensData.links_fn,
            header=None, skiprows=[0],
            names=[t[0] for t in MovieLensData.links_columns],
            dtype=dict(MovieLensData.links_columns)
        ).drop_duplicates(subset='tmdb_id')
    
    def _raw_user_product_ratings(self):
        ratings = pd.read_csv(
            MovieLensData.user_ratings_fn,
            header=None, skiprows=[0],
            names=[t[0] for t in MovieLensData.user_ratings_columns],
            dtype=dict(MovieLensData.user_ratings_columns))
        ratings.drop('timestamp', axis=1, inplace=True)
        ratings.columns = ['user_id', 'product_id', 'rating']
        return ratings.sample(frac=self.frac)

    def _raw_product_descriptions(self):
        movies_meta = pd.read_csv(
            MovieLensData.movies_meta_fn,
            header=None, skiprows=[0],
            names=[t[0] for t in MovieLensData.movies_meta_columns],
            dtype=dict(MovieLensData.movies_meta_columns)
        ).drop_duplicates(subset='tmdb_id')
        movies_meta.overview.fillna('', inplace=True)
        merged = movies_meta.merge(self.links, how='left', on='tmdb_id')
        return (merged[['movie_id', 'overview']]
                .rename(columns={'movie_id': 'product_id',
                                 'overview': 'description'}))

    def _raw_user_product_reviews(self):
        return pd.DataFrame([], columns=['user_id', 'product_id', 'review'])
