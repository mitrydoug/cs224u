import math
from collections import Counter
import numpy as np 

def extract_data_for_user(user, ratings_table, tfidf_table, vocab=None):
    if vocab is None:
        vocab = extract_user_vocab(user)
    print(vocab)
    products_ratings = ratings_table.loc[ratings_table['user_id'] == user]
    products = list(products_ratings["product_id"])
    print(products)

    ratings = list(products_ratings["rating"]) # Note: not mean centered

    # matrix , row is product , columns is words
    X = np.array([[tfidf_table.loc[p][v] for v in vocab] for p in products])
    print(X.shape)
    X_no_nan = np.array([[X[i,j] if not math.isnan(X[i,j]) else 0  for i in range(X.shape[0])] for j in range(X.shape[1])])
    y = ratings
    return (X_no_nan,y, vocab)

def get_vocab(X, n_words=None):
    """Get the vocabulary for an RNN example matrix `X`,
    adding $UNK$ if it isn't already present.

    Parameters
    ----------
    X : list of lists of str
    n_words : int or None
        If this is `int > 0`, keep only the top `n_words` by frequency.

    Returns
    -------
    list of str

    """
    wc = Counter([w for ex in X for w in ex])
    wc = wc.most_common(n_words) if n_words else wc.items()
    vocab = {w for w, c in wc}
    vocab.add("$UNK")
    return sorted(vocab)

# import re
# re.sub("[,.;?!():\[\]]","" , "a[df,e(ar;af.a)s]df!a:c?")