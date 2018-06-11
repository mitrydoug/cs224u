from collections import defaultdict
import re

import numpy as np
import torch
import torch.utils.data

import utils

class UserProductRatingsDataset(torch.utils.data.Dataset):

    def __init__(self, data,
            desc_vocab_size, revw_vocab_size,
            transform=None):

        self.desc_vocab_size = desc_vocab_size
        self.revw_vocab_size = revw_vocab_size
        self.transform = transform

        utils.base_timer.start('copying required data')
        self.raw_product_descriptions = data['product_descriptions']
        self.raw_product_reviews = data['product_reviews']
        self.user_product_ratings = data['user_product_ratings']
        self.product_descriptions = data['product_descriptions'].copy()
        self.product_reviews = data['product_reviews'].copy()

        utils.base_timer.start('cleaning product descriptions and reviews')
        self._clean_descs_and_reviews()

        utils.base_timer.start('building product descriptions and review vocabs')
        self._build_vocab()

        utils.base_timer.start('building product description and review index sequence')
        self._build_desc_revw_sequences()

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
                       reverse=True)[:top_n-len(special_tokens)]))
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
            self.product_to_desc_seq[pid] = desc
        self.product_to_revw_seq = defaultdict(list)
        for _, row in self.product_reviews.iterrows():
            pid = row['product_id']
            revw = []
            for match in re.findall(r' *([a-z0-9.,]+) +', row['review']):
                if match in self.revw_vocab_to_idx:
                    revw.append(self.revw_vocab_to_idx[match])
                else:
                    revw.append(self.revw_vocab_to_idx['<UNK>'])
            self.product_to_revw_seq[pid].append(revw)

    def __len__(self):
        return len(self.user_product_ratings)

    def __getitem__(self, idx):
        row = self.user_product_ratings.iloc[idx]
        res = {
            'user_id': row.user_id,
            'product_desc': self.product_to_desc_seq[row.product_id],
            'product_revw': self.product_to_revw_seq[row.product_id],
            'rating': row.rating}
        if self.transform:
            res = self.transform(res)
        return res

class ReviewSampler(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        revw_idx = np.random.choice(len(sample['product_revw']))
        product_revw = sample['product_revw'][revw_idx]
        res = dict(sample)
        res['product_revw'] = product_revw
        return res

class CombineSequences(object):

    def __init__(self):
        pass

    def __call__(self, batch):

        user_ids = torch.tensor([item['user_id'] for item in batch], dtype=torch.long)
        ratings = torch.tensor([item['rating'] for item in batch], dtype=torch.float)

        # padded nicely
        desc = [item['product_desc'] for item in batch]
        desc = [(len(x), idx, x) for idx, x in enumerate(desc)]
        desc.sort(reverse=True)
        # store original batch order
        desc_lens, desc_idxs, desc = map(list,zip(*desc))
        max_len = len(desc[0])
        desc = torch.tensor(
                [ds + [0] * (max_len - len(ds)) for ds in desc],
                dtype=torch.long)
        desc_lens = torch.tensor(desc_lens, dtype=torch.long)
        desc_idxs = torch.tensor(desc_idxs, dtype=torch.long)

        revw = [item['product_revw'] for item in batch]
        revw = [(len(r), idx, r) for idx, r in enumerate(revw)]
        revw.sort(reverse=True)
        revw_lens, revw_idxs, revw = map(list, zip(*revw))
        max_len = len(revw[0])
        revw = torch.tensor(
                [rv + [0] * (max_len - len(rv)) for rv in revw],
                dtype=torch.long)
        revw_lens = torch.tensor(revw_lens, dtype=torch.long)
        revw_idxs = torch.tensor(revw_idxs, dtype=torch.long)

        return (user_ids, ratings, desc, desc_lens, desc_idxs,
                                   revw, revw_lens, revw_idxs)
