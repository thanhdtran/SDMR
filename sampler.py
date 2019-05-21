"""
Module containing functions for negative item sampling.
"""

import numpy as np
from scipy.sparse import csr_matrix
import global_constants as gc
np.random.seed(gc.SEED)

class Sampler(object):
    def __init__(self):
        super(Sampler, self).__init__()
        self.user_neg_items_map = dict()



    def init_user_item_seqs(self, user_all_items, num_users, num_items):
        self.n_users = num_users
        self.n_items = num_items
        self.user_items_map = user_all_items

    def set_interactions(self, interactions):
        csr_data = interactions.tocsr()
        self.build_neg_dict(csr_data)

    def build_neg_dict(self, csr_data):
        #for each user, store the unobserved values into a dict for sampling later.
        csr_data = csr_matrix(csr_data)
        n_users, n_items = csr_data.shape
        user_counts = np.zeros(n_users)
        for u in range(n_users): user_counts = csr_data[u].getnnz()
        pass

    def random_neg_items(self, user_ids=None, num_neg=4):
        neg_items = np.zeros(shape=(len(user_ids), num_neg), dtype=np.int64)
        for i, uid in enumerate(user_ids):
            user_pos_items = self.user_items_map[uid]
            local_neg_items = set()
            j = 0
            while j < num_neg:
                neg_item = np.random.randint(self.n_items)
                if neg_item not in user_pos_items and neg_item not in local_neg_items and neg_item != gc.PADDING_IDX:
                    local_neg_items.add(neg_item)
                    neg_items[i][j] = neg_item
                    j += 1
        return neg_items

