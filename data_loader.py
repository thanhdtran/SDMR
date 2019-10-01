import numpy as np
import pandas as pd
import global_constants as gc
from collections import defaultdict

np.random.seed(gc.SEED)





def load_data(path, sep = '\t', header=None, dataset=None):
    return Interactions(path, dataset=dataset)

class Interactions(object):


    def __init__(self,
                 data_path,
                 dataset='ml1m'
                 ):
        self._user_all_items = defaultdict(list) # get consumed items, key is user, value is the consumed items
        self._useritem_prev_items = defaultdict(list) #key is (user,item), value is previous items
        self._user_ids = []
        self._item_ids = []
        self._ratings = []
        self._timestamps = []
        self._num_users, self._num_items = None, None
        duplicate_user_item_pairs = set() #remove duplicate pairs
        with open(data_path, 'r') as f:
            #user ids and item ids must start at 1
            for line in f:
                tokens = line.strip().split('\t')
                uid, iid, rating, timestamp = int(tokens[0]), int(tokens[1]), float(tokens[2]), int(float(tokens[3]))
                duplicate_key = (uid, iid)
                if duplicate_key not in duplicate_user_item_pairs: duplicate_user_item_pairs.add((uid, iid))
                else: continue

                self._user_ids.append(uid)
                self._item_ids.append(iid)
                self._ratings.append(rating)
                self._timestamps.append(timestamp)


                prev_items = self._user_all_items[uid][:] if len(self._user_all_items[uid]) > 0 else [gc.PADDING_IDX]
                self._useritem_prev_items[(uid, iid)].extend(prev_items)
                self._user_all_items[uid].append(iid)


        self._user_ids, self._item_ids = np.asarray(self._user_ids), np.asarray(self._item_ids)
        self._ratings, self._timestamps = np.asarray(self._ratings), np.asarray(self._timestamps)

        self.num_users = self._num_users or int(np.max(self._user_ids) + 1)
        self.num_items = self._num_items or int(np.max(self._item_ids) + 1)

        self._max_len_user_seq = 0 #maximum number of consumed items in all users' transactions
        for uid in set(self._user_ids):
            self._max_len_user_seq = max(self._max_len_user_seq, len(self._user_all_items[uid]))


        self._dataset = dataset


    def get_batch_seqs(self, user_ids, item_ids, max_seq_len=100):
        '''

        :param user_ids:
        :param max_seq_len:
        :param type: two options: only_prev, all. all: extract all consumed items,
                                                  only_prev: only items consumed before this target item
        :return:
        '''
        batch_size = len(user_ids)
        seq_len = max_seq_len if max_seq_len != -1 else self._max_len_user_seq
        user_seqs = np.zeros((batch_size, seq_len), dtype=np.int64)
        
        for i, (uid, iid) in enumerate(zip(user_ids, item_ids)):
            user_seq = np.zeros(seq_len, dtype=np.int64)
            key = (uid, iid)
            tmp_seq = self._useritem_prev_items[key]
            if len(tmp_seq) == 0:
                tmp_seq = np.asarray(self._user_all_items[uid], dtype=np.int64)
                tmp_seq = tmp_seq[tmp_seq != iid]  # remove item iid in user seq

            # shorten the seq as of seq_len limitation
            if len(tmp_seq) > seq_len:
                tmp_seq = tmp_seq[-seq_len:]

            if len(tmp_seq) == 0:
                print 'data_loader.py, line 215: error -->',uid, iid

            user_seq[-len(tmp_seq):] = tmp_seq
            user_seqs[i] = user_seq
        return user_seqs
