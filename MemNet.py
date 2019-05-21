import param_initializer as initializer
from torch.autograd import Variable
import numpy as np
import pytorch_utils as my_utils
import torch
from torch import nn
import torch.nn.functional as F
import global_constants as gc



L1, L2 = ('l1', 'l2')
WEIGHT_TYINGS = ADJACENT, LAYER_WISE, _ = ('adjacent', 'layerwise', None)
GATE_TYINGS = GATE_GLOBAL, GATE_HOP_SPECIFIC = ('gate_global', 'gate_hop_specific')
def L2_pow2_func(x):
    #squared L2 distance
    return x **2

def L1_func(x):
    return torch.abs(x)

class MemoryModule(nn.Module):
    def __init__(self,
                 n_users, n_items, embedding_size,
                 item_seq_size, memory_size=None,
                 user_embeddings = None, item_embeddings = None, item_biases = None,
                 W1 = None, W2 = None,
                 ):
        super(MemoryModule, self).__init__()

        self._n_users, self._n_items, self._embedding_size = n_users, n_items, embedding_size
        self._memory_size, self._item_seq_size = memory_size, item_seq_size
        self._user_embeddings = (
            user_embeddings or
            nn.Embedding(n_users, embedding_size, padding_idx=gc.PADDING_IDX)
        )
        self._item_embeddings = (
            item_embeddings or
            nn.Embedding(n_items, embedding_size, padding_idx=gc.PADDING_IDX)
        )

        self._item_biases = item_biases or nn.Embedding(n_items, 1)


        self._W1 = (
            W1 or
            nn.Linear(2*embedding_size, embedding_size) #combination of target user u and target item j
        )
        self._W2 = (
            W2 or
            nn.Linear(2*embedding_size, embedding_size) #combination of [u, j] with consumed item i
        )

        self._reset_weight()

    def _reset_transform_identity(self):
        self._W1.weight.data.copy_(my_utils.numpy2tensor(
            np.concatenate(
                (np.identity(self._embedding_size), -np.identity(self._embedding_size))
                , axis=0).T
        )
        )
        self._W2.weight.data.copy_(my_utils.numpy2tensor(
            np.concatenate(
                (np.identity(self._embedding_size), np.identity(self._embedding_size))
                , axis=0).T)
        )

    def _reset_weight(self, type='he-normal'):
        self._user_embeddings.weight.data.normal_(0, 0.01)#1.0/self._embedding_size)
        self._user_embeddings.weight.data[gc.PADDING_IDX].fill_(0)
        self._item_embeddings.weight.data.normal_(0, 0.01)#1.0/self._embedding_size) #0.01)
        self._item_embeddings.weight.data[gc.PADDING_IDX].fill_(0)

        type = gc.SDM_INIT_TRANSFORM_TYPE
        if type == 'normal':
            initializer.normal_initialization(self._W1.weight, 1.0 / self._embedding_size)
            initializer.normal_initialization(self._W2.weight, 1.0 / self._embedding_size)
        elif type == 'xavier':
            initializer.xavier_normal_initialization(self._W1.weight)
            initializer.xavier_normal_initialization(self._W2.weight)
        elif type == 'lecun':
            initializer.lecun_uniform_initialization(self._W1.weight)
            initializer.lecun_uniform_initialization(self._W2.weight)
        elif type == 'he-normal':
            initializer.he_normal_initialization(self._W1.weight)
            initializer.he_normal_initialization(self._W2.weight)
        elif type == 'he-uniform':
            initializer.he_uniform_initialization(self._W1.weight)
            initializer.he_uniform_initialization(self._W2.weight)
        elif type == 'identity':
            self._reset_transform_identity()
        else:
            self._reset_transform_identity()



class OutputModule(nn.Module):


    def __init__(self, embedding_size, distance_type = L2, dropout_prob=0.2, non_linear = None, sum_mapping=True, seq_size=None):
        super(OutputModule, self).__init__()
        self._dist_func = L2_pow2_func if distance_type == L2 else L1_func
        self._dropout = nn.Dropout(p=dropout_prob)
        self._non_linear = non_linear
        self._sum_func = nn.Linear(2*embedding_size, 1) if sum_mapping else None
        
        self._transform = nn.Linear(embedding_size, embedding_size)
        #self._agg_layer = nn.Linear(seq_size, 1)

        self._reset_weights()


    def _reset_weights(self):
        initializer.lecun_uniform_initialization(self._sum_func.weight)

    
    def forward(self, weights, q_o, m_c, W2):
        '''

        :param weights: batch x num_consumed_items x embedding_size # batch x seq_len
        :param q_o: q_o = W1[u, j] #u: target user, j: target item # batch x embedding_size
        :param m_c: output embeddings of consumed items, batch x seq_len x embedding_size
        :param C: output memory
        :return:
        '''

        q_o = q_o.unsqueeze(1).expand_as(m_c) #batch x embedding_size --> batch x seq_len x embedding_size


        q_m = W2(torch.cat([q_o, m_c], dim=2))

        q_m = self._non_linear(q_m) if self._non_linear else q_m

        q_m = self._dropout(q_m)

        # d2_q_m = -self._dist_func(q_m)
        d2_q_m = self._dist_func(q_m)

        weights_expand = weights.unsqueeze(2).expand_as(d2_q_m)
        dist_vec = torch.mul(d2_q_m, weights_expand)
        #dist_vec_sum = dist_vec.sum(dim=1)# -sum{ alpha_k . DISTANCE( W2[ W1[u,j], i])    }
                                        # negative because the higher the distance, the lower the attention score.
                                        # dist_sum: batch x embedding_size, sum of the dist_vec_sum will return the total
                                        # distance from target item j to all consumed items of target user u.
        dist_vec_sum = F.relu(self._transform(dist_vec)).sum(dim = 1) #new: adding aggregation layer on March 18, 2019 instead of sum
        return -(dist_vec_sum)
        # return -torch.abs(dist_vec_sum)
        #return dist_vec_sum



class MaskedAttention(nn.Module):
    def __init__(self, distance_type=L2, dropout_prob=0.2, non_linear = None, sum_mapping=False, embedding_size=128):
        super(MaskedAttention, self).__init__()
        self._dist_func = L2_pow2_func if distance_type==L2 else L1_func
        self._dropout = nn.Dropout(p=dropout_prob)
        self._non_linear = non_linear
        self._sum_func = nn.Parameter(torch.ones(embedding_size)) if sum_mapping else None
        self._sum_mapping=sum_mapping


    def forward(self, q, m, W2, mask):
        '''

        :param q: the query
        :param m: the memory
        :param A_memory: get the transformation weights in the A_memory
        :return: softmax(-DISTANCE(W2[m, q])) where q = W1[target_user, target_item] = W1[u, j]
        '''

        q = q.unsqueeze(1).expand_as(m)
        q_m = W2(torch.cat([q, m], dim=2)) # transformation between item memory and query
        q_m = self._non_linear(q_m) if self._non_linear else q_m
        q_m = self._dropout(q_m)


        d2_q_m = -self._dist_func(q_m) #negative because the higher the distance, the lower the attention score.
        d2_q_m = d2_q_m.sum(dim=2)  # measure the square distance of each input item to the target item, batch x seq_len

        # we want the padding index has very low scores
        d2_q_m = torch.mul(d2_q_m, mask)
        weights = F.softmax(d2_q_m, dim=1)

        return weights

class SDM(nn.Module):

    def __init__(self,
                 n_users, n_items, embedding_size,
                 item_seq_size, memory_size = None,
                 weight_tying = LAYER_WISE,
                 n_hops=3,
                 nonlinear_func = 'none',
                 dropout_prob = 0.5,
                 gate_tying = GATE_GLOBAL,
                 sum_mapping = True, #was False before
                 ret_sum = True
                 ):
        super(SDM, self).__init__()
        assert weight_tying in WEIGHT_TYINGS, (
            'Available weight tying schemes are: {weight_tying_types}'
            .format(weight_tying_types=self.WEIGHT_TYINGS)
        )

        self._n_users, self._n_items, self._embedding_size = n_users, n_items, embedding_size
        self._item_seq_size, self._memory_size = item_seq_size, memory_size
        self._weight_tying = weight_tying
        self._n_hops = n_hops
        self._dropout_prob = dropout_prob
        self._dropout = nn.Dropout(dropout_prob)
        self._ret_sum = ret_sum #return some at the end of forward or not

        if nonlinear_func == 'relu': self._non_linear = F.relu
        elif nonlinear_func == 'tanh': self._non_linear = torch.tanh
        else: self._non_linear = None

        self._outputModule = OutputModule(embedding_size, dropout_prob=dropout_prob,
                                          non_linear=self._non_linear, seq_size = item_seq_size)

        self._attModule = MaskedAttention(dropout_prob=dropout_prob,
                                          non_linear=self._non_linear, embedding_size=embedding_size)


        self._sum_func = nn.Linear(embedding_size, 1) if sum_mapping else None

        #create memories:
        self.A_memories = nn.ModuleList() #input memory
        self.C_memories = nn.ModuleList() #output memory
        self._gate_transforms = nn.ModuleList() #gate transformation

        for i in range(n_hops):
            exist_prev = i > 0
            A_prev = self.A_memories[i-1] if exist_prev else None

            C_prev = self.C_memories[i-1] if exist_prev else None



            gate_prev_transform = self._gate_transforms[i-1] if exist_prev and gate_tying == GATE_GLOBAL else None

            if weight_tying == ADJACENT:
                A_user_embeddings = C_prev._user_embeddings if exist_prev else None
                A_item_embeddings  = C_prev._item_embeddings if exist_prev else None
                A_W1 = C_prev._W1 if exist_prev else None
                A_W2 = C_prev._W2 if exist_prev else None

                C_user_embeddings, C_item_embeddings = None, None
                C_W1, C_W2 = None, None

            elif weight_tying == LAYER_WISE:
                #same weights shared by all hops
                A_user_embeddings = A_prev._user_embeddings if exist_prev else None
                A_item_embeddings = A_prev._item_embeddings if exist_prev else None
                A_item_biases = A_prev._item_biases if exist_prev else None

                C_user_embeddings = C_prev._user_embeddings if exist_prev else None
                C_item_embeddings = C_prev._item_embeddings if exist_prev else None
                C_item_biases = C_prev._item_biases if exist_prev else None


                #same transformation shared by all hops
                A_W1 = A_prev._W1 if exist_prev else None
                A_W2 = A_prev._W2 if exist_prev else None
                C_W1 = C_prev._W1 if exist_prev else None
                C_W2 = C_prev._W2 if exist_prev else None

            else:
                A_user_embeddings, A_item_embeddings = None, None
                A_item_biases = None
                A_W1, A_W2 = None, None

                C_user_embeddings, C_item_embeddings = None, None
                C_W1, C_W2 = None, None
                C_item_biases = None

            self.A_memories.append(
                MemoryModule(
                    n_users, n_items, embedding_size,
                    item_seq_size, memory_size=None,
                    user_embeddings=A_user_embeddings,
                    item_embeddings=A_item_embeddings,
                    item_biases = A_item_biases,
                    W1 = A_W1, W2 = A_W2,

                )
            )

            self.C_memories.append(
                MemoryModule(
                    n_users, n_items, embedding_size,
                    item_seq_size, memory_size=None,
                    user_embeddings=C_user_embeddings,#self.A_memories[-1]._user_embeddings, #C_user_embeddings,
                                                      # #share same embeddings with A or not #not sharing is better
                    item_embeddings=C_item_embeddings,
                    item_biases = C_item_biases,
                    W1 = C_W1, W2 = C_W2,
                )
            )

            gate_transform = gate_prev_transform if gate_prev_transform else \
                                   nn.Linear(embedding_size, embedding_size) #global gate transformation
            self._gate_transforms.append(gate_transform)

        self._reset_weights()

    def _reset_weights(self):
        if self._sum_func:
            initializer.lecun_uniform_initialization(self._sum_func.weight)
            # self._sum_func.weight.data.copy_(my_utils.numpy2tensor(np.ones((self._embedding_size, 1)).T)) #normal sum func

    def _make_query(self, u_embed, j_embed, W1):
        # transform target user u and target item j
        q = torch.cat([u_embed, j_embed], dim=1)
        q = self._non_linear(W1(q)) if self._non_linear else W1(q)
        return nn.Dropout(p=self._dropout_prob)(q)

    def _make_output_query(self, u_embed, j_embed, W1):
        # transform target user u and target item j
        q = torch.cat([u_embed, j_embed], dim=1)
        q = self._non_linear(W1(q)) if self._non_linear else W1(q)
        return nn.Dropout(p=self._dropout_prob)(q)



    def _make_mask(self, x):
        mask = np.asarray(my_utils.tensor2numpy(x.data.cpu().clone()), dtype=np.float64)
        # mask = my_utils.tensor2numpy(x != 0)
        mask[mask != gc.PADDING_IDX] = 1.0
        # mask[mask <= 0] = float('inf')
        mask[mask <= 0] = 65535

        return my_utils.gpu(Variable(my_utils.numpy2tensor(mask)).type(torch.FloatTensor), use_cuda=my_utils.is_cuda(x))


    def _get_additional_l2_loss(self):
        item_reg_l2 = torch.norm(self.C_memories[-1]._item_embeddings.weight)
        user_reg_l2 = torch.norm(self.C_memories[-1]._user_embeddings.weight)
        # return item_reg + user_reg
        return item_reg_l2 + user_reg_l2


    def forward(self, x, u, j):
        """

        :param x: the consumed items of user u, format: batch_size x 1 x n_items
        :param u: user, format batch_size x 1
        :param j: next item, format: batch_size x 1
        :return:
        """
        mask = self._make_mask(x)

        o = None #output
        hops_output = []

        for i, (A, C) in enumerate(zip(self.A_memories, self.C_memories)):

            prev_o = i > 0

            #get user embedding:
            a_u = A._user_embeddings(u) # batch x embedding_size
            a_j = A._item_embeddings(j) # batch x embedding_size
            m_a = A._item_embeddings(x)  # get the item embeddings in input memory, return batch x seq_len x embedding_size

            #make query
            W1 = A._W1
            if i == 0: q_a = self._make_query(a_u, a_j, W1)  #batch x embedding_size

            if prev_o:
                    #gated multiple-hop design
                    gated = torch.sigmoid(self._gate_transforms[i](o))
                    q_a = torch.mul((1-gated), q_a) + torch.mul(gated, o) #residual connection.

            W2 = A._W2
            weights = self._attModule(q_a, m_a, W2, mask)

            c_j = C._item_embeddings(j)
            c_u = C._user_embeddings(u)
            m_c = C._item_embeddings(x)

            W1_output = C._W1
            q_o = self._make_output_query(c_u, c_j, W1_output) #output combination of target user u and target item j
            #q_o = q_a #shared query in here, shared query is good for dense data, not good for sparse datasets.

            # W2_output = self._W2
            W2_output = C._W2
            o = self._outputModule(weights, q_o, m_c, W2_output)
            hops_output.append((o, weights))

        if self._ret_sum:
            if self._sum_func:
                return -(self._sum_func(hops_output[-1][0]))
            else:
                return -(hops_output[-1][0].sum(dim=1))  # return output of the last hop.
        else:
            return hops_output[-1][0]



class SDP(nn.Module):
    '''
    generalized metric matrix factorization method
    '''
    def __init__(self, n_users, n_items, embedding_size=16, distance_type = 'l1', sum_mapping = True, ret_sum = True,
                 nonlinear_func='none', dropout_prob=0.2, num_layers = 1):
        super(SDP, self).__init__()
        self._n_users = n_users
        self._n_items = n_items
        self._n_factors = embedding_size
        self._embedding_size = embedding_size
        self._ret_sum = ret_sum

        self._dropout = nn.Dropout(p=dropout_prob)

        self._user_embeddings = nn.Embedding(n_users, embedding_size)


        self._item_embeddings = nn.Embedding(n_items, embedding_size)

        self._fcs = nn.ModuleList()
        in_emb_size = 2*embedding_size
        out_emb_size = in_emb_size/2
        for i in range(num_layers):
            self._fcs.append(nn.Linear(in_emb_size, out_emb_size, bias=True))
            in_emb_size = out_emb_size
            out_emb_size = in_emb_size/2

        if nonlinear_func == 'relu': self._non_linear = F.relu
        elif nonlinear_func == 'tanh': self._non_linear = torch.tanh
        else: self._non_linear = None

        self._dist_func = L2_pow2_func if distance_type == L2 else L1_func

        self._sum_func = nn.Linear(in_emb_size, 1) if sum_mapping else None
        ###################################

        self._reset_weights()

    def _reset_weights(self):
        self._user_embeddings.weight.data.normal_(0, 1.0 / self._embedding_size)
        self._user_embeddings.weight.data[gc.PADDING_IDX].fill_(0)
        self._item_embeddings.weight.data.normal_(0, 1.0 / self._embedding_size)
        self._item_embeddings.weight.data[gc.PADDING_IDX].fill_(0)

        if self._sum_func: initializer.lecun_uniform_initialization(self._sum_func.weight)


        for i , fc in enumerate(self._fcs):
            if i == 0:
                fc.weight.data.copy_(my_utils.numpy2tensor(
                        np.concatenate(
                            (np.identity(self._embedding_size), -np.identity(self._embedding_size))
                            , axis=0).T  # initialy, it is subtractions of target user u and target item j, u-j
                        ))
            else:
                fc.weight.data.normal_(0, 1.0 / self._embedding_size)

    def _get_additional_l2_loss(self):
        item_reg_l2 = torch.norm(self._item_embeddings.weight)
        user_reg_l2 = torch.norm(self._user_embeddings.weight)
        return item_reg_l2 + user_reg_l2

    def forward(self, x=None, uids = None, iids = None):
        user_embeds = self._user_embeddings(uids)
        item_embeds = self._item_embeddings(iids)

        user_embeds = my_utils.flatten(user_embeds)
        item_embeds = my_utils.flatten(item_embeds)
        v = torch.cat([user_embeds, item_embeds], dim=1)

        for i, _fc in enumerate(self._fcs):
            v = _fc(v)
            v = self._non_linear(v) if self._non_linear else v
            if i == 0:
                v=self._dropout(v)

        v = self._dist_func(v)
        if self._ret_sum:
            v = self._sum_func(v) if self._sum_func else v.sum(dim=1)
            return -v
        else:
            return v




class SDMR(nn.Module):
    def __init__(self,
                 n_users, n_items, embedding_size,
                 item_seq_size, memory_size = None,
                 weight_tying = LAYER_WISE,
                 n_hops=4,
                 nonlinear_func_sdp='tanh',
                 nonlinear_func_sdm='tanh',
                 nonlinear_func_sdmr = 'none',
                 dropout_prob = 0.1,
                 gate_tying = GATE_GLOBAL,
                 beta = 0.9):
        super(SDMR, self).__init__()
        self._n_users = n_users
        self._n_items = n_items
        self._sdm = SDM(n_users=n_users, n_items=n_items, embedding_size=embedding_size,
                                     item_seq_size=item_seq_size, memory_size = memory_size, weight_tying=weight_tying,
                                     n_hops=n_hops, nonlinear_func=nonlinear_func_sdm,
                                     dropout_prob=dropout_prob, gate_tying=gate_tying,
                                     ret_sum = False
                                     )
        self._sdp = SDP(n_users=n_users, n_items=n_items, embedding_size=embedding_size,
                        nonlinear_func=nonlinear_func_sdp,
                        ret_sum=False)
        self._beta = beta

        self._sum_func = nn.Linear(2*embedding_size, 1) #combine two distances
        self._Wa = nn.Linear(2*embedding_size, 2*embedding_size)
        self._Va = nn.Linear(2*embedding_size, 2*embedding_size)
        if nonlinear_func_sdmr == 'tanh':
            self._nonlinear_func = nn.Tanh()
        elif nonlinear_func_sdmr == 'relu':
            self._nonlinear_func = F.relu
        else:
            self._nonlinear_func = None

        self._dropout = nn.Dropout(p=dropout_prob)

        self._reset_weights()

    def _get_additional_l2_loss(self):
        return (self._sdp._get_additional_l2_loss(), self._sdm._get_additional_l2_loss())

    def _attention(self, dist):
        scores = self._Va(F.tanh(
            self._dropout(
                self._Wa(dist)
            )
         )
        )
        weights = F.softmax(scores, dim=1)
        return torch.mul(weights, dist)

    def _reset_weights(self):
        initializer.lecun_uniform_initialization(self._sum_func.weight)

    def forward(self,  x, u, j):

        dist1 = self._sdp(x=None, uids=u, iids=j) #direct distance between target item j and user u
        dist2 = self._sdm(x, u, j)  # distance between target item j and prev consumed items x



        dist = torch.cat([dist1, dist2], dim=1) #batch x 2.embedding_size
        res = self._sum_func(dist)
        if self._nonlinear_func is not None:
            res = self._nonlinear_func(res)

        return -self._dropout(res)


        











