import torch.nn.functional as F
import global_constants as gc
import sampler as my_sampler
import time
import pytorch_utils as my_utils
import numpy as np
import torch.optim as optim
import losses as my_losses
import torch
import evaluator as my_evaluator
from torch.autograd import Variable
from model_based import ModelBased
from torch.optim.lr_scheduler import StepLR
from MemNet import SDP, SDM, SDMR
import MemNet as memnet
import os

class Rec(ModelBased):
    def __init__(self,
                 n_factors = 8,
                 n_iter = 20,
                 batch_size = 256,
                 reg_sdp= 0.00001,  # L2, L1 regularization
                 reg_sdm = 0.0001,    # L2, L1 regularization
                 lr = 1e-2, # learning_rate
                 decay_step = 20,
                 decay_weight= 0.5,
                 optimizer_func = None,
                 use_cuda = False,
                 random_state = None,
                 num_neg_samples = 4, #number of negative samples for each positive sample.
                 dropout=0.2,
                 n_hops=3,
                 activation_func_sdm = 'tanh',
                 activation_func_sdp = 'tanh',
                 n_layers_sdp=1,
                 gate_tying=memnet.GATE_GLOBAL,
                 model='sdm',
                 beta=0.9, args=None
                 ):
        super(Rec, self).__init__()


        self._n_factors = n_factors
        self._embedding_size = n_factors

        self._n_iters = n_iter
        self._batch_size = batch_size
        self._lr = lr
        self._decay_step = decay_step
        self._decay_weight = decay_weight

        self._reg_sdp = reg_sdp
        self._reg_sdm = reg_sdm
        self._optimizer_func = optimizer_func

        self._use_cuda = use_cuda
        self._random_state = random_state or np.random.RandomState()
        self._num_neg_samples = num_neg_samples

        self._n_users = None
        self._n_items = None
        self._lr_schedule = None
        self._loss_func = None
        self._n_hops = n_hops
        self._dropout = dropout

        self._gate_tying = gate_tying
        self._model = model
        self._beta = beta


        #my_utils.set_seed(self._random_state.randint(-10**8, 10**8), cuda=self._use_cuda)
        my_utils.set_seed(gc.SEED)

        self._activation_func_sdm = activation_func_sdm
        self._activation_func_sdp = activation_func_sdp
        self._n_layers_sdp = n_layers_sdp


        self._sampler = my_sampler.Sampler()
        self._args = args

        #create checkpoint directory
        if not os.path.exists(args.saved_path):
            os.mkdir(args.saved_path)


        if not os.path.exists(args.saved_path):
            os.mkdir(args.saved_path)

    def _has_params(self):

        for params in self._net.parameters():
            if params.requires_grad:
                return True
        if self._model == 'sdmr':
            for params in self._net._sdp.parameters():
                if params.requires_grad:
                    return True
            for params in self._net._memnet.parameters():
                if params.requires_grad:
                    return True
        return False

    def _is_initialized(self):
        return self._net is not None

    def _initialize(self, interactions, max_seq_len=-1):
        self._interactions = interactions
        self._max_user_seq_len = interactions._max_len_user_seq if max_seq_len == -1 else max_seq_len
        # self._max_item_seq_len = interactions._max_len_item_seq if max_seq_len == -1 else max_seq_len

        (self._n_users, self._n_items) = (interactions.num_users, interactions.num_items)
        print 'total users: %d, total items: %d'%(self._n_users, self._n_items)
        if self._model == 'sdm':
            self._net = SDM(n_users=self._n_users, n_items=self._n_items, embedding_size=self._embedding_size,
                                  item_seq_size=self._max_user_seq_len, n_hops=self._n_hops,
                                  nonlinear_func=self._activation_func_sdm, dropout_prob=self._dropout,
                                  gate_tying=self._gate_tying
                                  )
        elif self._model == 'sdp':
            self._net = SDP(n_users=self._n_users, n_items=self._n_items, embedding_size=self._embedding_size,
                             nonlinear_func=self._activation_func_sdp,
                             num_layers=self._n_layers_sdp,
                             dropout_prob=self._dropout)
        else:
            self._net = SDMR(n_users=self._n_users, n_items=self._n_items, embedding_size=self._embedding_size,
                             item_seq_size=self._max_user_seq_len, n_hops=self._n_hops,
                             nonlinear_func_sdm=self._args.act_func_sdm,
                             nonlinear_func_sdp=self._args.act_func_sdp,
                             nonlinear_func_sdmr=self._args.act_func_sdmr,
                             dropout_prob=self._dropout,
                             gate_tying=self._gate_tying,
                             beta=self._beta
                             )

        self._net = my_utils.gpu(self._net, self._use_cuda)
        reg = 1e-6
        if self._args.model == 'sdp': reg = self._args.reg_sdp
        elif self._args.model == 'sdm': reg = self._args.reg_sdm
        else: reg = 1e-6

        print 'setting reg to :', reg

        if self._optimizer_func is None:
            self._optimizer = optim.Adam(
                self._net.parameters(),
                weight_decay=reg,
                lr=self._lr
            )
            decay_step = self._decay_step
            decay_percent = self._decay_weight
            self._lr_schedule = StepLR(self._optimizer, step_size=decay_step, gamma=decay_percent)

        else:
            self._optimizer = self._optimizer_func(self._net.parameters())

        self._loss_func = my_losses.bpr_loss


        self._sampler.init_user_item_seqs(interactions._user_all_items, interactions.num_users, interactions.num_items)

    def fit(self, interactions, topN=10,
            testRatings=None, testNegatives=None,
            max_seq_len=5, args=None):

        if not self._is_initialized():
            self._initialize(interactions, max_seq_len=max_seq_len)

        user_ids = interactions._user_ids.astype(np.int64)
        item_ids = interactions._item_ids.astype(np.int64)

        best_hit = 0.0
        best_ndcg = 0.0
        best_epoch = 0
        test_hit, test_ndcg = 0.0, 0.0

        if args.load_best_chkpoint > 0:

            best_hit, best_ndcg = self.load_checkpoint(args)
            print 'Results from best checkpoints ...'
            t1 = time.time()
            hits, ndcgs = my_evaluator.evaluate(self, testRatings, testNegatives, topN)
            t2 = time.time()
            eval_time = t2-t1
            best_hit, best_ndcg = hits, ndcgs
            print('| Eval time: %d '
                  '| Test hits@%d = %.3f | Test ndcg@%d = %.3f |'
                  % (eval_time, topN, hits, topN, ndcgs))
            topN = 10
            print 'End!'

        if args.eval:
            print 'Evaluation using the saved checkpoint done!'
            return

        if self._has_params():
            for epoch in range(self._n_iters):

                self._lr_schedule.step(epoch)

                self._net.train()  # set training environment

                users, items = my_utils.shuffle(user_ids,
                                                item_ids,
                                                random_state=self._random_state)
                users = np.asarray(users)
                items = np.asarray(items)
                neg_items = self._sampler.random_neg_items(users, num_neg=args.num_neg)


                # copy to GPU:
                user_vars = Variable(my_utils.gpu(my_utils.numpy2tensor(users).type(torch.LongTensor),
                                                  use_cuda=self._use_cuda))
                item_vars = Variable(my_utils.gpu(my_utils.numpy2tensor(items).type(torch.LongTensor),
                                                  use_cuda=self._use_cuda))
                neg_item_vars = Variable(my_utils.gpu(my_utils.numpy2tensor(neg_items).type(torch.LongTensor),
                                                           use_cuda=self._use_cuda))

                context_vars = None
                #if args.model != 'sdp':
                contexts = interactions.get_batch_seqs(users, items,
                                                            max_seq_len=self._max_user_seq_len)
                context_vars = Variable(my_utils.gpu(my_utils.numpy2tensor(contexts).type(torch.LongTensor),
                                                     self._use_cuda), requires_grad=False)





                epoch_loss = 0.0

                t1 = time.time()
                total_interactions = 0
                for (minibatch_idx, (batch_users, batch_items, batch_neg_items, batch_contexts )) in enumerate(
                                                                                        my_utils.minibatch(
                                                                                            user_vars,
                                                                                            item_vars,
                                                                                            neg_item_vars,
                                                                                            context_vars,
                                                                                            batch_size=self._batch_size)):
                    total_interactions += len(batch_users)  # or batch_size
                    if args.model == 'sdp':
                        batch_contexts = None

                    positive_prediction = self._net(batch_contexts, batch_users, batch_items)

                    negative_prediction = self._get_neg_pred(batch_contexts, batch_users, batch_neg_items)
                    self._optimizer.zero_grad()

                    loss = self._loss_func(positive_prediction,
                                           negative_prediction
                                           )


                    epoch_loss += my_utils.cpu(loss).data.numpy()

                    loss.backward()

                    self._optimizer.step()


                epoch_loss = epoch_loss / total_interactions
                t2 = time.time()
                epoch_train_time = t2 - t1

                t1 = time.time()

                hits, ndcgs = my_evaluator.evaluate(self, testRatings, testNegatives, topN)
                t2 = time.time()
                eval_time = t2 - t1
                print('|Epoch %d | Train time: %d | Train loss: %.3f | Eval time: %d '
                      '| Test hits@%d = %.3f | Test ndcg@%d = %.3f |'
                      % (epoch, epoch_train_time, epoch_loss, eval_time, topN, hits, topN, ndcgs))
                if hits > best_hit or (hits == best_hit and ndcgs >= best_ndcg):
                    best_hit, best_ndcg, best_epoch = hits, ndcgs, epoch
                    if args.out:
                        self.save_checkpoint(args, hits, ndcgs, epoch)  # save best params

                if np.isnan(epoch_loss) or epoch_loss == 0.0:
                    raise ValueError('Degenerate epoch loss: {}'
                                     .format(epoch_loss))

            print ('Best result: '
                   '| test hits@%d = %.3f | test ndcg@%d = %.3f | epoch = %d' % (topN,  best_hit, topN, best_ndcg,
                                                                                 best_epoch))
    def _get_neg_pred(self, item_seqs_var, batch_user_var, batch_neg_item_vars):
        '''
        user_ids are numpy data
        :param user_ids:
        :param user_seqs:
        :return:
        '''

        negative_prediction = None
        for i in range(self._args.num_neg):
            tmp_negative_prediction = self._net(item_seqs_var, batch_user_var, batch_neg_item_vars[:, i])
            if negative_prediction is None: negative_prediction = tmp_negative_prediction
            else: negative_prediction = torch.max(negative_prediction, tmp_negative_prediction)

        return negative_prediction



    def predict(self, user_ids, item_ids):
        max_seq_len = self._max_user_seq_len
        self._net.train(False)

        item_seqs = self._interactions.get_batch_seqs(user_ids, item_ids, max_seq_len=max_seq_len)
        item_seqs = Variable(my_utils.gpu(my_utils.numpy2tensor(item_seqs).type(torch.LongTensor),
                                          self._use_cuda), requires_grad=False)
        out = self._net(item_seqs,
                        Variable(my_utils.gpu(my_utils.numpy2tensor(user_ids), self._use_cuda)),
                        Variable(my_utils.gpu(my_utils.numpy2tensor(item_ids), self._use_cuda)))
        return my_utils.cpu(out).detach().data.numpy().flatten()
