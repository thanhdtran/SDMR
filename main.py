import pytorch_utils as my_utils
import time
import data_loader as data_loader
import os
from model_rec import Rec
import argparse
import multiprocessing
import global_constants as gc

cpus_count = multiprocessing.cpu_count()

parser = argparse.ArgumentParser("Description: Running SDMR recommendation")
parser.add_argument('--saved_path', default='chk_points', type=str)
parser.add_argument('--load_best_chkpoint', default=0, type=int, help='loading the best checking point from previous run? (1=yes/0=no)')

parser.add_argument('--path', default='data', help='Input data path', type=str)
parser.add_argument('--dataset', default='ml1m', help='Dataset name', type=str)

parser.add_argument('--epochs', default=50, help='Number of epochs to run', type=int)
parser.add_argument('--batch_size', default=256, help='Batch size', type=int)
parser.add_argument('--num_factors', default=128, help='number of latent factors', type=int)

parser.add_argument('--n_hops', default=3, help='number of hops', type=int)

parser.add_argument('--reg_sdp', nargs='?', default='0.00001', help ='Regularization term in SDP model', type=str)
parser.add_argument('--reg_sdm', nargs='?',  default='0.000001', help ='Regularization term in SDM model', type=str)
parser.add_argument('--num_neg', default=4, type=int, help='Number of negative instances for each positive sample')
parser.add_argument('--lr', default=0.001, type=float, help = 'Learning rate')
parser.add_argument('--max_seq_len', type=int, default=5, help='maximum number of users/items to represents for a item/user')
parser.add_argument('--dropout', default=0.2, type=float, help='dropout probability for dropout layer')
parser.add_argument('--act_func_sdm', default='tanh', type=str, help='activation function [none, relu, tanh]')
parser.add_argument('--act_func_sdp', default='tanh', type=str)
parser.add_argument('--act_func_sdmr', default='relu', type=str)

parser.add_argument('--n_layers_sdp', default=1, type=int)

parser.add_argument('--beta', default=0.9, type=float, help='contribution of SDM module in the SDMR.')

parser.add_argument('--gate_tying', default='gate_global', type=str, help='gate weights tying [gate_global, gate_hop_specific]')

parser.add_argument('--topk', type=int, default=10, help='evaluation top K such as: NDCG@K, HITS@K')



parser.add_argument('--model', default='sdm', help='Selecting the model type [sdm, dmr, sdp], dmr= sdm + sdp', type=str) # dmr: deep metric memory recommender

parser.add_argument('--optimizer', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')

parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model (checkpoint).')

parser.add_argument('--eval', type=int, default=0,
                        help='Whether to evaluate the saved check points only or to re-build the model')


parser.add_argument('--cuda', type=int, default=0,
                        help='using cuda or not')

parser.add_argument('--seed', type=int, default=98765,
                        help='random seed')

parser.add_argument('--decay_step', type=int, default=20, help='how many steps to decay the learning rate')
parser.add_argument('--decay_weight', type=float, default=0.5, help ='percent of decaying')

parser.add_argument('--sdmr_retrain', type=int, default=0,
                        help='Whether to retrain the whole model or not, default is False.')

parser.add_argument('--sdm_init_transform_type', default = 'he-normal', type=str, help='init transformation weights Wa,'
                                                                                  'Wb, Wc, Wd. '
                                                                                  'options:identity, '
                                                                                  'he-normal, he-uniform, '
                                                                                  'normal, xavier, lecun')
# parser.add_argument('--share_all', default=1, type=int, help ='sharing weights among output in input memory')

args = parser.parse_args()
# args.layers = eval(args.layers)
args.reg_sdp = eval(args.reg_sdp) 
args.reg_sdm = eval(args.reg_sdm) 



#save to global constant
gc.BATCH_SIZE = args.batch_size
#gc.DEBUG = bool(args.debug)
# gc.SHARE_ALL = args.share_all
gc.SEED = args.seed
gc.model_type = args.model
gc.SDM_INIT_TRANSFORM_TYPE = args.sdm_init_transform_type
print 'SDM init type:', gc.SDM_INIT_TRANSFORM_TYPE
#if gc.DEBUG: args.epochs=2
#reuse from neural collaborative filtering
def load_rating_file_as_list(filename):
    ratingList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            ratingList.append([user, item])
            line = f.readline()
    return ratingList

#reuse from neural collaborative filtering
def load_negative_file(filename):
    negativeList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            negatives = []
            for x in arr[1: ]:
                negatives.append(int(x))
            uid, iid = arr[0].replace('(','').replace(')','').split(',')
            negativeList.append(negatives)
            #for j in range(100):
            #    negatives.append(int(iid))
            line = f.readline()
    return negativeList

train_file = os.path.join(args.path, args.dataset, '%s.train.rating'%args.dataset )
test_file = os.path.join(args.path, args.dataset, '%s.test.rating'%args.dataset )
test_neg_file = os.path.join(args.path, args.dataset, '%s.test.negative'%args.dataset )

testRatings = load_rating_file_as_list(test_file)
testNegatives = load_negative_file(test_neg_file)

print args


rec_model = Rec(
                n_factors = args.num_factors,
                n_iter = args.epochs,
                batch_size = args.batch_size,
                reg_sdp=args.reg_sdp,    # L2 regularization
                reg_sdm=args.reg_sdm,    # L2 regularization
                lr = args.lr, # learning_rate
                decay_step = args.decay_step, #step to decay the learning rate
                decay_weight = args.decay_weight, #percentage to decay the learning rat.
                optimizer_func = None,
                use_cuda = args.cuda,
                random_state = None,
                num_neg_samples = args.num_neg, #number of negative samples for each positive sample.
                dropout=args.dropout,
                n_hops=args.n_hops,
                activation_func_sdm = args.act_func_sdm,
                activation_func_sdp = args.act_func_sdp,
                n_layers_sdp=args.n_layers_sdp,
                gate_tying = args.gate_tying,
                model = args.model,
                beta = args.beta,
                args = args)


MAX_SEQ_LEN = args.max_seq_len
gc.MAX_SEQ_LEN = MAX_SEQ_LEN

t0 = time.time()
t1 = time.time()
print 'parsing data'
train_iteractions = data_loader.load_data(train_file, dataset=args.dataset)
t2 = time.time()
print 'loading data time: %d (seconds)'%(t2-t1)

print 'building the model'

try:

    rec_model.fit(train_iteractions,
                      topN=10,
                      testRatings=testRatings, testNegatives=testNegatives,
                      max_seq_len=MAX_SEQ_LEN, args=args)

except KeyboardInterrupt:
    print 'Exiting from training early'

t10 = time.time()
print 'Total running time: %d (seconds)'%(t10-t0)
