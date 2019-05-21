import pytorch_utils as my_utils
import numpy as np
import glob
import os
import torch
import global_constants as gc
class ModelBased(object):
    def __init__(self):
        self._net = None
        self._optimizer = None

    def _get_model_name(self, args):
        return str(args.model)
    def _make_model_desc(self, args, model = ''):
        # model_desc = '%s_'%args.model #sdp, sdm, or dmr
        model_desc = 'nfactors_%d' % args.num_factors
        model_name = args.model if model == '' else model
        if model_name == 'sdp':
            model_desc += '_reg_%s' % str(args.reg_sdp)
            model_desc += '_act_%s'% str(args.act_func_sdp)
            # print model_desc
            return model_desc
        elif model_name == 'sdm':
            model_desc += '_reg_%s' % str(args.reg_sdm)
            model_desc += '_num-hops_%d' % args.n_hops
            model_desc += '_%s'%args.gate_tying
            model_desc += '_seq%d' % args.max_seq_len
            model_desc += '_act_%s'% str(args.act_func_sdm)
            return model_desc
        elif model_name == 'dmmr':
            model_desc += '_sdp-reg_%s' % str(args.reg_sdp)
            model_desc += '_sdm-reg_%s' % str(args.reg_sdm)
            model_desc += '_num-hops_%d' % args.n_hops
            model_desc += '_%s' % args.gate_tying
            model_desc += '_seq%d' % args.max_seq_len
            model_desc += '_act_%s' % str(args.act_func_sdmr)
            return model_desc


    def save_checkpoint(self, args, best_hit, best_ndcg, epoch):
        model_state_dict = self._net.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'optimizer': self._optimizer.state_dict(),
            'settings': args,
            'best_hits': best_hit, #best validation/development hit
            'best_ndcg': best_ndcg, #best validation/development ndcg
            'epoch': epoch}

        model_name = '%s_%s_%s_hits_%.3f_ndcg_%.3f.chkpt' % (args.dataset,
                                                             args.model,
                                                             self._make_model_desc(args),
                                                             best_hit, best_ndcg)
        model_path = os.path.join(args.saved_path, model_name)
        torch.save(checkpoint, model_path)


    def load_checkpoint(self, args):
        lst_models = ['sdm', 'sdp']
        if args.eval:
            lst_models = ['sdm', 'sdp', 'dmmr']
        if args.model in lst_models:
            best_hits = 0.0
            best_ndcg = 0.0
            best_saved_file = ''

            # if len(os.listdir(args.saved_path)) > 0:
                # for fname in os.listdir(os.path.join(args.saved_path, '*%s*.chkpt'%args.conv_short_desc)): #must match model architecture
                # for fname in os.listdir(args.saved_path):
            saved_file_pattern = '%s_%s_%s*'%(args.dataset, args.model, self._make_model_desc(args))
            for filepath in glob.glob(os.path.join(args.saved_path,saved_file_pattern)):
                # filepath = os.path.join(args.saved_path, fname)
                if os.path.isfile(filepath):
                    print("=> loading checkpoint '{}'".format(filepath))
                    checkpoint = torch.load(filepath)
                    hits = float(checkpoint['best_hits'])
                    ndcg = float(checkpoint['best_ndcg'])
                    if hits > best_hits or (hits == best_hits and ndcg > best_ndcg):
                    # if ndcg > best_ndcg or (ndcg == best_ndcg and hits > best_hits):
                        best_saved_file=filepath
                        best_hits = hits
                        best_ndcg = ndcg
            if best_saved_file != '':
                checkpoint = torch.load(best_saved_file)
                self._net.load_state_dict(checkpoint['model'])
                self._optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                              .format(best_saved_file, checkpoint['epoch']))

            return (best_hits, best_ndcg)

        else:
            #dmr : Deep Metric memory Recommender
            # print 'here'
            #load best sdp checkpoint
            best_sdp_file = ''
            best_sdp_hits, best_sdp_ndcgs = 0,0
            sdp_files_pattern = '%s_sdp_%s*'%(args.dataset, self._make_model_desc(args, 'sdp'))
            # print sdp_files_pattern
            for filepath in glob.glob(os.path.join(args.saved_path, sdp_files_pattern)):
                print filepath
                if os.path.isfile(filepath):
                    print("=> loading checkpoint '{}'".format(filepath))
                    if args.cuda:
                        checkpoint = torch.load(filepath)
                    else:
                        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
                    hits = float(checkpoint['best_hits'])
                    ndcg = float(checkpoint['best_ndcg'])
                    if hits > best_sdp_hits or (hits == best_sdp_hits and ndcg > best_sdp_ndcgs):
                    # if ndcg > best_sdp_ndcgs or (ndcg == best_sdp_ndcgs and hits > best_sdp_hits):
                        best_sdp_file=filepath
                        best_sdp_hits = hits
                        best_sdp_ndcgs = ndcg

            #load best sdm checkpoint
            best_sdm_file = ''
            best_sdm_hits, best_sdm_ndcgs = 0, 0
            sdm_files_pattern = '%s_sdm_%s*' % (args.dataset, self._make_model_desc(args, 'sdm'))
            for filepath in glob.glob(os.path.join(args.saved_path, sdm_files_pattern)):
                if os.path.isfile(filepath):
                    print("=> loading checkpoint '{}'".format(filepath))
                    if args.cuda:
                        checkpoint = torch.load(filepath)
                    else:
                        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)

                    hits = float(checkpoint['best_hits'])
                    ndcg = float(checkpoint['best_ndcg'])
                    if hits > best_sdm_hits or (hits == best_sdm_hits and ndcg > best_sdm_ndcgs):
                    # if ndcg > best_sdm_ndcgs or (ndcg == best_sdm_ndcgs and hits > best_sdm_hits):
                        best_sdm_file = filepath
                        best_sdm_hits = hits
                        best_sdm_ndcgs = ndcg


            #now loading best checkpoints from sdp and sdm for dmr:
            if best_sdp_file != '':
                #load checkpoint into cpu
                if args.cuda:
                    checkpoint = torch.load(best_sdp_file)
                else:
                    checkpoint = torch.load(best_sdp_file, map_location=lambda storage, loc: storage)
                self._net._sdp.load_state_dict(checkpoint['model'])
                # self._optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                                .format(best_sdp_file, checkpoint['epoch']))
              
                # no train the sdm and sdp?
                for params in self._net._sdp.parameters():
                    params.requires_grad = bool(args.sdmr_retrain)


                # self._net._sdp._user_embeddings.requires_grad = False
                # self._net._sdp._item_embeddings.requires_grad = False

            if best_sdm_file != '':
                #load checkpoint into cpu:
                if args.cuda:
                    checkpoint = torch.load(best_sdm_file)
                else:
                    checkpoint = torch.load(best_sdm_file, map_location=lambda storage, loc: storage)

                self._net._sdm.load_state_dict(checkpoint['model'])


                print("=> loaded checkpoint '{}' (epoch {})"
                                 .format(best_sdm_file, checkpoint['epoch']))
                #no train the sdm and sdp?
                for params in self._net._sdm.parameters():
                    params.requires_grad = False

                # for A_memory in self._net._sdm.A_memories:
                #     A_memory._user_embeddings.requires_grad = False
                #     A_memory._item_embeddings.requires_grad = False
                # for C_memory in self._net._sdm.C_memories:
                #     C_memory._user_embeddings.requires_grad = False
                #     C_memory._item_embeddings.requires_grad = False
            sum_weights = np.concatenate(
                                            (
                                                (1-args.beta) * my_utils.tensor2numpy(self._net._sdp._sum_func.weight.data),
                                                args.beta * my_utils.tensor2numpy(self._net._sdm._sum_func.weight.data)
                                            ),
                                            axis=1
                                         )
            sum_bias = (1-args.beta) * my_utils.tensor2numpy(self._net._sdp._sum_func.bias.data) + \
                       args.beta * my_utils.tensor2numpy(self._net._sdm._sum_func.bias.data)

            self._net._sum_func.weight.data.copy_(my_utils.numpy2tensor(sum_weights))
            self._net._sum_func.bias.data.copy_(my_utils.numpy2tensor(sum_bias))
            return 0,0




