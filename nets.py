import collections
import torch.nn as nn
import torch
import layers as layers
import pytorch_utils as my_utils
import torch.nn.functional as F
import norm_layer as norm_layer
import math
import global_constants as gc

class MF(nn.Module):
    def __init__(self, n_users, n_items, n_factors=8, pretrained_user_embeddings = None, pretrained_item_embeddings = None):
        super(MF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors

        if pretrained_user_embeddings is not None:
            self.user_embeddings = pretrained_user_embeddings
        else:
            self.user_embeddings = layers.ScaledEmbedding(n_users, n_factors)

        if pretrained_item_embeddings is not None:
            self.item_embeddings = pretrained_item_embeddings
        else:
            self.item_embeddings = layers.ScaledEmbedding(n_items, n_factors)

        self.user_bias = layers.ZeroEmbedding(n_users, 1)
        self.item_bias = layers.ZeroEmbedding(n_items, 1)

    def forward(self, uids, iids):
        user_embeds = self.user_embeddings(uids) #first dimension is batch size
        item_embeds = self.item_embeddings(iids)

        user_embeds = my_utils.flatten(user_embeds)
        item_embeds = my_utils.flatten(item_embeds)

        user_bias = self.user_bias(uids)
        item_bias = self.item_bias(iids)

        user_bias = my_utils.flatten(user_bias) #bias has size batch_size * 1
        item_bias = my_utils.flatten(item_bias) #bias has size batch_size * 1

        dot_product = (user_embeds * item_embeds).sum(1) #first dimension is batch_size, return dimension (batch_size)
        # dot_product = torch.mul(user_embeds, item_embeds).sum(1)  # first dimension is batch_size

        return dot_product + user_bias.squeeze() + item_bias.squeeze()

class GMF(nn.Module):
    def __init__(self, n_users, n_items, n_factors=8, pretrained_user_embeddings = None, pretrained_item_embeddings = None):
        super(GMF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors

        if pretrained_user_embeddings is not None:
            self.user_embeddings = pretrained_user_embeddings
        else:
            self.user_embeddings = layers.ScaledEmbedding(n_users, n_factors)

        if pretrained_item_embeddings is not None:
            self.item_embeddings = pretrained_item_embeddings
        else:
            self.item_embeddings = layers.ScaledEmbedding(n_items, n_factors)

        self.fc = nn.Linear(n_factors, 1, bias=True)

    def forward(self, uids, iids):
        user_embeds = self.user_embeddings(uids)  # first dimension is batch size
        item_embeds = self.item_embeddings(iids)

        user_embeds = my_utils.flatten(user_embeds)
        item_embeds = my_utils.flatten(item_embeds)

        wise_product = torch.mul(user_embeds, item_embeds)

        # return F.sigmoid(self.fc(wise_product))
        return self.fc(wise_product)
class GRMF(nn.Module):
    def __init__(self, n_users, n_items, n_factors=8, pretrained_user_embeddings = None, pretrained_item_embeddings = None):
        super(GRMF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors

        if pretrained_user_embeddings is not None:
            self.user_embeddings = pretrained_user_embeddings
        else:
            self.user_embeddings = layers.ScaledEmbedding(n_users, n_factors)

        if pretrained_item_embeddings is not None:
            self.item_embeddings = pretrained_item_embeddings
        else:
            self.item_embeddings = layers.ScaledEmbedding(n_items, n_factors)

        self.fc1 = nn.Linear(2*n_factors, n_factors, bias=True)

    def forward(self, uids, iids):
        user_embeds = self.user_embeddings(uids)  # first dimension is batch size
        item_embeds = self.item_embeddings(iids)

        user_embeds = my_utils.flatten(user_embeds)
        item_embeds = my_utils.flatten(item_embeds)
        v = torch.cat([user_embeds, item_embeds], dim=1)  # dim = 0 is the batch_size

        v = self.fc1(v)
        v = v**2 #square
        v = torch.sum(v, dim=1)
        v = torch.sqrt(v)
        return -v

        # return F.sigmoid(self.fc(wise_product))
        # return self.fc(wise_product)

class MLP(nn.Module):
    def __init__(self, n_users, n_items, layers_size = [64,32,16,8], pretrained_user_embeddings = None, pretrained_item_embeddings = None):
        super(MLP, self).__init__()
        self._n_users = n_users
        self._n_items = n_items
        if pretrained_user_embeddings is not None:
            self._user_embeddings = pretrained_user_embeddings
        else:
            self._user_embeddings = layers.ScaledEmbedding(n_users, layers_size[0]/2)

        if pretrained_item_embeddings is not None:
            self._item_embeddings = pretrained_item_embeddings
        else:
            self._item_embeddings = layers.ScaledEmbedding(n_items, layers_size[0]/2)

        self._fcs = nn.ModuleList()
        prev_layer_size = layers_size[0]
        for ls in layers_size:
            fc = nn.Linear(prev_layer_size, ls)
            self._fcs.append(fc)
            self._fcs.append(nn.ReLU())
            prev_layer_size = ls

        self._fcs.append(nn.Linear(prev_layer_size, 1, bias=True)) #final classification layer

        # self._relu = nn.ReLU()

    def forward(self, uids, iids):
        user_embeds = (self._user_embeddings(uids))  # first dimension is batch size
        item_embeds = (self._item_embeddings(iids))

        v = torch.cat([user_embeds, item_embeds], dim=1) #dim = 0 is the batch_size

        for i in range(len(self._fcs)):
            v = self._fcs[i](v)
            # v = F.relu(v)

        return v.squeeze() #v has dimension of (batch_size, 1)

class NCF(nn.Module):
    def __init__(self, n_users, n_items, n_factors, layers_size = [64,32,16,8],
                 pretrained_mlp_user_embeddings=None, pretrained_mlp_item_embeddings=None,
                 pretrained_gmf_user_embeddings=None, pretrained_gmf_item_embeddings=None):
        super(NCF, self).__init__()
        self._n_users = n_users
        self._n_items = n_items
        self._n_factors = n_factors

        if pretrained_mlp_user_embeddings is not None:
            self._mlp_user_embeddings = pretrained_mlp_user_embeddings
        else:
            self._mlp_user_embeddings = layers.ManualEmbedding(n_users, layers_size[0]/2)

        if pretrained_mlp_item_embeddings is not None:
            self._mlp_item_embeddings = pretrained_mlp_item_embeddings
        else:
            self._mlp_item_embeddings = layers.ManualEmbedding(n_items, layers_size[0]/2)

        if pretrained_gmf_user_embeddings is not None:
            self._gmf_user_embeddings = pretrained_gmf_user_embeddings
        else:
            self._gmf_user_embeddings = layers.ManualEmbedding(n_users, n_factors)

        if pretrained_gmf_item_embeddings is not None:
            self._gmf_item_embeddings = pretrained_gmf_item_embeddings
        else:
            self._gmf_item_embeddings = layers.ManualEmbedding(n_items, n_factors)

        #fully connected layers for mlp
        self._fcs = nn.ModuleList()
        prev_layer_size = layers_size[0]
        for ls in layers_size:
            fc = nn.Linear(prev_layer_size, ls)
            self._fcs.append(fc)
            self._fcs.append(nn.ReLU())
            prev_layer_size = ls

        self._last_fc = nn.Linear(prev_layer_size + n_factors, 1, bias=True)

    def forward(self, uids, iids):
        mlp_user_embeds = self._mlp_user_embeddings(uids).squeeze()
        mlp_item_embeds = self._mlp_item_embeddings(iids).squeeze()

        mlp_concate = torch.cat([mlp_user_embeds, mlp_item_embeds], dim=1)
        for i in range(len(self._fcs)):
            mlp_concate = self._fcs[i](mlp_concate)

        #now mlp_concate has dimension (batchsize, layers_size[-1]), here layers_size[-1] = 8 --> (batch_size, 8)

        #gmf layer
        gmf_user_embeds = self._gmf_user_embeddings(uids).squeeze() #(batch_size, 8)
        gmf_item_embeds = self._gmf_item_embeddings(iids).squeeze() #(batch_size, 8)
        gmf_layer = torch.mul(gmf_user_embeds, gmf_item_embeds)

        concate = torch.cat([mlp_concate, gmf_layer], dim = 1)
        return self._last_fc(concate)


class RME(nn.Module):
    def __init__(self, n_users, n_items, n_factors, layers_size = [64,32,16,8],
                 pretrained_mlp_user_embeddings=None, pretrained_gmf_user_embeddings=None,
                 pretrained_mlp_item_embeddings=None, pretrained_gmf_item_embeddings=None,
                 pretrained_mlp_liked_item_context_embeddings=None, pretrained_gmf_liked_item_context_embeddings=None):
        super(RME, self).__init__()

        self._n_users = n_users
        self._n_items = n_items
        self._n_factors = n_factors
        self._layers_size = layers_size

        self._UI_model = NCF(n_users, n_items, n_factors) #user-item interaction matrix learner.
        self._CII_model = NCF(n_items, n_items, n_factors,
                              pretrained_gmf_user_embeddings=self._UI_model._gmf_item_embeddings,
                              pretrained_mlp_user_embeddings=self._UI_model._mlp_item_embeddings
                              ) #coliked item-item SPPMIT matrix learner
        #self._MF_model share similar item embeddings with self._UI_model
        self._DII_model = NCF(n_items, n_items, n_factors,
                              pretrained_gmf_user_embeddings=self._UI_model._gmf_item_embeddings,
                              pretrained_mlp_user_embeddings=self._UI_model._mlp_item_embeddings
                              ) #codisliked-item-item SPPMIT matrix learner
        self._UU_model = NCF(n_users, n_users, n_factors,
                             pretrained_gmf_user_embeddings=self._UI_model._gmf_user_embeddings,
                             pretrained_mlp_user_embeddings=self._UI_model._mlp_user_embeddings)


    def _autograd_user_embeds(self, attach=True):
        my_utils.set_requires_grad(self._UI_model._gmf_user_embeddings, attach)
        my_utils.set_requires_grad(self._UI_model._mlp_user_embeddings, attach)

    def _autograd_user_context_embeds(self, attach=True):
        my_utils.set_requires_grad(self._UU_model.item_embeddings, attach) #item embeddings is the user-context embeddings.

    def _autograd_item_embeds(self, attach=True):
        my_utils.set_requires_grad(self._UI_model._mlp_item_embeddings, attach)
        my_utils.set_requires_grad(self._UI_model._gmf_item_embeddings, attach)

    def _autograd_liked_item_context_embeds(self, attach=True):
        my_utils.set_requires_grad(self._CII_model.item_embeddings, attach) #item embeddings is liked item-context embeddings.

    def _autograd_disliked_item_context_embeds(self, attach=True):
        my_utils.set_requires_grad(self._DII_model.item_embeddings, attach) #item embeddings is disliked item-context embeddings.


    def _gen_user_item(self, uids, iids):
        return self._UI_model(uids, iids)


    def _gen_liked_item_item(self, iids, cicids):
        return self._CII_model(iids, cicids)


    def _gen_disliked_item_item(self, iids, dicids):
        return self._DII_model(iids, dicids)

    def _gen_user_user(self, uids, ucids):
        return self._UU_model(uids, ucids)

    def forward(self, uids, iids, cicids, type='user-item'):

        # if type == 'user': # update user embeddings only
        #     self._autograd_user_embeds(attach=True) #do grad for users
        #     self._autograd_item_embeds(attach=False) #detach item grad
        #     self._autograd_liked_item_context_embeds(attach=False) #detach item context grad
        #     y1 = self._gen_user_item(uids, iids)
        #     return y1
        #
        # if type == 'item':
        #     self._autograd_user_embeds(attach=False)  # do grad for users
        #     self._autograd_item_embeds(attach=True)  # detach item grad
        #     self._autograd_liked_item_context_embeds(attach=False)  # detach item context grad
        #     y1 = self._gen_user_item(uids, iids)
        #     y2 = self._gen_liked_item_item(iids, cicids)
        #     return y1, y2
        #
        # if type == 'liked-item-context':
        #     self._autograd_user_embeds(attach=False)  # do grad for users
        #     self._autograd_item_embeds(attach=False)  # detach item grad
        #     self._autograd_liked_item_context_embeds(attach=True)  # detach item context grad
        #     y1 = self._gen_liked_item_item(iids, cicids)
        #     return y1

        if type == 'user-item':
            #update user-item interaction matrix only
            return self._gen_user_item(uids, iids)
        if type == 'liked-item-item':
            #update liked-item-item interaction matrix only
            return self._gen_liked_item_item(iids, cicids)


class CNNNet(nn.Module):
    """
    Module representing users through stacked causal atrous convolutions ([3]_, [4]_).

    To represent a sequence, it runs a 1D convolution over the input sequence,
    from left to right. At each timestep, the output of the convolution is
    the representation of the sequence up to that point. The convolution is causal
    because future states are never part of the convolution's receptive field;
    this is achieved by left-padding the sequence.

    In order to increase the receptive field (and the capacity to encode states
    further back in the sequence), one can increase the kernel width, stack
    more layers, or increase the dilation factor.
    Input dimensionality is preserved from layer to layer.

    Residual connections can be added between all layers.

    During training, representations for all timesteps of the sequence are
    computed in one go. Loss functions using the outputs will therefore
    be aggregating both across the minibatch and across time in the sequence.

    Parameters
    ----------

    num_items: int
        Number of items to be represented.
    embedding_dim: int, optional
        Embedding dimension of the embedding layer, and the number of filters
        in each convolutional layer.
    kernel_width: tuple or int, optional
        The kernel width of the convolutional layers. If tuple, should contain
        the kernel widths for all convolutional layers. If int, it will be
        expanded into a tuple to match the number of layers.
    dilation: tuple or int, optional
        The dilation factor for atrous convolutions. Setting this to a number
        greater than 1 inserts gaps into the convolutional layers, increasing
        their receptive field without increasing the number of parameters.
        If tuple, should contain the dilation factors for all convolutional
        layers. If int, it will be expanded into a tuple to match the number
        of layers.
    num_layers: int, optional
        Number of stacked convolutional layers.
    nonlinearity: string, optional
        One of ('tanh', 'relu'). Denotes the type of non-linearity to apply
        after each convolutional layer.
    residual_connections: boolean, optional
        Whether to use residual connections between convolutional layers.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.

    References
    ----------

    .. [3] Oord, Aaron van den, et al. "Wavenet: A generative model for raw audio."
       arXiv preprint arXiv:1609.03499 (2016).
    .. [4] Kalchbrenner, Nal, et al. "Neural machine translation in linear time."
       arXiv preprint arXiv:1610.10099 (2016).
    """

    def __init__(self, num_items,
                 embedding_dim=32,
                 kernel_width=3,
                 dilation=1,
                 num_layers=1,
                 nonlinearity='tanh',
                 residual_connections=True,
                 sparse=False,
                 benchmark=True,
                 item_embedding_layer=None):

        super(CNNNet, self).__init__()



        self.embedding_dim = embedding_dim
        self.kernel_width = my_utils._to_iterable(kernel_width, num_layers)
        self.dilation = my_utils._to_iterable(dilation, num_layers)
        if nonlinearity == 'tanh':
            self.nonlinearity = F.tanh
        elif nonlinearity == 'relu':
            self.nonlinearity = F.relu
        else:
            raise ValueError('Nonlinearity must be one of (tanh, relu)')
        self.residual_connections = residual_connections

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = layers.ScaledEmbedding(num_items, embedding_dim, padding_idx=gc.PADDING_IDX)

        self.item_biases = layers.ZeroEmbedding(num_items, 1, padding_idx=gc.PADDING_IDX)

        self.cnn_layers = [
            nn.Conv2d(embedding_dim,
                      embedding_dim,
                      (_kernel_width, 1),
                      dilation=(_dilation, 1)) for
            (_kernel_width, _dilation) in zip(self.kernel_width,
                                              self.dilation)
        ]

        for i, layer in enumerate(self.cnn_layers):
            self.add_module('cnn_{}'.format(i),
                            layer)

    def user_representation(self, item_sequences):
        """
        Compute user representation from a given sequence.

        Returns
        -------

        tuple (all_representations, final_representation)
            The first element contains all representations from step
            -1 (no items seen) to t - 1 (all but the last items seen).
            The second element contains the final representation
            at step t (all items seen). This final state can be used
            for prediction or evaluation.
        """

        # Make the embedding dimension the channel dimension
        sequence_embeddings = (self.item_embeddings(item_sequences)
                               .permute(0, 2, 1))
        # Add a trailing dimension of 1
        sequence_embeddings = (sequence_embeddings
                               .unsqueeze(3))

        # Pad so that the CNN doesn't have the future
        # of the sequence in its receptive field.
        receptive_field_width = (self.kernel_width[0] +
                                 (self.kernel_width[0] - 1) *
                                 (self.dilation[0] - 1))

        x = F.pad(sequence_embeddings,
                  (0, 0, receptive_field_width, 0))

        x = self.nonlinearity(self.cnn_layers[0](x))

        if self.residual_connections:
            residual = F.pad(sequence_embeddings,
                             (0, 0, 1, 0))
            x = x + residual

        for (cnn_layer, kernel_width, dilation) in zip(self.cnn_layers[1:],
                                                       self.kernel_width[1:],
                                                       self.dilation[1:]):
            receptive_field_width = (kernel_width +
                                     (kernel_width - 1) *
                                     (dilation - 1))
            residual = x
            x = F.pad(x, (0, 0, receptive_field_width - 1, 0))
            x = self.nonlinearity(cnn_layer(x))

            if self.residual_connections:
                x = x + residual

        x = x.squeeze(3)

        return x[:, :, :-1], x[:, :, -1]

    def forward(self, user_representations, targets):
        """
        Compute predictions for target items given user representations.

        Parameters
        ----------

        user_representations: tensor
            Result of the user_representation_method.
        targets: tensor
            Minibatch of item sequences of shape
            (minibatch_size, sequence_length).

        Returns
        -------

        predictions: tensor
            Of shape (minibatch_size, sequence_length).
        """

        target_embedding = (self.item_embeddings(targets)
                            .permute(0, 2, 1)
                            .squeeze())
        target_bias = self.item_biases(targets).squeeze()

        dot = ((user_representations * target_embedding)
               .sum(1)
               .squeeze())

        return target_bias + dot


class CNNUserNet(nn.Module):
    def __init__(self, n_users, embed_dims, max_positions=1024,
                 convolutions =((512, 3),)*2, # input_channels = 512, apply kernel size of 3x3
                 dropout = 0.2,
                 convolution_type = 'vanilla', #glu (gated linear unit or just vanilla?)
                 non_linear='relu',
                 norm_type='layer', #or layer
                 use_positional_embed=True,
                 use_residual_connection=True,
                 use_depwise_sep=True
                 ):
        '''

        :param n_users:
        :param embed_dims:
        :param max_positions:
        :param dropout:
        :param use_positional_embed: whether or not using the positional embeddings
        '''
        
        super(CNNUserNet, self).__init__()

        self._dropout = dropout
        self._n_users = n_users
        self._embed_dim = embed_dims
        self._max_positions = max_positions
        self._use_pos_emb = use_positional_embed
        self._use_residual_connection = use_residual_connection
        self._convolution_type = convolution_type
        self._norm_type = norm_type
        self._dropoutlayer = nn.Dropout(p=dropout) #use dropout in front of linear layer

        #define user embeddings space
        # self.user_embeds = layers.ManualEmbedding(n_users, embed_dims, padding_idx=gc.PADDING_IDX)
        self.user_embeds = nn.Embedding(n_users, embed_dims, padding_idx=gc.PADDING_IDX)
        if use_positional_embed:

            self.pos_embeds = layers.PositionEmbedding(max_positions, embed_dims,
                                                       padding_idx=gc.PADDING_IDX, left_pad=True)
        self._in_channels = convolutions[0][0]
        input_channels = self._in_channels

        # self.fc1 = layers.Linear(embed_dims, self._in_channels, dropout=dropout) # need a fully connected layer to map embed_dim to in_channels
                                                                                           # if they are different (otherwise, no need)
        self.fc1 = nn.Linear(embed_dims, self._in_channels)
        self.norm1 = norm_layer.LayerNorm(input_channels)


        self.projections = nn.ModuleList() #for matching size of residuals and convolution
        self.convolutions = nn.ModuleList()
        self.non_linearities = nn.ModuleList()
        self.pads = []
        self.use_residuals = []
        self.norms = nn.ModuleList()

        input_channels_sizes = [input_channels]
        for (output_channels, kernel_size, use_residual) in convolutions:
            if input_channels != output_channels and use_residual:
                #need a fully connected layer to map from input_channels to output_channels before convoluting
                fc =  layers.Linear(input_channels, output_channels)
                self.projections.append(fc)
                # self.projections.append(None)
            else:
                self.projections.append(None)

            if use_depwise_sep: groups = input_channels #setting for depth-wise separable convolution.
            else: groups = 1

            if use_residual_connection:
                #need to pad, input is BxCxT but after convolve --> reduce T
                if isinstance(kernel_size, collections.Iterable): kernel_size_on_T = kernel_size[0]
                else: kernel_size_on_T = kernel_size
                stride = 1 # if stride is not 1
                n_pads = (kernel_size_on_T - 1)/2 #number of paddings in each of two sides (left and right)
                n_pads = n_pads * stride
                self.pads.append(n_pads)

            self.convolutions.append(layers.ConvTBC(input_channels, output_channels, kernel_size, groups=groups,
                                                    dropout=dropout, padding=gc.PADDING_IDX))
            self.norms.append(
                # None
                # nn.BatchNorm1d(gc.BATCH_SIZE)
                nn.LayerNorm(output_channels)
                # norm_layer.LayerNorm(output_channels)
            )

            if non_linear == 'relu':
                self.non_linearities.append(nn.ReLU())
            elif non_linear == 'tanh':
                self.non_linearities.append(nn.Tanh())
            else:
                self.non_linearities.append(None)



            # self.convolutions.append(nn.Conv1d(input_channels, output_channels,kernel_size,
            #                                    padding=gc.PADDING_IDX, bias=True))
            input_channels = output_channels
            self.use_residuals.append(use_residual)

        print (self.convolutions)
        #need another fc layer to map into embedding dimension embed_dim
        self.fc2 = layers.Linear(input_channels, embed_dims)

    def forward(self, batch_useqs, batch_item_ids=None):
        '''

        :param batch_useqs:
        :param batch_item_ids: batch x item_ids : may need this for attention layer
        :return:
        '''

        u_embeds = self.user_embeds(batch_useqs)
        if self._use_pos_emb:
            #use position embeddings for sequences:
            u_pos_embeds = self.pos_embeds(batch_useqs)
            u_embeds = u_embeds + u_pos_embeds
            u_embeds = F.dropout(u_embeds, p=self._dropout, training=self.training)

        #u_embeds: size BxTxE --> T x B x E (E can be seen as C: channels)
        original_u_embeds = u_embeds #store another copy for residual connection later


        #first project into size of convolution (input channels: C)
        u_embeds = self._dropoutlayer(u_embeds) #add dropout in front of linear layer
        # x = u_embeds # can comment out this line and uncomment the next line
        x = self.fc1(u_embeds) # x has size  B x T x C
        x = self.norm1(x)
        x = x.permute(1, 0, 2)  # B x T x C to T x B x C
        #apply layernorm right after each linear projection.
        # if self._user_layernorm:
        #     x = nn.LayerNorm(x.size()[1:])(x)  # do layer norm, not considering batch size (first dimension)


        #convert from B x T x C to T x B x C
        # x = x.transpose(0,1) #swap dimension 0 and 1

        for i, (projection, convolution, non_linearity, use_residual, norm) in \
                enumerate(zip(self.projections, self.convolutions,
                              self.non_linearities, self.use_residuals, self.norms)):
            # print i
            # if i == 0 and self._use_residual_connection:

                # print 'after projection:', x.size()
            if self._use_residual_connection:
                residual = x #store a copy of original input for residual connection later

            if projection is not None:
                residual = projection(residual)

            #convert from T B C to C B T to pad then reconvert back
            x = x.transpose(0,2) # swap dimension 0 and 2
            x = F.pad(x, (self.pads[i], self.pads[i]), mode='constant', value=gc.PADDING_IDX) #padding before convolve
            x = x.transpose(0,2)  # swap dimension 0 and 2
            x = convolution(x)
            if norm is not None: x = norm(x)

            #convolution : local linear projection --> apply layer norm after that.
            # if self._user_layernorm:
            #     x = nn.LayerNorm(x.size()[1:])(x)  # do layer norm, not considering batch size (first dimension)


            if non_linearity is not None:
                x = non_linearity(x)
            # print ('after convolution:',x.size())
            if self._convolution_type == 'glu': x = F.glu(x, dim=2) #using gated linear unit or not?

            #before moving to the next Convolution block, adding residual in here.
            if self._use_residual_connection:
                x = (x + residual) * math.sqrt(0.5) #need to check if multiplying sqrt(0.5) is helpful?

        # x = x.transpose(1, 0) #transpose to get B x T x C
        x = self._dropoutlayer(x) #add dropout in front of linear layer
        x = self.fc2(x) #mapping back to embed_dim

        # if self._user_layernorm: x = nn.LayerNorm(x.size()[1:])(x)  # do layer norm before fetching input to a new layer


        x = x.permute(1,0,2) #convert from T x B x C back to B x T x C

        #without attention, we just average them. B x T x C --> B x 1 x C
        x = torch.mean(x, dim=1)
        return x

class CNNItemNet(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass



class MLPNet(nn.Module):
    def __init__(self,
                 n_users, embed_dims, dropout):
        super(MLPNet, self).__init__()
        self.user_embeds = nn.Embedding(n_users, embed_dims, padding_idx=gc.PADDING_IDX)

    def forward(self):
        pass

class MLPUserItemNet(nn.Module):
    def __init__(self,
                 n_users, u_embed_dim, u_dropout,
                 n_items, i_embed_dim, i_dropout):
        super(MLPUserItemNet, self).__init__()

    def forward(self, batch_useqs, batch_iseqs, batch_user_ids = None, batch_item_ids = None ):
        pass

class CNNUserItemNet(nn.Module):
    def __init__(self,
                 n_users, u_embed_dim, max_u_positions, u_convolutions, u_dropout,
                 n_items, i_embed_dim, max_i_positions, i_convolutions, i_dropout
                 ):
        super(CNNUserItemNet, self).__init__()
        #position embeddings: position starts at 1 because 0 is padding, so we need to add 1 to max_positions
        # since each item is represented by a list of users who purchased it--> represent items by user embeddings
        self._item_net = CNNUserNet(n_users, u_embed_dim, max_u_positions + 1,
                                    convolutions=u_convolutions,dropout=u_dropout, use_positional_embed=True,
                                    use_residual_connection=True, non_linear='relu' #relu or tanh?
                                    )
        #since each user is represented by a list of items he purchased --> represent users by item embeddings
        self._user_net = CNNUserNet(n_items, i_embed_dim, max_i_positions + 1,
                                    convolutions=i_convolutions, dropout=i_dropout, use_positional_embed=True,
                                    use_residual_connection=True, non_linear='relu' #relu or tanh?
                                    )
        self._u_fc = nn.Linear(u_embed_dim, u_embed_dim)
        self._i_fc = nn.Linear(i_embed_dim, i_embed_dim)
        self._u_dropoutlayer = nn.Dropout(p=u_dropout)
        self._i_dropoutlayer = nn.Dropout(p=i_dropout)

    def forward(self, batch_useqs, batch_iseqs, batch_user_ids = None, batch_item_ids = None):
        '''

        :param batch_useqs: first dimension is batch size, second dimension is max length of user sequences
        :param batch_iseqs: first dimension is batch size, second dimension is max length of item sequences
        :param batch_user_ids: need for attention?
        :param batch_item_ids: need for attention?
        :return:
        '''

        user_encoding = self._user_net(batch_useqs)
        user_encoding = self._u_dropoutlayer(user_encoding)
        user_encoding = F.tanh(self._u_fc(user_encoding))

        item_encoding = self._item_net(batch_iseqs)
        item_encoding = self._i_dropoutlayer(item_encoding)
        item_encoding = F.tanh(self._i_fc(item_encoding))
        # print 'user encoding:', user_encoding.size()
        # print item_encoding.size()
        # print user_encoding
        batch_size = user_encoding.size(0)

        # user_encoding = my_utils.flatten(user_encoding)
        # item_encoding = my_utils.flatten(item_encoding)
        #print user_encoding.size()
	    #print item_encoding.size()
        return F.sigmoid(torch.sum(user_encoding * item_encoding, dim=1))
