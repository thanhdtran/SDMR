# SDMR
This repo contains source code for our paper: "Signed Distance-based Deep Memory Recommender" published in WWW 2019 (TheWebConf 2019).
This source code is coming soon (expect to be pushed before May 20)!

# DATA FORMAT:
- Same as "Neural Collaborative Filtering" paper (https://arxiv.org/abs/1708.05031).

# PARAMETERS:
<!-- - <code>saved_path</code>: the folder to save the checkpoints [Default is <code>chk_points</code>]. -->
- <code>load_best_chkpoint</code>: whether or not loading the best saved checkpoint [1 = Yes, 0 = No, Default is 0].
<!-- - <code>path</code>: path of the dataset, default is <code>data</code>. -->
- <code>dataset</code>: the name of dataset [ml1m or epinions-full].
<!-- - <code>epochs</code>: Number of running epoches. Default is 50. -->
- <code>num_factors</code>: number of latent factors. Default is 128.
- <code>n_hops</code>: number of hops in SDM model. Default is 3.
- <code>reg_sdp</code>: regularization param for SDP model. Default is 1e-3.
- <code>reg_sdm</code>: regularization param for SDM model. Default is 1e-3.
- <code>num_neg</code>: number of negative samples. Default is 4.
- <code>max_seq_len</code>: number of previous consumed items, serving as context items. Default is 5.
- <code>dropout</code>: dropout probability. Default is 0.2.
- <code>act_func_sdm</code>: activation function of sdm. [Default is <code>tanh</code>, but you can choose <code>relu</code>, or <code>none</code> as an identity function].
- <code>act_func_sdp</code>: activation function of sdp. [Default is <code>tanh</code>, but you can choose <code>relu</code>, or <code>none</code> as an identity function].
- <code>act_func_sdmr</code>: activation function of sdmr. [Default is <code>relu</code>].
- <code>model</code>: which model we are running? [3 choices are: <code>sdp</code>, <code>sdm</code>, <code>sdmr</code>].
- <code>out</code>: whether saving the best checkpoint or not. [Default is <code>1</code>].
- <code>cuda</code>: whether running using CUDA environment or not. [Default is <code>0</code>].
- <code>sdmr_retrain</code>: whether or not re-training the weights of SDP and SDM. [Default is No, which is <code>0</code>]. Setting this to 0 will save us back propagation time, but doesn't reduce forward time.

# RUNNING EXAMPLES:

