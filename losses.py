
import torch


def bpr_loss(positive_predictions, negative_predictions, mask=None, average=False):
    """
    Bayesian Personalised Ranking [1]_ pairwise loss function.

    Parameters
    ----------

    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Tensor containing predictions for sampled negative items.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.

    References
    ----------

    .. [1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from
       implicit feedback." Proceedings of the twenty-fifth conference on
       uncertainty in artificial intelligence. AUAI Press, 2009.
    """

    loss = - torch.log(torch.sigmoid
                                 (-negative_predictions + positive_predictions)
                        )



    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    if average:
        return loss.mean()
    else:
        return loss.sum()

