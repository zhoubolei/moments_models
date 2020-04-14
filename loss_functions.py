'''
-- Our loss implementations for wlsep, lsep, warp, bp_mll and bce with optional weighted learning
--
-- If you use these implementations in your paper please cite our paper https://arxiv.org/abs/1911.00232
--
-- scores is the output of the model
-- labels is a binary vector with a 1 indicating a positive class for each batch member
-- Both scores and labels have size BxC with B = batch size and C = number of classes
-- weights is an optional tensor of size C used to weight the learning for unbalanced training sets
-- We used w_i = min(count)/count_i for the weights to train the Multi-Moments model where
   count_i is the number of examples in the training set with a positive label for class i
   and min(count) is the number of examples with a positive label for the least common class.
--
-- By Mathew Monfort, mmonfort@mit.edu
'''
import torch
from torch.nn import functional as F

# https://arxiv.org/abs/1911.00232
def wlsep(scores, labels, weights=None):
  mask = ((labels.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)) -
           labels.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1))) > 0).float()
  diffs = (scores.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1)) -
           scores.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)))
  if weights is not None:
    return F.pad(diffs.add(-(1-mask)*1e10),
                 pad=(0,0,0,1)).logsumexp(dim=1).mul(weights).masked_select(labels.bool()).mean()
  else:
    return F.pad(diffs.add(-(1-mask)*1e10),
                 pad=(0,0,0,1)).logsumexp(dim=1).masked_select(labels.bool()).mean()

# http://openaccess.thecvf.com/content_cvpr_2017/html/Li_Improving_Pairwise_Ranking_CVPR_2017_paper.html
def lsep(scores, labels, weights=None):
  mask = ((labels.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)) -
           labels.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1))) > 0).float()
  diffs = (scores.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1)) -
           scores.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)))
  return diffs.exp().mul(mask).sum().add(1).log().mean()

""" https://www.aaai.org/ocs/index.php/IJCAI/IJCAI11/paper/viewPaper/2926
We pre-compute the rank weights (rank_w) into a tensor as below:
rank_w = torch.zeros(num_classes)
sum = 0.
for i in range(num_classes):
  sum += 1./(i+1)
  rank_w[i] = sum
"""

def warp(scores, labels, rank_w, weights=None):
  mask = ((labels.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)) -
           labels.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1))) > 0).float()
  diffs = (scores.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1)) -
           scores.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1))).add(1)
  if weights is not None:
    return (diffs.clamp(0,1e10).mul(mask).sum(1).div(mask.sum(1)).mul(weights).masked_select(labels.bool())
                 .mul(rank_w.index_select(0,scores.sort(descending=True)[1].masked_select(labels.bool()))).mean())
  else:
    return (diffs.clamp(0,1e10).mul(mask).sum(1).div(mask.sum(1)).masked_select(labels.bool())
                 .mul(rank_w.index_select(0,scores.sort(descending=True)[1].masked_select(labels.bool()))).mean())

#https://ieeexplore.ieee.org/abstract/document/1683770
def bp_mll(scores, labels, weights=None):
  mask = ((labels.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)) -
           labels.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1))) > 0).float()
  diffs = (scores.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1)) -
           scores.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)))
  if weights is not None:
    return diffs.exp().mul(mask).sum(1).mul(weights).sum(1).mean()
  else:
    return diffs.exp().mul(mask).sum(1).sum(1).mean()

def bce(output, labels, weights=None):
  if weights is not None:
    return (((1.-weights)*labels + weights*(1.-labels))*
             bceCriterion(output, torch.autograd.Variable(labels))).sum(1).mean()
  else:
    return bceCriterion(output, torch.autograd.Variable(labels)).sum(1).mean()
