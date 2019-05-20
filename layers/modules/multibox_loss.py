# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import face as cfg
from ..box_utils import match_ensure_max_prior, log_sum_exp, match_default

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes from using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,lo,g) = (Lconf(x, c) + αLloc(x,lo,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            lo: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, bipartite=True,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.bipartite = bipartite
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net_raw.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tuple of tensors): Ground truth boxes and truth_conf for a batch,
                tuple shape: [batch_size] -> [tensor1, tensor2, ...]
                tensor shape: [num_objs,5] -> [[xmin, ymin, xmax, ymax, conf], [...], ...]
                    !!!conf is always 0.0 [xmin, ymin, xmax, ymax, 0.0]
        """
        loc_data, conf_data, priors = predictions
        batch_size, num_priors, _ = loc_data.shape

        loc_truth = torch.Tensor(batch_size, num_priors, 4)
        conf_truth = torch.LongTensor(batch_size, num_priors)

        # in each batch, or say, in each image
        for idx in range(batch_size):
            truth_loc = targets[idx][:, :-1].data
            truth_conf = targets[idx][:, -1].data
            defaults = priors.data

            # match priors (default boxes) and ground truth boxes
            # get loc_truth & conf_truth
            if self.bipartite:
                match_ensure_max_prior(self.threshold, truth_loc, defaults,
                                       self.variance, truth_conf, loc_truth, conf_truth, idx)
            else:
                match_default(self.threshold, truth_loc, defaults,
                              self.variance, truth_conf, loc_truth, conf_truth, idx)
        if self.use_gpu:
            loc_truth = loc_truth.cuda()
            conf_truth = conf_truth.cuda()

        # wrap targets
        # loc_truth = Variable(loc_truth, requires_grad=False)
        # conf_truth = Variable(conf_truth, requires_grad=False)

        pos = conf_truth > 0  # positive

        # pos = Variable(pos, requires_grad=False)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_predict = loc_data[pos_idx].view(-1, 4)
        loc_truth = loc_truth[pos_idx].view(-1, 4)

        # calculate Smooth L1 Loss (for those score>0 only)
        loss_l = F.smooth_l1_loss(loc_predict, loc_truth, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # calculate Softmax Loss
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_truth.view(-1, 1))

        loss_c = loss_c.view(batch_size, -1)
        # Hard Negative Mining*********************************
        loss_c[pos] = 0  # filter out pos boxes for now

        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples**********************
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_truth[(pos + neg).gt(0)]
        if conf_p.shape == torch.Size([0]):
            loss_c = 10
            N = 1
        else:
            loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
            # Sum of losses: L(x,c,lo,g) = (Lconf(x, c) + αLloc(x,lo,g)) / N
            N = num_pos.data.sum().float()

        if N == 0:
            N = batch_size
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
