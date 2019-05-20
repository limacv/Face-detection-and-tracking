import torch
from torch.autograd import Function
from ..box_utils import decode, nms, center_size
from data import face as cfg
import time
import numpy as np


class Detect:
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        """
        :param num_classes: =2
        :param bkg_label: =0
        :param top_k:  =750, the max amount of total boxes
        :param conf_thresh: =0.05, only conf bigger than it will be selected
        :param nms_thresh: =0.3
        """
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']
        self.nms_top_k = 5000

    def __call__(self, loc_data, conf_data, prior_data):
        """
        :parameter
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        :return

        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        # decode boxes **************************************************
        for i in range(num):
            # print (self.variance)
            # decoded_boxes format: [x1, y1, x2, y2]
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i]  # .clone()

            # select scores bigger than self.conf_thresh **********************************************
            # result:   conf_scores: [2, 25600]
            #           decoded_boxes: [4, 25600]
            # c_mask/l_mask are where score bigger than self.conf_thresh
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)  # cost
                t1 = time.time()
                c_mask_temp = c_mask.nonzero().squeeze()
                t2 = time.time()
                # print("time:{:.4f}ms".format((t2-t1)*1000))

                scores = conf_scores[cl][c_mask_temp]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)

                # nms ************************************************
                # result: scores / boxes
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, min(boxes.shape[0], self.nms_top_k))
                count = min(count, self.top_k)
                ids = ids[:count]
                output[i, cl, :count] = torch.cat((scores[ids].unsqueeze(1), boxes[ids]), 1)

        return output
