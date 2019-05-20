import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import scipy.io as sio
import time

from PIL import Image, ImageDraw
from pyramid_mb2_try3 import build_sfd_mobile as build_sfd_mobile_try3
from pyramid_mobile_try1 import build_sfd_mobile as build_sfd_mobile_try1
from pyramid_mobile_try2 import build_sfd_mobile as build_sfd_mobile_try2
from pyramid import build_sfd
from layers import *
import cv2
import numpy as np
import math

red = (0, 0, 255)
white = (255, 255, 255)


def detect_face(x, shrink):
    if shrink != 1:
        x = cv2.resize(x, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)

    height, width, _ = x.shape
    x = x.astype(np.float32)
    x -= np.array([104, 117, 123], dtype=np.float32)

    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.unsqueeze(0)
    x = x.cuda()

    t0 = time.clock()
    with torch.no_grad():
        y = net(x)
    t1 = time.clock()
    print('time: {:.5f}ms   |   {:.4f}MB'.format((t1 - t0) * 1000, torch.cuda.memory_cached(0) / 1024 / 1024))
    detections = y.data

    scale = torch.Tensor([width, height, width, height])

    boxes = []
    scores = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.4:
            score_ = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            boxes.append([pt[0], pt[1], pt[2], pt[3]])
            scores.append(score_)
            j += 1
            if j >= detections.size(2):
                break

    det_conf = np.array(scores)
    boxes = np.array(boxes)

    if boxes.shape[0] == 0:
        return np.array([[0, 0, 0, 0, 0.4]])

    det_xmin = boxes[:, 0] / shrink
    det_ymin = boxes[:, 1] / shrink
    det_xmax = boxes[:, 2] / shrink
    det_ymax = boxes[:, 3] / shrink
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

    keep_index = np.where(det[:, 4] >= 0)[0]
    det = det[keep_index, :]
    return det


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    torch.cuda.set_device(0)

    print('Loading model..')
    # net = build_sfd('test', 640, 2)
    # net.load_state_dict(torch.load(r'./net_weight/Res50_pyramid.pth',
    #                                map_location=torch.device(0)))
    # net.priorbox = PriorBoxLayer(640, 480)

    net = build_sfd_mobile_try3('test', 640, 2)
    net.load_state_dict(torch.load(r'./weights_of_mine/try3/Mobile_pyramid_21000.pth',
                                   map_location=torch.device(0)))
    net.priorbox = PriorBoxLayer(640, 480, stride=[4, 8, 16, 32, 64], box=(16, 32, 64, 128, 256))

    # net = build_sfd_mobile_try1('test', 640, 2)
    # net.load_state_dict(torch.load(r'./weights_of_mine/try1_from104000/Mobile_pyramid_40000.pth',
    #                                map_location=torch.device(0)))
    # net.priorbox = PriorBoxLayer(640, 480)

    # net = build_sfd_mobile_try2('test', 640, 2)
    # net.load_state_dict(torch.load(r'./weights_of_mine/try2_from46000/Mobile_pyramid_6000.pth',
    #                                map_location=torch.device(0)))
    # net.priorbox = PriorBoxLayer(640, 480)

    net.cuda()
    net.eval()
    print('Finished loading model!')

    cap = cv2.VideoCapture(0)
    start, end = 0.0, 0.0
    while True:
        ret, image = cap.read()

        det0 = detect_face(image, 1)  # origin test
        # dets = bbox_vote(det0)

        for box in det0:
            score = box[4]
            box = box[:-1].astype(np.int32)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), red, 1)
            cv2.putText(image, "{:.3f}".format(score), (box[0], box[1] - 2), cv2.FONT_HERSHEY_PLAIN, 0.8, red, 1)

        end = time.time()
        cv2.putText(image, "fps={:.5f}".format(1/(end - start)), (1, 50), cv2.FONT_HERSHEY_PLAIN, 1, white, 1)
        start = time.time()
        cv2.imshow('1', image)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
