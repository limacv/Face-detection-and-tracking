import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import scipy.io as sio
import time

from PIL import Image, ImageDraw
from pyramid_mb2_official import build_sfd_mobile as build_sfd_mobile_try3
from pyramid_mobile_try1 import build_sfd_mobile as build_sfd_mobile_try1
from pyramid_mobile_try2 import build_sfd_mobile as build_sfd_mobile_try2
from pyramid import build_sfd
from layers import *
from utils.calc_performance import calculate_iou, calculate_distance
import cv2
import numpy as np
import math

# <<<<<<<<<<<<<<<<<<<<<<<<<<parameter configer>>>>>>>>>>>>>>>>>>>>>>>>>>>>
use_iou = True
sigma_iou = 0.4

sigma_dis = 8
sigma_h = 0.6
t_min = 5
display_result = True
use_net = 'repo'
video_file = './image_and_anno/video/video8'  # dont need .mp4
red = (0, 0, 255)

#  <<<<<<<<<<<<<<<<<<<<<<<end of config parameter>>>>>>>>>>>>>>>>>>>>>>>>>>


def detect_face(x, shrink):
    if shrink != 1:
        x = cv2.resize(x, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)

    height, width, _ = x.shape
    x = x.astype(np.float32)
    x -= np.array([104, 117, 123], dtype=np.float32)

    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.unsqueeze(0)
    x = x.cuda()

    t1 = time.time()
    with torch.no_grad():
        y = net(x)
    t2 = time.time()
    # print('{:.5f}'.format(t2-t1))
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
    cap = cv2.VideoCapture(video_file + '.mp4')
    frame_w, frame_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    torch.cuda.set_device(0)
    print('Loading model..')
    if use_net == 'repo':
        net = build_sfd('test', 640, 2)
        net.load_state_dict(torch.load(r'./net_weight/Res50_pyramid.pth',
                                       map_location=torch.device(0)))
        net.priorbox = PriorBoxLayer(640, 480)
    elif use_net == 'try3':
        net = build_sfd_mobile_try3('test', 640, 2)
        net.load_state_dict(torch.load(r'net_weight/Mobile_pyramid_try3.pth',
                                       map_location=torch.device(0)))
        net.priorbox = PriorBoxLayer(640, 480, stride=[4, 8, 16, 32, 64], box=(16, 32, 64, 128, 256))
    else:
        raise()
    net.cuda()
    net.eval()
    print('Finished loading model!')

    fps = cap.get(cv2.CAP_PROP_FPS)
    spf = 1000 / fps

    # initialization
    frame_num = 0
    tracks_active = []
    tracks_finished = []
    while True:
        frame_num += 1
        ret, image = cap.read()
        if not ret:
            print("finished tracking!")
            break
        image = cv2.resize(image, (640, 480))
        det0 = detect_face(image, 1)  # origin test

        # <<<<<<<<<<<<<<iou_tracker algorithm>>>>>>>>>>>>>>>
        dets = det0.tolist()
        updated_tracks = []
        for track in tracks_active:
            if len(dets) > 0:
                if use_iou:
                    iou = calculate_iou(np.array(dets)[:, :4], np.array([track['bboxes'][-1]]))
                    best_match = iou.argmax()
                    matched = iou[best_match] > sigma_iou
                else:
                    iou = calculate_distance(np.array(dets)[:, :4], np.array([track['bboxes'][-1]]))
                    best_match = iou.argmin()
                    matched = iou[best_match] < sigma_dis

                if matched:
                    track['bboxes'].append(dets[best_match][:4])
                    track['max_score'] = max(track['max_score'], dets[best_match][4])

                    updated_tracks.append(track)
                    del dets[best_match]
                else:
                    if track['max_score'] > sigma_h and len(track['bboxes']) > t_min:
                        tracks_finished.append(track)

        new_tracks = [{
            'bboxes': [det[:4], ],
            'max_score': det[4],
            'start_frame': frame_num
        } for det in dets]
        tracks_active = updated_tracks + new_tracks
        # <<<<<<<<<<<<<<<<<<<<end of track>>>>>>>>>>>>>>>>>>>>>>>
        # display intermidia process
        if display_result:
            for box in det0:
                score = box[4]
                box = box[:-1].astype(np.int32)
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), red, 1)
                cv2.putText(image, "{:.3f}".format(score), (box[0], box[1] - 2), cv2.FONT_HERSHEY_PLAIN, 0.8, red, 1)

        cv2.imshow('1', image)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # <<<<<<<<<<<<<<<begin of iou tracker>>>>>>>>>>>>>>>>>
    tracks_finished += [track for track in tracks_active
                        if track['max_score'] > sigma_h and len(track['bboxes']) >= t_min]
    # save to file
    np.save(video_file + ".npy", np.array(tracks_finished))
    print("file saved to: " + video_file + '.npy')
    # <<<<<<<<<<<<<<<<<<end of iou tracker>>>>>>>>>>>>>>>>>>>
