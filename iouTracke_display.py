import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import time

from utils.calc_performance import calculate_iou
import cv2
import numpy as np
import math

dis_video_width = 800
video_file = './image_and_anno/video/video8'  # dont need .mp4

if __name__ == '__main__':

    cap = cv2.VideoCapture(video_file + '.mp4')
    oriframe_w, oriframe_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = dis_video_width
    frame_h = int(oriframe_h * frame_w / oriframe_w)

    fps = cap.get(cv2.CAP_PROP_FPS)
    spf = 1000 / fps

    # initialization
    tracks_finished = np.load(video_file + '.npy').tolist()
    tracks_active = []
    frame_num = 0
    drawboard = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    w_ratio, h_ratio = frame_w / 640, frame_h / 480
    while True:
        frame_num += 1
        ret, image = cap.read()
        if not ret:
            print("over!")
            k = cv2.waitKey(0)
            break

        image = cv2.resize(image, (frame_w, frame_h))
        # <<<<<<<<<<<<<<<<<<<<draw track line>>>>>>>>>>>>>>>>>>>>>>>>>
        for track in tracks_finished:
            if track['start_frame'] == frame_num:
                track['color'] = cv2.cvtColor(np.uint8([[(np.random.randint(0, 360), 255, 255)]]),
                                              cv2.COLOR_HSV2BGR)[0, 0].tolist()
                tracks_active.append(track)

        for i, track in enumerate(tracks_active):
            dis_index = frame_num - track['start_frame']
            if dis_index >= len(track['bboxes']):
                del tracks_active[i]
            elif dis_index > 0:
                x10, y10, x20, y20 = track['bboxes'][dis_index - 1]
                cx0, cy0 = int(w_ratio * (x10 + x20) / 2), int(h_ratio * (y10 + y20) / 2)
                x11, y11, x21, y21 = track['bboxes'][dis_index]
                cx1, cy1 = int(w_ratio * (x11 + x21) / 2), int(h_ratio * (y11 + y21) / 2)
                pt1 = (int(w_ratio * x11), int(h_ratio * y11))
                pt2 = (int(w_ratio * x21), int(h_ratio * y21))
                thickness = int((x21 + y21 - y11 - x11) / 80) + 1
                drawboard = cv2.line(drawboard, (cx0, cy0), (cx1, cy1), track['color'], thickness)
                image = cv2.rectangle(image, pt1, pt2, track['color'], 1)

        # <<<<<<<<<<<<<<<<<<<<end of draw track line>>>>>>>>>>>>>>>>>>>>>>>
        mask = cv2.cvtColor(drawboard, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)
        image = cv2.bitwise_and(image, image, mask=mask)
        image = cv2.add(image, drawboard)

        cv2.imshow('1', image)
        k = cv2.waitKey(int(spf))
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite(video_file + '.jpg', image)
            print('image saved')

    cap.release()
    cv2.destroyAllWindows()
