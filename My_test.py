import argparse
import time
import os
import cv2
import numpy as np
import torch
import torch.utils.data as data

from layers import *
from pyramid import build_sfd as build_net_repo
from pyramid_mb2_try3 import build_sfd_mobile as build_net_try3
from pyramid_mb2_try4 import build_sfd_mobile as build_net_try4
from pyramid_mobile_try1 import build_sfd_mobile as build_net_try1
from utils.data_collector import Data_collector
from utils.calc_performance import calc_pr


def str2bool(string):
    return string in ["True"]


def detect_face(x):
    height, width, _ = x.shape
    x = x.astype(np.float32)
    x -= np.array([104, 117, 123], dtype=np.float32)

    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.unsqueeze(0)
    x = x.cuda()

    if args.net == 'repo' or args.net == 'repo_my' or args.net == 'try1' or args.net == 'try2':
        net.priorbox = PriorBoxLayer(width, height)
    elif args.net == 'try3' or args.net == 'try4' or args.net == 'try5':
        net.priorbox = PriorBoxLayer(width, height, stride=[4, 8, 16, 32, 64], box=(16, 32, 64, 128, 256))
    net.firstTime = True
    net.detect = Detect(2, 0, 750, args.threshold, 0.35)  # todo: is that right?

    with torch.no_grad():
        y = net(x)

    detections = y.data

    scale = torch.Tensor([width, height, width, height])

    boxes = []
    scores = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= args.threshold:
            score_ = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            boxes.append([pt[0], pt[1], pt[2], pt[3]])
            scores.append(score_)
            j += 1
            if j >= detections.size(2):
                break

    det_conf = np.array(scores).astype(np.float)
    boxes = np.array(boxes)

    if boxes.shape[0] == 0:
        return np.array([[0, 0, 0, 0, 0.4]])

    det_xmin = boxes[:, 0]
    det_ymin = boxes[:, 1]
    det_xmax = boxes[:, 2]
    det_ymax = boxes[:, 3]
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

    keep_index = np.where(det[:, 4] >= 0)[0]
    det = det[keep_index, :]
    return det


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
    parser.add_argument('--device', default=0, type=int, help='Testing device')
    parser.add_argument('--net', default="repo", type=str, help='net type to be tested')
    parser.add_argument('--threshold', default=0, type=float, help='score threshold for net')
    parser.add_argument('--display', default=True, type=str2bool, help='if show the image')
    parser.add_argument('--data_save_folder',
                        default="./draw_curve/", type=str,
                        help='tf_conf save folder')
    parser.add_argument('--img_save_folder',
                        default="./image_and_anno/test_image/", type=str,
                        help='test_image save folder')
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    print("testing on device: " + str(args.device))
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not os.path.exists(args.data_save_folder):
        os.mkdir(args.data_save_folder)
    args.data_save_folder += "data/"
    if not os.path.exists(args.data_save_folder):
        os.mkdir(args.data_save_folder)

    if not os.path.exists(args.img_save_folder):
        os.mkdir(args.img_save_folder)
    args.img_save_folder += args.net + "/"
    if not os.path.exists(args.img_save_folder):
        os.mkdir(args.img_save_folder)
    anno_path_eval = "./image_and_anno/anno/gen_anno_file_val"
    iou_thresh = 0.5
    tf_conf = np.array([[], []])
    truth_num = 0
    img_cnt = 0
    red = (0, 0, 255)
    green = (0, 255, 0)
    white = (255, 255, 255)

    print('Loading model to net_' + args.net + '...')
    if args.net == "repo_my":
        net = build_net_repo('test', 640, 2)
        net.load_state_dict(torch.load(r'./weights_of_mine/repo_Res50/Res50_pyramid_15000.pth',
                                       map_location=torch.device(0)))
    elif args.net == "repo":
        net = build_net_repo('test', 640, 2)
        net.load_state_dict(torch.load(r'./net_weight/Res50_pyramid.pth'))
    elif args.net == "try1":
        net = build_net_try1('test', 640, 2)
        net.load_state_dict(torch.load(r'./net_weight/Mobile_pyramid_try1.pth',
                                       map_location=torch.device(0)))
    elif args.net == "try3":
        net = build_net_try3('test', 640, 2)
        net.load_state_dict(torch.load(r'./net_weight/Mobile_pyramid_try3.pth',
                                       map_location=torch.device(0)))
    elif args.net == "try4":
        net = build_net_try4('test', 640, 2)
        net.load_state_dict(torch.load(r'./net_weight/Mobile_pyramid_try4.pth',
                                       map_location=torch.device(0)))
    else:
        net = None
        print("net type not recognized!")
    net = net.cuda()
    net.eval()
    print('Finished loading model!')

    dataset_eval = Data_collector(anno_path_eval)
    for image, target, img_id in iter(dataset_eval):
        print("testing image " + str(img_id) + '...')
        predict = detect_face(image)  # origin test

        if args.display:
            for box in target:
                cv2.rectangle(image, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), green, 1)

            for box in predict:
                score = box[4]
                box = box[:-1].astype(np.int32)
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), red, 1)
                cv2.putText(image, "{:.3f}".format(score), (box[0], box[1] - 2), cv2.FONT_HERSHEY_PLAIN, 0.7, red, 1)

            cv2.imshow('1', image)
            k = cv2.waitKey(1000)
            if k == ord('s'):
                cv2.imwrite(args.img_save_folder +
                            args.net+"_thresh_"+str(args.threshold)+"_"+str(img_cnt)+".jpg",
                            image)
                print("image saved in " + args.img_save_folder)
                img_cnt += 1

        tf_conf_, truth_num_ = calc_pr(predict, target, iou_thresh=iou_thresh)
        tf_conf = np.hstack((tf_conf, tf_conf_))
        truth_num += truth_num_

    print("finished testing all images")
    print("sortting and saving")
    tf_conf = tf_conf[:, np.argsort(tf_conf[1, :])[::-1]]
    np.save(args.data_save_folder + 'data_of_' + args.net + '.npy',
            np.hstack((tf_conf, [[0], [truth_num]])))
    print("Goodbye!")
