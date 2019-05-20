import cv2
import numpy as np
import time
from utils.data_collector import Data_collector
from utils.calc_performance import calc_pr
import torch
import torch.nn.functional as F
from FACEBOX.networks import FaceBox
from FACEBOX.encoderl import DataEncoder


def detect(im):
    im = cv2.resize(im, (1024, 1024))
    im_tensor = torch.from_numpy(im.transpose((2, 0, 1)))
    im_tensor = im_tensor.float().div(255)  # normalized pixel value to (0,1)
    # print(im_tensor.shape)
    im_tensor = im_tensor.cuda()

    # t1 = time.time()
    # utilize the network
    with torch.no_grad():
        loc, conf = net(torch.unsqueeze(im_tensor, 0))

    loc = loc.detach().squeeze(0)
    conf = F.softmax(conf.squeeze(0), dim=1).detach()
    # t2 = time.time()

    loc, conf = loc.cpu(), conf.cpu()
    # boxes_, labels, probs_ = data_encoder.decode_tensor(loc.data.squeeze(0),
    #                                                     F.softmax(conf.squeeze(0), dim=1).data)
    boxes_, probs_ = data_encoder.decode_np(loc, conf)
    # t3 = time.time()
    # print("network cost: {:.5f}".format(t2-t1))
    # print("decoder cost: {:.5f}".format(t3-t2))
    # print("********************")
    return boxes_, probs_


if __name__ == '__main__':
    net = FaceBox()
    net.load_state_dict(torch.load('./faceboxes.pt', map_location=torch.device(0)))
    net.cuda()
    net.eval()

    data_encoder = DataEncoder()

    anno_path_eval = "../image_and_anno/anno/gen_anno_file_val"
    data_save_folder = "../draw_curve/data/"
    display = True
    iou_thresh = 0.5
    tf_conf = np.array([[], []])
    truth_num = 0
    green = (0, 255, 0)
    red = (0, 0, 255)

    dataset_eval = Data_collector(anno_path_eval)
    for image, target, img_id in iter(dataset_eval):
        print("testing image " + str(img_id) + '...')
        boxes, probs = detect(image)
        h, w, _ = image.shape
        predict = np.column_stack((boxes * np.array([w, h, w, h]), probs))

        if display:
            for box in target:
                cv2.rectangle(image, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), green, 1)

            for box in predict:
                score = box[4]
                box = box[:-1].astype(np.int32)
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), red, 1)
                cv2.putText(image, "{:.3f}".format(score), (box[0], box[1] - 2), cv2.FONT_HERSHEY_PLAIN, 0.7, red, 1)

            cv2.imshow('1', image)
            k = cv2.waitKey(1)
        if len(predict) == 0:
            tf_conf_, truth_num_ = np.array([[], []]), target.shape[0]
        else:
            tf_conf_, truth_num_ = calc_pr(predict, target, iou_thresh=iou_thresh)
        tf_conf = np.hstack((tf_conf, tf_conf_))
        truth_num += truth_num_

    print("finished testing all images")
    print("sortting and saving")
    tf_conf = tf_conf[:, np.argsort(tf_conf[1, :])[::-1]]
    np.save(data_save_folder + 'data_of_facebox.npy',
            np.hstack((tf_conf, [[0], [truth_num]])))
    print("Goodbye!")
