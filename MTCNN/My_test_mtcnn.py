import cv2
import numpy as np
import time
from MTCNN.mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from utils.data_collector import Data_collector
from utils.calc_performance import calc_pr


if __name__ == '__main__':
    pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt",
                                        r_model_path="./original_model/rnet_epoch.pt",
                                        o_model_path="./original_model/onet_epoch.pt", use_cuda=True)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

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
        if img_id in [2852, ]:
            continue
        predict, _ = mtcnn_detector.detect_face(image)

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
    np.save(data_save_folder + 'data_of_mtcnn.npy',
            np.hstack((tf_conf, [[0], [truth_num]])))
    print("Goodbye!")
