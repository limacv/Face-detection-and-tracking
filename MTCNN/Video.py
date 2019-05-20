import cv2
import numpy as np
import time
from MTCNN.mtcnn.core.detect import create_mtcnn_net, MtcnnDetector

green = (0, 255, 0)
white = (255, 255, 255)


def cal_overlap(box1, box2, min_mode=True):
    x1, y1, x2, y2, _ = box1
    x_1, y_1, x_2, y_2, _ = box2
    area = (x2 - x1) * (y2 - y1)
    area_ = (x_2 - x_1) * (y_2 - y_1)
    inter = max((min(x_2, x2) - max(x1, x_1)), 0) * max(0, (min(y_2, y2) - max(y1, y_1)))
    if min_mode:
        return inter / min(area, area_)
    else:
        return inter / (area_ + area - inter)


def add_elements2img(boxes, marks, img, color=green):
    for box in boxes:
        score = box[4]
        box = box[:-1].astype(np.int32)
        cv2.putText(img, "score={:.3f}".format(score), (box[0], box[1]-2), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)

    marks = marks.reshape(-1, 2).astype(np.int32)
    for mark in marks:
            cv2.circle(img, (mark[0], mark[1]), 2, color, -1)
    return


def box_id_filter(boxes_cur, boxes_his):
    """
    1. 判断前后是否是同一个box
    2. 使box位置变化更加平缓
    :param boxes_cur:
    :param boxes_his:
    :return:
    """
    # for box in boxes_cur:

    return boxes_cur


if __name__ == '__main__':

    pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt", r_model_path="./original_model/rnet_epoch.pt", o_model_path="./original_model/onet_epoch.pt", use_cuda=True)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    cap = cv2.VideoCapture(0)
    start, end = 0, 0
    while True:
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bboxs, landmarks = mtcnn_detector.detect_face(frame)
        # if bboxs.shape[0] > 1:
        #     print(bboxs)
        # bboxs_his = bboxs * 0.6 + bboxs_his * 0.4
        add_elements2img(bboxs, landmarks, frame)

        end = time.time()
        cv2.putText(frame, "fps={:.3f}".format(1/(end-start)), (1, 50), cv2.FONT_HERSHEY_PLAIN, 1, white, 1)
        start = time.time()

        cv2.imshow("faces", frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
