from FACEBOX.networks import FaceBox
from FACEBOX.encoderl import DataEncoder

import torch
import torch.nn.functional as F
import cv2
import time

use_cuda = True
front = cv2.FONT_HERSHEY_COMPLEX_SMALL
light_green = (50, 150, 50)
green = (0, 255, 0)
white = (255, 255, 255)
blue = (255, 255, 255)


def detect(im):
    im = cv2.resize(im, (1024, 1024))
    im_tensor = torch.from_numpy(im.transpose((2, 0, 1)))
    im_tensor = im_tensor.float().div(255)  # normalized pixel value to (0,1)
    # print(im_tensor.shape)
    if use_cuda:
        im_tensor = im_tensor.cuda()

    # t1 = time.time()
    # utilize the network
    with torch.no_grad():
        t0 = time.clock()
        loc, conf = net(torch.unsqueeze(im_tensor, 0))
        print('time: {:.5f}ms   |   {:.4f}MB'.format(((time.clock()-t0)*1000), torch.cuda.memory_cached(0)/1024/1024))

    loc = loc.detach().squeeze(0)
    conf = F.softmax(conf.squeeze(0), dim=1).detach()
    # t2 = time.time()

    if use_cuda:
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
    net.load_state_dict(torch.load('./faceboxes.pt'))  # , map_location=lambda storage, loc: storage))
    net.cuda()
    net.eval()

    data_encoder = DataEncoder()

    cap = cv2.VideoCapture(0)
    start, end = 0.0, 0.0
    while True:
        ret, frame = cap.read(0)
        boxes, probs = detect(frame)
        h, w, _ = frame.shape
        for i, box in enumerate(boxes):
            x1 = int(box[0] * w)
            x2 = int(box[2] * w)
            y1 = int(box[1] * h)
            y2 = int(box[3] * h)

            cv2.rectangle(frame, (x1, y1 + 4), (x2, y2), green, 1)
            cv2.putText(frame, "{:.3f}".format(probs[i]), (x1, y1), front, 0.7, green)

        end = time.time()
        cv2.putText(frame, "fps={:.3f}".format(1 / (end - start)), (1, 50), cv2.FONT_HERSHEY_PLAIN, 1, white, 1)
        start = time.time()

        cv2.imshow("faces", frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
