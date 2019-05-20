import cv2
import time
import numpy as np
import torch
from torch.autograd.variable import Variable
from MTCNN.mtcnn.core.models import PNet, RNet, ONet
import MTCNN.mtcnn.core.utils as utils
import MTCNN.mtcnn.core.image_tools as image_tools
"""

"""


def create_mtcnn_net(p_model_path=None, r_model_path=None, o_model_path=None, use_cuda=True):
    """
    including:
        1. load p/r/o net module from file
        2. copy to CPU/GPU
        3. set modules to evaluation mode
    :return
        pnet, rnet, onet objects
    """
    pnet, rnet, onet = None, None, None

    if p_model_path is not None:
        pnet = PNet(use_cuda=use_cuda)
        if(use_cuda):
            # print('p_model_path:{0}'.format(p_model_path))

            # 从文件中load网络中的各个权重参数，并copy到pnet_object中
            pnet.load_state_dict(torch.load(p_model_path))
            pnet.cuda()  # move object to cuda
        else:
            # compute in CPU
            pnet.load_state_dict(torch.load(p_model_path, map_location=lambda storage, loc: storage))
        # set pnet in evalution mode
        pnet.eval()

    if r_model_path is not None:
        rnet = RNet(use_cuda=use_cuda)
        if (use_cuda):
            # print('r_model_path:{0}'.format(r_model_path))

            # 从文件中load网络中的各个权重参数，并copy到rnet_object中
            rnet.load_state_dict(torch.load(r_model_path))
            rnet.cuda()  # move object to cuda
        else:
            rnet.load_state_dict(torch.load(r_model_path, map_location=lambda storage, loc: storage))
        rnet.eval()

    if o_model_path is not None:
        onet = ONet(use_cuda=use_cuda)
        if (use_cuda):
            # print('o_model_path:{0}'.format(o_model_path))

            # 从文件中load网络中的各个权重参数，并copy到onet_object中
            onet.load_state_dict(torch.load(o_model_path))
            onet.cuda()  # move object to cuda
        else:
            onet.load_state_dict(torch.load(o_model_path, map_location=lambda storage, loc: storage))
        onet.eval()

    return pnet, rnet, onet


class MtcnnDetector:
    """
        P,R,O net face detection and landmarks align
        :param
            scale_factor: 图像金字塔，相邻层的缩放比例
            threshold[0/1/2]: classification threshold in p/r/o net
    """
    def __init__(self,
                 pnet=None,
                 rnet=None,
                 onet=None,
                 min_face_size=12,
                 stride=2,
                 threshold=(0.6, 0.6, 0.35),
                 scale_factor=0.709,
                 ):

        self.pnet_detector = pnet
        self.rnet_detector = rnet
        self.onet_detector = onet
        self.min_face_size = min_face_size
        self.stride = stride
        self.thresh = threshold
        self.scale_factor = scale_factor

    @staticmethod
    def unique_image_format(im):
        if not isinstance(im, np.ndarray):
            if im.mode == 'I':
                im = np.array(im, np.int32, copy=False)
            elif im.mode == 'I;16':
                im = np.array(im, np.int16, copy=False)
            else:
                im = np.asarray(im)
        return im

    @staticmethod
    def square_bbox(bbox):
        """
        convert bbox to square box
            - based on center point
        Parameters:
        ----------
            bbox: numpy array , shape n x m
                input bbox
        Returns:
        -------
            a square bbox
        """
        square_bbox = bbox.copy()  # deep copy

        # x2 - x1
        # y2 - y1
        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        la = np.maximum(h, w)
        # x1 = x1 + w*0.5 - la*0.5
        # y1 = y1 + h*0.5 - la*0.5
        square_bbox[:, 0] = bbox[:, 0] + w*0.5 - la*0.5
        square_bbox[:, 1] = bbox[:, 1] + h*0.5 - la*0.5

        # x2 = x1 + la - 1
        # y2 = y1 + la - 1
        square_bbox[:, 2] = square_bbox[:, 0] + la - 1
        square_bbox[:, 3] = square_bbox[:, 1] + la - 1
        return square_bbox

    @staticmethod
    def generate_bounding_box(fmap, reg, scale, threshold):
        """
            generate bbox from feature fmap
        Parameters:
        ----------
            fmap: numpy array , n x m x 1
                detect score for each position
            reg: numpy array , n x m x 4
                bbox
            scale: float number
                scale of this detection
            threshold: float number
                detect threshold
        Returns:
        -------
            bbox array
        """
        stride = 2
        cellsize = 12  # receptive field

        # t_index[0]: x_coordinate ; [1]: y_coordinate
        t_index = np.where(fmap > threshold)

        # find nothing
        if t_index[0].size == 0:
            return np.array([])

        # abtain score of classification which larger than threshold
        score = fmap[t_index[0], t_index[1], 0]
        # choose bounding box whose socre are larger than threshold
        reg = np.array([reg[0, t_index[0], t_index[1], i] for i in range(4)])

        # lefteye_dx, lefteye_dy, righteye_dx, righteye_dy, nose_dx, nose_dy, \
        # leftmouth_dx, leftmouth_dy, rightmouth_dx, rightmouth_dy = [landmarks[0, t_index[0], t_index[1], i] for i in range(10)]
        #
        # landmarks = np.array([lefteye_dx, lefteye_dy, righteye_dx, righteye_dy, nose_dx, nose_dy, leftmouth_dx, leftmouth_dy, rightmouth_dx, rightmouth_dy])

        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),             # x1 of prediction box in original image
                                 np.round((stride * t_index[0]) / scale),             # y1 of prediction box in original image
                                 np.round((stride * t_index[1] + cellsize) / scale),  # x2 of prediction box in original image
                                 np.round((stride * t_index[0] + cellsize) / scale),  # y2 of prediction box in original image
                                                                                      # reconstruct the box in original image
                                 score,
                                 reg,
                                 # landmarks
                                 ])

        return boundingbox.T

    @staticmethod
    def pad(bboxes, w, h):
        """
        when the box is out of the image,
        the out-of-range zone is filled with 0
        Parameters:
        ----------
            bboxes: numpy array, n x_ori 5
                input bboxes
            w: float number
                width of the input image
            h: float number
                height of the input image
        Returns :
        ------
            y_tar, x_tar : numpy array, n x_ori 1
                start point of the bbox in target image
            ey_tar, ex_tar : numpy array, n x_ori 1
                end point of the bbox in target image
            y_ori, x_ori : numpy array, n x_ori 1
                start point of the bbox in original image
            ex_ori, ex_ori : numpy array, n x_ori 1
                end point of the bbox in original image
            tmph, tmpw: numpy array, n x_ori 1
                height and width of the bbox
        """
        # width and height
        tmpw = (bboxes[:, 2] - bboxes[:, 0] + 1).astype(np.int32)
        tmph = (bboxes[:, 3] - bboxes[:, 1] + 1).astype(np.int32)
        numbox = bboxes.shape[0]

        # initialize x_tar,y_tar; ex_tar,ey_tar
        x_tar = np.zeros((numbox, ))
        y_tar = np.zeros((numbox, ))
        ex_tar, ey_tar = tmpw.copy()-1, tmph.copy()-1
        # initialize x_ori,y_ori; ex_ori,xy
        x_ori, y_ori, ex_ori, ey_ori = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        # 开始处理当box在界外时的情况*******************

        tmp_index = np.where(ex_ori > w-1)
        # tmpw - 1 - (ex_ori - w + 1)
        ex_tar[tmp_index] = tmpw[tmp_index] + w - 2 - ex_ori[tmp_index]
        ex_ori[tmp_index] = w - 1

        tmp_index = np.where(ey_ori > h-1)
        # tmph - 1 - (ey_ori - h + 1)
        ey_tar[tmp_index] = tmph[tmp_index] + h - 2 - ey_ori[tmp_index]
        ey_ori[tmp_index] = h - 1

        tmp_index = np.where(x_ori < 0)
        x_tar[tmp_index] = 0 - x_ori[tmp_index]
        x_ori[tmp_index] = 0

        tmp_index = np.where(y_ori < 0)
        y_tar[tmp_index] = 0 - y_ori[tmp_index]
        y_ori[tmp_index] = 0

        return_list = [y_tar, ey_tar, x_tar, ex_tar, y_ori, ey_ori, x_ori, ex_ori, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list

    def detect_pnet(self, im):
        """PNet forward

        Parameters:
        ----------
        im: numpy array
            input image array
            one batch

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_align: numpy array
            boxes after calibration (with box regression)
        """

        # im = self.unique_image_format(im)

        # original wider face data
        # h, w, c = im.shape

        net_size = 12

        current_scale = net_size / self.min_face_size    # find initial scale
        im_resized = cv2.resize(im, None, fx=current_scale, fy=current_scale, interpolation=cv2.INTER_LINEAR)
        current_height, current_width, _ = im_resized.shape

        # fcn
        all_boxes = list()
        # 每个循环处理图像金字塔的一层
        while min(current_height, current_width) > net_size:
            image_tensor = image_tools.convert_image_to_tensor(im_resized)
            feed_imgs = torch.stack([image_tensor])

            if self.pnet_detector.use_cuda:
                feed_imgs = feed_imgs.cuda()

            # self.pnet_detector is a trained pnet torch model

            # receptive field is 12×12
            # cla_map --> **score_map
            # reg --> **bounding box regression
            cls_map, reg = self.pnet_detector(feed_imgs)

            cls_map_np = image_tools.convert_chwTensor_to_hwcNumpy(cls_map.cpu())
            reg_np = image_tools.convert_chwTensor_to_hwcNumpy(reg.cpu())
            # print(cls_map_np.shape, reg_np.shape) # cls_map_np = (1, n, m, 1) reg_np.shape = (1, n, m 4)
            # time.sleep(5)
            # landmark_np = image_tools.convert_chwTensor_to_hwcNumpy(landmark.cpu())

            # self.threshold[0] = 0.6
            # print(cls_map_np[0,:,:].shape)
            # time.sleep(4)

            # boxes = [x1, y1, x2, y2, score, reg]
            boxes = MtcnnDetector.generate_bounding_box(cls_map_np[0, :, :], reg_np, current_scale, self.thresh[0])

            # generate pyramid images
            current_scale *= self.scale_factor  # self.scale_factor = 0.709
            im_resized = cv2.resize(im, None, fx=current_scale, fy=current_scale, interpolation=cv2.INTER_LINEAR)
            current_height, current_width, _ = im_resized.shape

            if boxes.size == 0:
                continue

            # 每层金字塔都进行一次non-maximum suppresion
            # 由于是单层金字塔内的，每个候选框面积一样，所以overlap threshold可以设置的
            keep = utils.nms(boxes[:, :5], 0.4, 'Minimum')  # dont need reg
            boxes = boxes[keep]
            all_boxes.append(boxes)
        # end of while:

        # got no boxes
        if len(all_boxes) == 0:
            return None, None

        all_boxes = np.vstack(all_boxes)

        # merge the detection from first stage
        keep = utils.nms(all_boxes[:, :5], 0.6, 'Union')
        all_boxes = all_boxes[keep]

        # box_width = x2 - x1
        # box_height = y2 - y1
        bw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        # 用regression校正bbox
        # boxes = boxes = [x1, y1, x2, y2, score, reg] reg= [px1, py1, px2, py2] (in prediction)
        align_topx = all_boxes[:, 0] + all_boxes[:, 5] * bw
        align_topy = all_boxes[:, 1] + all_boxes[:, 6] * bh
        align_bottomx = all_boxes[:, 2] + all_boxes[:, 7] * bw
        align_bottomy = all_boxes[:, 3] + all_boxes[:, 8] * bh

        # refine the boxes
        boxes_align = np.vstack([align_topx,
                                 align_topy,
                                 align_bottomx,
                                 align_bottomy,
                                 all_boxes[:, 4],
                                 ])
        boxes_align = boxes_align.T

        return boxes, boxes_align

    def detect_rnet(self, im, dets):
        """Get face candidates using rnet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of pnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_align: numpy array
            boxes after calibration
        """
        # im: an input image
        h, w, c = im.shape

        if dets is None:
            return None, None

        # (705, 5) = [x1, y1, x2, y2, score, reg]

        # convert to square boxes(匹配感受域)
        dets = self.square_bbox(dets)
        # rounds
        dets[:, 0:4] = np.round(dets[:, 0:4])

        # 处理当box位置在img外面的情况(padding)
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]

        '''
        # helper for setting RNet batch size
        batch_size = self.rnet_detector.batch_size
        ratio = float(num_boxes) / batch_size
        if ratio > 3 or ratio < 0.3:
            print "You may need to reset RNet batch size if this info appears frequently, \
        face candidates:%d, current batch_size:%d"%(num_boxes, batch_size)
        '''

        # 将pnet得到的box从图上切下来放入tmp, resize之后convert2tensor
        cropped_ims_tensors = []
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = im[y[i]:ey[i]+1, x[i]:ex[i]+1, :]

            crop_im = cv2.resize(tmp, (24, 24))

            cropped_ims_tensors.append(image_tools.convert_image_to_tensor(crop_im))

        feed_imgs = torch.stack(cropped_ims_tensors)

        # send data to GPU
        if self.rnet_detector.use_cuda:
            feed_imgs = feed_imgs.cuda()

        # receptive field is 24x24
        # cls_map --> score_map
        # reg --> **bounding box regression
        cls_map, reg = self.rnet_detector(feed_imgs)

        cls_map = cls_map.cpu().data.numpy()
        reg = reg.cpu().data.numpy()
        # landmark = landmark.cpu().data.numpy()

        # keep the boxes whose score > thresh[1]
        keep_inds = np.where(cls_map > self.thresh[1])[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            cls = cls_map[keep_inds]
            reg = reg[keep_inds]
            # landmark = landmark[keep_inds]
        else:
            return None, None

        # 再来一轮nms，实质是对pnet regression之后的box再进行nms
        keep = utils.nms(boxes, 0.6)

        if len(keep) == 0:
            return None, None

        # ★★★★★
        # cls -> [score_of_each_box_in_rnet]
        # boxes -> [x1, y1, x2, y2, score_in_pnet]
        # reg -> [dx1, dy1, dx2, dy2]
        keep_cls = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg = reg[keep]
        # keep_landmark = landmark[keep]

        # ***********整理数据格式***********
        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1

        keep_boxes[:, 4] = keep_cls[:, 0]  # replace the score in pnet with onet

        # 用regression校正bbox
        align_topx = keep_boxes[:, 0] + keep_reg[:, 0] * bw
        align_topy = keep_boxes[:, 1] + keep_reg[:, 1] * bh
        align_bottomx = keep_boxes[:, 2] + keep_reg[:, 2] * bw
        align_bottomy = keep_boxes[:, 3] + keep_reg[:, 3] * bh

        boxes_align = np.vstack([align_topx,
                                 align_topy,
                                 align_bottomx,
                                 align_bottomy,
                                 keep_cls[:, 0],
                                 ])
        boxes_align = boxes_align.T

        return keep_boxes, boxes_align

    def detect_onet(self, im, dets):
        """Get face candidates using onet
        内容基本和rnet一样，不同的是神经网络返回了landmarks，
        并且只返回校正之后的box和landmark
        *Parameters:
            im: numpy array
                input image array
            dets: numpy array -> [x1, y1, x2, y2, score_in_onet]
                detection results of rnet
        *Returns:
            （都是绝对座标）
            boxes_align: -> [x1, y1, x2, y2, score_in_0net]
                boxes after calibration
            landmarks_align: -> [x1, y1, x2, y2, ..., x5, y5]
                landmarks after calibration
        """
        h, w, c = im.shape

        if dets is None:
            return None, None

        # convert to square boxes(匹配感受域)
        dets = self.square_bbox(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        # 处理当box位置在img外面的情况(padding)
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]

        # cropped_ims_tensors = np.zeros((num_boxes, 3, 24, 24), dtype=np.float32)
        cropped_ims_tensors = []
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            # crop input image
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]

            crop_im = cv2.resize(tmp, (48, 48))

            cropped_ims_tensors.append(image_tools.convert_image_to_tensor(crop_im))

        feed_imgs = Variable(torch.stack(cropped_ims_tensors))

        if self.rnet_detector.use_cuda:
            feed_imgs = feed_imgs.cuda()

        # landmark is calculated
        cls_map, reg, landmark = self.onet_detector(feed_imgs)

        cls_map = cls_map.cpu().data.numpy()
        reg = reg.cpu().data.numpy()
        landmark = landmark.cpu().data.numpy()

        keep_inds = np.where(cls_map > self.thresh[2])[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            cls = cls_map[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None

        # 将nms放到regression之后
        # keep = utils.nms(boxes, 0.6)

        # ★★★★★
        # cls -> [score_of_each_box_in_onet]
        # boxes -> [x1, y1, x2, y2, score_in_rnet]
        # reg -> [dx1, dy1, dx2, dy2]
        # landmark -> [x1, y1, x2, y2, ..., x5, y5]
        keep_cls = cls  # [keep]
        keep_boxes = boxes  # [keep]
        keep_reg = reg  # [keep]
        keep_landmark = landmark  # [keep]

        # ***********整理数据格式***********
        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1

        # 用regression校正bbox
        align_topx = keep_boxes[:, 0] + keep_reg[:, 0] * bw
        align_topy = keep_boxes[:, 1] + keep_reg[:, 1] * bh
        align_bottomx = keep_boxes[:, 2] + keep_reg[:, 2] * bw
        align_bottomy = keep_boxes[:, 3] + keep_reg[:, 3] * bh

        boxes_align = np.vstack([align_topx,
                                 align_topy,
                                 align_bottomx,
                                 align_bottomy,
                                 keep_cls[:, 0],  # score in onet
                                 ])

        boxes_align = boxes_align.T

        # 得到landmark在原始图像中的绝对坐标
        align_landmark_topx = keep_boxes[:, 0]
        align_landmark_topy = keep_boxes[:, 1]
        landmark = np.vstack([
                                 align_landmark_topx + keep_landmark[:, 0] * bw,
                                 align_landmark_topy + keep_landmark[:, 1] * bh,
                                 align_landmark_topx + keep_landmark[:, 2] * bw,
                                 align_landmark_topy + keep_landmark[:, 3] * bh,
                                 align_landmark_topx + keep_landmark[:, 4] * bw,
                                 align_landmark_topy + keep_landmark[:, 5] * bh,
                                 align_landmark_topx + keep_landmark[:, 6] * bw,
                                 align_landmark_topy + keep_landmark[:, 7] * bh,
                                 align_landmark_topx + keep_landmark[:, 8] * bw,
                                 align_landmark_topy + keep_landmark[:, 9] * bh,
                                 ])

        landmark_align = landmark.T

        keep = utils.nms(boxes_align, 0.5, mode="Minimum")
        boxes_align = boxes_align[keep]
        landmark_align = landmark_align[keep]

        return boxes_align, landmark_align

    def detect_face(self, img):
        """
        ***the most important
        Detect face over image, pnet -> rnet -> onet
        returns:
            bboxs, landmarks
        """
        boxes_align = np.array([])
        landmark_align = np.array([])

        t = time.clock()
        # pnet
        if self.pnet_detector is not None:
            boxes, boxes_align = self.detect_pnet(img)
            if boxes_align is None:
                return np.array([]), np.array([])
        p_time = time.clock() - t

        t = time.clock()
        # rnet
        if self.rnet_detector is not None:
            boxes, boxes_align = self.detect_rnet(img, boxes_align)
            if boxes_align is None:
                return np.array([]), np.array([])
        r_time = time.clock() - t

        t = time.clock()
        # onet
        if self.onet_detector is not None:
            boxes_align, landmark_align = self.detect_onet(img, boxes_align)
            if boxes_align is None:
                return np.array([]), np.array([])
        o_time = time.clock() - t
        print("time: {:.5f}ms".format((p_time+r_time+o_time)*1000))
        print(torch.cuda.memory_cached(0)/1024/1024)
        return boxes_align, landmark_align
