import numpy as np
import time


def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = np.true_divide(inter,(box_area + area - inter))
    #ovr = inter / (box_area + area - inter)
    return ovr


def convert_to_square(bbox):
    """Convert bbox to square

    Parameters:
    ----------
    bbox: numpy array , shape n x 5
        input bbox

    Returns:
    -------
    square bbox
    """
    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w*0.5 - max_side*0.5
    square_bbox[:, 1] = bbox[:, 1] + h*0.5 - max_side*0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox


# non-maximum suppression: eleminates the box which have large interception with the box which have the largest score
def nms(dets, thresh, mode="Union"):
    """
    method:
        1. 按照score由大到小对所有候选框进行排序
        2. 以score最大的候选框为基准，计算余下候选框的与该基准的重叠部分的面积
        3. 按照overlap <= thresh的标准，舍去余下的候选框
        4. 对余下的候选框重复2~3的操作
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap <= thresh, 越大舍去的越少，越小舍去的越多
    :param mode: 'Union'/'Minimum', Union相对舍去的比较少一点，最好别用min
    :return: indexes to keep
    """

    x1, y1, x2, y2, scores = dets.transpose()

    areas = (x2 - x1) * (y2 - y1)
    # 对scores排序(从小到大)，[::-1] reverse the order
    order = scores.argsort()[::-1]

    # eleminates the box which have large interception with the box which have the largest score in order
    # matain the box with largest score and boxes don't have large interception with it
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 此时i为score最大时的位置

        # 计算i和除了i之外所有box的重叠区域(每个element代表一个框)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        # cacaulate the IOU between box which have largest score with other boxes
        if mode == "Union":
            # area[i]: the area of largest score
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])

        # 第一个不要，后面的按照overlap < thresh的标准来取舍，取重叠少的
        order = order[1:][ovr < thresh]

        # # 得到所有overlap小于阈值的索引值
        # inds = np.where(ovr < thresh)[0]
        # # 选定所有符合条件的候选框，也即删去了所有不符合条件的候选框
        # order = order[inds + 1]  # +1: 因为ovr数组相比order数组要少第一个元素
    return keep
