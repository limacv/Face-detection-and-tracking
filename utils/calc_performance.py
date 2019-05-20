import numpy as np


def intersect(box_a, box_b):
    """ We resize both arrays to [A,B,2] without new malloc:
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
        then calculate the inter area
    Args:
        box_a: npArray
            bounding boxes, Shape: [A,4]. [x1, y1, x2, y2]
        box_b: npArray
            bounding boxes, Shape: [B,4]. [x1, y1, x2, y2]
    Return: npArray
        intersection area, Shape: [A,B].
    """
    A = box_a.shape[0]
    B = box_b.shape[0]
    box_a_x1y1 = np.repeat(np.expand_dims(box_a[:, 2:], 1), B, 1)
    box_a_x2y2 = np.repeat(np.expand_dims(box_a[:, :2], 1), B, 1)
    box_b_x1y1 = np.repeat(np.expand_dims(box_b[:, 2:], 0), A, 0)
    box_b_x2y2 = np.repeat(np.expand_dims(box_b[:, :2], 0), A, 0)

    max_xy = np.minimum(box_a_x1y1, box_b_x1y1)
    min_xy = np.maximum(box_a_x2y2, box_b_x2y2)

    max_xy -= min_xy

    max_xy = np.maximum(max_xy, 0)
    res = max_xy[:, :, 0] * max_xy[:, :, 1]
    return res


def calculate_distance(box_a, box_b):
    A = box_a.shape[0]
    B = box_b.shape[0]
    box_a_x1y1 = np.repeat(np.expand_dims(box_a[:, 2:], 1), B, 1)
    box_a_x2y2 = np.repeat(np.expand_dims(box_a[:, :2], 1), B, 1)
    box_b_x1y1 = np.repeat(np.expand_dims(box_b[:, 2:], 0), A, 0)
    box_b_x2y2 = np.repeat(np.expand_dims(box_b[:, :2], 0), A, 0)

    a_dxdy = box_a_x1y1 - box_a_x2y2
    b_dxdy = box_b_x1y1 - box_b_x2y2
    ca_xy = (box_a_x1y1 + box_a_x2y2) / 2
    cb_xy = (box_b_x1y1 + box_b_x2y2) / 2
    delt_xy = cb_xy - ca_xy
    delt_dxdy = a_dxdy - b_dxdy
    delt_z = (delt_dxdy[:, :, 0] + delt_dxdy[:, :, 1]) / 2
    dis = delt_z * delt_z + delt_xy[:, :, 0] * delt_xy[:, :, 0] + delt_xy[:, :, 1] * delt_xy[:, :, 1]
    dis = dis ** 0.25
    return dis  # [A,B]


def calculate_iou(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.
    Args:
        box_a: npArray
            Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: npArray
            Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: npArray Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    A, B = inter.shape

    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_a = np.repeat(np.expand_dims(area_a, 1), B, 1)

    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    area_b = np.repeat(np.expand_dims(area_b, 0), A, 0)

    union = area_a + area_b - inter
    return inter / union  # [A,B]


def calc_pr(predict, truth, iou_thresh=0.5):
    """
    :param predict: npArray(num_box, 5) dtype = float
        [left_x, bottom_y, right_x, top_y, score]
    :param truth: npArray(num_box, 4)
        [left_x, bottom_y, width, height]
    :param iou_thresh: float
    :return:
        npArray.shape(2, pred_num), truth_num
        [[if_bigger_than_iou_threshs], [scores]]
    """
    truth = np.hstack((truth[:, :2], truth[:, 2:] + truth[:, :2]))
    iou = calculate_iou(truth, predict[:, :4])
    truth_num, _ = iou.shape
    tf = (np.max(iou, 0) > iou_thresh).astype(np.int32)
    return np.vstack((tf, predict[:, 4])), truth_num
