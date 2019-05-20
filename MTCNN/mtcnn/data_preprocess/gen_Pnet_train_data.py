"""
    generate positive, negative, positive 12*12_faces
        and their labels based on anno file & image
    input:
        anno_train_modified.txt (anno file)
        WIDER_train\images (images)
    output:
        \train\12\xxx (12*12 faces)
        \anno_store\xxx_12.txt (anno file)
"""
import sys
import numpy as np
import cv2
import os

from mtcnn.data_preprocess.utils import compute_iou
project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())

prefix = ''
anno_file = project_dir + r"\anno_store\anno_train_modified.txt"
im_dir = project_dir + r"\data_set\face_detection\WIDER_train\images"
pos_save_dir = project_dir + r"\data_set\train\12\positive"
part_save_dir = project_dir + r"\data_set\train\12\part"
neg_save_dir = project_dir + r"\data_set\train\12\negative"

if not os.path.exists(pos_save_dir):
    os.mkdir(pos_save_dir)
if not os.path.exists(part_save_dir):
    os.mkdir(part_save_dir)
if not os.path.exists(neg_save_dir):
    os.mkdir(neg_save_dir)

# store labels of positive, negative, part images
f1 = open(os.path.join(project_dir, 'anno_store', 'pos_12.txt'), 'w')
f2 = open(os.path.join(project_dir, 'anno_store', 'neg_12.txt'), 'w')
f3 = open(os.path.join(project_dir, 'anno_store', 'part_12.txt'), 'w')

# anno_file: store labels of the wider face training data
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print("%d pics in total" % num)

p_idx = 0  # positive index
n_idx = 0  # negative index
d_idx = 0  # dont care index
idx = 0  # image index
box_idx = 0  # box index
# 对于anno文件中的每一行（每一幅图片）：
for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = os.path.join(prefix, annotation[0])
    print(im_path)
    # bbox = list(map(int, annotation[1:]))
    # boxes = np.array(annotation[1:], dtype=np.int32).reshape(-1, 4)
    boxes = np.array(annotation[1:], dtype=np.int32, copy=True).reshape(-1, 4)
    img = cv2.imread(im_path)
    idx += 1
    if idx % 100 == 0:
        print(idx, "images done")

    height, width, channel = img.shape

    # 从图中任意(位置和大小都任意)截取，以IOU < 0.3为标准，获得negative的图片
    neg_num = 0
    while neg_num < 50:
        size = np.random.randint(12, min(width, height) / 2, dtype=np.int32)
        nx = np.random.randint(0, width - size, dtype=np.int32)
        ny = np.random.randint(0, height - size, dtype=np.int32)
        crop_box = np.array([nx, ny, nx + size, ny + size])

        Iou = compute_iou(crop_box, boxes)

        cropped_im = img[ny: ny + size, nx: nx + size, :]
        resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
            f2.write(save_file + ' 0\n')  # anno file
            cv2.imwrite(save_file, resized_im)  # image
            n_idx += 1
            neg_num += 1

    # 对于每幅图中的每个box
    for box in boxes:
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        # w = x2 - x1 + 1
        # h = y2 - y1 + 1
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue

        # generate negative examples that have overlap with gt
        for i in range(5):
            size = np.random.randint(12, min(width, height) / 2)
            # delta_x and delta_y are offsets of (x1, y1)

            delta_x = np.random.randint(max(-size, -x1), w)
            delta_y = np.random.randint(max(-size, -y1), h)
            nx1 = max(0, x1 + delta_x)
            ny1 = max(0, y1 + delta_y)
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])
            Iou = compute_iou(crop_box, boxes)

            cropped_im = img[ny1: ny2, nx1: nx2, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1

        # generate positive examples and part faces
        for i in range(20):
            size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            # delta here is the offset of box center
            delta_x = np.random.randint(-w * 0.2, w * 0.2)
            delta_y = np.random.randint(-h * 0.2, h * 0.2)

            nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
            ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])
            box_ = box.reshape(1, -1)
            Iou = compute_iou(crop_box, box_)

            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)

            cropped_im = img[int(ny1): int(ny2), int(nx1): int(nx2), :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            if Iou >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            elif Iou >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1
        box_idx += 1
        print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))

f1.close()
f2.close()
f3.close()
