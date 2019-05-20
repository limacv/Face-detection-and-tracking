import cv2
import torch
import torch.utils.data as data
import numpy as np


class Data_collector(data.Dataset):
    """
    input is image, target is annotation
    Arguments:
        transform: object of PyramidAugmentation(640, (104, 117, 123)) in augmentations.py
            will image_transform the image & box to enhance the performance

        target_transform: object of AnnotationTransform() in widerface.py
            pass in caption string, return tensor of word indices
        dataset_name
    """

    def __init__(self, anno_file):
        self.anno_file = anno_file
        self.ids = list()
        self.annotation = list()
        for line in open(self.anno_file, 'r'):
            filename = line.strip().split()[0]
            self.ids.append(filename)
            self.annotation.append(line.strip().split()[1:])

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < len(self):
            im, gt = self.pull_item(self.count)
            self.count += 1
        else:
            raise StopIteration

        return im, gt, self.count

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_path = self.ids[index]

        target = self.annotation[index]
        num = int(target[0])
        del target[0]
        target = np.array(target).astype(np.int32)
        target = target.reshape(num, 4)

        img = cv2.imread(img_path)
        height, width, _ = img.shape

        return img, target
