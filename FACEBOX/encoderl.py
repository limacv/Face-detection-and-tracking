# encoding:utf-8

import torch
import math
import itertools
import cv2
import numpy as np
import time


class DataEncoder:
	def __init__(self):
		"""
		compute default boxes
			what is default boxes?
			ANSWER: *all the boxes that have been detected by the network (in order same as the network output),
					*these boxes' position & size are determined by the network's kernel size, etc.
					*because the computations calculating these boxes have lots of overlap,
						so the speed is still fast despite the large quantity of boxes
		"""
		scale = 1024.
		steps = [s / scale for s in (32, 64, 128)]
		sizes = [s / scale for s in (32, 256, 512)]  # 当32改为64时，achor与label匹配的正样本数目更多
		aspect_ratios = ((1, 2, 4), (1,), (1,))
		feature_map_sizes = (32, 16, 8)

		density = [[-3, -1, 1, 3], [-1, 1], [0]]  # density for output layer1
		# density = [[0],[0],[0]] # density for output layer1
		
		num_layers = len(feature_map_sizes)
		boxes = []
		for i in range(num_layers):
			fmsize = feature_map_sizes[i]
			# print(len(boxes))
			for h, w in itertools.product(range(fmsize), repeat=2):
				cx = (w + 0.5)*steps[i]
				cy = (h + 0.5)*steps[i]

				s = sizes[i]
				for j, ar in enumerate(aspect_ratios[i]):
					if i == 0:
						for dx, dy in itertools.product(density[j], repeat=2):
							boxes.append((cx+dx/8.*s*ar, cy+dy/8.*s*ar, s*ar, s*ar))
					else:
						boxes.append((cx, cy, s*ar, s*ar))
		
		self.default_boxes = torch.Tensor(boxes)
		self.default_boxes_np = self.default_boxes.numpy()

	@staticmethod
	def test_iou(self):
		box1 = torch.Tensor([0, 0, 10, 10])
		box1 = box1[None, :]
		box2 = torch.Tensor([[5, 0, 15, 10], [5, 0, 15, 10]])
		# print('iou', self.iou(box1, box2))

	@staticmethod
	def iou(box1, box2):
		"""Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].

		Args:
			box1: (tensor) bounding boxes, sized [b_n,4].
			box2: (tensor) bounding boxes, sized [b_m,4].

		Return:
			(tensor) iou, sized [b_n,b_m].
		"""
		b_n = box1.size(0)
		b_m = box2.size(0)

		lt = torch.max(  # left top
			box1[:, :2].unsqueeze(1).expand(b_n, b_m, 2),  # [b_n,2] -> [b_n,1,2] -> [b_n,b_m,2]
			box2[:, :2].unsqueeze(0).expand(b_n, b_m, 2),  # [b_m,2] -> [1,b_m,2] -> [b_n,b_m,2]
		)

		rb = torch.min(  # right bottom
			box1[:, 2:].unsqueeze(1).expand(b_n, b_m, 2),  # [b_n,2] -> [b_n,1,2] -> [b_n,b_m,2]
			box2[:, 2:].unsqueeze(0).expand(b_n, b_m, 2),  # [b_m,2] -> [1,b_m,2] -> [b_n,b_m,2]
		)

		wh = rb - lt  # [b_n,b_m,2]
		wh[wh < 0] = 0  # clip at 0
		inter = wh[:, :, 0] * wh[:, :, 1]  # [b_n,b_m]

		area1 = (box1[:, 2]-box1[:, 0]) * (box1[:, 3]-box1[:, 1])  # [b_n,]
		area2 = (box2[:, 2]-box2[:, 0]) * (box2[:, 3]-box2[:, 1])  # [b_m,]
		area1 = area1.unsqueeze(1).expand_as(inter)  # [b_n,] -> [b_n,1] -> [b_n,b_m]
		area2 = area2.unsqueeze(0).expand_as(inter)  # [b_m,] -> [1,b_m] -> [b_n,b_m]

		iou = inter / (area1 + area2 - inter)
		return iou

	def test_encode(self, boxes, img, label):
		# box = torch.Tensor([ 0.4003,0.0000,0.8409,0.4295])
		# box = box[None,:]
		# label = torch.LongTensor([1])
		# label = label[None,:]
		loc, conf = self.encode(boxes, label)
		print('conf', type(conf), conf.size(), conf.long().sum())
		print('loc', loc)
		# img = cv2.imread('test1.jpg')
		w, h, _ = img.shape
		for box in boxes:
			cv2.rectangle(img, (int(box[0]*w),int(box[1]*w)), (int(box[2]*w), int(box[3]*w)), (0,255,0))
		
		print(type(conf))
		for i in range(len(self.default_boxes)):
			if conf[i] != 0:
				print(i)
	
		im = img.copy()
		# for i in range(42):
		# 	print(self.default_boxes[i]*w)

		for i in range(32*32*21):
			box_item = self.default_boxes[i]*w
			centerx, centery = int(box_item[0]), int(box_item[1])
			if conf[i] != 0:
				cv2.circle(im, (centerx, centery), 4, (0,255,0))
			else:
				cv2.circle(im, (centerx, centery), 1, (0,0,255))
		box = self.default_boxes[0]
		cv2.rectangle(im, (0,0), (int(box[2]*w), int(box[3]*w)), (0,255,0))
		box = self.default_boxes[16]
		cv2.rectangle(im, (0,0), (int(box[2]*w), int(box[3]*w)), (0,255,0))
		box = self.default_boxes[20]
		cv2.rectangle(im, (0,0), (int(box[2]*w), int(box[3]*w)), (0,255,0))
		cv2.imwrite('test_encoder_0.jpg', im)

		im = img.copy()
		for i in range(32*32*21, 32*32*21+16*16):
			box_item = self.default_boxes[i]*w
			centerx, centery = int(box_item[0]), int(box_item[1])
			if conf[i] != 0:
				cv2.circle(im, (centerx, centery), 4, (0,255,0))
			else:
				cv2.circle(im, (centerx, centery), 2, (0,0,255))
		box = self.default_boxes[32*32*21]
		cv2.rectangle(im, (0,0), (int(box[2]*w), int(box[3]*w)), (0,255,0))
		cv2.imwrite('test_encoder_1.jpg', im)

		im = img.copy()
		for i in range(32*32*21+16*16, len(self.default_boxes)):
			box_item = self.default_boxes[i]*w
			centerx, centery = int(box_item[0]), int(box_item[1])
			if conf[i] != 0:
				cv2.circle(im, (centerx, centery), 4, (0,255,0))
			else:
				cv2.circle(im, (centerx, centery), 2, (0,0,255))
		box = self.default_boxes[32*32*21+16*16]
		cv2.rectangle(im, (0,0), (int(box[2]*w), int(box[3]*w)), (0,255,0))
		cv2.imwrite('test_encoder_2.jpg', im)

		# for i in range(conf.size(0)):
		# 	if conf[i].numpy != 0:
		# 		print()

	def encode(self, boxes, classes, threshold=0.35):
		"""
		boxes:[num_obj, 4]
		default_box (x1,y1,x2,y2)
		return:boxes: (tensor) [num_obj,21824,4]
		classes:class label [obj,]
		"""
		boxes_org = boxes
		
		# print(boxes,classes)
		default_boxes = self.default_boxes  # [21824,4]
		num_default_boxes = default_boxes.size(0)
		num_obj = boxes.size(0)  # 人脸个数
		# print('num_faces {}'.format(num_obj))
		iou = self.iou(
			boxes,
			torch.cat([default_boxes[:, :2] - default_boxes[:, 2:]/2,
					   default_boxes[:, :2] + default_boxes[:, 2:]/2], 1))
		# iou = self.iou(boxes, default_boxes)
		# print('iou size {}'.format(iou.size()))
		max_iou, max_iou_index = iou.max(1)  # 为每一个bounding box不管IOU大小，都设置一个与之IOU最大的default_box
		iou, max_index = iou.max(0)  # 每一个default_boxes对应到与之IOU最大的bounding box上
		
		# print(max(iou))
		max_index.squeeze_(0)  # torch.LongTensor 21824
		iou.squeeze_(0)
		# print('boxes', boxes.size(), boxes, 'max_index', max_index)

		max_index[max_iou_index] = torch.LongTensor(range(num_obj))

		boxes = boxes[max_index]  # [21824,4] 是图像label
		variances = [0.1, 0.2]
		cxcy = (boxes[:, :2] + boxes[:, 2:])/2 - default_boxes[:, :2]  # [21824,2]
		cxcy /= variances[0] * default_boxes[:, 2:]
		wh = (boxes[:, 2:] - boxes[:, :2]) / default_boxes[:, 2:]  # [21824,2]  为什么会出现0宽度？？
		wh = torch.log(wh) / variances[1]  # Variable
		
		inf_flag = wh.abs() > 10000
		if inf_flag.long().sum() is not 0:
			print('inf_flag has true', wh, boxes)
			print('org_boxes', boxes_org)
			print('max_iou', max_iou, 'max_iou_index', max_iou_index)
			raise inf_error

		loc = torch.cat([cxcy, wh], 1)  # [21824,4]
		conf = classes[max_index]  # 其实都是1 [21824,]
		conf[iou < threshold] = 0  # iou小的设为背景
		conf[max_iou_index] = 1
		# 这么设置有问题，loc loss 会导致有inf loss，从而干扰训练，\
		# 去掉后，损失降的更稳定些，是因为widerFace数据集里有的label
		# 做的宽度为0，但是没有被滤掉，是因为max(1)必须为每一个object选择一个
		# 与之对应的default_box，需要修改数据集里的label。

		# ('targets', Variable containing:
		# 318.7500   -1.2500      -inf      -inf
		# org_boxes 0.1338  0.3801  0.1338  0.3801

		return loc, conf

	@staticmethod
	def nms_np(bboxes, scores, threshold=0.5, mode="Union"):
		"""
		method:
			1. 按照score由大到小对所有候选框进行排序
			2. 以score最大的候选框为基准，计算余下候选框的与该基准的重叠部分的面积
			3. 按照overlap <= thresh的标准，舍去余下的候选框
			4. 对余下的候选框重复2~3的操作
		:return: indexes to keep
		"""

		x1, y1, x2, y2 = bboxes.transpose()

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
			order = order[1:][ovr < threshold]

		# # 得到所有overlap小于阈值的索引值
		# inds = np.where(ovr < thresh)[0]
		# # 选定所有符合条件的候选框，也即删去了所有不符合条件的候选框
		# order = order[inds + 1]  # +1: 因为ovr数组相比order数组要少第一个元素
		return keep

	@staticmethod
	def nms(bboxes, scores, threshold=0.5):
		"""
		bboxes(tensor) [N,4]
		scores(tensor) [N,]
		"""
		x1 = bboxes[:, 0]
		y1 = bboxes[:, 1]
		x2 = bboxes[:, 2]
		y2 = bboxes[:, 3]
		areas = (x2-x1) * (y2-y1)

		_, order = scores.sort(0, descending=True)
		keep = []
		while order.numel() > 0:
			try:
				i = order[0]
			except IndexError:
				i = order
			keep.append(i)

			if order.numel() == 1:
				break

			xx1 = x1[order[1:]].clamp(min=x1[i])
			yy1 = y1[order[1:]].clamp(min=y1[i])
			xx2 = x2[order[1:]].clamp(max=x2[i])
			yy2 = y2[order[1:]].clamp(max=y2[i])

			w = (xx2-xx1).clamp(min=0)
			h = (yy2-yy1).clamp(min=0)
			inter = w*h

			ovr = inter / (areas[i] + areas[order[1:]] - inter)
			ids = (ovr <= threshold).nonzero().squeeze()
			if ids.numel() == 0:
				break
			order = order[ids+1]
		return torch.LongTensor(keep)

	def decode_np(self, loc, conf, conf_thres=0.35):
		"""
		將预测出的 loc/conf转换成真实的人脸框
		loc [21842,4]
		conf [21824,2]
		"""
		st = time.time()
		score = conf[:, 1].numpy()
		ids = np.where(score > conf_thres)[0]

		cxcy = loc[ids, :2].numpy() * 0.1 * self.default_boxes_np[ids, 2:] + self.default_boxes_np[ids, :2]
		wh = np.exp(loc[ids, 2:].numpy() * 0.2) * self.default_boxes_np[ids, 2:]
		boxes = np.hstack([cxcy-wh/2, cxcy+wh/2])

		keep = self.nms_np(boxes, score[ids])
		end = time.time()
		# print("innertime: {:.5f}".format(end - st))
		return boxes[keep], score[ids][keep]

	def decode_tensor(self, loc, conf):
		"""
		將预测出的 loc/conf转换成真实的人脸框
		loc [21842,4]
		conf [21824,2]
		"""
		variances = [0.1, 0.2]  # decode corresponding to encoder

		# cxcy -> center point [x, y]
		cxcy = loc[:, :2] * variances[0] * self.default_boxes[:, 2:] + self.default_boxes[:, :2]
		# wh -> box [wid, hei]
		wh = torch.exp(loc[:, 2:] * variances[1]) * self.default_boxes[:, 2:]
		boxes = torch.cat([cxcy-wh/2, cxcy+wh/2], 1)  # [21824,4]

		conf[:, 0] = 0.4

		max_conf, labels = conf.max(1)  # [21842,1]
		# print(max_conf)
		# print('labels', labels.long().sum())
		if labels.long().sum() is 0:
			sconf, slabel = conf.max(0)
			max_conf[slabel[0:5]] = sconf[0:5]
			labels[slabel[0:5]] = 1

		ids = labels.nonzero().squeeze(1)
		# print('ids', ids)
		# print('boxes', boxes.size(), boxes[ids])

		keep = self.nms(boxes[ids], max_conf[ids])  # .squeeze(1))

		return boxes[ids][keep], labels[ids][keep], max_conf[ids][keep]


if __name__ == '__main__':
	dataencoder = DataEncoder()
	# dataencoder.test_iou()
	dataencoder.test_encode()
	# print((dataencoder.default_boxes))
	# boxes = torch.Tensor([[-8,-8,24,24],[400,400,500,500]])/1024
	# dataencoder.encode(boxes,torch.Tensor([1,1]))
