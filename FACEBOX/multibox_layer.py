# encoding:utf-8
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.autograd import Variable


class MultiBoxLayer(nn.Module):

	num_anchors = [21, 1, 1]
	in_channels = [128, 256, 256]

	def __init__(self):
		super(MultiBoxLayer, self).__init__()
		
		self.loc_layers = nn.ModuleList()
		self.conf_layers = nn.ModuleList()
		for i in range(len(self.in_channels)):
			self.loc_layers.append(nn.Conv2d(self.in_channels[i], self.num_anchors[i] * 4, kernel_size=3, padding=1))
			# *4 -> number of location
			self.conf_layers.append(nn.Conv2d(self.in_channels[i], self.num_anchors[i] * 2, kernel_size=3, padding=1))
			# *2 -> yes or not face

	def forward(self, xs):
		"""
		xs:list of 之前的featuremap
		retrun: loc_preds: [n_,21842,4]
				conf_preds:[n_,24842,2]
		"""
		y_locs = []
		y_confs = []
		for i, x in enumerate(xs):
			y_loc = self.loc_layers[i](x)  # n_, anhors*4, H, W
			n_ = y_loc.size(0)
			y_loc = y_loc.permute(0, 2, 3, 1).contiguous()
			y_loc = y_loc.view(n_, -1, 4)
			y_locs.append(y_loc)

			y_conf = self.conf_layers[i](x)
			y_conf = y_conf.permute(0, 2, 3, 1).contiguous()
			y_conf = y_conf.view(n_, -1, 2)
			y_confs.append(y_conf)

		loc_preds = torch.cat(y_locs, 1)
		conf_preds = torch.cat(y_confs, 1)
		return loc_preds, conf_preds
