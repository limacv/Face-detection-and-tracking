import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import face
import time
import os


class SSHContext(nn.Module):
    def __init__(self, channels, xchannels=256):
        """
        :param channels: input channels #
        :param xchannels: output channels# = xc *2
        size remains unchange
        """
        super(SSHContext, self).__init__()

        self.conv1 = nn.Conv2d(channels, xchannels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, xchannels // 2, kernel_size=3, dilation=2, stride=1, padding=2)
        self.conv2_1 = nn.Conv2d(xchannels // 2, xchannels // 2, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(xchannels // 2, xchannels // 2, kernel_size=3, dilation=2, stride=1, padding=2)
        self.conv2_2_1 = nn.Conv2d(xchannels // 2, xchannels // 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x), inplace=True)
        x2 = F.relu(self.conv2(x), inplace=True)
        x2_1 = F.relu(self.conv2_1(x2), inplace=True)
        x2_2 = F.relu(self.conv2_2(x2), inplace=True)
        x2_2 = F.relu(self.conv2_2_1(x2_2), inplace=True)

        return torch.cat([x1, x2_1, x2_2], 1)


class SSHContext_my(nn.Module):
    def __init__(self, channels, xchannels=256):
        """
        :param channels: input channels #
        :param xchannels: output channels# = xc *2
        size remains unchange
        """
        super(SSHContext_my, self).__init__()

        self.conv1 = Mobilenetv1(channels, xchannels, kernel_size=3, stride=1, padding=1)
        self.conv2 = Mobilenetv1(channels, xchannels // 2, kernel_size=3, dilation=2, stride=1, padding=2)
        self.conv2_1 = Mobilenetv1(xchannels // 2, xchannels // 2, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = Mobilenetv1(xchannels // 2, xchannels // 2, kernel_size=3, dilation=2, stride=1, padding=2)
        self.conv2_2_1 = Mobilenetv1(xchannels // 2, xchannels // 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x), inplace=True)
        x2 = F.relu(self.conv2(x), inplace=True)
        x2_1 = F.relu(self.conv2_1(x2), inplace=True)
        x2_2 = F.relu(self.conv2_2(x2), inplace=True)
        x2_2 = F.relu(self.conv2_2_1(x2_2), inplace=True)

        return torch.cat([x1, x2_1, x2_2], 1)


class ContextTexture(nn.Module):
    def __init__(self, up, main):
        """
        :param up: input channels (from upper layer)
        :param main: output channels, also this layer's channel #
        """
        super(ContextTexture, self).__init__()
        self.up_conv = nn.Conv2d(up, main, kernel_size=1)
        self.main_conv = nn.Conv2d(main, main, kernel_size=1)

    def forward(self, up, main):
        up = self.up_conv(up)
        main = self.main_conv(main)
        _, _, H, W = main.size()
        res = F.interpolate(up, scale_factor=2, mode='bilinear', align_corners=False)
        if res.size(2) != H or res.size(3) != W:
            res = res[:, :, 0:H, 0:W]
        res = res + main
        return res


# MobileNetv1
class Mobilenetv1(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(Mobilenetv1, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=bias, groups=in_channel, dilation=dilation)
        self.bn = nn.BatchNorm2d(in_channel)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.activate(x)
        x = self.conv2(x)
        return x


# MobileNetv2
class Mobilenetv2(nn.Module):

    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1, t=4,
                 padding=1, activation=nn.ReLU6, dilation=1, side_way=False, bias=False):
        super(Mobilenetv2, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes * t, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes * t)
        self.conv2 = nn.Conv2d(inplanes * t, inplanes * t, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=bias, groups=inplanes * t, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(inplanes * t)
        self.conv3 = nn.Conv2d(inplanes * t, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.activation = activation(inplace=True)
        self.stride = stride
        self.inplanes = inplanes
        self.outplanes = outplanes
        if side_way:
            assert stride == 1 and self.inplanes == self.outplanes
        self.side_way = side_way

    def forward(self, x):
        if self.side_way:
            ori_x = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.side_way:
            x += ori_x
        return x


class SFD_mobile(nn.Module):
    def __init__(self, phase, num_classes, size):
        # here: num_c=2, size=640
        super(SFD_mobile, self).__init__()
        self.firstTime = True
        self.phase = phase
        self.num_classes = num_classes
        self.priorbox = PriorBoxLayer(size, size, stride=[4, 8, 16, 32, 64, 128])
        self.priors = None  # PriorBoxLayer(width, height)
        self.priorbox_head = PriorBoxLayer(size, size, stride=[8, 16, 32, 64, 128, 128])
        self.priors_head = None  # PriorBoxLayer(width, height)
        self.priorbox_body = PriorBoxLayer(size, size, stride=[16, 32, 64, 128, 128, 128])
        self.priors_body = None  # PriorBoxLayer(width, height)
        self.t = 4
        self.size = size
        self.in_planes = 64

        self.conv1_my = Mobilenetv1(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        # block = BottleNeck
        self.layer1_my = nn.Sequential(
            Mobilenetv2(64, 64, kernel_size=3, stride=1, padding=1, side_way=True),
            Mobilenetv2(64, 64, kernel_size=3, stride=1, padding=1, side_way=True),
            Mobilenetv2(64, 64, kernel_size=3, stride=1, padding=1, side_way=True)
        )
        self.layer1_adj = nn.Conv2d(64, 256, kernel_size=1, bias=False)
        self.layer2_my = nn.Sequential(
            Mobilenetv2(64, 64, kernel_size=3, stride=2, padding=1),
            Mobilenetv2(64, 64, kernel_size=3, stride=1, padding=1, side_way=True),
            Mobilenetv2(64, 64, kernel_size=3, stride=1, padding=1, side_way=True),
            Mobilenetv2(64, 128, kernel_size=3, stride=1, padding=1)
        )
        self.layer2_adj = nn.Conv2d(128, 512, kernel_size=1, bias=False)
        self.layer3_my = nn.Sequential(
            Mobilenetv2(128, 128, kernel_size=3, stride=2, padding=1, t=2),
            Mobilenetv2(128, 128, kernel_size=3, stride=1, padding=1, t=2, side_way=True),
            Mobilenetv2(128, 128, kernel_size=3, stride=1, padding=1, t=2, side_way=True),
            Mobilenetv2(128, 128, kernel_size=3, stride=1, padding=1, t=2, side_way=True),
            Mobilenetv2(128, 128, kernel_size=3, stride=1, padding=1, t=2, side_way=True),
            Mobilenetv2(128, 256, kernel_size=3, stride=1, padding=1, t=2)
        )
        self.layer3_adj = nn.Conv2d(256, 1024, kernel_size=1, bias=False)
        self.layer4_my = nn.Sequential(
            Mobilenetv2(256, 256, kernel_size=3, stride=2, padding=1),
            Mobilenetv2(256, 256, kernel_size=3, stride=1, padding=1, side_way=True),
            Mobilenetv2(256, 512, kernel_size=3, stride=1, padding=1)
        )
        self.layer4_adj = nn.Conv2d(512, 2048, kernel_size=1, bias=False)
        self.layer5_my = Mobilenetv2(512, 512, kernel_size=3, stride=2, bias=True)
        self.layer6_my = Mobilenetv2(512, 256, kernel_size=3, stride=2, bias=True)

        self.smooth_c3_my = Mobilenetv1(256, 256, kernel_size=3, padding=1, bias=True)
        self.smooth_c4_my = Mobilenetv1(512, 512, kernel_size=3, padding=1, bias=True)
        self.smooth_c5_my = Mobilenetv1(1024, 1024, kernel_size=3, padding=1, bias=True)

        self.latlayer_fc_my = nn.Conv2d(2048, 2048, kernel_size=1, groups=4)
        self.latlayer_c6_my = nn.Conv2d(512, 512, kernel_size=1, groups=2)
        self.latlayer_c7_my = nn.Conv2d(256, 256, kernel_size=1, groups=1)

        self.conv3_ct_py = ContextTexture(up=512, main=256)
        self.conv4_ct_py = ContextTexture(up=1024, main=512)
        self.conv5_ct_py = ContextTexture(up=2048, main=1024)

        self.conv2_SSH = SSHContext(256, 256)
        self.conv3_SSH = SSHContext(512, 256)
        self.conv4_SSH = SSHContext(1024, 256)
        self.conv5_SSH = SSHContext(2048, 256)
        self.conv6_SSH = SSHContext(512, 256)
        self.conv7_SSH = SSHContext(256, 256)

        loc = []
        conf = []
        for i in range(6):
            loc.append(nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1))
            conf.append(nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1))

        self.face_loc = nn.ModuleList(loc)
        self.face_conf = nn.ModuleList(conf)

        head_loc = []
        head_conf = []
        for i in range(5):
            head_loc.append(nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1))
            head_conf.append(nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1))

        self.head_loc = nn.ModuleList(head_loc)
        self.head_conf = nn.ModuleList(head_conf)

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 750, 0.3, 0.5)

    def forward(self, x):
        # Bottom-up

        sources = list()
        loc = list()
        conf = list()
        head_loc = list()
        head_conf = list()

        t0 = time.time()
        # BackBone Layer - SF3D ***************************************************
        c1 = F.relu(self.bn1(self.conv1_my(x)), inplace=True)
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1_my(c1)  # S4
        c3 = self.layer2_my(c2)  # S8
        c4 = self.layer3_my(c3)  # S16
        c5 = self.layer4_my(c4)  # S32
        c6 = self.layer5_my(c5)  # S64
        c7 = self.layer6_my(c6)  # S128

        c2 = self.layer1_adj(c2)
        c3 = self.layer2_adj(c3)
        c4 = self.layer3_adj(c4)
        c5 = self.layer4_adj(c5)
        intermedia = [c2, c3, c4, c5, c6, c7]

        t1 = time.time()
        # LFPN ****************************************************
        c5_lat = self.latlayer_fc_my(c5)
        c6_lat = self.latlayer_c6_my(c6)
        c7_lat = self.latlayer_c7_my(c7)

        c4_fuse = self.conv5_ct_py(c5_lat, c4)
        c3_fuse = self.conv4_ct_py(c4_fuse, c3)
        c2_fuse = self.conv3_ct_py(c3_fuse, c2)

        c2_fuse = self.smooth_c3_my(c2_fuse)
        c3_fuse = self.smooth_c4_my(c3_fuse)
        c4_fuse = self.smooth_c5_my(c4_fuse)

        t2 = time.time()
        # Context-sensitive Predict Layers part 1********************************

        # get sources
        c2_fuse = self.conv2_SSH(c2_fuse)
        sources.append(c2_fuse)
        c3_fuse = self.conv3_SSH(c3_fuse)
        sources.append(c3_fuse)
        c4_fuse = self.conv4_SSH(c4_fuse)
        sources.append(c4_fuse)
        c5_lat = self.conv5_SSH(c5_lat)
        sources.append(c5_lat)
        c6_lat = self.conv6_SSH(c6_lat)
        sources.append(c6_lat)
        c7_lat = self.conv7_SSH(c7_lat)
        sources.append(c7_lat)
        sources = [c2_fuse, c3_fuse, c4_fuse, c5_lat, c6_lat, c7_lat]

        # Calculate the prior boxes for face/head/body ######<begin>
        # this block of code will only be executed once
        if self.firstTime:
            self.firstTime = False
            prior_boxs = []
            prior_head_boxes = []
            for idx, f_layer in enumerate(sources):
                # print('source size:',sources[idx].size())
                prior_boxs.append(self.priorbox(idx, f_layer.shape[3], f_layer.shape[2]))
                if idx > 0:
                    prior_head_boxes.append(self.priorbox_head(idx - 1, f_layer.shape[3], f_layer.shape[2]))
                # if idx > 1:
                #     prior_body_boxes.append(self.priorbox_body.forward(idx-2, f_layer.shape[3], f_layer.shape[2]))

            self.priors = torch.cat([p for p in prior_boxs], 0).cuda()
            self.priors_head = torch.cat([p for p in prior_head_boxes], 0).cuda()
            # self.priors_body = torch.cat([p for p in prior_body_boxes],0)
        # ######<end>

        tt = time.time()
        # Context-sensitive Predict Layers part 2***********************************
        # get face_loc & face_conf(max-in-out)
        for idx, (x, loc_net, conf_net) in enumerate(zip(sources, self.face_loc, self.face_conf)):
            if idx == 0:
                tmp_conf = conf_net(x)
                a, b, conf_net, pos_conf = tmp_conf.chunk(4, 1)
                neg_conf = torch.cat([a, b, conf_net], 1)
                max_conf, _ = neg_conf.max(1)
                max_conf = max_conf.view_as(pos_conf)
                conf.append(torch.cat([max_conf, pos_conf], 1).permute(0, 2, 3, 1).contiguous())
            else:
                tmp_conf = conf_net(x)
                neg_conf, a, b, conf_net = tmp_conf.chunk(4, 1)
                pos_conf = torch.cat([a, b, conf_net], 1)
                max_conf, _ = pos_conf.max(1)
                max_conf = max_conf.view_as(neg_conf)
                conf.append(torch.cat([neg_conf, max_conf], 1).permute(0, 2, 3, 1).contiguous())
            loc.append(loc_net(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # get head_loc & head_conf(max-in-out)
        for idx, (x, loc_net, conf_net) in enumerate(zip(sources[1:], self.head_loc, self.head_conf)):
            head_loc.append(loc_net(x).permute(0, 2, 3, 1).contiguous())
            head_conf.append(conf_net(x).permute(0, 2, 3, 1).contiguous())

        head_loc = torch.cat([o.view(o.size(0), -1) for o in head_loc], 1)
        head_conf = torch.cat([o.view(o.size(0), -1) for o in head_conf], 1)

        t3 = time.time()
        # output layer *************************************************

        if self.phase == "test":
            loc = loc.view(loc.size(0), -1, 4)
            conf = self.softmax(conf.view(conf.size(0), -1, 2))

            output = self.detect(
                loc,  # loc preds
                conf,  # conf preds
                self.priors  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, 2),
                self.priors,
                head_loc.view(head_loc.size(0), -1, 4),
                head_conf.view(head_conf.size(0), -1, 2),
                self.priors_head
            )
        t4 = time.time()
        # print("backbone:{:.4f}ms | LFPN:{:.4f}ms | CPL1:{:.4f}ms | CPL2:{:.4f}ms | O:{:.4f}ms"
        #       .format((t1-t0)*1000, (t2-t1)*1000, (tt-t2)*1000, (t3-tt)*1000, (t4-t3)*1000))
        return output, intermedia, sources

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            pretrained_model = torch.load(base_file, map_location=lambda storage, loc: storage)
            model_dict = self.state_dict()
            pretrained_model = {k: v for k, v in pretrained_model.items() if k in model_dict}
            model_dict.update(pretrained_model)
            self.load_state_dict(model_dict)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def build_sfd_mobile(phase, size=640, num_classes=2):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 640:
        print("Error: Sorry only 640 is supported currently!")
        return
    return SFD_mobile(phase, num_classes, size)
