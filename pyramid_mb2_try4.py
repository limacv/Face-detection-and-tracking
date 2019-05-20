import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import face
import time
import os


def conv_bn(inp, oup, stride, use_batch_norm=True, onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 7, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 7, stride, 1, bias=False),
            ReLU(inplace=True)
        )


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


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_batch_norm=True, onnx_compatible=False):
        super(InvertedResidual, self).__init__()
        ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )
        else:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SFD_mobile(nn.Module):
    def __init__(self, phase, num_classes, size):
        # here: num_c=2, size=640
        super(SFD_mobile, self).__init__()
        self.firstTime = True
        self.phase = phase
        self.num_classes = num_classes
        self.priorbox = PriorBoxLayer(size, size, stride=[4, 8, 16, 32, 64], box=(16, 32, 64, 128, 256))
        self.priors = None  # PriorBoxLayer(width, height)
        self.priorbox_head = PriorBoxLayer(size, size, stride=[8, 16, 32, 64, 64], box=(16, 32, 64, 128, 256))
        self.priors_head = None  # PriorBoxLayer(width, height)
        self.size = size
        self.in_planes = 32
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building inverted residual blocks
        layers = [conv_bn(3, self.in_planes, 2)]
        for t, c, n, s in self.cfgs:
            layers.append(InvertedResidual(self.in_planes, c, s, t))
            self.in_planes = c
            for i in range(1, n):
                layers.append(InvertedResidual(self.in_planes, c, 1, t))
                self.in_planes = c
        self.features = nn.Sequential(*layers)
        #   0       layer0  conv_bn(3, 32, 2),
        #  1        layer1  self._make_layer(1, 16, 1, 1),
        #  2 3      layer2  self._make_layer(6, 24, 2, 2),
        # 4 5 6     layer3  self._make_layer(6, 32, 3, 2),
        # 7 8 9 10  layer4_0  self._make_layer(6, 64, 4, 2),
        # 11 12 13  layer4_1  self._make_layer(6, 96, 3, 1),
        # 14 15 16  layer5_0  self._make_layer(6, 160, 3, 2),
        #   17      layer5_1  self._make_layer(6, 320, 1, 1)]
        self.layer6 = InvertedResidual(320, 160, 2, 6)

        self.conv2_ct_py = ContextTexture(up=32, main=24)
        self.conv3_ct_py = ContextTexture(up=96, main=32)
        self.conv4_ct_py = ContextTexture(up=320, main=96)

        self.smooth_c2 = nn.Sequential(InvertedResidual(24, 24, 1, 4),
                                       nn.Conv2d(24, 24, kernel_size=3, padding=1))
        self.smooth_c3 = nn.Sequential(InvertedResidual(32, 32, 1, 4),
                                       nn.Conv2d(32, 32, kernel_size=3, padding=1))
        self.smooth_c4 = nn.Sequential(InvertedResidual(96, 96, 1, 2),
                                       nn.Conv2d(96, 96, kernel_size=3, padding=1))
        self.smooth_c5 = nn.Conv2d(320, 320, kernel_size=1, padding=1)
        self.smooth_c6 = nn.Conv2d(160, 160, kernel_size=1, padding=1)

        self.conv2_SSH = SSHContext(24, 128)
        self.conv3_SSH = SSHContext(32, 128)
        self.conv4_SSH = SSHContext(96, 128)
        self.conv5_SSH = SSHContext(320, 128)
        self.conv6_SSH = SSHContext(160, 128)

        loc = []
        conf = []
        for i in range(6):
            loc.append(nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1))
            conf.append(nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1))

        self.face_loc = nn.ModuleList(loc)
        self.face_conf = nn.ModuleList(conf)

        head_loc = []
        head_conf = []
        for i in range(5):
            head_loc.append(nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1))
            head_conf.append(nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1))

        self.head_loc = nn.ModuleList(head_loc)
        self.head_conf = nn.ModuleList(head_conf)

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 750, 0.2, 0.35)

    def forward(self, x):
        # Bottom-up

        sources = list()
        loc = list()
        conf = list()
        head_loc = list()
        head_conf = list()

        t0 = time.time()
        # BackBone Layer - SF3D ***************************************************
        c2 = self.features[3](self.features[2](self.features[1](self.features[0](x))))
        # c2 = self.layer2(self.layer1(self.layer0(x)))
        c3 = self.features[6](self.features[5](self.features[4](c2)))
        # c3 = self.layer3(c2)
        c4 = self.features[10](self.features[9](self.features[8](self.features[7](c3))))
        c4 = self.features[13](self.features[12](self.features[11](c4)))
        # c4 = self.layer4_1(self.layer4_0(c3))
        c5 = self.features[17](self.features[16](self.features[15](self.features[14](c4))))
        # c5 = self.layer5_1(self.layer5_0(c4))
        c6 = self.layer6(c5)

        t1 = time.time()
        # LFPN ****************************************************
        c6 = self.smooth_c6(c6)
        c5 = self.smooth_c5(c5)

        c4 = self.conv4_ct_py(c5, c4)
        c3 = self.conv3_ct_py(c4, c3)
        c2 = self.conv2_ct_py(c3, c2)

        c2 = self.smooth_c2(c2)
        c3 = self.smooth_c3(c3)
        c4 = self.smooth_c4(c4)

        t2 = time.time()
        # Context-sensitive Predict Layers part 1********************************

        # get sources
        c2 = self.conv2_SSH(c2)
        sources.append(c2)
        c3 = self.conv3_SSH(c3)
        sources.append(c3)
        c4 = self.conv4_SSH(c4)
        sources.append(c4)
        c5 = self.conv5_SSH(c5)
        sources.append(c5)
        c6 = self.conv6_SSH(c6)
        sources.append(c6)

        # Calculate the prior boxes for face/head/body ######<begin>
        # this block of code will only be executed once
        # <
        if self.firstTime:
            self.firstTime = False
            prior_boxs = []
            prior_head_boxes = []
            for idx, f_layer in enumerate(sources):
                # print('source size:',sources[idx].size())
                prior_boxs.append(self.priorbox(idx, f_layer.shape[3], f_layer.shape[2]))
                if idx > 0:
                    prior_head_boxes.append(self.priorbox_head(idx - 1, f_layer.shape[3], f_layer.shape[2]))

            self.priors = torch.cat([p for p in prior_boxs], 0).cuda()
            self.priors_head = torch.cat([p for p in prior_head_boxes], 0).cuda()
        # >

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
        return output

    def zero_grad_pretrained_layer(self):
        self.features.zero_grad()

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            pretrained_model = torch.load(base_file, map_location=lambda storage, loc: storage)
            model_dict = self.state_dict()
            pretrained_model = {k: v for k, v in pretrained_model.items() if k in model_dict}
            # del pretrained_model["features.0.0.weight"]
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
