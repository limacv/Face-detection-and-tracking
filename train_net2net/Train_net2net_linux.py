import pdb
import os
import sys
sys.path.append("../")
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import face, AnnotationTransform, Detection, detection_collate
from utils.augmentations import PyramidAugmentation
from layers.modules import MultiBoxLoss
from train_net2net.pyramid_train import build_sfd, SFD
from train_net2net.pyramid_train_mobile import build_sfd_mobile, SFD_mobile
import numpy as np
import time
from layers import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--device', 	default=0,	type=int, help='Training device')
parser.add_argument('--batch_size',     default=6,     type=int, help='Batch size for training')
parser.add_argument('--resume',         default=None, type=str,
                    help='Resume from checkpoint')
parser.add_argument('--start_iter',    default=0,   type=int, help='iteration number of last breakpoint')
parser.add_argument('--num_workers',    default=0,      type=int, help='Number of workers used in dataloading')
parser.add_argument('--train',    default="intermedia",      type=str,
                    help=r"--train 'intermedia' or 'source'")
parser.add_argument('--cuda',                   default=True,   type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate',  default=1e-3,   type=float,
                    help='initial learning rate')
parser.add_argument('--save_folder',            default='./weights_of_mine/',
                    help='Location to save checkpoint models')
parser.add_argument('--annoPath',
                    default="../image_and_anno/anno/gen_anno_file_train",
                    help='Location of annotation')
parser.add_argument('--iter', default=500000, type=int, help='The total iteration time of the training process')
parser.add_argument('--save_point', default=2000, type=int, help='save loss and .pth in every # times of iters')
args = parser.parse_args()

torch.cuda.set_device(args.device)
print("training on device: " + str(args.device))

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

cfg = face

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

num_classes = 2
batch_size = args.batch_size
stepvalues = (3000000, 4000000, 1000000)
gamma = 0.3
moment = 0.5
ssd_net_mobile = build_sfd_mobile('train', 640, num_classes)
ssd_net = build_sfd('train', 640, num_classes)
# intermedia_loss_weight = [6, 6, 8, 8, 10, 12]
intermedia_loss_weight = [12, 12, 12, 12, 8, 8]
source_loss_weight = 128
overall_loss_weight = [4, 1, 0, 4, 1, 0]
if args.cuda:
    # net_raw = torch.nn.DataParallel(ssd_net_mobile, device_ids=[2,])
    # cudnn.benchmark = True
    net_raw = ssd_net_mobile
    net_raw = net_raw.cuda()
    net_tar = ssd_net
    net_tar = net_tar.cuda()
else:
    net_raw = ssd_net_mobile
    net_tar = ssd_net


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        if 'bias' in m.state_dict().keys():
            m.bias.data.zero_()

    if isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)
        if 'bias' in m.state_dict().keys():
            m.bias.data.zero_()

    if isinstance(m, nn.BatchNorm2d):
        m.weight.data[...] = 1
        m.bias.data.zero_()


for layer in net_raw.modules():
    layer.apply(weights_init)

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    net_raw.load_weights(args.resume)
else:
    print('Initializing weights...')

net_tar.load_weights("../net_weight/Res50_pyramid.pth")

optimizer = optim.SGD(net_raw.parameters(), lr=args.lr, momentum=moment, weight_decay=0.0001)
criterion = nn.MSELoss()


def train():
    net_raw.train()
    net_tar.eval()
    # loss counters

    dataset = Detection(args.annoPath, PyramidAugmentation(640, (104, 117, 123)), AnnotationTransform())
    epoch_size = len(dataset) // args.batch_size
    step_index = 0
    batch_iterator = None
    dis_loss = 0.0
    loss_save_idx = 0
    loss_save = np.zeros(args.save_point + 1)

    data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
    print("start training " + args.train + "!")
    for iteration in range(args.start_iter+1, args.iter+1):
        t0 = time.time()
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
        if iteration in stepvalues:
            print("adjusting lr...")
            step_index += 1
            adjust_learning_rate(optimizer, gamma, step_index)

        # images are what only needed
        images, _ = next(batch_iterator)

        if args.cuda:
            images = images.cuda()

        # the most crutial part
        # forward***********************************************************************
        t1 = time.time()
        # pdb.set_trace()
        # intermedia****************************
        if args.train == "intermedia":
            _, intermedia_raw, _ = net_raw(images)
            with torch.no_grad():
                _, intermedia_tar, _ = net_tar(images)

            optimizer.zero_grad()
            loss = []
            final_loss = torch.Tensor([0, ])
            for raw, tar, w in zip(intermedia_raw, intermedia_tar, intermedia_loss_weight):
                temp = criterion(raw, tar)
                loss.append(float(temp))
                final_loss += temp * w

        # source****************************
        elif args.train == "source":
            _, _, src_raw = net_raw(images)
            with torch.no_grad():
                _, _, src_tar = net_tar(images)

            optimizer.zero_grad()
            loss = []
            final_loss = torch.Tensor([0, ])
            for raw, tar in zip(src_raw, src_tar):
                temp = criterion(raw, tar)
                loss.append(float(temp))
                final_loss += temp * source_loss_weight
                # source****************************
        elif args.train == "overall":
            overal_raw, _, _ = net_raw(images)
            with torch.no_grad():
                overal_tar, _, _ = net_tar(images)
            optimizer.zero_grad()
            loss = []
            final_loss = torch.Tensor([0, ])
            for raw, tar, w in zip(overal_raw, overal_tar, overall_loss_weight):
                temp = criterion(raw, tar)
                loss.append(float(temp))
                final_loss += temp * w
        else:
            print("args.train should be intermedia or source")
            break

        final_loss.backward()
        optimizer.step()
        t2 = time.time()
        # end*******************************************

        loss_save[loss_save_idx] = float(final_loss)
        loss_save_idx += 1
        dis_loss += float(final_loss)
        if iteration % 20 == 0:
            print("-----------------------------------------------")
            print('net_raw time: {} sec. || total time: {} sec'.format((t2 - t1), (t2 - t0)))
            print('iter ' + repr(iteration) + ' || Loss: %.4f' % (dis_loss / 20))
            dis_loss = 0.0
            print('each:' + repr(loss))
            print('lr: {}'.format(optimizer.param_groups[0]['lr']))

        if iteration % args.save_point == 0:
            print('Saving state and loss, iter:', iteration)
            torch.save(ssd_net_mobile.state_dict(), args.save_folder + args.train + '_net_' +
                       repr(iteration) + '.pth')
            np.save(args.save_folder + args.train + "_loss_" + repr(iteration) + '.npy', loss_save)
            loss_save_idx = 0
            
    torch.save(ssd_net_mobile.state_dict(), args.save_folder + args.train + '_net_final' + '.pth')


def adjust_learning_rate(optimr, gama, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gama ** step)
    for param_group in optimr.param_groups:
        # param_group['lr'] = param_group['lr'] * gama
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
