import pdb
import os
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
from pyramid_mb2_try4 import build_sfd_mobile, SFD_mobile
import numpy as np
import time
from layers import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--device', 	default=0,	type=int, help='Training device')
parser.add_argument('--batch_size',     default=14,     type=int, help='Batch size for training')
parser.add_argument('--resume',         default="./net_weight/mb2.pth", type=str,
                    help='Resume from checkpoint')
parser.add_argument('--start_iter',    default=0,   type=int, help='iteration number of last breakpoint')
parser.add_argument('--num_workers',    default=0,      type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda',                   default=True,   type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate',  default=1e-4,   type=float,
                    help='initial learning rate')
parser.add_argument('--save_folder',            default='weights_of_mine/try4',
                    help='Location to save checkpoint models')
parser.add_argument('--iter', default=5000000, type=int, help='The total iteration time of the training process')
parser.add_argument('--save_point', default=3000, type=int, help='save loss and .pth in every # times of iters')
parser.add_argument('--train_pretrain', default=500, type=int, help='after #, start training pretrained')
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

annoPath = "./image_and_anno/anno/gen_anno_file_train"
annoPath_val = "./image_and_anno/anno/gen_anno_file_val"
num_classes = 2
batch_size = args.batch_size
stepvalues = (3000000, 4000000, 5000000)
gamma = 0.5
moment = 0.3
ssd_net = build_sfd_mobile('train', 640, num_classes)
eval_freq = 500

if args.cuda:
    # net_raw = torch.nn.DataParallel(ssd_net_mobile, device_ids=[2,])
    # cudnn.benchmark = True
    net = ssd_net
    net = net.cuda()
else:
    net = ssd_net


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


for layer in ssd_net.modules():
    layer.apply(weights_init)

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    ssd_net.load_weights(args.resume)
else:
    print('Initializing weights...')

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=moment, weight_decay=0.0001)
criterion = MultiBoxLoss(num_classes=num_classes,
                         overlap_thresh=0.35,
                         prior_for_matching=True,
                         bkg_label=0,
                         neg_mining=True,
                         neg_pos=3,
                         neg_overlap=0.35,
                         encode_target=False,
                         bipartite=False,
                         use_gpu=args.cuda)  # ★


def train():
    net.train()
    # loss counters
    epoch = 0
    total_loss_eval = list()
    loss_dis = 0
    total_loss = np.zeros(args.save_point + 1)
    face_loss_loc, face_loss_conf = np.zeros(args.save_point + 1), np.zeros(args.save_point + 1)
    head_loss_loc, head_loss_conf = np.zeros(args.save_point + 1), np.zeros(args.save_point + 1)
    loss_save_idx = 0
    print('Loading Dataset...')

    dataset = Detection(annoPath, PyramidAugmentation(640, (104, 117, 123)), AnnotationTransform())
    dataset_eval = Detection(annoPath_val, PyramidAugmentation(640, (104, 117, 123)), AnnotationTransform())
    epoch_size = len(dataset) // args.batch_size

    step_index = 0

    batch_iterator = None
    data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
    data_loader_eval = data.DataLoader(dataset_eval, 2, shuffle=True,
                                       collate_fn=detection_collate, pin_memory=True)
    print("start training!")
    for iteration in range(args.start_iter+1, args.iter+1):
        t0 = time.time()
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
        if iteration in stepvalues:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

            # reset epoch loss counters
            epoch += 1

        # load train data
        images, targets = next(batch_iterator)

        if args.cuda:
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]

        # the most crutial part
        # forward***********************************************************************
        t1 = time.time()
        # pdb.set_trace()
        out = net(images)
        # backprop*********************************************************************
        optimizer.zero_grad()
        loss_l, loss_c = criterion(tuple(out[0:3]), targets)
        loss_l_head, loss_c_head = criterion(tuple(out[3:6]), targets)

        loss = loss_l + loss_c + 0.5 * loss_l_head + 0.5 * loss_c_head
        loss.backward()

        if iteration < args.train_pretrain:
            net.zero_grad_pretrained_layer()

        optimizer.step()
        t2 = time.time()
        # end***************************************************************************

        # save the loss
        total_loss[loss_save_idx] = float(loss)
        face_loss_loc[loss_save_idx], face_loss_conf[loss_save_idx] = float(loss_l), float(loss_c)
        head_loss_loc[loss_save_idx], head_loss_conf[loss_save_idx] = float(loss_l_head), float(loss_c_head)
        loss_save_idx += 1

        loss_dis += float(loss)

        # print information------------------------------------------------------------------
        if iteration % 20 == 0:
            print("-----------------------------------------------")
            print('net time: {:5f} sec. || total time: {:5f} sec'.format((t2 - t1), (t2 - t0)))
            print('iter ' + repr(iteration) + ' || Loss: %.4f' % (loss_dis/20))
            loss_dis = 0
            print('Loss conf: {:5f} || Loss loc: {:5f}'.format(float(loss_c), float(loss_l)))
            print('Loss head conf: {:5f} || Loss head loc: {:5f}'.format(float(loss_c_head), float(loss_l_head)))
            print('lr: {}'.format(optimizer.param_groups[0]['lr']))

        # evaluation-----------------------------------------------------------------------------------------
        if iteration % eval_freq == 0 or iteration == 1:
            print("start evaluating...")
            net.eval()
            loss_val = 0.0
            eval_num = 0
            for img_eval, target_eval in iter(data_loader_eval):
                eval_num += 1
                if args.cuda:
                    img_eval = img_eval.cuda()
                    target_eval = [anno.cuda() for anno in target_eval]
                with torch.no_grad():
                    out_eval = net(img_eval)
                    loss_l_val, loss_c_val = criterion(tuple(out_eval[0:3]), target_eval)
                    loss_l_head_val, loss_c_head_val = criterion(tuple(out_eval[3:6]), target_eval)
                loss_val += float(loss_l_val + loss_c_val + 0.5 * loss_l_head_val + 0.5 * loss_c_head_val)
                if eval_num > 50:
                    break
            total_loss_eval.append(loss_val/eval_num)
            net.train()
            print("evaluating over!" + " eval_loss = {:.5f}".format(loss_val/eval_num))
        # save point--------------------------------------------------------------------------
        if iteration % args.save_point == 0:
            print('Saving state and loss, iter:', iteration)
            torch.save(ssd_net.state_dict(), args.save_folder + 'Mobile_pyramid_' +
                       repr(iteration) + '.pth')
            np.save(args.save_folder + "Mobile_train_loss_" + repr(iteration) + '.npy',
                    np.vstack((total_loss, face_loss_loc, face_loss_conf, head_loss_loc, head_loss_conf)))
            loss_save_idx = 0
            np.save(args.save_folder + "Mobile_eval_loss_" + repr(iteration) + '.npy',
                    np.array(total_loss_eval))
            total_loss_eval = list()

    torch.save(ssd_net.state_dict(), args.save_folder + 'Mobile_pyramid_final' + '.pth')


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
