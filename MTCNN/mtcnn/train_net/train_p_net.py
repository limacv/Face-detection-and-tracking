import argparse
import sys
import os
from mtcnn.core.imagedb import ImageDB
from mtcnn.train_net.train import train_pnet
import mtcnn.config as config
project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())


def train_net(annotation_file, model_path,
              end_epoch=16, frequency=200, learn_rate=0.01, batch_size=128, use_cuda=False):

    imagedb = ImageDB(annotation_file)
    gt_imdb = imagedb.load_imdb()
    gt_imdb = imagedb.append_flipped_images(gt_imdb)
    train_pnet(model_store_path=model_path, end_epoch=end_epoch, imdb=gt_imdb, batch_size=batch_size, frequent=frequency, base_lr=learn_rate, use_cuda=use_cuda)


def parse_args():
    parser = argparse.ArgumentParser(description='Train PNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--anno_file', dest='annotation_file',
                        default=os.path.join(config.ANNO_STORE_DIR, config.PNET_TRAIN_IMGLIST_FILENAME), help='training data annotation file', type=str)
    parser.add_argument('--model_path', dest='model_store_path', help='training model store directory',
                        default=config.MODEL_STORE_DIR, type=str)
    parser.add_argument('--end_epoch', dest='end_epoch', help='end epoch of training',
                        default=config.END_EPOCH, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=200, type=int)
    parser.add_argument('--lr', dest='lr', help='learning rate',
                        default=config.TRAIN_LR, type=float)
    parser.add_argument('--batch_size', dest='batch_size', help='train batch size',
                        default=config.TRAIN_BATCH_SIZE, type=int)
    parser.add_argument('--gpu', dest='use_cuda', help='train with gpu',
                        default=config.USE_CUDA, type=bool)
    parser.add_argument('--prefix_path', dest='', help='training data annotation images prefix root path', type=str)

    args = parser.parse_args()
    return args


anno_file = project_dir + r'\anno_store\imglist_anno_12.txt'
model_store_path = project_dir + r'\model_store'
if __name__ == '__main__':
    # args = parse_args()
    print('train Pnet argument:')
    # print(args)

    train_net(anno_file, model_store_path,
              end_epoch=10, frequency=200, learn_rate=0.01, batch_size=512, use_cuda=True)

    # train_net(annotation_file=args.annotation_file, model_store_path=args.model_store_path,
    #             end_epoch=args.end_epoch, frequent=args.frequent, lr=args.lr, batch_size=args.batch_size, use_cuda=args.use_cuda)
