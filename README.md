# Face Detector and Tracking

This work implement three tasks:

1. evaluate performance of [MTCNN](https://arxiv.org/abs/1604.02878) & [Faceboxes](https://arxiv.org/abs/1708.05234) & [PyramidBox](https://arxiv.org/abs/1803.07737) ([click to jump](#title1))
2. simplify PyramidBox network structure using [MobileNetV2](https://arxiv.org/abs/1801.04381) structure ([click to jump](#title2))
3. implement multi-face offline tracking with PyramidBox & [IoU tracker](http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf) ([click to jump](#title3))

## Requirements

python 3.6.8  
pytorch 1.1  
other common packages like cv2, numpy, time, argparse...

## Pre-preparation

### 1. Obtain dataset & annotation  

1. download train/evaluate datasets & annotations from [WIDER-FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html)  
2. training images -> ./image_and_anno/images;  
evaluating images -> ./image_and_anno/images_val;  
wider_face_train_bbx_gt.txt & wider_face_val_bbx_gt.txt -> ./image_and_anno/anno
3. run `$ python ./image_and_anno/anno/gen_anno_file_*` to generate anno files in the format we used
4. if it outputs "`error in line: #`" during generating anno files, delete the specified line in generated anno file (this prevents the case when the image has no face, but presents `*.jpg 1 0 0 0 0` in anno file)

### 2. Get pretrained models

1. download pretrained net weight files from <https://pan.baidu.com/s/1yNGFegHFbKIRfs1g93TYgw>  
Extraction code: ntec
2. original_model/ -> ./MTCNN/  
faceboxes.pt -> ./FACEBOX/  
mb2_pretrained.pth, resnet50_pretrained.pth, Res50_pyramid.pth -> ./net_weight/

-----------------------------------
<span id='title1'> </span>

## Task1: performance evaluation  

1. to get detect results in eval dataset for each algorithm, run below commands:  

```terminal
$ python FACEBOX/My_test_facebox.py  
$ python MTCNN/My_test_mtcnn.py  
$ python My_test.py --net repo  
```

this will generate `*.npy` in ./draw_curve/data/, which contains detected bounding boxes and confidences for each picture in eval dataset

2. modify the `data_file_list` in ./draw_curve/draw_pr_roc.py, run and you will get the PR, ROC curve of the corresponding detectors

<span id='title2'> </span>

## ★Task2: simplify PyramidBox

The original PyramidBox is based on ResNet50.  We tried to modify the network structure in pyramid.py and retrain the network, we've tried 5 different modifications. But only try1, try3 seem to get acceptable results.  

- In try[1-2], we modify the network with MobileNetV2 structure, and then pre-train the backbone network with the idea of "__train network with network__"
- In try[3-5], we directly use the __SSDLite__ pretrain model.

### Pretrain backbone(only for try[1-2])

1. to pre-train the backboen network, run this command:  

```terminal
$ python train_net2net/Train_net2net_linux.py
```

2. when the loss converges, move the final *.pth in ./train_net2net/weights_of_mine -> ./net_weight , as the pretrained net weight

### Train overall network

to train the overall network, run:

```terminal
$ python MyTrain_repo.py --resume net_weight/xxx.pth
or
$ python MyTrain_mobile.py --net xxx --resume net_weight/xxx.pth
```

>specify the network type after `--net`, e.g. `--net try1`  
>specify the pretrained model after `--resume`, for try[1-2], it's the result from _Pretrain backbone_; for try[3-5], it's mb2_pretrained.pth; for repo, it's resnet50_pretrained.pth
> if the train process is interrupted midway, you can easily resume training by specifying `--resume` and `--start_iter`

### Test

1. to test the network, run:

```terminal
$ python My_test.py --net xxx
```

this will generate `*.npy` in ./draw_curve/data/, which contains detected bounding boxes and confidences for each picture in eval dataset

> specify the network type after `--net`, e.g. `--net try1`  
> if you don't want to intuitively see the detection result, use `--display False`

2. modify the `data_file_list` in ./draw_curve/draw_pr_roc.py, run and you will get the PR, ROC curve of the corresponding detectors

<span id='title3'> </span>

## Task3: multi-face offline tracking

1. modify the variable `video_file` in iouTracke_cal.py & iouTracke_display.py to the path of the video to be tested
2. run iouTracke_cal.py, this will generate .npy files containing tracking datas in the same path as the video file
3. run iouTracke_display.py, and you will see the tracking result

## Other features

### draw loss curve

you can see the loss curve of all the training process by running ./draw_curve/draw_loss.py. But firstly you will need to add loss files to draw_loss.py

### real-time camera face detection

run Video.py, and you can see a real-time face detection with your web camera(if available)

## Code Reference

MTCNN: <https://github.com/kpzhang93/MTCNN_face_detection_alignment>  
FaceBoxes: <https://github.com/lxg2015/faceboxes>  
PyramidBox: <https://github.com/Goingqs/PyramidBox>  
IoU tracker: <https://github.com/bochinski/iou-tracker>  
