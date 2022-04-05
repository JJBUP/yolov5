# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
import threading
from datetime import time
from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import pyautogui
import torch
import torchvision.transforms
from PIL import ImageGrab
from matplotlib import pyplot as plt

import matplotlib

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh,
                           xywh2xyxy)

from utils.torch_utils import select_device, time_sync


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''  #
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)  # 按双三次插值进行resize
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_box_coords(img_scaled_shape, coords, img_orgin_shape, ratio_pad=None):
    """
    将预测的坐标信息转换回原图尺度
    :param img_scaled_shape: 缩放后的图像尺度
    :param coords: 预测的box信息
    :param img_orgin_shape: 缩放前的图像尺度
    :param ratio_pad: 缩放过程中的缩放比例以及pad
    :return:
    """
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img_scaled_shape[0] / img_orgin_shape[0],
                   img_scaled_shape[1] / img_orgin_shape[1])  # gain  = old / new
        pad = (img_scaled_shape[0] - img_orgin_shape[0] * gain) / 2, (
                img_scaled_shape[1] - img_orgin_shape[1] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img_orgin_shape)
    return coords


@torch.no_grad()
def realtime_detect_crop(
        img_scaled_size=(640, 640),  # 模型进行推断时的尺寸
        conf_thres=0.25,  # 置信度 的过滤阈值
        iou_thres=0.45,  # NMS交并比阈值
        max_det=30,  # 图片中最大检测数量
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        img=None,  # 抓取屏幕图片
        model=None,  # 传入模型
        screen_size=None  # 当前屏幕分辨率，可能与抓取的屏幕图片不同
):
    # 如果不存在图片，返回none
    if img == None:
        return None
    # 1. 将图片裁剪，我们只要屏幕中间2/1的尺寸
    iw, ih = img.size
    print("imgsize:{} * {}".format(iw, ih))
    img_crop = img.crop(box=(iw / 4, ih / 4, iw * 3 / 4, ih * 3 / 4))

    # matplotlib.use('TkAgg')  # 大小写无所谓 tkaGg ,TkAgg 都行
    # plt.figure("Image")  # 图像窗口名称
    # plt.imshow(img_crop)
    # plt.show()

    # 2. 图片缩放到模型指定尺寸
    img_scaled = letterbox_image(image=img_crop, size=img_scaled_size)
    # matplotlib.use('TkAgg')  # 大小写无所谓 tkaGg ,TkAgg 都行
    # plt.figure("Image")  # 图像窗口名称
    # plt.imshow(img_scaled)
    # plt.show()

    # t1 = time_sync()
    # 3.转变魏tensor并放到GPU上，
    to_tensor = torchvision.transforms.ToTensor()
    img_scaled = to_tensor(img_scaled).to(device)
    img_scaled = img_scaled.half() if model.fp16 else img_scaled.float()  # uint8 to fp16/32
    if len(img_scaled.shape) == 3:
        img_scaled = img_scaled[None]  # 添加batch 维度

    # 4.进行预测
    pred = model(img_scaled)  # [batch,3*ny*nx ,xywh o c ]
    # t3 = time_sync()

    # 5. 非极大值抑制
    # NMS # [batch,3*ny*nx ,xywh o c ] -->[batch,3*ny*nx ,xyxy o c ]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    # 单张图片检测，我们去掉batch [3*ny*nx ,xywh o c ]
    pred = pred[0]
    pred = xyxy2xywh(pred)  # [batch,3*ny*nx ,xyxy o c ]-->[batch,3*ny*nx ,xywh o c ]

    # 6. 将图片恢复到原尺寸（恢复到屏幕原尺寸，而不是原图的尺寸，因为二者可能不同）
    if len(pred) and screen_size:
        # 恢复到原图大小，但是发现分辨率和抓取图片大小不一样 所以放大到屏幕分辨率才行
        sw, sh = screen_size  # 屏幕原尺寸
        sw_crop, sh_crop = sw / 2, sh / 2  # 裁剪后屏幕尺寸
        # 将裁减图片 将相对模型输入的预测 恢复到 相对屏幕尺寸裁剪后的 大小
        pred = scale_box_coords(img_scaled_shape=img_scaled_size, coords=pred, img_orgin_shape=(sw_crop, sh_crop))
        sw_offset, sh_offset = sw / 4, sh / 4
        # 将裁剪后的部分的偏移量
        pred[:, [0]] += sw_offset  # x padding
        pred[:, [1]] += sh_offset  # y padding
        # print(pred)
    else:
        # 非极大值抑制后没有任何预测输出
        return None
    return pred
    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

@torch.no_grad()
def realtime_detect(
        img_scaled_size=(320,320),  # 模型进行推断时的尺寸
        conf_thres=0.25,  # 置信度 的过滤阈值
        iou_thres=0.45,  # NMS交并比阈值
        max_det=30,  # 图片中最大检测数量
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        img=None,  # 抓取屏幕图片
        model=None,  # 传入模型
        screen_size=None  # 当前屏幕分辨率，可能与抓取的屏幕图片不同
):
    # 如果不存在图片，返回none
    if img == None:
        return None
    # 1. 将图片裁剪，这里是原图
    iw, ih = img.size
    print("imgsize:{} * {}".format(iw, ih))

    # 2. 图片缩放到模型指定尺寸
    img_scaled = letterbox_image(image=img, size=img_scaled_size)
    # matplotlib.use('TkAgg')  # 大小写无所谓 tkaGg ,TkAgg 都行
    # plt.figure("Image")  # 图像窗口名称
    # plt.imshow(img_scaled)
    # plt.show()

    # t1 = time_sync()
    # 3.转变魏tensor并放到GPU上，
    to_tensor = torchvision.transforms.ToTensor()
    img_scaled = to_tensor(img_scaled).to(device)
    img_scaled = img_scaled.half() if model.fp16 else img_scaled.float()  # uint8 to fp16/32
    if len(img_scaled.shape) == 3:
        img_scaled = img_scaled[None]  # 添加batch 维度

    # 4.进行预测
    pred = model(img_scaled)  # [batch,3*ny*nx ,xywh o c ]
    # t3 = time_sync()

    # 5. 非极大值抑制
    # NMS # [batch,3*ny*nx ,xywh o c ] -->[batch,3*ny*nx ,xyxy o c ]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    # 单张图片检测，我们去掉batch [3*ny*nx ,xywh o c ]
    pred = pred[0]
    pred = xyxy2xywh(pred)  # [batch,3*ny*nx ,xyxy o c ]-->[batch,3*ny*nx ,xywh o c ]

    # 6. 将图片恢复到原尺寸（恢复到屏幕原尺寸，而不是原图的尺寸，因为二者可能不同）
    if len(pred) and screen_size:
        # 恢复到原图大小，但是发现分辨率和抓取图片大小不一样 所以放大到屏幕分辨率才行
        pred = scale_box_coords(img_scaled_shape=img_scaled_size, coords=pred, img_orgin_shape=screen_size)
        # print(pred)
    else:
        # 非极大值抑制后没有任何预测输出
        return None
    return pred
    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

# 抓取屏幕图片
def image_grab():
    image = ImageGrab.grab()
    return image


def main():
    # print("5秒后开始执行程序")
    # time.sleep(5)
    thread_imagegrab = threading.Thread(target=image_grab)
    thread_imagedetect = threading.Thread(target=realtime_detect)
    thread_imagegrab.start()
    thread_imagedetect.start()
    width, height = pyautogui.size()
    # 屏幕分辨率
    center_x, center_y = width / 2, height / 2

    '''
        创建了两个并行的线程。realtime_detect(image)用来检测image_grab()抓取的图像。
        并不是把抓取的每个图像都检测一遍，抓取的图像会有遗失部分（如果抓取速度大于检测的速度，
        但是几乎不会影响机器的判断
    '''
    print("开始执行程序！")
    # Load model
    weights = ROOT / 'runs/train/exp2/weights/last.pt'
    device = '0'
    device = select_device(device)
    model = DetectMultiBackend(weights=weights, device=device)
    pyautogui.moveTo(center_x, center_y, duration=0.1)
    while True:
        width, height = pyautogui.size()
        print("屏幕分辨率：{} * {}".format(width, height))
        # pred = realtime_detect(
        pred = realtime_detect_crop(
            device=device,
            model=model,
            img=image_grab(),
            screen_size=(width, height)
        )
        # 恐怖分子头0，恐怖分子身体1，反恐精英头2，反恐精英身体3
        if pred is not None:
            # 优先选择头部
            is_head = pred[:, 5] == 0
            is_body = pred[:, 5] == 1
            if True in is_head:
                head_pred = pred[is_head]
                pyautogui.moveTo(head_pred[0][0], head_pred[0][1], duration=0.001)
                pyautogui.doubleClick()
                # print("鼠标位置：x{} , y{}".format(head_pred[0][0], head_pred[0][1]))
            elif True in is_body:
                # 没有就选择身子的第一个
                pyautogui.moveTo(pred[0][0], pred[0][1], duration=0.001)
                pyautogui.doubleClick()


if __name__ == "__main__":
    main()
