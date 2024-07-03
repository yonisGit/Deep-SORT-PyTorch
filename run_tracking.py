import argparse
import os
import time
from distutils.util import strtobool

import cv2
import numpy as np
import torch

from VMD.vmd import VMD
from deep_sort import DeepSort
from reid.builder import build_reid
from reid.utils import crop_imgs
from util import adjust_normalized_boxes
from util import draw_bboxes

FPS = 25.

VMD_CONFIG = "VMD/configs/Altitude=100_motion=False_resolution=(512, 640).yaml"
height, width = (1080, 1920)


class Detector(object):
    def __init__(self, args):
        self.args = args
        use_cuda = bool(strtobool(self.args.use_cuda))

        self.vdo = cv2.VideoCapture()
        self.vmd = VMD.from_yaml(VMD_CONFIG)
        self.deepsort = DeepSort(args.deepsort_checkpoint, use_cuda=use_cuda)
        self.mask_irrelevant_classes = False
        self.reid = build_reid()

    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self):

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(self.args.save_path, fourcc, FPS, (width, height))
        while True:
            start = time.time()
            ret, frame = self.vdo.read()

            if ret:
                results = self.vmd(frame).to_numpy()

                cls_conf = np.ones(results.shape[0])
                cls_ids = np.zeros_like(cls_conf)

                bbox_xcycwh = adjust_normalized_boxes(results, frame.shape[0], frame.shape[1])

                if len(bbox_xcycwh) > 0:
                    if self.mask_irrelevant_classes:
                        bbox_xcycwh, cls_conf = mask_irrelevant_classes(bbox_xcycwh, cls_conf, cls_ids)
                    outputs = self.deepsort.update(bbox_xcycwh, cls_conf,
                                                   frame)  # outputs is a list of the form: <[[bbox coordinates],id]>

                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        frame = draw_bboxes(frame, bbox_xyxy, identities)

                end = time.time()
                print("time: {}s, fps: {}".format(end - start, 1 / (end - start)))
                video.write(frame)
            else:
                video.release()
                break

        if self.vdo:
            self.vdo.release()


def mask_irrelevant_classes(bbox_xcycwh, cls_conf, cls_ids):
    cls_ids_clone = cls_ids
    cls_ids_clone += 1  # added 1 because comparison with 0 didn't work for some reason...
    mask = cls_ids_clone == 1  # looking only for person class
    bbox_xcycwh = bbox_xcycwh[mask]
    bbox_xcycwh[:, 3:] *= 1.2
    cls_conf = cls_conf[mask]
    return bbox_xcycwh, cls_conf


def reid_testing(bbox_xcycwh, frame):
    img_metas = {}
    crops = crop_imgs(img=frame, img_metas=img_metas, bboxes=torch.tensor(bbox_xcycwh).clone(),
                      rescale=False)
    embeds = self.reid.simple_test(crops)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--VIDEO_PATH", type=str)
    parser.add_argument("--yolo_cfg", type=str, default="YOLOv3/cfg/yolo_v3.cfg")
    parser.add_argument("--yolo_weights", type=str, default="YOLOv3/yolov3.weights")
    parser.add_argument("--yolo_names", type=str, default="YOLOv3/cfg/coco.names")
    parser.add_argument("--conf_thresh", type=float, default=0.2)
    parser.add_argument("--nms_thresh", type=float, default=0.1)
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--max_dist", type=float, default=0.2)
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="out/demo.mp4")
    parser.add_argument("--use_cuda", type=str, default="True")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with Detector(args) as det:
        det.detect()
