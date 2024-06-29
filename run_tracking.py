import os
import cv2
import time
import argparse
import numpy as np
from distutils.util import strtobool

from YOLOv3 import YOLOv3
from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes
from reid.builder import build_reid
from reid.utils import crop_imgs
import torch
class Detector(object):
    def __init__(self, args):
        self.args = args
        use_cuda = bool(strtobool(self.args.use_cuda))

        if args.display:
            pass
            # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        self.yolo3 = YOLOv3(args.yolo_cfg, args.yolo_weights, args.yolo_names, is_xywh=True,
                            conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, use_cuda=use_cuda)
        self.deepsort = DeepSort(args.deepsort_checkpoint, use_cuda=use_cuda)
        self.class_names = self.yolo3.class_names
        self.reid = build_reid()

    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width, self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self):
        while (True):
            start = time.time()
            ret, frame = self.vdo.read()

            if ret == True:

                # cv2.imshow("lala", frame)
                # cv2.waitKey(0)
                bbox_xcycwh, cls_conf, cls_ids, = self.yolo3(frame)
                img_metas = {}
                crops = crop_imgs(img=frame, img_metas=img_metas, bboxes=torch.tensor(bbox_xcycwh).clone(),
                                 rescale=False)
                embeds = self.reid.simple_test(crops)

                if bbox_xcycwh is not None:
                    # select class person
                    # mask = cls_ids == (2 or 7)

                    # bbox_xcycwh = bbox_xcycwh[mask]
                    bbox_xcycwh[:, 3:] *= 1.2

                    # cls_conf = cls_conf[mask]
                    outputs = self.deepsort.update(bbox_xcycwh, cls_conf,
                                                   frame)  # outputs is a list of the form: <[[bbox coordinates],id]>
                    # print(outputs)
                    if len(outputs) > 0:
                        # print(bbox_xcycwh, cls_conf, cls_ids)
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        frame = draw_bboxes(frame, bbox_xyxy, identities)

                end = time.time()
                print("time: {}s, fps: {}".format(end - start, 1 / (end - start)))

                self.output.write(frame)
                cv2.imshow('tracks', frame)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    pass

            # Break the loop
            else:
                break

        if self.vdo:
            self.vdo.release()
        if self.args.save_path:
            self.output.release()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--VIDEO_PATH", type=str)
    # parser.add_argument("--display", dest='feature', action='store_false') # DOES NOT WORK YET
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
    parser.add_argument("--use_cuda", type=str, default="False")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with Detector(args) as det:
        det.detect()
