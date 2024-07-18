import numpy as np

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker
from .tracker_config import TrackerConfig

__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path, max_dist=TrackerConfig.MAX_DIST, use_cuda=True):
        self.min_confidence = TrackerConfig.MIN_CONFIDENCE
        self.nms_max_overlap = 1.0
        self.extractor = Extractor(model_path, use_cuda=use_cuda)
        metric = NearestNeighborDistanceMetric("cosine", max_dist, TrackerConfig.NN_BUDGET)
        self.tracker = Tracker(metric)

    def update(self, bbox_xywh, confidences, img):
        self.height, self.width = img.shape[:2]
        features = self._get_features(bbox_xywh, img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)

        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confidences) if
                      conf > self.min_confidence]

        detections = self.update_detections_after_nms(detections)

        self.tracker.predict()
        self.tracker.update(detections)

        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            outputs.append(np.array([x1, y1, x2, y2, track.track_id], dtype=np.int32))

        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    def update_detections_after_nms(self, detections):
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        return detections

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        bbox_xywh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_xywh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_xywh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features
