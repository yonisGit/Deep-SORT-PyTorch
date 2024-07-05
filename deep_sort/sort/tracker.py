# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from ..tracker_config import TrackerConfig


#   TODO: IMPORTANT
class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=TrackerConfig.MAX_IOU, max_age=TrackerConfig.MAX_AGE,
                 n_init=TrackerConfig.N_INIT):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        self.update_matching_tracks_with_their_matching_detection(detections, matches)
        self.mark_existing_track_as_missed(unmatched_tracks)
        self.initialize_new_tracks(detections, unmatched_detections)
        self.update_tracks_without_deleted_ones()

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if
                          t.is_confirmed()]  # adding new confirmed tracks to the list and removing missed tracks

        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features  # For every confirmed track we append all the items in
            # its list of feature embeddings.
            targets += [track.track_id for _ in track.features]  # For every added feature embedding item in
            # the features list above, assign the track_id to connect between id and feature.
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets),
            active_targets)  # Just updating the dict of samples for every track with the
        # new detected features (for new tracks they will get a list of size n_init).

    def update_matching_tracks_with_their_matching_detection(self, detections, matches):
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])

    def update_tracks_without_deleted_ones(self):
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def mark_existing_track_as_missed(self, unmatched_tracks):
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

    def initialize_new_tracks(self, detections, unmatched_detections):
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            detections_features = np.array([dets[i].feature for i in detection_indices])
            target_tracks_ids = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(detections_features,
                                               target_tracks_ids)  # creates a matrix of distances between all the
            # track samples and detections.
            # TODO: include an option to compare aggregated features in the distance instead of one feature at a time.
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        # Confirmed are tracks that appeared in at least n_init frames.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]  # already confirmed tracks.
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]  # fresh tracks that wasn't confirmed yet.

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
