import cv2
import numpy as np

from norfair import Detection, Tracker, Video, draw_tracked_objects

class NorfairTracker():

    def __init__(self, input_video_path, output_video_path, distance_function="frobenius", distance_threshold=100):
        self.video = Video(input_path=input_video_path, output_path=output_video_path)
        self.tracker = Tracker(distance_function=distance_function, distance_threshold=distance_threshold)

    def detections_to_norfair(self, detections, scores):
        "gets all detections in a frame and convert them to the norfair accepted type"
        for (detection_center, score) in zip(detections, scores):
            self.norfair_detections.append(Detection(points=np.array(detection_center), scores=np.array(score), label=int(2)))

        return self.norfair_detections

    def update_tracks(self, detections, scores):
        self.norfair_detections = []
        self.norfair_detections = self.detections_to_norfair(detections, scores)
        self.tracked_objects = self.tracker.update(detections=self.norfair_detections)

        return self.tracked_objects

    def draw_tracks(self, frame, tracked_objects, save_video=False):
        draw_tracked_objects(frame, tracked_objects)
        if save_video:
            self.video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
