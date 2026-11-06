import numpy as np

from src.TrackedFace import TrackedFace
from src.utils.Detection import Detection


class FaceTracker:
    # how far a detection can be from a tracked face and still counting as the same face. measured relative to face edge diagonal
    MAX_MATCH_DIST_FACTOR = 0.5

    def __init__(self, sface_threshold, candidate_threshold):
        self.sface_threshold = sface_threshold
        self.candidate_threshold = candidate_threshold
        self.trackedFaces: list[TrackedFace] = []

    def process_raw_detections(self, detections: list[Detection]):
        """takes in raw detections, this happens every single frame"""
        matches, new_detections = self.match(detections)

        # (temporary) removes frames that didnt get a detection signal
        kept_faces: list[TrackedFace] = []
        for tracked_i, detected_i in matches:
            detection = detections[detected_i]
            tracked_face = self.trackedFaces[tracked_i]
            tracked_face.process_new_data(detection.pos, detection.size, detection.scores)
            kept_faces.append(tracked_face)

        for detected_i in new_detections:
            detection = detections[detected_i]
            kept_faces.append(TrackedFace(detection.pos, detection.size, detection.scores, self.sface_threshold, self.candidate_threshold))

        self.trackedFaces = kept_faces

    def match(self, detections: list[Detection]) -> tuple[list[tuple[int, int]], list[int]]:
        """finds out which detection is conected with each trackedFace, returns a list of pair<trackedFace, detection>"""
        # make pair<track, detection> for each track x direction, but only if close enough
        pairs = []
        for track_i, track in enumerate(self.trackedFaces):
            for detection_i, detection in enumerate(detections):
                dist = np.linalg.norm(detection.pos - track.pos)
                if dist <= track.diagonal * self.MAX_MATCH_DIST_FACTOR:
                    pairs.append((dist, track_i, detection_i))

        # shorter distance at the top
        pairs.sort()

        # taking the shortest first bc theyre more certain
        matches: list[tuple[int, int]] = []
        used_tracks = set()
        used_detections = set()
        for dist, track_i, detection_i in pairs:
            if track_i in used_tracks or detection_i in used_detections:
                continue
            matches.append((track_i, detection_i))
            used_tracks.add(track_i)
            used_detections.add(detection_i)

        # new detections not conected to a TrackedFace
        newDetections = [detected_i for detected_i in range(len(detections)) if detected_i not in used_detections] # = detections.indices.filter { it !in usedDetections }
        return matches, newDetections # would have used the objs directly if this was kt
