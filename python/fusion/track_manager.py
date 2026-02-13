import numpy as np
from scipy.spatial.distance import cdist
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from python.radar_processing.radar_tracking import RadarTrack


class RadarTrackManager:
    def __init__(self, dist_threshold=2.5, max_missed=5):
        self.tracks = []
        self.next_id = 0
        self.dist_threshold = dist_threshold
        self.max_missed = max_missed

    def update(self, objects):
        # --------------------------------------------------
        # 1. Predict all existing tracks
        # --------------------------------------------------
        for t in self.tracks:
            t.predict()

        # --------------------------------------------------
        # 2. Handle case: no detections
        # --------------------------------------------------
        if len(objects) == 0:
            for t in self.tracks:
                t.missed += 1

            self.tracks = [t for t in self.tracks if t.missed < self.max_missed]
            return self.tracks

        # --------------------------------------------------
        # 3. If no tracks exist → create all
        # --------------------------------------------------
        if len(self.tracks) == 0:
            for obj in objects:
                self._create_track(obj)
            return self.tracks

        # --------------------------------------------------
        # 4. Build cost matrix (XY distance)
        # --------------------------------------------------
        track_pos = np.array([t.x[:2] for t in self.tracks])
        obj_pos = np.array([o["center"][:2] for o in objects])

        cost = cdist(track_pos, obj_pos)

        # --------------------------------------------------
        # 5. Greedy one-to-one matching (IMPORTANT FIX)
        # --------------------------------------------------
        matches = []
        used_tracks = set()
        used_objs = set()

        pairs = [
            (cost[i, j], i, j)
            for i in range(cost.shape[0])
            for j in range(cost.shape[1])
        ]

        pairs.sort(key=lambda x: x[0])  # smallest distance first

        for dist, i, j in pairs:
            if dist > self.dist_threshold:
                continue
            if i in used_tracks or j in used_objs:
                continue

            matches.append((i, j))
            used_tracks.add(i)
            used_objs.add(j)

        # --------------------------------------------------
        # 6. Update matched tracks
        # --------------------------------------------------
        matched_tracks = set()
        matched_objs = set()

        for i, j in matches:
            self.tracks[i].update(objects[j])
            matched_tracks.add(i)
            matched_objs.add(j)

        # --------------------------------------------------
        # 7. Create new tracks for unmatched objects
        # --------------------------------------------------
        for j, obj in enumerate(objects):
            if j not in matched_objs:
                self._create_track(obj)

        # --------------------------------------------------
        # 8. Age & prune tracks
        # --------------------------------------------------
        survivors = []
        for i, t in enumerate(self.tracks):
            if i not in matched_tracks:
                t.missed += 1

            if t.missed < self.max_missed:
                survivors.append(t)

        self.tracks = survivors
        return self.tracks

    # ------------------------------------------------------
    # Helper: create new track
    # ------------------------------------------------------
    def _create_track(self, obj):
        track = RadarTrack(obj, self.next_id)
        self.tracks.append(track)
        self.next_id += 1
