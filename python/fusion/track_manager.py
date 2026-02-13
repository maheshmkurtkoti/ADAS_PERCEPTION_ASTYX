import numpy as np
from scipy.spatial.distance import cdist
from python.radar_processing.radar_tracking import RadarTrack

class RadarTrackManager:
    def __init__(self):
        self.tracks =[]
        self.next_id = 0

    def update(self, objects):
        for t in self.tracks(objects):
            t.predict()

        if len(self.tracks) == 0:
            for obj in objects:
                self._create_track(obj)
            return self.tracks
        
        track_pos = np.array([t.x[:2] for t in self.tracks])
        obj_pos = np.array([o["center"][:2] for t in self.tracks])

        cost = cdist(track_pos, obj_pos)

        matched_tracks = set()
        matched_objs = set()

        for i,j in zip(*np.where(cost <2.5)):
            self.tracks[i].update(objects[j])
            matched_tracks.add(i)
            matched_objs.add(j)

        for j, obj in enumerate (objects):
            if j not in matched_objs:
                self._create_track(obj)

        survivors = []

        for i, t in enumerate(self.tracks):
            if i not in matched_tracks:
                t.missed += 1
            if t.missed < 5:
                survivors.append(t)

        self.tracks = survivors
        return self.tracks
    
    def _create_track(self,obj):
        self.tracks.append(RadarTrack(obj,self.next_id))
        self.next_id += 1

        

        
