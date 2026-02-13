import numpy as np
from python.fusion.kalman_filter import KalmanFilterCV

class RadarTrack:
    def __init__(self, obj, track_id):
        self.id = track_id
        self.age = 0
        self.missed = 0
        self.kf = KalmanFilterCV()

        self.x = np.array([
            obj["center"][0],
            obj["center"][1],
            0,0
        ])

        self.P = np.eye(4)


    def predict(self):
        self.x,self.P = self.kf.predict(self.x,self.P)
        self.age += 1

    def update(self, obj):
        z = obj["center"][:2]
        self.x,self.P = self.kf.update(self.x,self.P,z)
        self.missed = 0
