import numpy as np

def build_radar_objects(xyz,vr,labels):
    objects = []

    for lab in np.unique(labels):
        if lab == -1:
            continue #noise
        pts =xyz[labels == lab]
        vrs =vr[labels == lab]

        obj = {
            "center": pts.mean(axis=0),
            "velocity":vrs.mean(),
            "size": pts.std(axis=0),
            "num_points": len(pts)
        }

        objects.append(obj)

    return objects