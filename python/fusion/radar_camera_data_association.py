import numpy as np
from scipy.spatial.distance import cdist

def compute_bbox_centers(detections):
    """
    detectctions: list of camera detections
    returns: NX2 array
    """
    if detections is None:
        return np.empty((0,2))
    
    centers = np.array([det["center"] for det in detections])
    return centers

def compute_track_pixels(tracks, project_fn, calib, image):
    if len(tracks) == 0:
        return np.empty((0,2)), []
    
    pts = np.array([t.x[:3] for t in tracks])

    pixels = project_fn(pts, calib, image)

    return pixels, tracks

def associate_radar_camera(
        tracks,
        detections,
        project_fn,
        calib,
        image,
        dist_threshold = 80
):
    #get centers of bounding box
    cam_centers = compute_bbox_centers(detections)
    track_pixels, tracks = compute_track_pixels(tracks, project_fn, calib, image)

    if len(track_pixels) == 0 or len(cam_centers) == 0:
        return [], tracks, detections
    #distance matrix
    cost = cdist(track_pixels,cam_centers)

    matches = []
    matched_tracks = set()
    matched_dets = set()
    
    #greedy matching
    while True:
        i, j = np.unravel_index(np.argmin(cost), cost.shape)
        min_val = cost[i,j]

        if min_val > dist_threshold:
            break

        matches.append((tracks[i], detections[j]))
        matched_tracks.add(i)
        matched_dets.add(j)

        cost[i,:] = np.inf
        cost[:,j] =np.inf

    unmatched_tracks = [
        t for k, t in enumerate(tracks) if k not in matched_tracks
    ]

    unmatched_detections = [
        d for k,d in enumerate(detections) if k not in matched_dets
    ]

    return matches, unmatched_tracks, unmatched_detections


