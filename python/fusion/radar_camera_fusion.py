#fused object list is 
def build_fused_objects(matches):
    fused_objects = []

    for radar_track, detection in matches:
        fused_obj = {}

        #radar info
        fused_obj["track_id"] = radar_track.id
        fused_obj["age"] = radar_track.age

        #state vector [x, y, vx, vz]
        fused_obj["radar_state"] = radar_track.x.copy()
        fused_obj["covariance"] = radar_track.P.copy()

        #camera info
        fused_obj["bbox"] = detection["bbox"]
        fused_obj["class_name"] = detection["class_name"]
        fused_obj["score"] = detection["score"]

        fused_objects.append(fused_obj)

    return fused_objects

