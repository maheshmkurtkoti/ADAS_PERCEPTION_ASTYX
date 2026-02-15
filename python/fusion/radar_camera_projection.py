import numpy as np
import cv2

def get_sensor_transform(calib, sensor_name):
    for s in calib["sensors"]:
        if s["sensor_uid"] == sensor_name:
            return np.array(s["calib_data"]["T_to_ref_COS"])
    raise ValueError("{sensor_name} not found in calibration")
    
def get_camera_intrinsics(calib):
    for s in  calib["sensors"]:
        if s["sensor_uid"] == "camera_front":
            return np.array(s["calib_data"]["K"])
    raise ValueError("camera_front not found in calibration")
    
def draw_radar_on_image(image, pixels):
    vis = image.copy()
    for u, v in pixels:
        if 0 <= u < vis.shape[1] and 0 <= v < vis.shape[0]:
            cv2.circle(vis,(u,v), 3, (0,255,0), -1)
    return vis

def draw_tracks_on_image(image, pixels,tracks):
    h,w =image.shape[:2]

    for (u,v), t in zip(pixels, tracks):
        u, v = int(u), int(v)

        if 0 <= u < w and 0 <= v < h:
            #draw circle
            cv2.circle(image, (u,v), 6, (0, 255, 0), -1)

            #draw track id
            cv2.putText(
                image,
                f"ID:{t.id}",
                (u+5, v-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                1
            )

    return image

def project_radar_to_image(radar_xyz, calib, image):

    T_r = get_sensor_transform(calib,"radar_6455")
    T_c = get_sensor_transform(calib,"camera_front")
    K = get_camera_intrinsics(calib)

    #radar to camera transforms
    #T_r2c = T_c @ np.linalg.inv(T_r)
    T_r2c = np.linalg.inv(T_c) @ T_r

    #Homogenous radar coordinates
    N = radar_xyz.shape[0]
    radar_h = np.hstack([radar_xyz,np.ones((N,1))]).T #4XN
    cam_pts = T_r2c @ radar_h
    cam_pts = cam_pts[:3,:] # 3XN
    #keep only points in front of the camera
    print("cam_points shape", cam_pts.shape)
    print("z stats:", cam_pts[2,:].min(),cam_pts[2,:].max())
    print("z>0 count", (cam_pts[2,:] > 0).sum())
    valid = cam_pts[2, :] > 0.1
    cam_pts = cam_pts[:, valid]
    #projection
    pixels = K @ cam_pts
    pixels /= pixels[2,:]
    return pixels[:2, :].T.astype(int)
    


    