import os
import sys
import numpy as np
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from python.dataset_loader.astyx_loader import AstyxLoader
from python.radar_processing.clustering import cluster_radar_points
from python.radar_processing.object_list import build_radar_objects
from python.fusion.track_manager import RadarTrackManager
from python.camera_detection.camera_tracking import CameraDetector
from python.fusion.radar_camera_data_association import associate_radar_camera
from python.fusion.radar_camera_projection import project_radar_to_image
from python.visualization.draw_fusion import draw_radar_camera_fusion

loader = AstyxLoader("F:\\radar_dataset_astyx\\dataset_astyx_hires2019")

track_manager = RadarTrackManager()

camera_object_detector = CameraDetector()

for i in range (70,140):
    img, radar, calib = loader.get_sample(i)
    
    #-----radar tracks generation-------
    labels = cluster_radar_points(radar["xyz"],eps =1.0, min_samples= 6)
    objects = build_radar_objects(radar["xyz"], radar["vr"],labels)
    print("Radar objects:", len(objects))
    tracks = track_manager.update(objects)
    print("active tracks:",len(tracks))

    #-----camera detections generation -----
    detections = camera_object_detector.detect(img)

    #------------association-----------------
    matches, ut, ud = associate_radar_camera(tracks,detections,project_radar_to_image,calib,img)
    print("Matches:", len(matches))
    vis = draw_radar_camera_fusion(img,matches)
    cv2.imshow("Fusion",vis)
    cv2.waitKey(0)




   