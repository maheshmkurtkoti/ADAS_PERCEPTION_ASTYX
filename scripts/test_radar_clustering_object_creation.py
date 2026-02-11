import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from python.dataset_loader.astyx_loader import AstyxLoader
from python.radar_processing.clustering import cluster_radar_points
from python.radar_processing.object_list import build_radar_objects

loader = AstyxLoader("F:\\radar_dataset_astyx\\dataset_astyx_hires2019")
img, radar, calib = loader.get_sample(0)

labels = cluster_radar_points(radar["xyz"],eps =1.0, min_samples= 6)

objects = build_radar_objects(radar["xyz"], radar["vr"],labels)

print("Radar objects:", len(objects))
print(objects[:3])