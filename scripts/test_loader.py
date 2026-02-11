import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

from python.dataset_loader.astyx_loader import AstyxLoader

loader = AstyxLoader("F:\\radar_dataset_astyx\\dataset_astyx_hires2019")

img, radar, calib = loader.get_sample(0)

print("Image", img.shape)
print("Radar objects", len(radar["xyz"]))
print("Calibration keys:", calib.keys())