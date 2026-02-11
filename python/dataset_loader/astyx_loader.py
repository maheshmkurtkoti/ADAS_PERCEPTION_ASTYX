import os
import cv2
import json
import numpy as np 

class AstyxLoader:
    def __init__(self,root_path):
        self.root = os.path.normpath(root_path)
        self.image_dir = os.path.join(root_path,"camera_front")
        self.radar_dir = os.path.join(root_path,"radar_6455")
        self.calib_dir = os.path.join(root_path,"calibration")
        self.frames = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.frames)

    def load_image(self, idx):
        img_path = os.path.join(self.image_dir,self.frames[idx])
        img = cv2.imread(img_path)
    
        if img is None:
            raise FileNotFoundError("Image Not Found :  {img_path}")
        
        return img

    def load_radar(self, idx):
        radar_file = self.frames[idx].replace(".jpg",".txt")
        radar_path = os.path.join(self.radar_dir, radar_file)

        if radar_path is None:
            raise FileNotFoundError("Radar file Not Found :  {radar_path}")

        radar_points = []

        radar_data = np.loadtxt(radar_path, skiprows=2)

        if radar_data.ndim == 1:
            radar_data = radar_data.reshape(1,-1)

        radar_points = {
            "xyz": radar_data[:,0:3],
            "vr": radar_data[:,3],
            "mag": radar_data[:,4]
        }

        return radar_points

    def load_calibration(self, idx):
        calib_file = self.frames[idx].replace(".jpg",".json")
        calib_path = os.path.join(self.calib_dir, calib_file)

        if calib_path is None:
            raise FileNotFoundError("Radar file Not Found :  {calib_path}")

        with open(calib_path, "r") as f:
            calib = json.load(f)

        return calib

    def get_sample(self,idx):
        img = self.load_image(idx)
        radar = self.load_radar(idx)
        calib = self.load_calibration(idx)

        return img, radar, calib