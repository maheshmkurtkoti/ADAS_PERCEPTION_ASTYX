import cv2

def draw_radar_camera_fusion(image,matches):
    vis = image.copy()

    for track,det in matches:
        x1, y1, x2, y2 = det["bbox"]

        #draw camera box
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),5)
        cx,cy = det["center"]
        #draw track id
        cv2.putText(
            vis,
            f"ID:{track.id}",
            (cx,cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            2,
        )

    return vis