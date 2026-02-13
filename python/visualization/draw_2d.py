import cv2

def draw_camera_detections(image, detections):
    vis = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f'{det["class_name"]}:{det["score"]:.2f}'
        cv2.rectangle(vis,(x1,y1),(x2,y2),(255,0,0),5)
        cv2.putText(vis,label,(x1, y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 1)

    return vis