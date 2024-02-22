import numpy as np
import cv2
import onnxruntime as ort
import time
from model.detector import Detector

STEP = 480
FRAME_SIZE = 640

# 5198, 3462

def processing_img(image, model):
    output_image = image.copy()
    start_time = time.time()
    count = 0
    for row in range(0, image.shape[0] - FRAME_SIZE, STEP):
        for col in range(0,image.shape[1] - FRAME_SIZE, STEP):
            frame = image[row:row+FRAME_SIZE,col:col+FRAME_SIZE]
            results = model(frame)
            combined_img = model.draw_detections(frame)
            output_image[row:row+FRAME_SIZE,col:col+FRAME_SIZE] = combined_img
            cv2.imshow("Output", combined_img)
            cv2.imshow("hi",frame)
            cv2.waitKey(0)
            count += 1
    # cv2.destroyAllWindows()
    print("Iterations: " + str(count))
    print("Frametime: " + str(time.time() - start_time))
    return output_image


filename = 'IMG_6762.png'
img = cv2.imread(filename)

model_path = "best.onnx"
detector = Detector(model_path, conf_thres=0.3, iou_thres=0.5)

output_image = cv2.resize(processing_img(image=img, model=detector), (1700,1080))
cv2.imshow("hi", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()