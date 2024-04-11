import numpy as np
import cv2
import onnxruntime as ort
import time
from model.detector import Detector
from threading import Thread

STEP = 1080
FRAME_SIZE = 1280
ITERATIONS = 12

# 5198, 3462

def run_model(input_tensor, results, index):
    boxes, _, _ = model(input_tensor)
    results[index] = boxes
    print(len(boxes))

def processing_img(image):
    threads = [None] * ITERATIONS
    results = [None] * ITERATIONS
    output_image = image.copy()
    start_time = time.time()
    count = 0
    for row in range(0, image.shape[0] - FRAME_SIZE, STEP):
        for col in range(0,image.shape[1] - FRAME_SIZE, STEP):
            
            frame = image[row:row+FRAME_SIZE,col:col+FRAME_SIZE]
            resized = cv2.resize(frame, (640,640))
            resized = cv2.cvtColor(cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)[:,:,0], cv2.COLOR_BGR2RGB) / 255
            resized = resized.transpose(2, 0, 1)
            input_tensor = resized[np.newaxis, :, :, :].astype(np.float16)

            # run model on separate thread
            threads[count] = Thread(target = run_model, args=(input_tensor,results,count))
            threads[count].start()
            count += 1

    for i in range(ITERATIONS):
        threads[i].join()
    for i in range(ITERATIONS):
        print(results[i])

    
    print("Iterations: " + str(count))
    print("Frametime: " + str(time.time() - start_time))
    return output_image


filename = 'IMG_6762.png'
img = cv2.imread(filename)

model_path = "best.onnx"
model = Detector(model_path, conf_thres=0.3, iou_thres=0.5)

output_image = processing_img(image=img)
cv2.imwrite("output.png", output_image)
# cv2.imshow("hi", output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()