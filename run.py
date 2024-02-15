import numpy as np
import cv2
import onnxruntime as ort
import math

STEP = 400
FRAME_SIZE = 640

# 5198, 3462

def processing_img(image, model):
    for row in range(0, image.shape[0] - FRAME_SIZE, STEP):
        for col in range(0,image.shape[1] - FRAME_SIZE, STEP):
            frame = image[row:row+FRAME_SIZE,col:col+FRAME_SIZE]
            input = frame.astype(np.float32)
            input = input.transpose((2, 0, 1))
            input = np.expand_dims(input, axis=0)
            result = model.run(None, {'images':input})[0]
           
            cv2.imshow("hi",frame)
            cv2.waitKey(0)
    cv2.destroyAllWindows()


filename = 'IMG_6762.png'
img = cv2.imread(filename)

model = ort.InferenceSession("best.onnx", providers=['CUDAExecutionProvider'])
processing_img(image=img, model=model)
