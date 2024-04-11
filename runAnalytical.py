import numpy as np
import cv2
import time

STEP = 480
FRAME_SIZE = 640

# 5198, 3462

def processing_img(image):
    output_image = image.copy()
    start_time = time.time()
    count = 0
    for row in range(0, image.shape[0] - FRAME_SIZE, STEP):
        for col in range(0,image.shape[1] - FRAME_SIZE, STEP):
            frame = image[row:row+FRAME_SIZE,col:col+FRAME_SIZE]
            hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            sat_val  = cv2.addWeighted(hsv_img[:,:,1],0.2,hsv_img[:,:,2],0.4,0)
            gray  = cv2.addWeighted(hsv_img[:,:,0],1,sat_val,1,0)
            thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,27,5)
            output = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            output_image[row:row+FRAME_SIZE,col:col+FRAME_SIZE] = output
    print("Iterations: " + str(count))
    print("Frametime: " + str(time.time() - start_time))
    return output_image


filename = 'IMG_6762.png'
img = cv2.imread(filename)

output_image = processing_img(image=img)
cv2.imwrite("output.png", output_image)
# cv2.imshow("hi", output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()