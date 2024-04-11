from pipeline import TargetShapeText
import cv2
import time

STEP = 1080
FRAME_SIZE = 1280
ITERATIONS = 12

urmom = TargetShapeText()


filename = 'IMG_6762.png'
img = cv2.imread(filename)

start = time.time()
urmom.run(img)
box_results = urmom.get_boxes()
for box in box_results:
    cv2.rectangle(img, (box[0],box[1]), (box[2],box[3]), (255,0,0), 5)

print("time:",time.time() - start)
cv2.imwrite("output.png",img)

