import cv2
import numpy as np
from targets import Targets

folder_name = "diffusion/synthetic_data"
target_sampler = Targets()

img, data = target_sampler.sample()
img = cv2.resize(img,(120,120))

# cv2.imshow("original", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite(folder_name+"/sampleimg.png", img)
