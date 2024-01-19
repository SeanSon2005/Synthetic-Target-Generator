from targets import Targets
from background import Backgrounds
import numpy as np
import cv2
import numba
import glob
import os
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

SAMPLE_SIZE = 50
MAX_NUM_OBJECTS = 2
PADDING = 20
CORE_COUNT = 64

# pastes an image on top of another image taking alpha into account
@numba.jit(nopython=True)
def alpha_paste(bg_img, img):
    # Handle Shapes and Sizes
    bg_img_size = bg_img.shape[0]
    img_size = img.shape[0]
    center_x = np.random.randint(PADDING,bg_img_size-PADDING)
    center_y = np.random.randint(PADDING,bg_img_size-PADDING)
    frame_x = center_x - img_size // 2
    frame_y = center_y - img_size // 2

    # Handle Pasting
    for ay in range(img_size):
            for ax in range(img_size):
                y_i = frame_y + ay
                x_i = frame_x + ax
                if frame_x < 0 or frame_y < 0 or \
                y_i >= bg_img_size or \
                x_i >= bg_img_size:
                    continue
                if img[ay][ax][3] == 255:
                    bg_img[y_i][x_i][0] = min(max(img[ay][ax][0]+np.random.randint(-5,6),0),255)
                    bg_img[y_i][x_i][1] = min(max(img[ay][ax][1]+np.random.randint(-5,6),0),255)
                    bg_img[y_i][x_i][2] = min(max(img[ay][ax][2]+np.random.randint(-5,6),0),255)
    return bg_img, (center_x,center_y)

# generates a sample image
def generate_sample(bg_sampler:Backgrounds, tgt_sampler:Targets, idx):
    bg_img = bg_sampler.sample()
    bg_img_size = bg_img.shape[0]
    with open("base_labels/data"+str(idx)+".txt","w") as file:
        for i in range(np.random.randint(0,MAX_NUM_OBJECTS+1)):
            img, data = tgt_sampler.sample()
            img_size = img.shape[0]
            # rotate image (make smaller to fit in image frame)
            rotation = np.random.randint(0,360)
            M = cv2.getRotationMatrix2D(
                (img_size // 2, img_size // 2), rotation, 0.75)
            img = cv2.warpAffine(img, M, (img_size, img_size))

            # paste image
            bg_img, center_pt = alpha_paste(bg_img, img)
            c_x = str(center_pt[0] / bg_img_size)
            c_y = str(center_pt[1] / bg_img_size)
            px_size = str(img_size / bg_img_size)
            file.write("0 "+c_x+" "+c_y+" "+px_size+" "+px_size+"\n")
        file.close()
    cv2.imwrite("base_images/data"+str(idx)+".png",bg_img)

if __name__ == '__main__':
    print("Starting generator...")
    # clean folder
    files = glob.glob("base_images/*")
    for f in files:
        os.remove(f)
    files = glob.glob("base_labels/*")
    for f in files:
        os.remove(f)

    backgrounds = Backgrounds()
    targets = Targets()

    try:
        pool = ThreadPool(processes=CORE_COUNT)

        # Process Images Standard.
        for i in tqdm(range(SAMPLE_SIZE), desc="generating data"):
            # Multi Process Images
            index = i * CORE_COUNT
            iter = np.zeros(CORE_COUNT, dtype=np.object_)
            for j in range(CORE_COUNT):
                iter[j] = (backgrounds, targets, index + j)
            pool.starmap(generate_sample, iter)

    finally:
        pool.close()
        pool.join()



