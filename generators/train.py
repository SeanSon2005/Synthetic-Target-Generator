import cv2
import numpy as np
from sklearn.cluster import KMeans
from targets import Targets

SAMPLE_SIZE = 20

target_sampler = Targets()

SHAPE_DICT = [
    2, # quarter
    5, # pentagon
    6, # star
    2,
    1, # semi
    5,
    6,
    3, # triangle
    7, # cross
    4, # rectangle
    6,
    3,
    0, # circle
    0,
    4,
    4,
    7,
    1,
    0
]

for i in range(1,20):
    filename = "diffusion/target_data/t" + str(i) + ".png"
    img = cv2.imread(filename=filename)
    center_img = img[60-SAMPLE_SIZE:60+SAMPLE_SIZE,60-SAMPLE_SIZE:60+SAMPLE_SIZE]
    center_img = center_img.reshape((center_img.shape[0] * center_img.shape[1], 3))

    # GET COLOR
    kmeans = KMeans(n_clusters = 5)
    kmeans.fit(center_img)
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    idx = np.where(counts == counts.max())[0]
    avg_color = tuple(map(int,kmeans.cluster_centers_[idx][0]))

    img, data = target_sampler.sample(avg_color, SHAPE_DICT[i-1], True)
    cv2.imwrite("diffusion/synthetic_data/t"+str(i)+".png", img)
