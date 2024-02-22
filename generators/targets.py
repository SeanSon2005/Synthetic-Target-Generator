import cv2
import math
import numpy as np

SEED = 2024
SIZE = 60
TEXT_SIZE = 0.8

COLORS = {  # the 4th value is for alpha channel
    0: (0,0,0,255),  # black
    1: (255,255,255,255),  # white
    2: (0,0,255,255),  # red
    3: (0,255,0,255),  # green
    4: (255,0,0,255),  # blue
    5: (128,0,128,255),  # purple
    6: (0,165,255,255),  # orange
    7: (19,69,139,255),  # brown
}
COLOR_NAMES = ['black',
               'white',
               'red',
               'green',
               'blue',
               'purple',
               'orange',
               'brown']
SHAPE_NAMES = ['circle',
               'semicircle',
               'quartercircle',
               'triangle',
               'rectangle',
               'pentagon',
               'star',
               'cross']

FONT = cv2.FONT_HERSHEY_SIMPLEX

class Targets:
    def __init__(self):
        try:
            self.images = np.load("generators/images.npy")
            self.data = np.load("generators/data.npy")
        except:
            raise Exception("run targets.py")
    def sample(self):
        id = np.random.randint(0,len(self.images))
        return self.images[id], self.data[id]

# alphanumeric: 0 - 35
def add_text(img, alphanum, color):
    alphanum += 48
    if alphanum > 57:
        alphanum += 7
    txt = str(chr(alphanum))
    return cv2.putText(img,
                    txt,
                    (SIZE//2-9,SIZE//2+9),
                    FONT,
                    TEXT_SIZE,
                    color,
                    2)

# Base Shape generators
def polygon_points(sides = 4):
    if sides == 3:
        adjust = 0.523599
    elif sides == 4:
        adjust = 0.785398
    else:
        adjust = -0.3
    center = SIZE / 2
    r = SIZE / 2
    points = []
    for i in range(sides):
        x = int(center + r * math.cos(2 * math.pi * i / sides + adjust))
        y = int(center + r * math.sin(2 * math.pi * i / sides + adjust))
        points.append((x,y))
    return [np.array(points)]
def star_points(size):
    x, y, a = size, size, size
    points = []
    for i in range(10):
        angle = i * 36 - 90
        if i % 2 == 0:
            aX = x + a * math.cos(math.radians(angle))
            aY = y + a * math.sin(math.radians(angle))
        else:
            aX = x + a * math.cos(math.radians(angle)) * 0.45
            aY = y + a * math.sin(math.radians(angle)) * 0.45
        points.append((int(aX), int(aY)))
    return [np.array(points)]
def cross_points(size):
    points = [(size//2,0),
              (int(size*1.5),0),
              (int(size*1.5),size//2),
              (size*2,size//2),
              (size*2,int(size*1.5)),
              (int(size*1.5),int(size*1.5)),
              (int(size*1.5),size*2),
              (size//2,size*2),
              (size//2,int(size*1.5)),
              (0,int(size*1.5)),
              (0,size//2),
              (size//2,size//2)]
    return [np.array(points)]

def polygon(sides, color, color2, alphanum):
    img = np.zeros((SIZE, SIZE, 4))
    return add_text(cv2.drawContours(img,
                            polygon_points(sides),
                            0,color,-1),
                            alphanum, color2)
# type 0: circle, type 1: half, type 2: semi
def circle(type, color, color2, alphanum):
    img = np.zeros((SIZE, SIZE, 4))
    center = SIZE // 2
    if type == 0:
        return add_text(cv2.circle(img, (center, center), center, color, -1),
                        alphanum, color2)
    if type == 1:
        return add_text(cv2.ellipse(img, (center, int(center * 1.5)), 
                           (center,center), 
                           0, 0, -180, color, -1),
                           alphanum, color2)
    if type == 2:
        return add_text(cv2.ellipse(img, 
                           (int(center/2.5),int(center/2.5)), 
                           (int(center*1.5),int(center*1.5)),
                            0, 0, 90, color, -1),
                            alphanum, color2)
def star(color, color2, alphanum):
    img = np.zeros((SIZE, SIZE, 4))
    center = SIZE // 2
    return add_text(cv2.drawContours(img, 
                            star_points(center), 
                            0, color, -1),
                            alphanum, color2)
def cross(color, color2, alphanum):
    img = np.zeros((SIZE,SIZE,4))
    center = SIZE//2
    return add_text(cv2.drawContours(img, 
                            cross_points(center), 
                            0, color, -1),
                            alphanum, color2)

# Actual Image Generation
if __name__ == '__main__':
    images = []
    data = []
    for color_idx in range(8):
        for color_idx2 in range(8):
            # skip same color
            if color_idx == color_idx2:
                continue
            for alphanum_id in range(36):
                color = COLORS[color_idx]
                color_name = COLOR_NAMES[color_idx]
                color2 = COLORS[color_idx2]
                images.append(circle(0,color,color2,alphanum_id))
                images.append(circle(1,color,color2,alphanum_id))
                images.append(circle(2,color,color2,alphanum_id))
                images.append(polygon(3,color,color2,alphanum_id))
                images.append(polygon(4,color,color2,alphanum_id))
                images.append(polygon(5,color,color2,alphanum_id))
                images.append(star(color,color2,alphanum_id))
                images.append(cross(color,color2,alphanum_id))
                for i in range(8):
                    data.append((color_idx,i,alphanum_id))

    np.save("generators/images.npy", np.array(images))
    np.save("generators/data.npy", np.array(data))

