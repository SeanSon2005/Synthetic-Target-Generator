import cv2
import math
import numpy as np

SEED = 2024
SIZE = 80
TEXT_SIZE = 0.9

SHAPE_NAMES = ['circle',
               'semicircle',
               'quartercircle',
               'triangle',
               'rectangle',
               'pentagon',
               'star',
               'cross']

FONT = cv2.FONT_HERSHEY_SIMPLEX

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


class Targets:
    def __init__(self):
        pass
    def sample(self, color = None, shape_id = None, black = False):
        # SHAPE AND ALLPHANUMERIC
        alphanum_id = np.random.randint(0,36)

        if shape_id is None:
            shape_id = np.random.randint(0,8)
        
        if color is None:
            # COLOR 1
            r = np.random.randint(0, 256)
            g = np.random.randint(0, 256)
            b = np.random.randint(0, 256)
            color = (b, g, r, 255)
        else:
            b = int(color[0])
            g = int(color[1])
            r = int(color[2])
            color = (b, g, r, 255)

        # COLOR 2
        if black:
            color2 = (0,0,0,255)
        else:
            r2 = np.random.randint(0, 256)
            g2 = np.random.randint(0, 256)
            b2 = np.random.randint(0, 256)
            while(True):
                if (abs(r - r2) + abs(g - g2) + abs(b - b2) > 0.35):
                    break
                r2 = np.random.randint(0, 256)
                g2 = np.random.randint(0, 256) 
                b2 = np.random.randint(0, 256)

            color2 = (b2, g2, r2, 255)
        
        if shape_id == 0:
            img = circle(0,color,color2,alphanum_id) # 0 circle
        elif shape_id == 1:
            img = circle(1,color,color2,alphanum_id) # 1 semi-circle
        elif shape_id == 2:
            img = circle(2,color,color2,alphanum_id) # 2 quarter-circle
        elif shape_id == 3:
            img = polygon(3,color,color2,alphanum_id) # 3 triangle
        elif shape_id == 4:
            img = polygon(4,color,color2,alphanum_id) # 4 rectangle
        elif shape_id == 5:
            img = polygon(5,color,color2,alphanum_id) # 5 pentagon
        elif shape_id == 6:
            img = star(color,color2,alphanum_id) # 6 star
        else:
            img = cross(color,color2,alphanum_id) # 7 cross

        # Data augmentation
        noise_level = np.random.randint(10,15)
        noise = np.random.normal(0, noise_level, img.shape)
        img = cv2.add(img, noise)
        img = cv2.GaussianBlur(img, (7,7), 0)
        
        data = (shape_id, alphanum_id)
        return img, data

