import os
import json 
import cv2
import numpy as np
import sys
import random
from PIL import Image
from matplotlib import pyplot as plt
from evaluate.lane import LaneEval

## Constants
colors = [[255,0,0],[0,255,0],[0,0,255],[0,255,255],[255,255,0]]  # r, g, b, y
train_base_path = '../data/lane_detection/train_set'
test_base_path = '../data/lane_detection/test_set'

def load_gt():
    gt_base_path = '../data/lane_detection/'
    gt_records_path = 'test_label.json'
    gt_path = os.path.join(gt_base_path, gt_records_path)
    return [json.loads(line) for line in open(gt_path)]


def load_test_clip(raw_file, frame=20, grey=False):
    if frame != 20: raw_file =  raw_file.replace("20", str(frame))
    if grey: img = cv2.imread(os.path.join(test_base_path, raw_file), 0)
    else: img = cv2.imread(os.path.join(test_base_path, raw_file))
    return img


def load_train_data():
    paths = [path for path in os.listdir(train_base_path) if 'json' in path]
    jsons = []
    for path in paths:
        with open(os.path.join(train_base_path, path), 'r') as f:
            jsons.append(json.load(f)) 

    l = []
    for js in jsons: l += js
    
    return l


def load_train_clip(raw_file, grey=False):
    if grey:
        img = cv2.imread(os.path.join(train_base_path, raw_file), 0)
    else:
        img = cv2.imread(os.path.join(train_base_path, raw_file))

    return img 


def show(img):
    while(1):
        cv2.imshow('img', img)
        k = cv2.waitKey(33)
        if k==27:    # Esc key to stop
            break
        elif k==-1:  # normally -1 returned,so don't print it
            continue


def draw_lanes(img, pred, style='lines', thickness=None, markers=False):
    gt_lanes = pred['lanes']
    y_samples = pred['h_samples']
    gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
    img_vis = img.copy()

    if style=='lines':
        for j, lane in enumerate(gt_lanes_vis):
            t = 5 if thickness is None else thickness
            cv2.polylines(img_vis, np.int32([lane]), isClosed=False, color=colors[j], thickness=t)
    elif style=='circles':
        for lane in gt_lanes_vis:
            for pt in lane:
                cv2.circle(img_vis, pt, radius=5, color=(0, 255, 0))
    elif style=='threshold':
        _, _, _, threshs = LaneEval.bench(gt_lanes, gt_lanes, y_samples, 1)
        for i, lane in enumerate(gt_lanes_vis):
            left, right = lane.copy(), lane.copy()
            for j in range(len(lane)):
                left[j] = (max(0, lane[j][0] + threshs[i]), lane[j][1])
                right[j] = (max(0, lane[j][0] - threshs[i]), lane[j][1])
            cv2.polylines(img_vis, np.int32([left]), isClosed=False, color=(0,255,0), thickness=3)
            cv2.polylines(img_vis, np.int32([right]), isClosed=False, color=(0,255,0), thickness=3)
            cpy = img_vis.copy()
            right = np.flipud(right)
            points = np.concatenate((left, right)).astype(np.int32)
            cv2.fillPoly(img_vis, [points], color=[0,255,0])
            alpha = 0.6
            img_vis=cv2.addWeighted(img_vis, alpha, cpy,1-alpha, gamma=0)
    else:
        raise ValueError(f"Unspported style {style}. Supported styles are 'lines' and 'circles'.")

    return img_vis
    

def draw_output(img, out, N_SEG=5):
    cpy = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    viz = np.zeros(img.shape, dtype=np.uint8)
    colors = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,255,0]]
    for i in range(1, N_SEG+1):
        viz = color_lanes(viz, out, i, colors[i-1], img.shape[0], img.shape[1])

    return cv2.addWeighted(cpy, 1, viz, 0.4, 0)

###################################################
# START from https://github.com/fabvio/Cascade-LD #
###################################################
def color_lanes(image, classes, i, color, HEIGHT, WIDTH):
    print(color)
    print(color[0])
    buffer_c1 = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    buffer_c1[classes == i] = color[0]
    image[:, :, 0] += buffer_c1
    buffer_c2 = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    buffer_c2[classes == i] = color[1]
    image[:, :, 1] += buffer_c2
    buffer_c3 = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    buffer_c3[classes == i] = color[2]
    image[:, :, 2] += buffer_c3
    return image

#################################################
# END from https://github.com/fabvio/Cascade-LD #
#################################################


if __name__=='__main__':
    jsons = load_train_data()

    i = 0
    for i in range(300):
        img = load_train_clip(jsons[i]['raw_file'])
        lanes = draw_lanes(img, jsons[i], style='lines')
        show(lanes)