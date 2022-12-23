import json
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib
from evaluate.lane import LaneEval
matplotlib.use('TkAgg')

gt_base_path = '../data/lane_detection/'
gt_records_path = 'test_label.json'
gt_path = os.path.join(gt_base_path, gt_records_path)
pred_path = gt_path


json_pred = [json.loads(line) for line in open(pred_path).readlines()]
json_gt = [json.loads(line) for line in open(gt_path)]

#print(json_gt[0]['lanes'])
#print(json_gt[0]['h_samples'])

"""
NOTE:   Only some example images are loaded to git, and the randomly copied
        directories do NOT map to consecutive values in the ground-truth
        dictionary. valid i values are:

        [1, 47, 56, 60, 65, 71, 78, 83, 93, 101, 115, 128, 161, 186]
"""
i = 1
pred, gt = json_pred[i], json_gt[i]
pred_lanes = pred['lanes']
run_time = 0.12345 # pred['run_time']
gt_lanes = gt['lanes']
y_samples = gt['h_samples']
raw_file = gt['raw_file']

img = plt.imread(os.path.join(gt_base_path, 'example', raw_file))
plt.imshow(img)
plt.show()


gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
img_vis = img.copy()

for lane in gt_lanes_vis:
    for pt in lane:
        cv2.circle(img_vis, pt, radius=5, color=(0, 255, 0))

plt.imshow(img_vis)
plt.show()


gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
pred_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in pred_lanes]
img_vis = img.copy()

#for lane in gt_lanes_vis:
#    cv2.polylines(img_vis, np.int32([lane]), isClosed=False, color=(0,255,0), thickness=5)

plt.imshow(img_vis)
plt.show()


np.random.shuffle(pred_lanes)
# Overall Accuracy, False Positive Rate, False Negative Rate
_, _, _, threshs = LaneEval.bench(pred_lanes, gt_lanes, y_samples, run_time)
colors = [[255,0,0],[0,255,0],[0,0,255],[255,255,0]]
for i, lane in enumerate(gt_lanes_vis):
    left, right = lane.copy(), lane.copy()
    for j in range(len(lane)):
        left[j] = (max(0, lane[j][0] + threshs[i]), lane[j][1])
        right[j] = (max(0, lane[j][0] - threshs[i]), lane[j][1])
    cv2.polylines(img_vis, np.int32([left]), isClosed=False, color=colors[i], thickness=3)
    cv2.polylines(img_vis, np.int32([right]), isClosed=False, color=colors[i], thickness=3)
    cpy = img_vis.copy()
    right = np.flipud(right)
    points = np.concatenate((left, right)).astype(np.int32)
    cv2.fillPoly(img_vis, [points], color=colors[i])
    alpha = 0.5
    img_vis=cv2.addWeighted(img_vis, alpha, cpy,1-alpha, gamma=0)

plt.imshow(img_vis)
plt.title('$\\theta$-specific true-positive regions', fontsize=26, y=1.04, usetex=True)
plt.show()