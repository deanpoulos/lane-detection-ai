from utils import *
from eval import *
from matplotlib import pyplot as plt
import random
import cv2
import matplotlib
matplotlib.use('TkAgg')

if __name__=='__main__':
    #for i in range(100):
    i = 9
    fig, ax = plt.subplots(1, 2)
    gt = load_gt()
    img = resize(load_test_clip(gt[i]['raw_file']))
    erfnet, params = load_model('04-22_01:08:34')
    out = erfnet(cv_to_tensor(img)).cpu().max(dim=1)[1].numpy()
    coords = np.where(out == 1)[1:]
    coords = [coords[-1], coords[-2]]
    templist = list()
    for tup in zip(coords[0], coords[1]):
        templist.append(tup)
    random.shuffle(templist)
    N = 150
    x= [a for a,b in templist[0:N]]
    y= [b for a,b in templist[0:N]]
    gt_lanes, y_samples = gt[i]['lanes'], gt[i]['h_samples']
    #gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]

    for e, mode in enumerate(['mean', 'linear']):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[e].imshow(img, alpha=0.8)
        if e==0: ax[e].scatter(x, y, linewidths=0.5, zorder=0.5, facecolors='none', edgecolors='lime', label='segmented class')
        else: ax[e].scatter(x, y, linewidths=0.5, zorder=0.5, facecolors='none', edgecolors='lime')
        for y_sample in y_samples:
            ax[e].plot([0,639], [y_sample/2]*2, color='red', linewidth=0.1)
        lanes = fit_model(out[0], y_samples, params, mode=mode)
        lane_drawings = [(x, y/2) for (x, y) in zip(lanes[0], y_samples) if x >= 0]
        ax[e].plot(*zip(*lane_drawings), color='red', marker='x', label=f'{mode} poly-line')
        ax[e].set_title(f'{mode} poly-line model fit')
    plt.tight_layout()
    plt.suptitle("Two Poly-line Fitting Methods", y=0.9)
    plt.show()