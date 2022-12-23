"""

    @date:    07/04/21
    @author:  Dean Poulos

    @purpose: Frames-to-video visualiser with bounding boxes

    Usage: 
        python watch_video.py [ --clip | -c ] clip_number
                              [ --src | -s ] source_clips_path

"""

import os
import cv2
import argparse
import json
import time

def load_image(path, clip, frame):
    img_path = os.path.join(path, str(clip), "imgs", f"0{str(frame).rjust(2, '0')}.jpg")
    return cv2.imread(img_path)

def draw_bbox(img, bbox):
    res = img.copy()
    ix, iy, x, y = bbox['left'], bbox['top'], bbox['right'], bbox['bottom']
    cv2.rectangle(res,(ix,iy),(x,y),(0,255,0),2)

    return res

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--clip", default='1', help="clip number for labelling")        
    ap.add_argument("-s", "--src", default='evaluation/clips', help="path to input clips folder")        
    args = vars(ap.parse_args())

    path, clip, frame = args['src'], args['clip'], 1
    # get clips/annotations paths from os
    clip_path = os.path.join(path, clip)
    path_to_annotations = os.path.join(clip_path, 'annotation.json')
    # read annotations dict if exists, otherwise make it
    assert os.path.isfile(path_to_annotations), f"No annotations found for clip {clip}"
    with open(path_to_annotations, 'r') as f:
        annotations = json.load(f)

    while(1):
        img = load_image(path, clip, frame)
        img = draw_bbox(img, annotations[str(frame)])        
        cv2.imshow('image',img)
        if frame == 40: 
            time.sleep(1)
            frame = 1
        else: frame += 1

        k = cv2.waitKey(1) & 0xFF
        if k == 27: break

        time.sleep(0.015)


    cv2.destroyAllWindows()      
