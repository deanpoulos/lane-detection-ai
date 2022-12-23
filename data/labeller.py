"""

    @date:    07/04/21
    @author:  Dean Poulos

    @purpose: Car-detection image labelling tool.

    Usage: 
        python labeller.py [ --clip | -c ] clip_number
                           [ --frame | -f ] frame_number
                           [ --src | -s ] source_clips_path
                           [ --dst | -d ] path_to_outputs

"""

import os
import cv2
import numpy as np
import argparse
import json

def load_image(path, clip, frame):
    img_path = os.path.join(path, str(clip), "imgs", f"0{frame.rjust(2, '0')}.jpg")
    return cv2.imread(img_path)

def load_annotations(path, clip):
    with open(os.path.join(path, f'{clip}/annotation.json'), 'r') as f:
        annotations = json.load(f)

    return annotations

def make_bbox(ix, iy, x, y):
    return({ 'left': min(ix, x), 'top': min(iy, y), 'right': max(ix, x), 'bottom': max(iy, y) })

def next(clip, frame):
    clip = str(int(clip) + 1) if int(frame)==40 else clip
    frame = '1' if int(frame)==40 else str(int(frame) + 1)
    return clip, frame

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--clip", default='1', help="clip number for labelling")        
    ap.add_argument("-f", "--frame", default='1', help="starting frame of clip")        
    ap.add_argument("-s", "--src", default='benchmark_velocity_test/clips', help="path to input clips folder")        
    ap.add_argument("-d", "--dst", default='evaluation/clips', help="path to output clips folder")        
    args = vars(ap.parse_args())

    path, clip, frame, out = args['src'], args['clip'], args['frame'], args['dst']
    name = f"clip-{clip}, frame-{frame}"
    # get clips/annotations paths from os
    clip_path = os.path.join(out, clip)
    annotations_path = os.path.join(clip_path, 'annotation.json')
    # read annotations dict if exists, otherwise make it
    if os.path.isfile(annotations_path):
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
    else:
        if not os.path.exists(clip_path):
            os.makedirs(clip_path)
        annotations = {}

    # load image and copy
    original = load_image(path, clip, frame)
    img = original.copy()

    drawing = False # true if mouse is pressed
    update=True
    ix,iy = -1,-1

    # mouse callback function
    def draw_rectangle(event,x,y,flags,param):
        global ix,iy,drawing,img,frame,original,clip,clip_path,annotations,annotations_path

        if event == cv2.EVENT_LBUTTONDOWN:
            # save starting coordinates
            drawing = True
            ix,iy = x,y

        elif event == cv2.EVENT_MOUSEMOVE:
            # update drawn bounding box
            update = True
            if drawing == True:
                img = original.copy()
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            # draw bounding box
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),1)
            # write bbox dict to annotations[frame]
            bbox = make_bbox(ix, iy, x, y)
            print(f"clip: {clip}, frame: {frame}", end=': ')
            print(bbox)
            annotations[frame] = make_bbox(ix, iy, x, y)
            # write annotations to file
            with open(annotations_path, 'w+') as f:
                json.dump(annotations, f)
            # write image to folder
            imgs_path = os.path.join(clip_path, "imgs")
            if not os.path.exists(imgs_path):
                os.makedirs(imgs_path)
            cv2.imwrite(os.path.join(imgs_path, f"{frame.rjust(3, '0')}.jpg"), original)
            # get next clip/frame numbers
            clip, frame = next(clip, frame)
            # update clip/annotations paths
            clip_path = os.path.join(out, clip)
            annotations_path = os.path.join(clip_path, 'annotation.json')
            # update image according to new clips/frames
            original = load_image(path, clip, frame)
            img = original.copy()
            # read annotations dict if exists, otherwise make it
            if os.path.isfile(annotations_path):
                with open(annotations_path, 'r') as f:
                    annotations = json.load(f)
            else:
                if not os.path.exists(clip_path):
                    os.makedirs(clip_path)
                annotations = {}

    cv2.namedWindow(name)
    cv2.setMouseCallback(name,draw_rectangle)

    while(1):
        cv2.imshow(name,img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27: break

    cv2.destroyAllWindows()      