from train import *

if __name__=='__main__':
    gt = load_gt()
    img = load_test_clip(gt[0]['raw_file'])
    show(generate_labels(resize(img), gt[0]))
    show(draw_lanes(img, gt[0], style='circles'))