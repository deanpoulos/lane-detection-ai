import imageio
import torch
import cv2
from tqdm import tqdm
from eval import load_model, fit_model
from utils import load_gt, load_test_clip, show, draw_lanes
from train import resize, cv_to_tensor

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda': map_location=lambda storage, loc: storage.cuda()
    else: map_location='cpu'
    erfnet, params = load_model('erfnet_tusimple.pth')
    WIDTH, HEIGHT, N_SEG, RATIO = params['WIDTH'], params['HEIGHT'], params['N_SEG'], params['RATIO']
    erfnet = erfnet.eval()
    gt = load_gt()
    num_clips = 10
    fit_method = 'mean'
    imgs = []
    for j in tqdm(range(num_clips), f"Generating .gif"):
        for f in range(1,21):
            tst = resize(load_test_clip(gt[j]['raw_file'], frame=f))
            t = cv_to_tensor(tst) 
            out = erfnet(t)
            out = out.max(dim=1)[1]
            out = out.cpu().numpy()[0]
            y_samples = gt[j]['h_samples']
            lanes = fit_model(out, y_samples, params, mode=fit_method) 
            pred = {'lanes': lanes, 'h_samples': [int(round(y/RATIO)) for y in y_samples]}
            lanes = [[x * RATIO for x in lane] for lane in lanes]
            imgs.append(cv2.cvtColor(draw_lanes(tst, pred, style='lines'), cv2.COLOR_BGR2RGB))

    print("Exporting to gif...")
    imageio.mimsave('../screenshots/lanes.gif', imgs)