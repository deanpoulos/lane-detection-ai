import numpy as np
import cv2
import torch
import json
from train import resize, cv_to_tensor, tensor_to_cv
from tqdm import tqdm
from model import ERFNet
from utils import *
import matplotlib
matplotlib.use('TkAgg')

global WIDTH, HEIGHT, N_SEG, RATIO

def load_model(path):
    # use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda': map_location=lambda storage, loc: storage.cuda()
    else: map_location='cpu'

    # get model weights and parameters
    models_dir = 'models'
    model_path = os.path.join(models_dir, path)
    json_path = os.path.join(models_dir, f'{path}.json' )
    with open(json_path, 'r') as f: params = json.load(f)
    segmentation_model = torch.load(model_path, map_location = map_location)

    # load model using weights 
    erfnet = ERFNet(params['settings']['N_SEG']).to(device)
    erfnet.load_state_dict(segmentation_model)

    return erfnet, params
 

def fit_model(out, y_samples, params, mode='mean'):

    # extract information from params
    settings = params['settings']
    w, h, n, r = settings['WIDTH'], settings['HEIGHT'], settings['N_SEG'], settings['RATIO']

    modes = ['mean', 'linear']
    assert mode in modes, f"{mode} not available. Implemented modes are {modes}"

    if mode=='linear':
        polys = []
        y_scaled = [round(y/r) for y in y_samples]
        for lane in range(1, n+1):
            ys, xs = np.where(out==lane)
            if len(xs): polys.append(np.poly1d(np.polyfit(ys, xs, 1)))
            else: polys.append(lambda x: -2)
    
    lanes = [[-2]*len(y_samples) for _ in range(n)]
    for i, y in enumerate(y_samples):
        y = round(y/r)
        for lane in range(1, n+1):
            if mode=='mean':
                # get the mean of all segment x-values and assign to poly-line
                x_indices = np.where(out[y,:] == lane)[0]
                lanes[lane-1][i] = int(np.mean(x_indices)) if len(x_indices) else -2
            elif mode=='linear':
                # fit a line through all segment coordinates and sample for poly-line
                if polys[lane-1](12345) == -2 or not len(np.where(out[y,:] == lane)[0]):
                    lanes[lane-1][i] = -2
                else:
                    lanes[lane-1][i] = polys[lane-1](y)

    return lanes


if __name__ == '__main__':

    plot=True
    path = 'adam_5'  # sorted(os.listdir('models'))[-5]
    erfnet, params = load_model(path)
    settings = params['settings']
    WIDTH, HEIGHT, N_SEG, RATIO = settings['WIDTH'], settings['HEIGHT'], settings['N_SEG'], settings['RATIO']
    erfnet = erfnet.eval()

    gt = load_gt()
    num_evals = len(gt)
    fit_method = 'mean'
    average_accuracy = 0

    for j in tqdm(range(9, num_evals), f"Evalutating model {path} on {num_evals} test examples"):
        tst = resize(load_test_clip(gt[j]['raw_file']))
        t = cv_to_tensor(tst)
        out = erfnet(t)
        out = out.max(dim=1)[1]
        out = out.cpu().numpy()[0]

        y_samples = gt[j]['h_samples']
        lanes = fit_model(out, y_samples, params, mode=fit_method)
        pred = {'lanes': lanes, 'h_samples': [int(round(y/RATIO)) for y in y_samples]} 
        lanes = [[x * RATIO for x in lane] for lane in lanes]
        accuracy, x, y = LaneEval.bench(gt[j]['lanes'], lanes, y_samples, 1)[:3]
        average_accuracy += accuracy / num_evals
        if plot:
            seg = draw_output(tst.copy(), out, N_SEG=5)
            mean_lanes = fit_model(out, y_samples, params, mode='mean')
            linr_lanes = fit_model(out, y_samples, params, mode='linear')
            mean_pred = {'lanes': mean_lanes, 'h_samples': [int(round(y/RATIO)) for y in y_samples]} 
            linr_pred = {'lanes': linr_lanes, 'h_samples': [int(round(y/RATIO)) for y in y_samples]} 
            mean = draw_lanes(tst.copy(), mean_pred, style='lines')
            linear = draw_lanes(tst.copy(), linr_pred, style='lines')
            fig, ax = plt.subplots(2,2)
            ax[0,0].imshow(cv2.cvtColor(tst, cv2.COLOR_BGR2RGB))
            ax[0,0].set_xlabel('Original')
            ax[0,1].imshow(seg)
            ax[0,1].set_xlabel('Network Output')
            ax[1,0].imshow(cv2.cvtColor(mean, cv2.COLOR_BGR2RGB))
            ax[1,0].set_xlabel('Mean Line-Fitting')
            ax[1,1].imshow(cv2.cvtColor(linear, cv2.COLOR_BGR2RGB))
            ax[1,1].set_xlabel('Linear Regression Line-Fitting')
            fig.tight_layout()
            plt.show()
    
    # export average_accuracy and number of test examples to json
    print(f"{average_accuracy*100}% average accuracy")
    if 'evaluation' not in params: params['evaluation'] = {}
    params['evaluation'][fit_method] = {
        'average_accuracy': average_accuracy,
        'number_of_evaluation_images': num_evals
    }
    model_path = os.path.join('models', path)
    with open(f"{model_path}.json", 'w+') as f: json.dump(params, f, indent=4)