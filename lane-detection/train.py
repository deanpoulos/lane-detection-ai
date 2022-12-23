import numpy as np
import cv2
import torch
import math
import torch.nn
import torchvision.transforms as transforms
import pickle as pkl
import time
from datetime import datetime
import json
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor
from model import ERFNet
from utils import *
from loss import InstanceLoss
from torch.nn import BCELoss
from torch.utils.data import Dataset, DataLoader

# constants
N_SEG = 5       # number of lanes for segmentation
HEIGHT = 360    # img resize in pixels
WIDTH = 640
RATIO = 2

# use GPU if available
#device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cuda': map_location=lambda storage, loc: storage.cuda()
else: map_location='cpu'

def tensor_to_cv(tensor, transpose=False):
    if transpose: cv = np.transpose(tensor.squeeze(0).cpu().numpy(), (1, 2, 0))
    else: cv = tensor.squeeze(0).cpu().numpy()
    return cv.astype(np.int64)

def labels_to_tensor(lbl):
    im_tensor = torch.from_numpy(lbl).to(torch.float).to(device).unsqueeze(0)
    print(im_tensor.shape)
    return im_tensor

def cv_to_tensor(img):
    op_transforms = transforms.Compose([transforms.ToTensor()])
    im_tensor = torch.unsqueeze(op_transforms(img), dim=0)
    im_tensor = im_tensor.to(device)
    return im_tensor

def resize(img, WIDTH=WIDTH, HEIGHT=HEIGHT): return cv2.resize(img, (WIDTH, HEIGHT))

def generate_labels(img, gt):
    mask = np.zeros_like(img)
    gt_lanes = [[round(x/RATIO) for x in lane] for lane in gt['lanes']]
    y_samples = [round(y/RATIO) for y in gt['h_samples']]
    gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
    colors = [[255,0,0],[0,255,0],[0,0,255],[0,255,255],[255,255,0]]
    for i in range(len(gt_lanes_vis)): cv2.polylines(mask, np.int32([gt_lanes_vis[i]]), isClosed=False,color=colors[i], thickness=5)
    label = np.zeros((img.shape[0], img.shape[1]),dtype = np.uint8)
    for i in range(len(colors)): label[np.where((mask == colors[i]).all(axis=2))] = 20*(i+1)
    #labels = np.zeros((N_SEG, img.shape[0], img.shape[1]), dtype = np.uint8)
    #for i in range(len(colors)): labels[i][np.where((mask == colors[i]).all(axis=2))] = i
    #return labels
    return label

class TuSimpleDataset(Dataset):
    def __init__(self, data, lazy_load=True, save=False):
        self.filenames = [data[i]['raw_file'] for i in range(len(data))]
        self.images = []
        lazy_load_path = 'data/images.pkl' 
        if lazy_load: 
            with open(lazy_load_path, 'rb') as f: self.images = pkl.load(f)
        else:
            for i in tqdm(range(len(self.filenames)), "pre-processing images"):
                self.images.append(resize(load_train_clip(self.filenames[i])))
            if save==True: 
                with open(lazy_load_path, 'wb') as f: pkl.dump(self.images, f)
        self.data = data


    def __getitem__(self, index):
        inp = cv_to_tensor(self.images[index])
        label = cv_to_tensor(generate_labels(self.images[index], self.data[index]))
        #label = labels_to_tensor(generate_labels(self.images[index], self.data[index]))
        return inp, label

    def __len__(self): return len(self.images)


def main():
    # instantiate network
    pretrained=False
    if(pretrained): 
        path = 'erfnet_tusimple.pth'
        erfnet, params = load_model()
    else: erfnet = ERFNet(N_SEG).to(device)
    erfnet = erfnet.train()  # set training mode

    # instantiate optimiser
    lrn_rate = 5e-4
    weight_decay = 0.0
    loss_obj = InstanceLoss() #BCELoss() 
    optimiser = 'Adam'
    if optimiser=='Adam': optimizer = torch.optim.Adam(erfnet.parameters(), lr=lrn_rate, weight_decay=weight_decay)
    elif optimiser=='SGD': optimizer = torch.optim.SGD(erfnet.parameters(), lr=lrn_rate, momentum=0.05)
    else: raise NotImplementedError(f"optimiser {optimiser} not implemented.")

    # load pre-processed dataset
    batch_size = 10
    with open('data/train_data.pkl', 'rb') as f: data = pkl.load(f)
    dataset_train = TuSimpleDataset(data)
    loader = DataLoader(dataset_train, batch_size=batch_size)

    # batching
    num_examples = len(data)
    num_batches = num_examples // batch_size + 1
    epochs = 15
    early_stop = False
    epoch_loss = [0.]*epochs

    t1 = tqdm(range(0, epochs), "All Epochs")
    for epoch in t1:
        try:
            d = f"Epoch {epoch+1}/{epochs}"
            t2 = tqdm(loader, desc=d, leave=False)
            if epoch==0: start_time = time.time()
            for (batchidx, batch) in enumerate(t2):
                X = batch[0].squeeze(1)
                Y = batch[1].squeeze(1)
                optimizer.zero_grad()
                out = erfnet(X)
                loss_val = loss_obj(out, Y)   # a tensor
                d = f"Epoch {epoch+1}/{epochs}, Batch Loss: {loss_val.item():.3f}"
                t2.set_description(d); t2.refresh()
                epoch_loss[epoch] += loss_val.item()  # accumulate
                loss_val.backward()  # compute all gradients
                optimizer.step()     # update all wts, biases
            if epoch==0: total_time = time.time() - start_time
        except KeyboardInterrupt:
            print(" Stopping training early...")
            ans = ''
            while ans not in ['y', 'n']: ans = input("Would you like to save the model? (y/n) ")
            if ans == 'y':
                early_stop=True
                epochs = epoch
                break
            else: exit()

        t1.set_description(f"All Epochs, Epoch Loss: {epoch_loss[epoch]:.3f}")
        t1.refresh()
    
    params = { 'early_stop': early_stop,
               'optimiser': str(optimizer), 
               'batch_shape': { 'batch_size': batch_size, 'num_batches': num_batches },
               'settings' : { 'N_SEG': N_SEG, 'HEIGHT': HEIGHT, 'WIDTH': WIDTH, 'RATIO': RATIO },
               'loss': { 'loss_function': str(loss_obj), 'epochs': epochs, "epoch_loss": epoch_loss[:epochs] },
               'training_time': total_time*epochs}

    dt = path if pretrained else datetime.now().strftime('%m-%d_%H:%M:%S')
    PATH = os.path.join('models', dt)
    torch.save(erfnet.state_dict(), PATH)
    with open(os.path.join('models', f'{dt}.json'), 'w+') as f: json.dump(params, f, indent=4)

if __name__ == '__main__':
    main()