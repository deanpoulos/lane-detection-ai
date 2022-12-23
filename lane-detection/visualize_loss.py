import json
from matplotlib import pyplot as plt

if __name__ == '__main__':
    optims = ['sgd', 'adam']
    for bs in [1, 5, 10]:
        for optim in optims:
            with open(f"models/{optim}_{bs}.json", 'r') as f: params = json.load(f)
            average_accuracy = params['evaluation']['mean']['average_accuracy']
            if bs==1 and optim=='sgd': optim='sgd (no momentum)'
            plt.plot(params['loss']['epoch_loss'], linestyle='-', marker='x', label=optim) 
        plt.title(f'Loss for batch_size={bs}')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Raw Loss')
        plt.show()
