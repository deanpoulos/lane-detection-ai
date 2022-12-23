import json
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Qt5Agg')


if __name__ == '__main__':
    mode = 'Model-Fit Method'  # 'Training Time'
    i = 0
    markers = ['o', '+', 'x', '*', '.', 'X']
    colours = ['orange', 'blue']
    optims = ['sgd', 'adam']
    bss = [5] if mode=='Model-Fit Method' else [1,5,10]
    for bs in bss:
        for optim in optims:
            with open(f"models/{optim}_{bs}.json", 'r') as f: params = json.load(f)
            if mode=='Model-Fit Method':
                j = 0
                for k, v in params['evaluation'].items():
                    a = params['evaluation'][k]['average_accuracy']
                    print(f"{k}, {a} for {bs} and {optim}")
                    if j==0:
                        plt.plot( k, a, marker=markers[i], linestyle="None", 
                                markersize=10, color=colours[i],
                                label=f"{optim}, batch_size={bs}")
                    else:
                        plt.plot( k, a, marker=markers[i], linestyle="None", 
                                markersize=10, color=colours[i])
                    j += 1
                i += 1
            else:
                a = params['evaluation']['mean']['average_accuracy'] * 100
                t = params['training_time']
                e = params['loss']['epochs']
                if bs==1 and optim=='sgd': optim='sgd, momentum=0'
                plt.plot(
                    t, a, marker=markers[i], linestyle="None", markersize=10, 
                    label=f'{optim}, batch_size={bs}, epochs={e}') 
                i += 1

    if mode != 'Model-Fit Method':
        plt.grid()
    plt.legend()
    plt.title(f'Accuracy vs {mode}')
    if mode=='Training Time':
        plt.xlabel(f"{mode} (s)")
    plt.ylabel('Accuracy (%)')
    plt.show()    

    if mode=='Model-Fit Method':
        fig, ax = plt.subplots()
        labels = ['Mean', 'Linear Regression']
        width = 0.35
        x = np.arange(len(labels))
        adams = np.array([0.905, 0.894]) * 100
        sgds = np.array([0.877, 0.860]) * 100
        rects1 = ax.bar(x - width/2, adams, width, label='Adam')
        rects2 = ax.bar(x + width/2, sgds, width, label='SGD')
        ax.set_ylabel('Accuracy (%)')
        ax.set_xlabel('Model Fitting Method')
        ax.set_title('Accuracy per model fitting method')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc='lower left')

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        plt.show()