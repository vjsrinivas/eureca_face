import os
import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
from preprocessing import readNoiseText
import numpy as np
import copy

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_file', type=str, default='noise.txt')
    parser.add_argument('--result_dir', type=str, default='./results')
    return parser.parse_args()

def graph_2(noise_dict, result_root):
    col = plt.cm.cool([0.25,0.75,1])
    for item in noise_dict.items():
            mAP = {}
            plt.figure()
            plt.title(item[0])
            x = item[1]

            for r in item[1]:
                file_path = os.path.join(result_root, 'retinaface', '%s_%0.6f_map.txt'%(item[0], r))
                with open(file_path, 'r') as f:
                    contents = list(map(str.strip, f.readlines()))
                    for con in contents:
                        cat, val = con.split(',')
                        val = float(val)
                        if not cat in mAP:
                            mAP[cat] = [val]
                        else:
                            mAP[cat].append(val)
        
            legend = list(mAP.keys())
            for i, map_item in enumerate(mAP.items()):
                y = map_item[1]
                plt.plot(x,y, color=col[i])
            if item[0] == 'speckle':
                plt.gca().invert_xaxis()
            plt.legend(legend)
            plt.show()

def graph_1(noise_dict, result_root, models):

    def sort_first(x):
        return x[0][0]

    eval_modes = ['easy_val', 'medium_val', 'hard_val']
    for item in noise_dict.items():
        avg = []
        colors = cm.rainbow(np.linspace(0,1,len(models)*len(eval_modes))) 
        fig = plt.figure(figsize=(10,5.5))
        x = item[1]

        model_container = {i:[] for i in models}

        for mod in models:
            emd = {i:[] for i in eval_modes}
            
            for r in item[1]:
                file_path = os.path.join(result_root, mod, '%s_%0.6f_map.txt'%(item[0], r))    
                with open(file_path, 'r') as f:
                    contents = list(map(str.strip, f.readlines()))
                    for con in contents:
                        cat, val = con.split(',')
                        val = float(val)
                        emd[cat].append(val)
            model_container[mod].append(emd)

        plt.title(determineProperName(item[0]))
        legend = []
        plot_y = []
        for m in model_container.items():
            for _emd in m[1]:
                for k,n in enumerate(_emd.items()):
                    plot_y.append((n[1], '%s %s'%(m[0], n[0])))
                    plot_y.sort(key=sort_first)
                    if len(avg) == 0:
                        avg = copy.copy(n[1])
                    else:
                        for j,nval in enumerate(n[1]):
                            avg[j] = avg[j] + nval
                    #plt.plot(x,n[1], color=colors[i])
        
        for _n,_c in zip(plot_y, colors):
            print(_n)
            plt.plot(x, _n[0], color=_c)
            legend.append(_n[1])
        
        avg = [_avg/len(legend) for _avg in avg]
        plt.plot(x,avg, color='black', linewidth=3)
        legend.append('average')

        # flip x-axis for speckle:
        if item[0] == 'speckle':
            plt.gca().invert_xaxis()

        plt.ylabel('mAP (%)')
        plt.xlabel(determineXLabel(item[0]))

        plt.ylim([0,1])
        plt.legend(legend, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('./results/graphs/overview_%s_graph.png'%item[0])

def determineXLabel(input_str):
    if input_str == 'gaussian_noise':
        return 'Standard Deviation'
    elif input_str == 'salt_pepper':
        return 'Percentage of Image'
    elif input_str == 'poisson':
        return 'Lambda'
    elif input_str == 'speckle':
        return 'Uniform Low Value'


def determineProperName(input_str):
    if input_str == 'gaussian_noise':
        return 'Gaussian Noise'
    elif input_str == 'salt_pepper':
        return 'Salt and Pepper'
    elif input_str == 'poisson':
        return 'Poisson'
    elif input_str == 'speckle':
        return 'Speckle'
    else:
        raise Exception('Invalid input_str specified!')


if __name__ == '__main__':
    args = parseArgs()

    with open(args.noise_file, 'r') as f:
        contents = list(map(str.strip, f.readlines()))
        noise_dict = readNoiseText(contents)
    
    #graph_2(noise_dict, args.result_dir)
    graph_1(noise_dict, args.result_dir, ['retinaface', 'tinaface', 'dsfd'])
        