import os
import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse

from scipy.special.orthogonal import legendre

from preprocessing import readNoiseText
import numpy as np
import copy

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_file', type=str, default='noise.txt')
    parser.add_argument('--correction_file', type=str, default='corrections.txt')
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

def graph_3(noise_dict, correct_dict, result_root, models):
    def sort_inner(x):
        return x[0]

    val_weights = [0.33,0.33,0.33]

    # assemble all maps from result_root:
    for correct in correct_dict.items():
        for noise in noise_dict.items():
            for mod in models:
                fig = plt.figure(figsize=(10,5.5))
                x = copy.copy(correct[1])
                legend = []
                
                colors = cm.rainbow(np.linspace(0,1,len(noise[1])+1))
                noise_map = []
                
                for noi in noise[1]:
                    legend.append('%s @ %f'%(determineProperName(noise[0]), noi))
                    mAP = []

                    # grab noise-only value to represent the baseline
                    default_path = os.path.join(result_root, mod, "%s_%0.6f_map.txt"%(noise[0], noi))
                    avg = 0
                    with open(default_path, 'r') as f:
                        contents = list(map(str.strip, f.readlines()))
                        for i, con in enumerate(contents):
                            cat, val = con.split(',')
                            val = float(val)
                            avg += val*val_weights[i]
                    mAP.append(avg)

                    for cor in correct[1]:
                        #print(correct)
                        _path = os.path.join(result_root, mod, "%s_%s_%0.6f_%0.6f_map.txt"%(noise[0], correct[0], noi, cor) )
                        print(_path)
                        with open(_path, 'r') as f:
                            contents = list(map(str.strip, f.readlines()))
                            avg = 0
                            for i, con in enumerate(contents):
                                cat, val = con.split(',')
                                val = float(val)
                                avg += val*val_weights[i]
                            mAP.append(avg)
                    noise_map.append(mAP)

                # add the default on the y:
                x.insert(0, 1)

                # plot out:
                for c, _noise in enumerate(noise_map):
                    print(x, _noise)
                    plt.plot(x, _noise, color=colors[c])
                
                if correct[0] == 'median':
                    plt.xlabel('Kernel Size')
                elif correct[0] == 'nplf':
                    plt.xlabel('Non-linear Mean Filter')

                plt.ylabel('mAP (%)')
                plt.title('Correcting %s with %s on %s'%(determineProperName(noise[0]), determineProperName(correct[0]), determineProperName(mod)))
                plt.xticks(x)
                plt.ylim([0,1])
                plt.legend(legend, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()

                plt.savefig('./results/graphs/overview_%s_%s_%s.png'%(correct[0], noise[0], mod))

# for histogram equiliazation only:
def graph_4(noise_dict, result_root, models):
    val_weights = [0.33,0.33,0.33]
    colors = ['red', 'orange']
    legend = ['Baseline', 'Histogram Equalization']

    for i, noise in enumerate(noise_dict.items()):
        for mod in models:
            mAP = []
            fig = plt.figure(figsize=(10,5.5))
            x = noise[1]
            baseline = {'easy_val':[], 'medium_val':[], 'hard_val':[], 'avg':[]}
            
            for noi in noise[1]:
                # grab noise-only value to represent the baseline
                default_path = os.path.join(result_root, mod, "%s_%0.6f_map.txt"%(noise[0], noi))
                print(default_path)
                avg = 0
                with open(default_path, 'r') as f:
                    contents = list(map(str.strip, f.readlines()))
                    for i, con in enumerate(contents):
                        cat, val = con.split(',')
                        val = float(val)
                        avg += val*val_weights[i]
                        #baseline[cat].append(val)
                    baseline['avg'].append(avg)
            mAP.append(baseline)
            
            line = {'easy_val':[], 'medium_val':[], 'hard_val':[], 'avg':[]}
            for noi in noise[1]:
                avg = 0
                _path = os.path.join(result_root, mod, "%s_%s_%0.6f_%0.6f_map.txt"%(noise[0], 'he', noi, 0) )
                with open(_path, 'r') as f:
                    contents = list(map(str.strip,f.readlines()))
                    for i, con in enumerate(contents):
                        cat, val = con.split(',')
                        val = float(val)
                        avg += val*val_weights[i]
                        line[cat].append(val)
                    line['avg'].append(avg)
            mAP.append(line)

            for j, m in enumerate(mAP):
                for val in m.items():
                    if val[1] != 'avg':
                        pass
                        #plt.plot(val[1], color='cornflowerblue')
                plt.plot(x, m['avg'], color=colors[j])
            plt.legend(legend, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.ylabel('mAP (%)')
            plt.title('Correcting %s with %s on %s'%(determineProperName(noise[0]), 'Histogram Equalization', determineProperName(mod)))
            plt.xticks(x)
            plt.ylim([0,1])
            plt.tight_layout()
            #plt.show()
            plt.savefig('./results/graphs/overview_%s_%s_%s.png'%('he', noise[0], mod))

def determineXLabel(input_str):
    if input_str == 'gaussian_noise':
        return 'Standard Deviation'
    elif input_str == 'salt_pepper':
        return 'Percentage of Image'
    elif input_str == 'poisson':
        return 'Lambda'
    elif input_str == 'speckle':
        return 'Uniform Low Value'
    elif input_str == 'gamma':
        return 'Gamma Shape'


def determineProperName(input_str):
    if input_str == 'gaussian_noise':
        return 'Gaussian Noise'
    elif input_str == 'salt_pepper':
        return 'Salt and Pepper'
    elif input_str == 'poisson':
        return 'Poisson'
    elif input_str == 'speckle':
        return 'Speckle'
    elif input_str == 'median':
        return 'Median'
    elif input_str == 'gamma':
        return 'Gamma'
    elif input_str == 'he':
        return 'Histogram Equalization'
    elif input_str == 'dsfd':
        return 'DSFD'
    elif input_str == 'retinaface':
        return 'RetinaFace'
    else:
        raise Exception('Invalid input_str specified!')


if __name__ == '__main__':
    args = parseArgs()

    with open(args.noise_file, 'r') as f:
        contents = list(map(str.strip, f.readlines()))
        noise_dict = readNoiseText(contents)

    with open(args.correction_file, 'r') as f:
        contents = list(map(str.strip, f.readlines()))
        correction_list = readNoiseText(contents)
    
    graph_2(noise_dict, args.result_dir)
    graph_1(noise_dict, args.result_dir, ['retinaface', 'tinaface', 'dsfd'])
    graph_3(noise_dict, correction_list, args.result_dir, ['retinaface', 'dsfd'])
    graph_4(noise_dict, args.result_dir, ['retinaface', 'dsfd'])