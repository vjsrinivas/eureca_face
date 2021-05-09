import os
import sys
import numpy as np
import cv2
import imageprocessing as ip
import argparse
from tqdm import tqdm
import scipy.io as sio
import cv2

MODELS = ['retinaface', 'tinaface', 'dsfd']

def generateNoiseImage(input_path, output_path, noise, level):
    img = cv2.imread(input_path)
    assert type(img) != type(None)
    #img_name = input_path.split('/')[-1]

    if noise == 'gaussian_noise':
        _img = ip.gaussian(img, level)
    elif noise == 'salt_pepper':
        _img = ip.salt_pepper(img, level)
    elif noise == 'poisson':
        _img = ip.poisson(img, level)
    elif noise == 'speckle':
        _img = ip.speckle(img, 0.3, level)
    elif noise == 'gamma':
        _img = ip.gamma(img, level)
    else:
        raise Exception('Specified noise invalid!')
    cv2.imwrite(output_path, _img)

def WIDER_procedure(noise, level, output_path, mat_file, original_path):
    EVENTS = mat_file['event_list']
    FILE_LIST = mat_file['file_list']

    for i, event in enumerate(tqdm(EVENTS)):
        _event = event[0][0]
        os.mkdir(
            os.path.join(output_path, 'images', _event)
        )
        for filename_list in FILE_LIST[i]:
            for filename in filename_list:
                _filename = filename[0][0]+'.jpg'
                _in = os.path.join(original_path, 'WIDER_val/images', _event, _filename)
                _out = os.path.join(output_path, 'images', _event, _filename)
                generateNoiseImage(_in, _out, noise, level)

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str)
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--noise_list', type=str)
    parser.add_argument('--correction_list', type=str)
    return parser.parse_args()

def readNoiseText(contents):
    # content is a list of the file contents:
    noise_dict = {}
    for con in contents:
        noise, params = con.split(' ')
        param = params.split(',')
        param = list(map(float, param))
        noise_dict[noise] = param
    return noise_dict

def generateValList(parent_dir, wider_val_mat):
    EVENTS = wider_val_mat['event_list']
    FILE_LIST = wider_val_mat['file_list']
    
    with open(os.path.join(parent_dir, 'val.txt'), 'w') as f:
        for i, event in enumerate(EVENTS):
            for filename_list in FILE_LIST[i]:
                for filename in filename_list:
                    _event = event[0][0]
                    _filename = filename[0][0]+'.jpg'
                    _out = os.path.join(parent_dir, 'images', _event, _filename)
                    f.write('%s\n'%_out)

if __name__ == '__main__':
    args = parseArgs()
    wider_val_mat = sio.loadmat('./WIDERFACE/eval_tools/ground_truth/wider_face_val.mat')
    ROOT = os.getcwd()
    print(ROOT)

    # make sure that a folder in the parent level created:
    ROOT_NOISE = 'NOISES'
    CORRECT_ROOT = 'CORRECTIONS'
    EVENTS = wider_val_mat['event_list']

    if not os.path.exists(ROOT_NOISE):
        os.mkdir(ROOT_NOISE)

    if not os.path.exists(CORRECT_ROOT):
        os.mkdir(CORRECT_ROOT)

    # parse out the noises you need to run:
    with open(args.noise_list, 'r') as f:
        content = f.readlines()
        content = list(map(str.strip, content))
        noises = readNoiseText(content)

    # do the same for corrections:
    with open(args.correction_list,'r') as f:
        content = f.readlines()
        content = list(map(str.strip, content))
        corrections = readNoiseText(content)
    
    print("Now running following noises: ", noises.keys())
    for item in noises.items():
        print("Running: ", item[0])
        NOISE_DIR = os.path.join(ROOT_NOISE, "%s"%(item[0]) )
        if not os.path.exists( NOISE_DIR ):
            os.mkdir(NOISE_DIR)

        for param in tqdm(item[1]):
            SUB_NOISE_DIR = os.path.join(NOISE_DIR, '%0.6f'%(param))
            if not os.path.exists(SUB_NOISE_DIR):
                os.mkdir(SUB_NOISE_DIR)
                os.mkdir(os.path.join(SUB_NOISE_DIR, 'detections'))
                os.mkdir(os.path.join(SUB_NOISE_DIR, 'images'))

            # write out ./SUB_NOISE_DIR/detections/[model] folders:
            for model in MODELS:
                _model_path = os.path.join(SUB_NOISE_DIR, 'detections', model)
                if not os.path.exists(_model_path):
                    os.mkdir(_model_path)

                for event in EVENTS:
                    class_model = os.path.join(_model_path, event[0][0])
                    if not os.path.exists(class_model):
                        os.mkdir(class_model)

            FULL_SUB_PATH = os.path.join(ROOT, SUB_NOISE_DIR)

            # write out all correction folders needed:
            for correct in corrections.items():
                CORRECT_PATH = os.path.join(CORRECT_ROOT, '%s_%s'%(item[0], correct[0]))
                if not os.path.exists(CORRECT_PATH):
                    os.mkdir(CORRECT_PATH)

                for correct_param in correct[1]:
                    CORRECT_PARAM_PATH = os.path.join(CORRECT_PATH, "%0.6f_%0.6f"%(param, correct_param))
                    if not os.path.exists(CORRECT_PARAM_PATH):
                        os.mkdir(CORRECT_PARAM_PATH)

                    # write out ./CORRECTIONS/[noise_correction]/[model] folders:
                    for model in MODELS:
                        _model_path = os.path.join(CORRECT_PARAM_PATH, model)
                        if not os.path.exists(_model_path):
                            os.mkdir(_model_path)

                        for event in EVENTS:
                            class_model = os.path.join(_model_path, event[0][0])
                            if not os.path.exists(class_model):
                                os.mkdir(class_model)


            #create if val file exists:
            if not os.path.exists(
                    os.path.join(FULL_SUB_PATH, 'val.txt')
                ):
                generateValList(FULL_SUB_PATH, wider_val_mat)
            
            WIDER_procedure(item[0], param, SUB_NOISE_DIR, wider_val_mat, args.data_root)
        