import cv2
import sys
import numpy as np
import datetime
import os
import glob

from retinaface import RetinaFace
import argparse
from tqdm import tqdm

# import imageprocessing for corrections:
sys.path.append('../')
import imageprocessing as ip

# Purpose: This script will only output detections for the matlab eval to compute the mAP.

# parse args:
parser = argparse.ArgumentParser()
parser.add_argument('--val_file', type=str)
parser.add_argument('--noise', type=str, default='', required=False)
parser.add_argument('--noise_param', type=str, default='', required=False)
parser.add_argument('--correction', type=str, default="", required=False)
parser.add_argument('--correction_param', type=str, required=False)
args = parser.parse_args()

with open(args.val_file, 'r') as f:
    content = list(map(str.strip, f.readlines()))

thresh = 0.02 # set as same as test_widerface
gpuid = 0
detector = RetinaFace('./weights/R50', 0, gpuid, 'net3')

# read in val.txt:
for img_path in tqdm(content):
    img = cv2.imread(img_path)
    
    if args.correction != '':
        # range is done one-at-a-time, looped in the bash script
        crange = float(args.correction_param)
        if args.correction == 'median':
            img = ip.median(img, int(crange))
        elif args.correction == 'nlm':
            img = ip.nonlocal_means(img, crange)

    #target_size = 1200
    #max_size = 1600
    #target_size = 1504
    #max_size = 2000
    target_size = 1600
    max_size = 2150
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    scales = [im_scale]
    flip = False

    faces, landmarks = detector.detect(img,
                                    thresh,
                                    scales=scales,
                                    do_flip=flip)

    # make new path for detection text:
    if args.correction != '':
        # special path for corrections:
        image_id = img_path.replace('jpg', 'txt').split('/')[-2:]
        det_path = os.path.join('../CORRECTIONS', '%s_%s'%(args.noise, args.correction), "%s_%s"%(args.noise_param, args.correction_param), 'retinaface', '/'.join(image_id))
    else:
        det_path = img_path.replace('images', 'detections/retinaface').replace('jpg', 'txt')
        last_slash = det_path.rfind('/')
        det_folder_path = det_path[:last_slash]
        if not os.path.exists(det_folder_path):
            os.mkdir(det_folder_path)

    with open(det_path, 'w') as det:
        det.write('%s\n'%img_path.split('/')[-1].replace('.jpg',''))
        if faces is not None:
            det.write('%i\n'%(faces.shape[0]))
            for i in range(faces.shape[0]):
                score = faces[i][4]
                box = faces[i].astype(np.int)
                x1,y1,x2,y2= box[0], box[1], box[2], box[3]
                w,h = x2-x1, y2-y1
                det.write("%d %d %d %d %f\n"%(x1,y1,w,h,score))
        else:
            det.write('0\n')       