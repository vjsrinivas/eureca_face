import argparse

import cv2
import numpy as np
from numpy.core.arrayprint import _void_scalar_repr
import torch

import os
from tqdm import tqdm

from vedacore.image import imread, imwrite
from vedacore.misc import Config, color_val, load_weights
from vedacore.parallel import collate, scatter
from vedadet.datasets.pipelines import Compose
from vedadet.engines import build_engine


def parse_args():
    parser = argparse.ArgumentParser(description='Infer a detector')
    parser.add_argument('config', type=str)
    parser.add_argument('--val_file', type=str)

    args = parser.parse_args()
    return args


def prepare(cfg):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'
    engine = build_engine(cfg.infer_engine)

    engine.model.to(device)
    load_weights(engine.model, cfg.weights.filepath)

    data_pipeline = Compose(cfg.data_pipeline)
    return engine, data_pipeline, device


def plot_result(result, imgfp, class_names, outfp='out.jpg'):
    font_scale = 0.5
    bbox_color = 'green'
    text_color = 'green'
    thickness = 1

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    img = imread(imgfp)

    bboxes = np.vstack(result)
    labels = [
        np.full(bbox.shape[0], idx, dtype=np.int32)
        for idx, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox[:4].astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness)
        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    imwrite(img, outfp)

def writeDetectionFile(results, output_file, image_file):
    results = results[0]
    with open(output_file, 'w') as f:
        f.write("%s\n"%image_file)
        f.write("%i\n"%len(results))
        for res in results:
            x1,y1,x2,y2,conf = res
            w,h = x2-x1, y2-y1
            f.write('%d %d %d %d %f\n'%(x1,y1,w,h,conf))

def main():

    args = parse_args()
    cfg = Config.fromfile(args.config)
    class_names = cfg.class_names

    engine, data_pipeline, device = prepare(cfg)

    with open(args.val_file, 'r') as f:
        contents = list(map(str.strip, f.readlines()))
        for val_file in tqdm(contents): 
            #print(val_file)
            data = dict(img_info=dict(filename=val_file), img_prefix=None)
            data = data_pipeline(data)
            data = collate([data], samples_per_gpu=1)
            
            if device != 'cpu':
                # scatter to specified GPU
                data = scatter(data, [device])[0]
            else:
                # just get the actual data from DataContainer
                data['img_metas'] = data['img_metas'][0].data
                data['img'] = data['img'][0].data

            result = engine.infer(data['img'], data['img_metas'])[0]
            output_file = val_file.replace('images', 'detections')
            output_arr = output_file.split('/')
            output_arr.insert(-2, 'tinaface')
            output_arr[-1] = output_arr[-1].replace('jpg', 'txt')
            output_file = '/'.join(output_arr)

            image_name = val_file.split('/')[-1].replace('.jpg', '')

            writeDetectionFile(result, output_file, image_name)
    
            #plot_result(result, '../t2.jpg', class_names)


if __name__ == '__main__':
    main()
