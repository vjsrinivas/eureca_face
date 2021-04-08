from ast import parse
import glob
import os
import cv2
import time
import face_detection
import argparse
from tqdm import tqdm

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_file', type=str)
    return parser.parse_args()

def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)

def write_txt(bboxes, output_path, image_name):
    with open(output_path, 'w') as f:
        f.write('%s\n'%image_name)
        f.write("%d\n"%len(bboxes))
        for bbox in bboxes:
            x0, y0, x1, y1, score = [float(_) for _ in bbox]
            w,h = x1-x0, y1-y0
            f.write('%d %d %d %d %f\n'%(x0,y0,w,h,score))
        
        #cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)

if __name__ == "__main__":
    args = parseArgs()
    impaths = "images"
    impaths = glob.glob(os.path.join(impaths, "*.jpg"))
    detector = face_detection.build_detector(
        "DSFDDetector",
        max_resolution=2160,
        confidence_threshold=0.01,
        nms_iou_threshold=0.3
    )

    with open(args.val_file) as f:
        val_files = list(map(str.strip, f.readlines()))
        for val in tqdm(val_files):
            output_file = val.replace('images', 'detections')
            output_arr = output_file.split('/')
            output_arr.insert(-2, 'dsfd')
            output_arr[-1] = output_arr[-1].replace('jpg', 'txt')
            output_file = '/'.join(output_arr)
            image_name = val.split('/')[-1].replace('.jpg', '')
            im = cv2.imread(val)
            dets = detector.detect(
                im[:, :, ::-1]
            )[:,:]
            write_txt(dets, output_file, image_name)