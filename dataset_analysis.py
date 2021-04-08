import imageprocessing as ip
import cv2
import os
import sys
import numpy as np
from preprocessing import readNoiseText
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import random
from scipy.io import loadmat
from tqdm import tqdm

ROOT='./WIDERFACE/WIDER_val/'
NOISE_TEXT='noise.txt'
WIDER_test = os.path.join(ROOT, 'wider_face_split')
WIDER_val = os.path.join(ROOT, 'WIDER_val')
WIDER_train = os.path.join(ROOT, 'WIDER_train')
WIDER_devkit = os.path.join(ROOT, 'wider_face_split')

def batchImages(noise, levels:list):
    return 0

def vizDetBatchImages(noises, levels, image_path, random_img=False):
    if not random_img:
        if image_path == '':
            raise Exception('Image path cannot be empty!')
        else:
            img = cv2.imread(image_path)
    else:
        # look in WIDERFACE Val:
        sample_folder = os.path.join(ROOT, 'images/0--Parade')
        for _,_,sample_img in os.walk(sample_folder):
            pass
        print(sample_img)
        image_path = os.path.join(ROOT, 'images/0--Parade', random.choice(sample_img))
        img = cv2.imread(image_path)
    
    splits = image_path.split('/')
    class_path = splits[-2]
    image_id = splits[-1].replace('.jpg', '')
    print(image_id)
    
    # get detection paths:
    for level in levels:
        with open(os.path.join('NOISES', noises, '%0.6f'%level, 'detections', 'retinaface', class_path, image_id+'.txt'), 'r') as f:
            det = list(map(str.strip, f.readlines()))
            print(det)

def displayNoiseMatrix(img, noise_list, range):
    fig, ax = plt.subplots(1,5)
    ax[0].imshow(img[130:250,130:250,:])
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    for i,noise in enumerate(noise_list):
        print(noise)
        r_img = np.copy(img)
        if noise == 'gaussian_noise':
            o_img = ip.gaussian(r_img, range[i])
            text = 'Gaussian Noise'
        elif noise == 'salt_pepper':
            o_img = ip.salt_pepper(r_img, range[i])
            text = 'Salt & Pepper'
        elif noise == 'poisson':
            o_img = ip.poisson(r_img, range[i])
            text = 'Poisson'
        else:
            o_img = ip.speckle(r_img, 0.3, 1.5)
            text = 'Speckle'
        ax[i+1].imshow(o_img[130:250,130:250,:])
        ax[i+1].set_title(text)
        ax[i+1].axis('off')
    plt.show()

def displaySSIMDemo(img, noise, range_list):
    r_img = np.copy(img)
    fig, ax = plt.subplots(1,len(range_list), figsize=(10,3))
    if noise == 'gaussian_noise':
        fig.suptitle('SSIM on Range of Gaussian Noises', fontweight="bold")
    for i, r in enumerate(range_list):
        if noise == 'gaussian_noise':
            # run gaussian_noise
            out_img = ip.gaussian(r_img,r)
            ssim_value = ssim(img, out_img, multichannel=True)
            ax[i].imshow(out_img[130:250,130:250,:])
            ax[i].axis('off')
            ax[i].set_title('SSIM: %0.6f'%(ssim_value))
    plt.savefig('ssim_demo.png')

def displayMedianDemo(img, range_list):
    # destroyed image: img
    r_img = np.copy(img)
    fig, ax = plt.subplots(1,len(range_list), figsize=(10,2))
    for i, r in enumerate(range_list):
        _img = ip.median(r_img, r)
        ax[i].imshow(_img)
        ax[i].axis('off')
        ax[i].set_title("Filter Size: %d"%r)

    plt.savefig('median_demo.png')

def displayNLMF(img, range_list):
    # destroyed image: img
    r_img = np.copy(img)
    fig, ax = plt.subplots(1,len(range_list), figsize=(10,2))
    for i, r in enumerate(range_list):
        _img = ip.nonlocal_means(r_img, r)
        ax[i].imshow(_img)
        ax[i].axis('off')
        ax[i].set_title("Filter Size: %d"%r)
    plt.savefig('./results/graphs/nlmf_demo.png')

def displaySSIM(img, noises, range_list):
    fig, ax = plt.subplots(2,2, figsize=(14,10))
    for i, noise in enumerate(noises):
        img_list = []
        x_label, y_label = '', 'SSIM Score'
        for r in range_list[i]:
            r_img = np.copy(img)
            if noise == 'gaussian_noise':
                out_img = ip.gaussian(r_img, r)
                x_label = 'Standard Deviation'

            elif noise == 'salt_pepper':
                out_img= ip.salt_pepper(r_img, r)
                x_label = 'Percentage of Noise to Image'

            elif noise == 'poisson':
                out_img = ip.poisson(r_img, r)
                x_label = 'Intensity'

            elif noise == 'speckle':
                out_img = ip.speckle(r_img, 1, r)
                x_label = 'Uniform High' 

            img_list.append(out_img)
        ssim_list = ip.ssim_list(img_list, img)
        
        ax[i%2][i//2].plot(range_list[i], ssim_list)
        ax[i%2][i//2].set_title("SSIM of %s"%noise)
        ax[i%2][i//2].set_xlabel(x_label)
        ax[i%2][i//2].set_ylabel(y_label)
        
        if noise == 'speckle':
            ax[i%2][i//2].invert_xaxis()
    plt.savefig('./results/graphs/ssim_comparison.png')


def displayAveragedSSIM(noises, range_list):
    fig, ax = plt.subplots(2,2, figsize=(14,10))
    avg_list = {key:[] for key in noises}
    
    with open('./NOISES/gaussian_noise/0.000000/val.txt', 'r') as f:
        template = list(map(str.strip, f.readlines()))

    template = template[:20]
    invert = True
    
    for k, item in enumerate(tqdm(template)):
        orig_img = cv2.imread(item)
        broken = item.split('/')
        # -4 -> level
        # -5 -> noise_type

        for i, noise in enumerate(noises):
            broken[-5] = noise

            img_list = []
            x_label, y_label, noise_title = '', 'SSIM Score', ''
            for j, r in enumerate(range_list[i]):
                broken[-4] = "%0.6f"%r
                img_path = '/'.join(broken)
                out_img = cv2.imread(img_path) 

                if noise == 'gaussian_noise':
                    x_label = 'Standard Deviation'
                    noise_title = 'Gaussian Noise'
                elif noise == 'salt_pepper':
                    x_label = 'Percentage of Noise to Image'
                    noise_title = 'Salt & Pepper'
                elif noise == 'poisson':
                    x_label = 'Intensity'
                    noise_title = 'Poisson Noise'
                elif noise == 'speckle':
                    x_label = 'Uniform High' 
                    noise_title = 'Speckle Noise'

                img_list.append(out_img)

            ssim_list = ip.ssim_list(img_list, orig_img)
            if len(avg_list[noise]) == 0:
                avg_list[noise] = ssim_list
            else:
                for x in range(len(avg_list[noise])):
                    avg_list[noise][x] = avg_list[noise][x] + ssim_list[x]
            
            ax[i%2][i//2].plot(range_list[i], ssim_list, color='cornflowerblue')
            ax[i%2][i//2].set_title("SSIM of %s"%noise_title)
            ax[i%2][i//2].set_xlabel(x_label)
            ax[i%2][i//2].set_ylabel(y_label)
            
            if noise == 'speckle' and invert:
                ax[i%2][i//2].invert_xaxis()
                invert = False
    
    for i, item in enumerate(avg_list.items()):
        ax[i%2][i//2].plot(range_list[i], [x/(k+1) for x in item[1]], linewidth=3, color='red')

    #plt.savefig('./results/graphs/ssim_comparison.png')
    plt.savefig('./results/graphs/ssim_avg_comparison.png')

if __name__ == '__main__':
    with open(NOISE_TEXT, 'r') as f:
        contents = f.readlines()
        noise_contents = list(map(str.strip, contents))
        noise_dict = readNoiseText(noise_contents)
    
    randomImage = 'dog.png'
    img = cv2.imread(randomImage)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # For SSIM Demo Image:
    #noise_ranges = [noise for i, noise in enumerate(noise_dict['gaussian_noise']) if i % 3 == 0 ]
    #displaySSIMDemo(img, "gaussian_noise", noise_ranges)
    
    # For showing Median Image Demo:
    #displayMedianDemo(ip.gaussian(img, 50), [2,3,4,5])
    
    # For showing nonlocal-means filter:
    #displayNLMF(ip.gaussian(img, 100), [10,30,50,70])

    # For showing Noises Demo:
    #displayNoiseMatrix(img, list(noise_dict.keys()), [item[1][5] for item in noise_dict.items()])

    # For showing SSIM Matrix:
    displayAveragedSSIM(noise_dict.keys(), [item[1] for item in noise_dict.items()])

    #vizDetBatchImages('gaussian_noise', noise_dict['gaussian_noise'], './WIDERFACE/WIDER_val/images/0--Parade/0_Parade_marchingband_1_20.jpg')