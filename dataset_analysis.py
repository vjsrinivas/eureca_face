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

from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import cm
from matplotlib.patches import Rectangle
import imageio

ROOT='./WIDERFACE/WIDER_val/'
NOISE_TEXT='noise.txt'
CORRECT_TEXT='corrections.txt'
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
    fig, ax = plt.subplots(1,5, figsize=(10,5))
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
        elif noise == 'speckle':
            o_img = ip.speckle(r_img, 0.3, 1.5)
            text = 'Speckle'
        elif noise == 'gamma':
            o_img = ip.gamma(r_img, 100)
            text = 'Gamma'
        ax[i+1].imshow(o_img[130:250,130:250,:])
        ax[i+1].set_title(text)
        ax[i+1].axis('off')
    plt.tight_layout()
    plt.savefig('./results/graphs/noise_matrix.png')

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
    plt.savefig('./results/graphs/ssim_demo.png')

def displayMedianDemo(img, range_list):
    # destroyed image: img
    r_img = np.copy(img)
    fig, ax = plt.subplots(1,len(range_list), figsize=(10,2))
    for i, r in enumerate(range_list):
        _img = ip.median(r_img, r)
        ax[i].imshow(_img)
        ax[i].axis('off')
        ax[i].set_title("Filter Size: %d"%r)

    plt.savefig('./results/graphs/median_demo.png')

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
                elif noise == 'gamma':
                    x_label = 'Shape' 
                    noise_title = 'Gamma Noise'

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


def exampleDegradeRecover(image_path, noise, correction_dict):
    CORRECT_DIR = './CORRECTIONS/salt_pepper_median/'
    img = cv2.imread(image_path)
    image_id = image_path.split('/')[-1].replace('.jpg', '.txt')
    folder_name = image_path.split('/')[-2]
    
    median_list = correction_dict['median']
    fig, ax = plt.subplots(1,len(median_list)+1, figsize=(20,10))

    # default image:
    with open('./NOISES/salt_pepper/%0.6f/detections/retinaface/%s/%s'%(noise,folder_name,image_id), 'r') as f:
            content = list(map(str.strip,f.readlines()))
            bbox = []
            for con in content[2:]:
                x,y,w,h,conf = con.split(' ')
                x,y,w,h,conf = int(x), int(y), int(w), int(h), float(conf)
                bbox.append((x,y,w,h,conf))

    d_img = np.copy(img)
    bb_len = 0
    for bb in bbox:
        if bb[4] > 0.20:
            cv2.rectangle(d_img, (bb[0], bb[1]), (bb[0]+bb[2], bb[1]+bb[3]), (0,0,255), 3)
            bb_len += 1
    d_img = cv2.cvtColor(d_img, cv2.COLOR_RGB2BGR)
    ax[0].imshow(d_img)
    ax[0].set_xlabel('Faces Detected: %i'%(bb_len))
    ax[0].set_title('Original Image')
    ax[0].set_xticks([])
    ax[0].set_xticks([], minor=True)
    ax[0].set_yticks([])
    ax[0].set_yticks([], minor=True)

    i = 1
    for med in median_list:
        _ax = ax[i]
        noise_det_path = os.path.join(CORRECT_DIR, "%0.6f_%0.6f"%(noise, med), "retinaface", folder_name, image_id)
        with open(noise_det_path, 'r') as f:
            content = list(map(str.strip,f.readlines()))
            bbox = []
            for con in content[2:]:
                x,y,w,h,conf = con.split(' ')
                x,y,w,h,conf = int(x), int(y), int(w), int(h), float(conf)
                bbox.append((x,y,w,h,conf))

        r_img = np.copy(img)
        d_img = ip.median(r_img, int(med))
        bb_len = 0
        for bb in bbox:
            if bb[4] > 0.20:
                cv2.rectangle(d_img, (bb[0], bb[1]), (bb[0]+bb[2], bb[1]+bb[3]), (0,0,255), 3)
                bb_len += 1

        d_img = cv2.cvtColor(d_img, cv2.COLOR_RGB2BGR)
        _ax.imshow(d_img)
        _ax.set_xlabel('Faces Detected: %i'%(bb_len))
        _ax.set_title('Median Filter @ %0.6f'%(med))
        _ax.set_xticks([])
        _ax.set_xticks([], minor=True)
        _ax.set_yticks([])
        _ax.set_yticks([], minor=True)
        i += 1
    plt.tight_layout()
    plt.savefig('./results/graphs/median_fix_demo.png')

def exampleDegradeHERecover(image_path):
    WF_class = '1--Handshaking'
    noise = 200
    txt_name = image_path.replace('jpg', 'txt')
    orig_txt_path = os.path.join('NOISES', 'gamma', '%0.6f'%(noise), 'detections', 'dsfd', WF_class, txt_name)
    he_txt_path = os.path.join('CORRECTIONS', 'gamma_he', '%0.6f_%0.6f'%(noise, 0), 'retinaface', WF_class, txt_name)

    he_path = os.path.join('NOISES', 'gamma', '%0.6f'%(noise), 'images', WF_class, image_path)
    org_img = cv2.imread(he_path)
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    he_img = np.copy(org_img)
    he_img = ip.hist_equalization(he_img)

    with open(orig_txt_path, 'r') as f:
        bb_org = 0
        content = list(map(str.strip, f.readlines()))
        for bbox in content[2:]:
            x,y,w,h,c = bbox.split()
            x,y,w,h,c = int(x), int(y), int(w), int(h), float(c)
            if c > 0.2:
                org_img = cv2.rectangle(org_img, (x,y), (x+w,y+h), (255,0,0), 2)
                bb_org += 1

    with open(he_txt_path, 'r') as f:
        bb_he = 0
        content = list(map(str.strip, f.readlines()))
        for bbox in content[2:]:
            x,y,w,h,c = bbox.split()
            x,y,w,h,c = int(x), int(y), int(w), int(h), float(c)
            if c > 0.2:
                he_img = cv2.rectangle(he_img, (x,y), (x+w,y+h), (255,0,0), 2)
                bb_he += 1

    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(org_img)
    ax[0].set_title('Original Image')
    ax[0].set_xlabel('Faces Detected: %i'%(bb_org))
    ax[1].imshow(he_img)
    ax[1].set_title('Histogram Equalization')
    ax[1].set_xlabel('Faces Detected: %i'%(bb_he))
    
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_yticks([])
    #plt.show()
    plt.tight_layout()
    plt.savefig('./results/graphs/he_fix_demo.png')

def animatedBanner(image_base, noise, noise_lvl_list, correction, correct_lvl_list, model='retinaface', MIN_CONF = 0.001):
    org_img = cv2.imread(os.path.join('WIDERFACE/WIDER_val/images',image_base)) 
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    with open(os.path.join('NOISES/gaussian_noise/%0.6f/detections/%s'%(0, model),image_base.replace('.jpg', '.txt'))) as f:
        content = list(map(str.strip,f.readlines()))
        dets = []
        for con in content[2:]:
            x,y,w,h,conf = con.split(' ')
            x,y,w,h,conf = int(x), int(y), int(w), int(h), float(conf)
            dets.append((x,y,w,h,conf))

    for j, n_lvl in enumerate(noise_lvl_list):

        # grab noise image:
        print(os.path.join('NOISES', noise, '%0.6f'%(n_lvl), 'images', image_base))
        noise_img = cv2.imread(os.path.join('NOISES', noise, '%0.6f'%(n_lvl), 'images', image_base))
        noise_img = cv2.cvtColor(noise_img, cv2.COLOR_BGR2RGB)

        # grab dets from noise:
        with open(os.path.join('NOISES', noise, '%0.6f'%(n_lvl), 'detections', model, image_base.replace('.jpg', '.txt'))) as f:
            content = list(map(str.strip,f.readlines()))
            noise_dets = []
            for con in content[2:]:
                x,y,w,h,conf = con.split(' ')
                x,y,w,h,conf = int(x), int(y), int(w), int(h), float(conf)
                noise_dets.append((x,y,w,h,conf))

        for i,c_lvl in enumerate(correct_lvl_list):
            
            # correction image:
            c_lvl = int(c_lvl)

            if correction == 'median':
                correct_img = ip.median(np.copy(noise_img), c_lvl)
            elif correction == 'he':
                correct_img = ip.hist_equalization(np.copy(noise_img))

            # grab dets from corrected:
            with open(os.path.join('CORRECTIONS', "%s_%s"%(noise, correction), '%0.6f_%0.6f'%(n_lvl, c_lvl), model, image_base.replace('.jpg', '.txt'))) as f:
                content = list(map(str.strip,f.readlines()))
                corr_dets = []
                for con in content[2:]:
                    x,y,w,h,conf = con.split(' ')
                    x,y,w,h,conf = int(x), int(y), int(w), int(h), float(conf)
                    corr_dets.append((x,y,w,h,conf))

            #black_cover = np.ndarray(img.shape)
            #black_cover.fill(0)
            black_cm = cm.get_cmap('summer', len(noise_dets))

            fig = plt.figure(figsize=(12,2))
            axs = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(1, 4),  # creates 2x2 grid of axes
                        axes_pad=0.1,  # pad between axes in inch.
                        cbar_location="right",
                        cbar_mode='single',
                        cbar_size='5%',
                        cbar_pad=0.05
                        )
            
            axs[0].imshow(org_img)
            axs[1].imshow(org_img)
            axs[2].imshow(noise_img)
            axs[3].imshow(correct_img)

            axs[0].set_title('Original Image', fontsize=10)
            axs[1].set_title('Image Detections', fontsize=10)
            axs[2].set_title('Salt & Pepper Noise @ %f'%n_lvl, fontsize=10)
            axs[3].set_title('Corrected with Median @ %d'%c_lvl, fontsize=10)

            for ax in axs:
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)

            # draw noise bboxes:
            for bb in dets:
                x1, y1, w, h, c = bb
                rect = plt.Rectangle((x1,y1), w, h, fc='none', ec=black_cm(c))
                axs[1].add_patch(rect)

            # draw noise bboxes:
            for bb in noise_dets:
                x1, y1, w, h, c = bb
                rect = plt.Rectangle((x1,y1), w, h, fc='none', ec=black_cm(c))
                axs[2].add_patch(rect)

            # draw correction bboxes:
            for bb in corr_dets:
                x1, y1, w, h, c = bb
                rect = plt.Rectangle((x1,y1), w, h, fc='none', ec=black_cm(c))
                axs[3].add_patch(rect)

            sm = plt.cm.ScalarMappable(cmap='summer', norm=plt.Normalize(vmin=MIN_CONF, vmax=1))
            fig.colorbar(sm, cax=axs.cbar_axes[0], label="Confidence")
            fig.tight_layout()
            plt.savefig('./tmp/%d_%d_frame.png'%(j,i))
            plt.close()

    # stitch for animation:
    with imageio.get_writer('anim_header_1.gif', mode='I') as writer:
        duration = 10
        for i,_ in enumerate(noise_lvl_list):
            for j,_ in enumerate(correct_lvl_list):
                name = '%d_%d_frame.png'%(i,j)
                for _ in range(duration):
                    image = imageio.imread(os.path.join('tmp',name))
                    writer.append_data(image)

if __name__ == '__main__':
    with open(NOISE_TEXT, 'r') as f:
        contents = f.readlines()
        noise_contents = list(map(str.strip, contents))
        noise_dict = readNoiseText(noise_contents)
    
    with open(CORRECT_TEXT, 'r') as f:
        contents = f.readlines()
        correct_content = list(map(str.strip, contents))
        correct_dict = readNoiseText(correct_content)
    
    randomImage = 'dog.png'
    img = cv2.imread(randomImage)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # For SSIM Demo Image:
    noise_ranges = [noise for i, noise in enumerate(noise_dict['gaussian_noise']) if i % 3 == 0 ]
    #displaySSIMDemo(img, "gaussian_noise", noise_ranges)
    
    # For showing Median Image Demo:
    #displayMedianDemo(ip.gaussian(img, 50), [3,5,7,9])
    
    # For showing nonlocal-means filter:
    #displayNLMF(ip.gaussian(img, 100), [10,30,50,70])

    # For showing Noises Demo:
    #displayNoiseMatrix(img, list(noise_dict.keys()), [item[1][4] for item in noise_dict.items()])

    # For showing SSIM Matrix:
    #displayAveragedSSIM(noise_dict.keys(), [item[1] for item in noise_dict.items()])

    # Viz of image degradation and recover with bbox:
    #noise = 0.50
    #exampleDegradeRecover('./NOISES/salt_pepper/%0.6f/images/61--Street_Battle/61_Street_Battle_streetfight_61_432.jpg'%noise, noise, correct_dict)

    # For showing HE Example:
    #exampleDegradeHERecover('1_Handshaking_Handshaking_1_602.jpg')

    # For creating animation example of degradation and enhancement:
    animatedBanner('61--Street_Battle/61_Street_Battle_streetfight_61_432.jpg', 'salt_pepper', [noise for i, noise in enumerate(noise_dict['salt_pepper'])], 'median', [correct for i, correct in enumerate(correct_dict['median'])])