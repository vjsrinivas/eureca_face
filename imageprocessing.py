from unicodedata import combining
import cv2
import os
from matplotlib.pyplot import hsv
import matplotlib.pyplot as plt
import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def poisson(image, time):
    # create possion noise:
    # code from https://stackoverflow.com/a/36331042
    possion_matrix = np.random.poisson(image/255.0*time)/time*255
    combined = image+np.random.poisson(possion_matrix)
    np.clip(combined, 0, 255, out=combined)
    return combined.astype('uint8')

def gaussian(image, std):
    # only control standard dev:
    normal_matrix = np.random.normal(2, std, size=image.shape)
    
    # print distribution
    #plt.hist(normal_matrix.ravel(), bins='auto')
    #plt.show()

    combined = image+normal_matrix
    np.clip(combined, 0, 255, out=combined)
    return combined.astype('uint8')

def light_intensity(image, intensity):
    assert intensity <= 1 and intensity >= 0
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # adjust value channel:
    value = hsv_img[:,:,2].astype('float64')*intensity
    hsv_img[:,:,2] = value.astype('uint8')
    back_to = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    return back_to

def salt_pepper(image, amount):
    # keep pepper constant
    # code refractored from: https://gist.github.com/Prasad9/077050d9a63df17cb1eaf33df4158b19 
    assert len(image.shape) == 3
    img = np.copy(image)
    ratio = 0.2
    h,w,c = img.shape
    pixel_num = h*w
    salt = np.ceil(amount * pixel_num * ratio)
    pepper = np.ceil(amount * pixel_num * (1-ratio))

    coords = [np.random.randint(0,i-1,int(salt)) for i in img.shape]
    # apply salt
    img[coords[0], coords[1], :] = 255
    
    # apply pepper
    coords = [np.random.randint(0,i-1, int(pepper)) for i in img.shape]
    img[coords[0], coords[1], :] = 0
    
    return img

def speckle(image, low, high):
    # create a noise matrix that was not gaussian-type:
    noise_matrix = np.random.uniform(low=low, high=high, size=image.shape)
    combined = noise_matrix*image
    np.clip(combined, 0, 255, out=combined)
    return combined.astype('uint8')

def ssim_graph(list_of_imgs, orig_img, x_axis):
    ssim_list = []
    for img in list_of_imgs:
        metric_ssim = ssim(img, orig_img, multichannel=True)
        print("SSIM:", metric_ssim)
        ssim_list.append(metric_ssim)
    plt.figure()
    plt.plot(x_axis, ssim_list)
    plt.show()

def ssim_list(list_of_imgs, orig_img):
    ssim_list = []
    for img in list_of_imgs:
        metric_ssim = ssim(img, orig_img, multichannel=True)
        #print("SSIM:", metric_ssim)
        ssim_list.append(metric_ssim)
    return ssim_list

# corrections:
def median(image, median_size):
    '''
    r,g,b = image[:,:,0], image[:,:,1], image[:,:,2]
    _r = scipy.ndimage.median_filter(r,median_size)
    _g = scipy.ndimage.median_filter(g,median_size)
    _b = scipy.ndimage.median_filter(b,median_size) 
    image[:,:,0], image[:,:,1], image[:,:,2] = _r, _g, _b
    '''
    image = cv2.medianBlur(image,median_size)
    return image

def median2(image, median_size):
    r,g,b = image[:,:,0], image[:,:,1], image[:,:,2]
    _r = scipy.ndimage.median_filter(r,median_size)
    _g = scipy.ndimage.median_filter(g,median_size)
    _b = scipy.ndimage.median_filter(b,median_size) 
    image[:,:,0], image[:,:,1], image[:,:,2] = _r, _g, _b
    return image

def nonlocal_means(image, h_lum):
    out_image = cv2.fastNlMeansDenoisingColored(image, None, h_lum, 10, 21, 7)
    return out_image

if __name__ == '__main__':
    img = cv2.imread('/home/vijay/Documents/devmk4/eureca/eureca_face/WIDERFACE/WIDER_val/images/0--Parade/0_Parade_marchingband_1_20.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    params = [i/8 for i in range(-8,9,1)]
    print(params)
    print(params)
    imgs = []
    for p in params:
        img2 = speckle(img, p, 1)
        cv2.imshow('test', img2)
        cv2.waitKey(-1)
        imgs.append(img2)

    ssim_graph(imgs, img, params)