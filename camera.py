import cv2
import os
import sys
import numpy as np

def equalization(img):
    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clache = cv2.createCLAHE(clipLimit=12.0, tileGridSize=(8,8))
    hist_img = clache.apply(img)
    hist_color_img = cv2.cvtColor(hist_img, cv2.COLOR_GRAY2BGR)

    return hist_color_img

def auto_brightness(img, value):
    diff = int(value - img.mean())
    if diff > 0:
        M = np.ones(img.shape, dtype = 'uint8') * diff
        return cv2.add(img, M)
    else:
        M = np.ones(img.shape, dtype = 'uint8') * (diff*-1)
        return cv2.subtract(img, M)
    
def threshold(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thr1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    return thr1

def erode(img):
    kernel = np.ones((5,5), np.uint8)
    erode = cv2.erode(img, kernel, iterations=20)

    return erode

def thr_erode(img):
    thr_img = threshold(img)
    erode_img = erode(thr_img)

    return erode_img

def dilate(img):
    kernel = np.ones((5,5), np.uint8)
    dilate = cv2.dilate(img, kernel, iterations=1)

    return dilate

def thr_dilate(img):
    thr_img = threshold(img)
    dilate_img = dilate(thr_img)

    return dilate_img

def normalize(img): 
    norm_img = cv2.normalize(img, None, alpha= 0.2, beta=1.0, norm_type= cv2.NORM_MINMAX, dtype=cv2.CV_32F)     
    norm_img = np.clip(norm_img, 0, 1)
    n_img = (255*norm_img).astype(np.uint8)

    return n_img

def bright_norm (img):
    bri_img = auto_brightness(img, 130)
    norm_img = normalize(bri_img)

    return norm_img

def hsv_bri(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    final_hsv = cv2.merge((h, s, v))
    hsv_to_bgr = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return hsv_to_bgr

def blackhat(img):  
    thr_img = threshold(img)
    kernel = np.ones((2,2), np.uint8)
    blackhat_img = cv2.morphologyEx(thr_img, cv2.MORPH_BLACKHAT, kernel)

    return blackhat_img


if __name__ == '__main__':
    img = cv2.imread('/home/seok/다운로드/20230411135301_OK_warp_노랑5핀오삽.bmp')
    #out_img = equalization(img)
    #out_img = auto_brightness(img, 200)
    #out_img = threshold(img)
    #out_img = thr_erode(img)
    #out_img = thr_dilate(img)
    #out_img = normalize(img)
    #out_img = bright_norm(img)
    #out_img = hsv_bri(img, 90)
    out_img = blackhat(img)


    print(out_img)
    img = cv2.resize(out_img, dsize=(1280,720), interpolation = cv2.INTER_AREA)
    cv2.imshow('test', img)
    cv2.waitKey()
    cv2.destoryAllWindows()
