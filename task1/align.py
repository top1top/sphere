from numpy import array, dstack, roll
from skimage.transform import rescale
import scipy
import scipy.ndimage
from scipy import misc
def fn(w, h, I1, I2):
    if w > 0:
        if h > 0:
            tmp1 = I1[abs(w):, abs(h):]
            tmp2 = I2[:width_cur - abs(w), :height_cur - abs(h)]
        else:
            tmp1 = I1[abs(w):, :height_cur - abs(h)]
            tmp2 = I2[ :width_cur - abs(w), abs(h):]
    else:
        if h > 0:
            tmp1 = I1[:width_cur - abs(w),abs(h):]
            tmp2 = I2[ abs(w):, :height_cur - abs(h)]
        else:
            tmp1 = I1[:width_cur - abs(w), :height_cur - abs(h)]
            tmp2 = I2[abs(w):, abs(h):]
    return (tmp1, tmp2)

def mse(I1, I2):
    answer = 100000000
    ans_h = 0
    ans_w = 0
    for w in range(-15, 16, 1):
        for h in range (-15, 16, 1):
            tmp_ans = ((tmp1 - tmp2) ** 2).sum()
            if tmp_ans < answer:
                answer = tmp_ans
                ans_h = h
                ans_w = w
    print (tmp_ans)
    return (ans_h, ans_w)

import math
def cross_cor(I1, I2):
    answer = -100000000
    ans_h = 0
    ans_w = 0
    for w in range(-15, 16, 1):
        for h in range (-15, 16, 1):
            tmp1, tmp2 = fn(w, h, I1, I2)
            tmp_ans = (tmp1*tmp2).sum()/math.sqrt(((tmp1 ** 2)*(tmp2 ** 2)).sum())
            if tmp_ans > answer:
                answer = tmp_ans
                ans_h = h
                ans_w = w
    return (ans_h, ans_w)

def align(bgr_image, g_coord):
    width, height = bgr_image.size
    new_height = (int)(height / 3)
    per_width = (int)(width * 0.05)
    per_height = (int)(new_height * 0.05)
    crop_w_l = per_width
    crop_w_r = width - per_width
    
    bbox = (crop_w_l, per_height, crop_w_r, new_height + per_height)
    im1 = img.crop(bbox)
    
    bbox = (crop_w_l, new_height + 3 *per_height, crop_w_r, 2 * new_height + 3 *per_height)
    im2 = img.crop(bbox)
    
    bbox = (crop_w_l, 2 * new_height + 5 *per_height, crop_w_r, 3 * new_height + 5 *per_height)
    im3 = img.crop(bbox)
    
    arr_im1 = scipy.misc.fromimage(im1, flatten=True, mode=None)
    arr_im2 = scipy.misc.fromimage(im2, flatten=True, mode=None)
    arr_im3 = scipy.misc.fromimage(im3, flatten=True, mode=None)
    
    width_cur, height_cur = arr_im3.shape
    b_row = b_col = r_row = r_col = 0
    h_1, w_1 = mse(arr_im2, arr_im1);
    h_2, w_2 = mse(arr_im1, arr_im2);
    b_row, b_col = g_coord
    b_row += h_1
    b_col += w_1
    r_row, r_col = g_coord
    r_row += h_2
    r_col += w_2
    im1_roll = numpy.roll(im1, h_1, axis = 0)
    im1_roll = numpy.roll(im1, w_1, axis = 1)
    im3_roll = numpy.roll(im1, h_2, axis = 0)
    im3_roll = numpy.roll(im1, w_2, axis = 1)
    
    bgr_image  = numpy.dstack(im3_roll, arr_im2)
    bgr_image  = numpy.dstack(bgr_image, im1_roll)
    
    return bgr_image, (b_row, b_col), (r_row, r_col)

