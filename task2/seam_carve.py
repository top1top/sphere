import numpy as np
from numpy import zeros
from sys import argv, stderr
from glob import iglob, glob
from skimage.io import imread
from pickle import load
from numpy import zeros

def seam_carve(img, mode, mask=None):
    if mode == 'horizontal shrink':
        if mask == None:
            i,j = img.shape[0:2]
            resized_img, resized_mask, carve = shrink(img, np.zeros((i,j)))
            return (resized_img, mask, carve)
        else:
            resized_img, resized_mask, carve = shrink(img, mask)
            return (resized_img, resized_mask, carve)
    if mode == 'vertical shrink':
        if mask == None:
            i,j = img.shape[0:2]
            img = np.swapaxes(img, 0, 1)
            resized_img, resized_mask, carve = shrink(img, np.zeros((j,i)))
            resized_carve = np.swapaxes(carve,0,1)
            resized_img = np.swapaxes(resized_img, 0,1)
            return (resized_img, mask, resized_carve)
        else:
            i,j = img.shape[0:2]
            img = np.swapaxes(img, 0, 1)
            mask = np.swapaxes(mask, 0, 1)
            resized_img, resized_mask, carve = shrink(img, mask)
            resized_carve = np.swapaxes(carve,0, 1)
            resized_img = np.swapaxes(resized_img, 0, 1)
            resized_mask = np.swapaxes(resized_mask, 0, 1)
            return (resized_img, resized_mask, resized_carve)
    
    if mode == 'horizontal expand':
        if mask == None:
            i,j = img.shape[0:2]
            resized_img, resized_mask, carve = expand(img, np.zeros((i,j)))
            return (resized_img, mask, carve)
        else:
            resized_img, resized_mask, carve = expand(img, mask)
            return (resized_img, resized_mask, carve)
    if mode == 'vertical expand':
        if mask == None:
            i,j = img.shape[0:2]
            img = np.swapaxes(img, 0, 1)
            resized_img, resized_mask, carve = expand(img, np.zeros((j,i)))
            resized_carve = np.swapaxes(carve,0,1)
            resized_img = np.swapaxes(resized_img, 0,1)
            return (resized_img, mask, resized_carve)
        else:
            i,j = img.shape[0:2]
            img = np.swapaxes(img, 0, 1)
            mask = np.swapaxes(mask, 0, 1)
            resized_img, resized_mask, carve = expand(img, mask)
            resized_carve = np.swapaxes(carve,0, 1)
            resized_img = np.swapaxes(resized_img, 0, 1)
            resized_mask = np.swapaxes(resized_mask, 0, 1)
            return (resized_img, resized_mask, resized_carve)

def find_seam(img,mask):
    Y = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    i,j = Y.shape

    xx1 = np.roll(Y,1,axis = 0)
    xx2 = np.roll(Y,i-1,axis = 0)
    Y_energy_x = xx1-xx2
    Y_energy_x[-1,:] = Y[-1,:] - Y[-2,:]
    Y_energy_x[0,:] = Y[1,:] - Y[0,:]
    
    
    yy1 = np.roll(Y,1,axis = 1)
    yy2 = np.roll(Y,j-1,axis = 1)
    Y_energy_y = yy1 - yy2
    Y_energy_y[:,0] = Y[:,1] - Y[:,0]
    Y_energy_y[:,-1] = Y[:,-1] - Y[:,-2]
   
    
    tmp = np.sqrt(Y_energy_x ** 2 + Y_energy_y ** 2)
    a = tmp + i*j*mask

    for k in range(1,i,1):
        a[k,0] = min(a[k-1,0],a[k-1,1]) + a[k,0]
        a[k,j-1] = min(a[k-1,j-1],a[k-1,j-2]) + a[k,j-1]
        for t in range(1,j-1,1):
            a[k,t] = min(a[k-1,t],min(a[k-1,t-1], a[k-1,t+1]))+a[k,t]
    
    return a

def shrink(img, mask):
    i,j = img.shape[0:2]
    tmp = find_seam(img,mask)
    
    carve_mask = np.ones_like(tmp, bool)
    cur_min = 100000000000
    cur_ind_min = 0
    
    
    for x in range(j):
        if tmp[i-1,x] < cur_min:
            cur_min = tmp[i-1,x]
            cur_ind_min = x
    carve_mask[i-1, cur_ind_min] = False
    
    for x in range(i-2,-1,-1):
        cur_min = 100000000000
        tmp_ind_min = 0
        for k in range(-1, 2, 1):
            if cur_ind_min+k > -1 and cur_ind_min+k < j:
                if tmp[x, cur_ind_min+k] < cur_min:
                    cur_min = tmp[x,cur_ind_min+k]
                    tmp_ind_min = cur_ind_min+k
        carve_mask[x,tmp_ind_min] = False
        cur_ind_min = tmp_ind_min
        
    resized_img = tmp[carve_mask].reshape(i,j-1)
    resized_mask = mask[carve_mask].reshape(i,j-1)
    return (resized_img, resized_mask, (~carve_mask).astype(np.float64))

def expand(img, mask):
      
    tmp = find_seam(img,mask)
    i,j = tmp.shape
    carve_mask = np.ndarray((i,j + 1), bool)
    carve_mask[:,:] = True
    resized_img = np.ndarray((i,j + 1, 3), int)
    resized_mask = np.ndarray((i,j + 1), int)
    cur_min = 10000000000
    cur_ind_min = 0
    
    for x in range(j):
        if tmp[i-1,x] < cur_min:
            cur_min = tmp[i-1,x]
            cur_ind_min = x
    carve_mask[i-1, cur_ind_min] = False
    mask[i-1,cur_ind_min] = 1
    
    for x in range(i-2,-1,-1):
        cur_min = 100000000000
        tmp_ind_min = 0
        for k in range(-1, 2, 1):
            if cur_ind_min+k > -1 and cur_ind_min+k < j:
                if tmp[x, cur_ind_min+k] < cur_min:
                    cur_min = tmp[x,cur_ind_min+k]
                    tmp_ind_min = cur_ind_min+k
        carve_mask[x,tmp_ind_min] = False
        mask[i-1,tmp_ind_min] = 1         
        cur_ind_min = tmp_ind_min
        
    for x in range(i):
        k = 0
        while carve_mask[x, k]:
            resized_img[x,k,:] = img[x,k,:]
            resized_mask[x,k] = mask[x,k]
            k = k + 1
        resized_img[x,k,:] = img[x,k,:]
        resized_mask[x,k] = mask[x,k]
        k = k + 1
        for t in range(k+1,j+1,1):
            resized_img[x,t,:] = img[x,t-1,:]
            resized_mask[x,t] = mask[x,t-1]
        if k == j:
            resized_img[x,k,:] = img[x,k-1,:]
        else:
            resized_img[x,k,:] = (img[x,k-1,:] + img[x,k,:])/2.
        resized_mask[x,k] = 0
    return (resized_img, resized_mask, (~carve_mask).astype(np.float64))
                
                

   
                    
                    
                    
                
