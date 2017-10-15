from numpy import ones
import numpy
from sklearn import svm
import math
import scipy
     
def extract_hog(img0):
    img0 = scipy.misc.imresize(img0,(64,64))
    img = numpy.float64(img0)
    i,j = (64,64)
    resized_img = numpy.zeros((i+2,j+2,3), numpy.float64)
    resized_img[0,1:-1,:] = img[0,:,:]
    resized_img[i-1,1:-1,:] = img[i-1,:,:]
    resized_img[1:-1,0,:] = img[:,0,:]
    resized_img[1:-1,j-1,:] = img[:,j-1,:]
    resized_img[1:-1,1:-1,:] = img[:,:,:]
    resized_img[0,0,:] = img[0,1,:]
    resized_img[i-1,0,:] = img[i-1,1,:]
    resized_img[0,j-1,:] = img[1,j-1,:]
    resized_img[i-1,j-1,:] = img[-2,j-1,:]
    
    height = i
    width = j
    i_x = numpy.zeros((i,j,3),numpy.float64)
    i_y = numpy.zeros((i,j,3),numpy.float64)
    i_x[:,:,:] = resized_img[2:,1:-1,:] - resized_img[0:-2,1:-1,:]
    i_y[:,:,:] = resized_img[1:-1,2:,:] - resized_img[1:-1,0:-2,:]
    '''i_x[0, 0:width, :] = img[1, 0:width, :] - img[0, 0:width, :]
    i_x[height - 1, 0:width, :] = img[height - 1, 0:width, :] - img[height - 2, 0:width, :]
    i_x[1:height - 1, 0:width, :] = img[2:height, 0:width, :] - img[0:height - 2, 0:width, :]
    i_y[0:height, 0, :] = img[0:height, 1, :] - img[0:height, 0, :]
    i_y[0:height, width - 1, :] = img[0:height, width - 1, :] - img[0:height, width - 2, :]
    i_y[0:height, 1:width - 1, :] = img[0:height, 2:width, :] - img[0:height, 0:width - 2, :]'''
    
    g = (i_y ** 2 + i_x ** 2) ** (0.5)
    
    new_g = numpy.zeros((i,j),numpy.float64)
    new_q = numpy.zeros((i,j),numpy.float64)
    tmp = numpy.argmax(g, axis=2)
    
    for ii in range(i):
        for jj in range(j):
            new_g[ii,jj] = g[ii,jj, tmp[ii,jj]]
            new_q[ii,jj] = math.atan2(i_y[ii,jj, tmp[ii,jj]],i_x[ii,jj, tmp[ii,jj]])
    
    new_q[new_q<0] += math.pi
    csx = csy = 8#pixels at cell
    qq = new_q[:i,:j]
    gg = new_g[:i,:j]

    max_angle = math.pi
    nbins = 9#binCount
    b_step = max_angle/nbins
    b0 = qq/b_step
    res = numpy.zeros((i,j,nbins),numpy.float64)
    #print(qq)
    
    '''for ii in range(i):
        for jj in range(j):
            if b0[ii,jj] == int(b0[ii,jj]):
                if int(b0[ii,jj]) == 9:
                    res[ii,jj,0] += gg[ii,jj] 
                else:
                    res[ii,jj,int(b0[ii,jj])] += gg[ii,jj] 
                continue
            for tt in range(nbins-1):
                if b0[ii,jj] > tt and b0[ii,jj] < tt + 1:
                    res[ii,jj,tt] += gg[ii,jj] * (b0[ii,jj] - tt)
                    res[ii,jj,tt+1] += gg[ii,jj] * (tt + 1 - b0[ii,jj])
            if b0[ii,jj] > nbins-1:
                res[ii,jj,nbins-1] += gg[ii,jj] * (b0[ii,jj] - (nbins-1))
                res[ii,jj,0] += gg[ii,jj] * (nbins - b0[ii,jj])                                
    '''
    n_cells_x = i//csx
    n_cells_y = j//csy    
    cells = numpy.zeros((n_cells_x, n_cells_y, nbins),numpy.float64) 
    
    for row in range (n_cells_x):
        for col in range (n_cells_y):
            rr = row * csx
            cc = col * csy        
            for x in range(csx):
                for y in range(csy):
                    ang = qq[rr + x,cc + y] / math.pi * nbins
                    num = int(ang)
                    if (ang == int(ang)):
                        pr = 1
                    else:
                        pr = ang - int(ang)
                    if (num == nbins):
                        num = 0
                    cells[row, col, num] += (1 - pr) * gg[rr + x,cc + y]
                    if (num != nbins - 1): 
                        cells[row, col, num + 1] += pr * gg[rr + x,cc + y]
                    else:
                        cells[row, col, 0] += pr * gg[rr + x,cc + y]
    
    '''      
    for xx in range(n_cells_x):
        for yy in range(n_cells_y):
            for tt in range(nbins):
                cells[xx,yy,tt] = res[xx*csx:xx*csx+csx, yy*csy:yy*csy+csy,tt].sum()
    '''
    n_block_x = n_block_y = 2
    amount_block_x = n_cells_x - n_block_x + 1
    amount_block_y = n_cells_y - n_block_y + 1
    eps = 0.000001
    
    v = numpy.ndarray((n_block_x * n_block_y * amount_block_x * amount_block_y*nbins),numpy.float64)
    
    cur = 0 
    for ii in range(amount_block_x):
        for jj in range(amount_block_y):
            z = cells[ii:ii+n_block_x,jj:jj+n_block_y,:]
            size1, size2 = z.shape[0:2]
            z = z / ((z ** 2).sum() + eps) ** 0.5
            size1, size2 = z.shape[0:2]
            for ff in range(size1):
                for rr in range(size2):
                    for cc in range(nbins):
                        v[cur] = z[ff,rr,cc]
                        cur += 1    
    #print("_______________________________________________")
    return v

def fit_and_classify(train_features, train_labels, test_features):
     clf = svm.LinearSVC()
     clf.fit(train_features, train_labels) 
     return clf.predict(test_features)
