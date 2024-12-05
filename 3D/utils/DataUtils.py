import numpy as np
import cv2

def __getCircle(radius):
    x = np.arange(-radius, radius)
    y = np.arange(-radius, radius)
    z = np.arange(-radius, radius)
    xx, yy, zz = np.meshgrid(x, y, z)
    return np.sqrt(xx**2 + yy**2 + zz**2) < radius

def __getGaussian(radius):
    x = np.linspace(-radius, +radius, radius*2)
    y = np.linspace(-radius, +radius, radius*2)
    z = np.linspace(-radius, +radius, radius*2)
    xx, yy, zz = np.meshgrid(x, y, z)
    d = np.sqrt(xx**2+yy**2+zz**2)
    sigma, mu = radius/2, 0.0
    gauss = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    gauss = ( gauss - np.min(gauss) ) / ( np.max(gauss) - np.min(gauss) ) # scalling between 0 to 1
    return gauss

def centroids2Images(point_list, im_num_row, im_num_col, im_num_slices, g_radius, th):
    circle_mat = __getGaussian(g_radius)
    circle_mat = circle_mat/np.max(circle_mat)
    threshold_circle = circle_mat>th
    temp_im = np.zeros((im_num_row+g_radius*2, im_num_col+g_radius*2, im_num_slices+g_radius*2))
    mask = np.zeros((im_num_row+g_radius*2, im_num_col+g_radius*2, im_num_slices+g_radius*2))
    
    count_ = 1
    for one_pnt in point_list:
        pnt_row = int(one_pnt[0])
        pnt_col = int(one_pnt[1])
        pnt_slice = int(one_pnt[2])

        current_patch = temp_im[g_radius+pnt_row-g_radius:g_radius+pnt_row+g_radius, g_radius+pnt_col-g_radius:g_radius+pnt_col+g_radius, g_radius+pnt_slice-g_radius:g_radius+pnt_slice+g_radius]
        indices = np.where(current_patch>circle_mat)
        temp_im[g_radius+pnt_row-g_radius:g_radius+pnt_row+g_radius, g_radius+pnt_col-g_radius:g_radius+pnt_col+g_radius, g_radius+pnt_slice-g_radius:g_radius+pnt_slice+g_radius] = np.maximum(current_patch, circle_mat)
        current_patch = mask[g_radius+pnt_row-g_radius:g_radius+pnt_row+g_radius, g_radius+pnt_col-g_radius:g_radius+pnt_col+g_radius, g_radius+pnt_slice-g_radius:g_radius+pnt_slice+g_radius]
        aux_circle = (threshold_circle)*count_
        aux_circle[indices] = 0
        mask[g_radius+pnt_row-g_radius:g_radius+pnt_row+g_radius, g_radius+pnt_col-g_radius:g_radius+pnt_col+g_radius, g_radius+pnt_slice-g_radius:g_radius+pnt_slice+g_radius] = np.maximum(current_patch, aux_circle)
        
        count_ +=1
    
    temp_im = temp_im[g_radius:-g_radius, g_radius:-g_radius, g_radius:-g_radius]
    mask = mask[g_radius:-g_radius, g_radius:-g_radius, g_radius:-g_radius]

    temp_im_aux = (temp_im*255.0).astype('uint8')
    temp_im_aux = (temp_im_aux/255.0>th)*1
    mask[temp_im_aux==0] = 0

    return temp_im, mask
