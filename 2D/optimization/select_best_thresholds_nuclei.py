#code to select the best thresholds for Golgi centroids detection
import tifffile
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import csv
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from math import*

# test parameters
_patch_size = 256 #patch size along x and y directions
_z_size = 64 #patch size along z direction
_step = 64 #overlap along x and y directions between consecutive patches extracted from the image
_step_z = 64 #overlap along z directions between consecutive patches extracted from the image

def square_rooted(x):
    return round(np.sqrt(sum([a*a for a in x])),3)

def measure_distance(x,y):
    length1 = square_rooted(x)
    length2 = square_rooted(y)
    return abs(length1-length2)

#Euclidean distance computed in um
def distance_um(p, q, dimx, dimy):
    dist_um = (((p[0]-q[0])*dimx)**2)+(((p[1]-q[1])*dimy)**2)
    return np.sqrt(dist_um) 

def select_best_nuclei(pred_dir, gt_dir, img_dir, x_spacing, y_spacing):

    thresholds_detection_n = np.arange(0.1,1.00,0.05)
    distances_ = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] #voxels (min dist, perc5)
    #distances_ = np.divide(distances_, 2)

    levels_ = np.arange(0.10, 0.80, 0.05)
    levels_ = np.flip(levels_)

    nuclei_thresholds = 22*levels_
    nuclei_thresholds = [nuclei_thresholds[0]]
    golgi_thresholds = 14*levels_
    golgi_thresholds = [golgi_thresholds[0]]

    levels_ = [0] #select the level based on which the best thresholds will be defined

    imgs = []
    for ii in os.listdir(img_dir):
        imgs.append(ii)

    N = len(imgs)    

    performance_metrics_nucleus = pd.DataFrame(columns = ["Image", "nth", "TPR", "FPR"])

    for img_name in tqdm(imgs):

        output_nuclei = np.load(os.path.join(pred_dir, img_name.replace('.npy','nuclei.npy')))

        for n_th_det in thresholds_detection_n:

                for dist_ in distances_:

                    nuclei_centroids = peak_local_max(output_nuclei, min_distance=dist_,threshold_rel=n_th_det)
                    nuclei_centroids = np.asarray(nuclei_centroids)
 
                    nuclei_centroids_gt = np.load(os.path.join(gt_dir,  img_name.replace('.npy','nuclei.npy'))) 

                    ## evaluate the performance of the assignment algorithm
                    ''' Assignment nuclei centroids '''
                    ## compute the Euclidean distance between the predicted and ground truth centroids
                    matrix = np.zeros((len(nuclei_centroids),len(nuclei_centroids_gt)))  ##this is the cost matrix

                    ## build the cost matrix
                    for i in range(0,len(nuclei_centroids)):
                        for j in range(0,len(nuclei_centroids_gt)):
                            matrix[i,j] = distance_um(nuclei_centroids[i], nuclei_centroids_gt[j], x_spacing, y_spacing)

                    matrix[matrix>3] = 2000 #perc5

                    ## method to solve the linear assignment problem
                    row_ind, col_ind = linear_sum_assignment(matrix)

                    ''' Compute the metrics for the vectors '''
                    for n_th, thlvl in zip(nuclei_thresholds, levels_):
                        metrics = []
                        for i in range(0, len(row_ind)):
                            n_coord = nuclei_centroids[row_ind[i]]
                            n_coord_gt = nuclei_centroids_gt[col_ind[i]]
                            dist_n_centroids = distance_um(n_coord, n_coord_gt, x_spacing, y_spacing)
                            if dist_n_centroids<=n_th:
                                metrics.append(1)

                        TP = len(metrics)
                        FP = np.shape(nuclei_centroids)[0] - len(metrics)
                        FN = np.shape(nuclei_centroids_gt)[0] - len(metrics)

                        if TP ==0 and FN == 0:
                            continue
                        else:
                            TPR = TP/(TP+FN)
                            if TP==0 and FP==0:
                                FPR = 0
                            else:
                                FPR = FP/(FP+TP)
                            FNR = FN/(FN+TP)

                            res = {"Image": img_name, "nth": str(n_th_det)+'_'+str(dist_),
                                   "TPR": TPR, "FPR": FPR}

                            row = len(performance_metrics_nucleus)
                            performance_metrics_nucleus.loc[row] = res

    performance_metrics_nucleus.to_csv(os.path.join(pred_dir, 'metrics_nuclei_detection.csv'), sep=';')

