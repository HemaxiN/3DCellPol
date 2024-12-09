#code to select the best thresholds
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

def select_best_golgi(pred_dir, gt_dir, img_dir, x_spacing, y_spacing):

    thresholds_detection_g = np.arange(0.1,1.00,0.05)
    distances_ = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] #voxels (min dist, perc5)

    levels_ = np.arange(0.10, 0.80, 0.05)
    levels_ = np.flip(levels_)

    nuclei_thresholds = 22*levels_
    nuclei_thresholds = [nuclei_thresholds[0]]
    golgi_thresholds = 14*levels_
    golgi_thresholds = [golgi_thresholds[0]]

    levels_ = [0] #select the level based on which the best thresholds will be defined

    performance_metrics_golgi = pd.DataFrame(columns = ["Image", "gth", "TPR", "FPR"])
    
    imgs = []
    for ii in os.listdir(img_dir):
        imgs.append(ii)

    N = len(imgs)    
    
    for img_name in tqdm(imgs):

        output_golgi = np.load(os.path.join(pred_dir, img_name.replace('.npy','golgi.npy')))

        for g_th_det in thresholds_detection_g:

                for dist_ in distances_:

                    golgi_centroids = peak_local_max(output_golgi, min_distance=dist_,threshold_rel=g_th_det)
                    golgi_centroids = np.asarray(golgi_centroids)

                    golgi_centroids_gt = np.load(os.path.join(gt_dir,  img_name.replace('.npy','golgi.npy'))) 
                    
                    ## evaluate the performance of the assignment algorithm
                    ''' Assignment nuclei centroids '''
                    ## compute the Euclidean distance between the predicted and ground truth centroids
                    matrix = np.zeros((len(golgi_centroids),len(golgi_centroids_gt)))  ##this is the cost matrix

                    ## build the cost matrix
                    for i in range(0,len(golgi_centroids)):
                        for j in range(0,len(golgi_centroids_gt)):
                            matrix[i,j] = distance_um(golgi_centroids[i], golgi_centroids_gt[j], x_spacing, y_spacing)

                    matrix[matrix>2.5] = 2000 #perc 5

                    ## method to solve the linear assignment problem
                    row_ind, col_ind = linear_sum_assignment(matrix)

                    ''' Compute the metrics for the vectors '''
                    for g_th, thlvl in zip(golgi_thresholds, levels_):
                        metrics = []
                        for i in range(0, len(row_ind)):
                            g_coord = golgi_centroids[row_ind[i]]
                            g_coord_gt = golgi_centroids_gt[col_ind[i]]
                            dist_g_centroids = distance_um(g_coord, g_coord_gt, x_spacing, y_spacing)
                            if dist_g_centroids<=g_th:
                                metrics.append(1)

                        TP = len(metrics)
                        FP = np.shape(golgi_centroids)[0] - len(metrics)
                        FN = np.shape(golgi_centroids_gt)[0] - len(metrics)

                        if TP ==0 and FN == 0:
                            continue
                        else:
                            TPR = TP/(TP+FN)
                            if TP==0 and FP==0:
                                FPR = 0
                            else:
                                FPR = FP/(FP+TP)
                            FNR = FN/(FN+TP)

                            res = {"Image": img_name, "gth": str(g_th_det)+'_'+str(dist_),
                                   "TPR": TPR, "FPR": FPR}

                            row = len(performance_metrics_golgi)
                            performance_metrics_golgi.loc[row] = res

    performance_metrics_golgi.to_csv(os.path.join(pred_dir, 'metrics_golgi_detection.csv'), sep=';')