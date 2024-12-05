#code to select the best thresholds for nuclei centroids detection
method = '3DCellPol'
pred_dir = r'/home/jovyan/discotreino/3DCellPol_Github/results_val' #directory to save the validation results
gt_dir = r'/home/jovyan/discotreino/3DCellPol_Github/dataset/val/vectors' #directory with the ground truth validation vectors
img_dir = r'/home/jovyan/discotreino/3DCellPol_Github/dataset/val/images' # directory with the validation image sub-volumes
x_spacing = 0.666 #resolution of the images across x
y_spacing = 0.666 #resolution of the images across y
z_spacing = 0.270 #resolution of the images across z

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

## image names
imgs = [f for f in os.listdir(img_dir) if f.endswith('.npy')]

thresholds_detection_n = np.arange(0.1,1.00,0.05)
distances_ = [4, 5, 6, 7, 8, 9] #voxels (min dist, perc5)

levels_ = np.arange(0.10, 0.80, 0.05)
levels_ = np.flip(levels_)

nuclei_thresholds = 16*levels_
nuclei_thresholds = [nuclei_thresholds[0]]
golgi_thresholds = 8*levels_
golgi_thresholds = [golgi_thresholds[0]]

levels_ = [0] #select the level based on which the best thresholds will be defined

performance_metrics_nucleus = pd.DataFrame(columns = ["Image", "nth", "TPR", "FPR"])

def square_rooted(x):
    return round(np.sqrt(sum([a*a for a in x])),3)

def measure_distance(x,y):
    length1 = square_rooted(x)
    length2 = square_rooted(y)
    return abs(length1-length2)

#Euclidean distance computed in um
def distance_um(p, q, dimx, dimy, dimz):
    dist_um = (((p[0]-q[0])*dimx)**2)+(((p[1]-q[1])*dimy)**2)+(((p[2]-q[2])*dimz)**2)
    return np.sqrt(dist_um) 
    
def inside_img(coord,img_dim_x,img_dim_y,img_dim_z,x_y_lim,z_lim):
    return coord[0]<img_dim_x-x_y_lim and coord[0]>x_y_lim and coord[1]<img_dim_y-x_y_lim and coord[1]>x_y_lim and coord[2]<img_dim_z-z_lim and coord[2]>0


for img_name in imgs:

    output_nuclei = np.load(os.path.join(pred_dir, img_name.replace('.npy','nuclei.npy')))

    for n_th_det in tqdm(thresholds_detection_n):

            for dist_ in distances_:

                nuclei_centroids = peak_local_max(output_nuclei, min_distance=dist_,threshold_rel=n_th_det)
                nuclei_centroids = np.asarray(nuclei_centroids)

                ## directory with the ground truth vectors
                nuclei_centroids_gt = np.load(os.path.join(gt_dir,  img_name.replace('.npy','nuclei.npy')))  
                
                #print('Evaluation')
                ## evaluate the performance of the assignment algorithm
                ''' Assignment nuclei centroids '''
                ## compute the Euclidean distance between the predicted and ground truth centroids
                matrix = np.zeros((len(nuclei_centroids),len(nuclei_centroids_gt)))  ##this is the cost matrix
                
                ## build the cost matrix
                for i in range(0,len(nuclei_centroids)):
                    for j in range(0,len(nuclei_centroids_gt)):
                        matrix[i,j] = distance_um(nuclei_centroids[i], nuclei_centroids_gt[j], x_spacing, y_spacing, z_spacing)
                
                matrix[matrix>5.5] = 2000 #perc5
                
                ## method to solve the linear assignment problem
                row_ind, col_ind = linear_sum_assignment(matrix)
                
                ''' Compute the metrics for the vectors '''
                for n_th, thlvl in zip(nuclei_thresholds, levels_):
                    metrics = []
                    for i in range(0, len(row_ind)):
                        n_coord = nuclei_centroids[row_ind[i]]
                        n_coord_gt = nuclei_centroids_gt[col_ind[i]]
                        dist_n_centroids = distance_um(n_coord, n_coord_gt, x_spacing, y_spacing, z_spacing)
                        if dist_n_centroids<=n_th:
                            metrics.append(1)

                    TP = len(metrics)
                    FP = np.shape(nuclei_centroids)[0] - len(metrics)
                    FN = np.shape(nuclei_centroids_gt)[0] - len(metrics)
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

performance_metrics_nucleus.groupby(['nth'], as_index=False).agg({"TPR": np.mean, "FPR": np.mean})
performance_metrics_nucleus.to_csv(os.path.join(pred_dir, 'metrics_nuclei_detection.csv'), sep=';')
