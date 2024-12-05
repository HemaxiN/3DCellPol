## import the necessary packages
import numpy as np
import os
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import csv
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from tqdm import tqdm
from math import*
 
def square_rooted(x):
    return round(np.sqrt(sum([a*a for a in x])),3)
 
def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)

#Euclidean distance computed in um
def distance_um(p, q, dimx, dimy, dimz):
    dist_um = (((p[0]-q[0])*dimx)**2)+(((p[1]-q[1])*dimy)**2)+(((p[2]-q[2])*dimz)**2)
    return np.sqrt(dist_um) 

#ignore the borders of the image
def inside_img(coord,img_dim_x,img_dim_y,img_dim_z,x_y_lim,z_lim):
    return coord[0]<img_dim_x-x_y_lim and coord[0]>x_y_lim and coord[1]<img_dim_y-x_y_lim and coord[1]>x_y_lim

def evaluation(img_dir, gt_dir, pred_dir, save_dir, x_spacing, y_spacing, z_spacing):

    levels_ = np.arange(0.10, 0.80, 0.05)
    levels_ = np.flip(levels_)
    nuclei_thresholds = 16*levels_
    golgi_thresholds = 8*levels_

    levels_ = np.arange(0,len(levels_),1)

    lvl_ = 5 #select a level to plot TP, FP and FN

    ## image names
    imgs = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
    numbers_ = np.arange(0,len(imgs))

    #info = ['test']*len(imgs)

    performance_metrics = pd.DataFrame(columns = ["NucleusTh", "GolgiTh", "Threshold_level",
                                                  "CosineSimilarityM",
                                                  "CosineSimilaritySTD", "VecErrorM","VecErrorSTD",
                                                  "DistanceNuM", "DistanceNuSTD", "DistanceGoM",
                                                  "DistanceGoSTD", "TP", "FP", "FN", "TPR", "FPR", "FNR"])

    metrics_stats = pd.DataFrame(columns = ["NucleusTh", "GolgiTh", "Threshold_level",
                                            "cosine similarity", "vec_error", "nuclei", "golgi"])

    allmetrics = pd.DataFrame(columns = ["NucleusTh", "GolgiTh", "Threshold_level",
                                          "index_tp_gt", "cosine similarity", "vec_error", "nuclei", "golgi"])


    method = '3DCellPol'

    for image_nb in tqdm(numbers_):

        ## directory containing the predicted nucleus and golgi centroids
        nuclei_centroids = np.load(os.path.join(pred_dir,  imgs[image_nb].replace('.tif','') + '_nuclei_centroids_nms.npy'))
        golgi_centroids = np.load(os.path.join(pred_dir, imgs[image_nb].replace('.tif','') + '_golgi_centroids_nms.npy'))
        
        ## directory with the ground truth vectors
        gt_vectors = os.path.join(gt_dir, imgs[image_nb].replace('.tif','') + '.csv')

        ## read the image and get its dimensions
        image = tifffile.imread(os.path.join(img_dir, imgs[image_nb]))

        (img_dim_x, img_dim_y, img_dim_z, channels) = np.shape(image)
        
        #limits to ignore vectors at the borders of the image
        x_y_lim = int(7/x_spacing)  #(voxels)  16
        z_lim = int(1/z_spacing)    #(voxels)  5

        print('Reading the csv file with the ground truth vectors')
        ## nuclei and golgi centroids
        nuclei_centroids_gt = [] 
        golgi_centroids_gt = []
        
        #open the csv file and save the gt nucleus and Golgi centroids
        file = open(gt_vectors, "rU")
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            if row[0] != 'YN,XN,ZN,YG,XG,ZG':
                aux = row[0].split(",")
                YN = float(aux[0])
                XN = float(aux[1])
                ZN = float(aux[4])
                YG = float(aux[2])
                XG = float(aux[3])
                ZG = float(aux[5])
                
                nuclei_centroids_gt.append((XN,YN,ZN))
                golgi_centroids_gt.append((XG,YG,ZG)) 


        #Remove ground-truth nuclei and Golgi at image borders
        nuclei_centroids_gt_final = []
        golgi_centroids_gt_final = []
        for nn, gg in zip(nuclei_centroids_gt, golgi_centroids_gt):
            if inside_img(nn, img_dim_x, img_dim_y, img_dim_z, x_y_lim, z_lim) and inside_img(gg, img_dim_x, img_dim_y, img_dim_z, x_y_lim, z_lim):
                nuclei_centroids_gt_final.append(nn)
                golgi_centroids_gt_final.append(gg)
        
        golgi_centroids_gt = np.asarray(golgi_centroids_gt_final)
        nuclei_centroids_gt = np.asarray(nuclei_centroids_gt_final)
        
        #Remove predicted nuclei and Golgi at image borders
        n_centroids = []
        g_centroids = []
        for nc, gc in zip(nuclei_centroids, golgi_centroids):
            if inside_img(nc, img_dim_x, img_dim_y, img_dim_z, x_y_lim, z_lim) and inside_img(gc,img_dim_x, img_dim_y, img_dim_z, x_y_lim, z_lim):
                n_centroids.append(nc)
                g_centroids.append(gc)
                
        nuclei_centroids = np.asarray(n_centroids)
        golgi_centroids = np.asarray(g_centroids)
        
        print('Evaluation')
        ''' Assignment nuclei centroids '''
        ## compute the Euclidean distance between the predicted and ground truth centroids
        matrix = np.zeros((len(nuclei_centroids),len(nuclei_centroids_gt)))  ##this is the cost matrix
        
        ## build the cost matrix
        for i in range(0,len(nuclei_centroids)):
            for j in range(0,len(nuclei_centroids_gt)):
                matrix[i,j] = distance_um(nuclei_centroids[i], nuclei_centroids_gt[j], x_spacing, y_spacing, z_spacing) + distance_um(golgi_centroids[i], golgi_centroids_gt[j], x_spacing, y_spacing, z_spacing)
        
        matrix[matrix>10] = 2000
        
        ## method to solve the linear assignment problem
        row_ind, col_ind = linear_sum_assignment(matrix)
        
        ''' Compute the metrics for the vectors '''
        for n_th, g_th, thlvl in zip(nuclei_thresholds, golgi_thresholds, levels_):
            metrics = pd.DataFrame(columns = ["NucleusTh", "GolgiTh", "Threshold_level",
                                          "cosine similarity", "vec_error", "nuclei", "golgi"])

            if thlvl==lvl_:
                index_tp = []  ## positions in vectors nuclei_centroids, golgi_centroids, that are
                                ## true positives
                                
                index_tp_gt = [] ## positions in vectors nuclei_centroids_gt and golgi_centroids_gt,
                                  ## that correspond to true positives
        
        
            for i in range(0, len(row_ind)):
                n_coord = nuclei_centroids[row_ind[i]]
                g_coord = golgi_centroids[row_ind[i]]
            
                vec = g_coord - n_coord
            
                n_coord_gt = nuclei_centroids_gt[col_ind[i]]
                g_coord_gt = golgi_centroids_gt[col_ind[i]]
            
                vec_gt = g_coord_gt - n_coord_gt
                
                dist_n_centroids = distance_um(n_coord, n_coord_gt, x_spacing, y_spacing, z_spacing)
                dist_g_centroids = distance_um(g_coord, g_coord_gt, x_spacing, y_spacing, z_spacing)
                vec_error = distance_um(vec, vec_gt, x_spacing, y_spacing, z_spacing)
                
                cos_sim = cosine_similarity(vec, vec_gt)
                
                if dist_n_centroids<=n_th and dist_g_centroids<=g_th:

                    res = {"NucleusTh": n_th, "GolgiTh": g_th,
                           "Threshold_level": thlvl,
                           "cosine similarity": abs(cos_sim), "vec_error": vec_error, 
                           "nuclei": dist_n_centroids, "golgi": dist_g_centroids}
                    
                    res_aux = {"NucleusTh": n_th, "GolgiTh": g_th,
                           "Threshold_level": thlvl, "index_tp_gt": col_ind[i],
                           "cosine similarity": abs(cos_sim), "vec_error": vec_error, 
                           "nuclei": dist_n_centroids, "golgi": dist_g_centroids}
                    
                    row_aux = len(allmetrics)
                    allmetrics.loc[row_aux] = res_aux
                    
                    row = len(metrics)
                    metrics.loc[row] = res
                    
                    row_stats = len(metrics_stats)
                    metrics_stats.loc[row_stats] = res
                    
                    if thlvl==lvl_:
                        index_tp.append(row_ind[i])
                        index_tp_gt.append(col_ind[i])
                    
            
            metrics_mean = metrics.mean()
            metrics_std = metrics.std()
            
            TP = len(metrics)
            FP = np.shape(golgi_centroids)[0] - len(metrics)
            FN = np.shape(golgi_centroids_gt)[0] - len(metrics)
            TPR = TP/(TP+FN)
            
            if TP==0 and FP!=0:
                FPR = 1
            else:
                FPR = FP/(FP+TP)

            FNR = FN/(FN+TP)

            res = {"NucleusTh": n_th, "GolgiTh": g_th, "Threshold_level": thlvl,
                   "CosineSimilarityM": metrics_mean['cosine similarity'],
                   "CosineSimilaritySTD": metrics_std['cosine similarity'], 
                   "VecErrorM": metrics_mean['vec_error'],
                   "VecErrorSTD": metrics_std['vec_error'],
                   "DistanceNuM": metrics_mean['nuclei'], 
                   "DistanceNuSTD": metrics_std['nuclei'], 
                   "DistanceGoM": metrics_mean['golgi'], 
                   "DistanceGoSTD": metrics_std['golgi'], 
                   "TP": TP, 
                   "FP": FP, 
                   "FN": FN, 
                   "TPR": TPR, 
                   "FPR": FPR,
                   "FNR": FNR}
            
            row = len(performance_metrics)
            performance_metrics.loc[row] = res
        
    final_metrics = performance_metrics.groupby(["Threshold_level"], as_index=False).agg({'CosineSimilarityM': np.mean,
                                                     "CosineSimilaritySTD": np.mean,
                                                     "VecErrorM": np.mean,
                                                     "VecErrorSTD": np.mean,
                                                     "DistanceNuM": np.mean, 
                                                     "DistanceNuSTD": np.mean, 
                                                     "DistanceGoM": np.mean, 
                                                     "DistanceGoSTD": np.mean, 
                                                     "TP": np.sum, 
                                                     "FP": np.sum, 
                                                     "FN": np.sum, 
                                                     "TPR": np.mean, 
                                                     "FPR": np.mean,
                                                     "FNR": np.mean})
            
            
    plt.figure()
    plot1 = plt.plot(np.asarray(final_metrics["Threshold_level"]), np.asarray(final_metrics["TPR"]), label='TPR')
    plot2 = plt.plot(np.asarray(final_metrics["Threshold_level"]), np.asarray(final_metrics["FPR"]), label='FPR')
    plot3 = plt.plot(np.asarray(final_metrics["Threshold_level"]), np.asarray(final_metrics["CosineSimilarityM"]), label='CosSim')
    plt.yticks(np.arange(0,1.1,0.1))
    plt.legend()
    plt.savefig(os.path.join(save_dir,'TPR_FPR_CosSim.jpg'), dpi=100)
    plt.close()

    plt.figure()
    plot1 = plt.plot(np.asarray(final_metrics["Threshold_level"]), np.asarray(final_metrics["DistanceNuM"]), label='ND')
    plot2 = plt.plot(np.asarray(final_metrics["Threshold_level"]), np.asarray(final_metrics["DistanceGoM"]), label='GD')
    plot3 = plt.plot(np.asarray(final_metrics["Threshold_level"]), np.asarray(final_metrics["VecErrorM"]), label='Error')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'ND_GD_Error.jpg'), dpi=100)
    plt.close()

    performance_metrics.to_csv(os.path.join(save_dir, method + '.csv'), index=False)

    allmetrics.to_csv(os.path.join(save_dir, method+'all.csv'), index=False)