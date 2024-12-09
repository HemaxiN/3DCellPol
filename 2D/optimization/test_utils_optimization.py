#import the required packages
from keras import backend as K
import tensorflow as tf
from functools import partial
from tensorflow.python.ops import *
import keras
import cv2
from scipy.ndimage import rotate
import random
import numpy as np 
import os
import math
import scipy.ndimage as ndi
from keras.models import load_model
import tifffile
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#custom loss functions:

#weighted mean squared error loss function
def mse(y_true, y_pred, sample_weight=None):
    squared  = math_ops.square(y_pred - y_true)
    if sample_weight==None:
        return tf.reduce_mean(squared)
    else:
        multiplication = math_ops.multiply(sample_weight, squared)
        return tf.reduce_mean(multiplication)

def weighted_mean_se(y_true, y_pred):
    [golgi_c_gt, nuclei_c_gt, golgi_weight, nuclei_weight, _, _] = tf.unstack(y_true, 6, axis=4)
    [golgi_c_pred, nuclei_c_pred, _, _] = tf.unstack(y_pred, 4, axis=4)
    mse_loss_golgi = mse(golgi_c_gt, golgi_c_pred, golgi_weight)
    mse_loss_nuclei = mse(nuclei_c_gt, nuclei_c_pred, nuclei_weight)
    return mse_loss_golgi + mse_loss_nuclei

def push_loss(E_tensor):
    # Compute the absolute differences between each element Ek and all other elements Ej
    E_diff = tf.abs(tf.expand_dims(E_tensor, axis=1) - tf.expand_dims(E_tensor, axis=0))
    # Compute the max(0, 2 - abs(Ek - Ej)) operation for each element in the differences tensor
    max_values = tf.maximum(0.0, 2.0 - E_diff)
    # Get the number of elements in the tensor
    num_elements = tf.shape(max_values)[0]
    # Create a mask to set diagonal elements to NaN
    mask = tf.eye(num_elements, dtype=tf.float32)
    # Set diagonal elements to NaN using tf.linalg.set_diag
    max_values = tf.where(tf.equal(mask, 1.0), tf.fill([num_elements, num_elements], float('nan')), max_values)
    # Sum the max(0, 1 - abs(Ek - Ej)) values for each Ek over all Ej, ignoring NaNs
    quantity = tf.reduce_mean(tf.boolean_mask(max_values, tf.is_finite(max_values)))  
    return quantity

def pull_aux_optimized(y, mask_g, embeddings_g, mask_n, embeddings_n):
    def true_condition(input_1, input_2):
        return input_1 + input_2
    def false_condition(input_):
        return input_
    pull = 0.
    E_tensor = []
    for filter_value in range(0, 40):
        if filter_value != 0:
            # Create a boolean mask where mask has the filter_value
            mask_filter_n = tf.equal(mask_n, filter_value)
            # Use the mask to extract values from matrix embeddings
            filtered_values_n = tf.boolean_mask(embeddings_n, mask_filter_n)
            # Create a boolean mask where mask has the filter_value
            mask_filter_g = tf.equal(mask_g, filter_value)
            # Use the mask to extract values from matrix embeddings
            filtered_values_g = tf.boolean_mask(embeddings_g, mask_filter_g)
            filtered_values = tf.concat([filtered_values_n, filtered_values_g], axis=0)
            mean_ = tf.reduce_mean(filtered_values)
            paux = tf.reduce_mean(tf.pow((filtered_values-mean_),2))
            condition =  tf.is_finite(paux)
            pull = tf.cond(condition, lambda: true_condition(pull, paux), lambda: false_condition(pull))
            E_tensor.append(mean_)            
        else:
            # Create a boolean mask where mask has the filter_value
            mask_filter_n = tf.equal(mask_n, filter_value)
            # Use the mask to extract values from matrix embeddings
            filtered_values_n = tf.boolean_mask(embeddings_n, mask_filter_n)
            # Create a boolean mask where mask has the filter_value
            mask_filter_g = tf.equal(mask_g, filter_value)
            # Use the mask to extract values from matrix embeddings
            filtered_values_g = tf.boolean_mask(embeddings_g, mask_filter_g)
            filtered_values = tf.concat([filtered_values_n, filtered_values_g], axis=0)        
            mean_ = tf.reduce_mean(filtered_values)            
            pull = pull + tf.abs(mean_)            
    tam_ = tf.cast(tf.shape(y)[0], tf.float32)-1
    E_tensor = tf.convert_to_tensor(E_tensor)
    E_tensor = E_tensor[:tf.shape(y)[0]-1]
    return pull/tam_, E_tensor, tam_

def pull_loss(y_true, y_pred, bi):
    [gctr, nctr, golgi_embeddings, nuclei_embeddings] = tf.unstack(y_pred, 4, axis=4)
    [gctr_gt, nctr_gt, wg, wn, golgi_mask_gt, nuclei_mask_gt] = tf.unstack(y_true, 6, axis=4)
    nuclei_embeddings = nuclei_embeddings[bi]
    golgi_embeddings = golgi_embeddings[bi]
    golgi_mask_gt = golgi_mask_gt[bi]
    nuclei_mask_gt = nuclei_mask_gt[bi]
    #reshape the necessary tensors
    nuclei_mask_gt = tf.reshape(nuclei_mask_gt,[-1])
    golgi_mask_gt = tf.reshape(golgi_mask_gt,[-1])
    nuclei_embeddings = tf.reshape(nuclei_embeddings,[-1])
    golgi_embeddings = tf.reshape(golgi_embeddings,[-1])
    #get unique objects ids
    y, _ = tf.unique(nuclei_mask_gt)
    pull, E, tam_ = pull_aux_optimized(y, golgi_mask_gt, golgi_embeddings, nuclei_mask_gt, nuclei_embeddings)
    return pull, E, tam_

def push_pull_loss(y_true, y_pred):
    def true_condition(E):
        push = push_loss(E)
        return push
    def false_condition():
        return 0.
    pull_list = []
    push_list = []
    for bi in range(8):
        pull, E, tam_ = pull_loss(y_true, y_pred, bi)
        condition = tf.greater_equal(tam_, 2)
        push = tf.cond(condition, lambda: true_condition(E), false_condition)
        pull_list.append(pull)
        push_list.append(push)
    push_list = tf.convert_to_tensor(push_list)
    pull_list = tf.convert_to_tensor(pull_list)
    return tf.reduce_mean(push_list)+tf.reduce_mean(pull_list)

def pull_loss_vis(y_true, y_pred):
    def true_condition(E):
        push = push_loss(E)
        return push
    def false_condition():
        return 0.
    pull_list = []
    push_list = []
    for bi in range(4):
        pull, E, tam_ = pull_loss(y_true, y_pred, bi)
        condition = tf.greater_equal(tam_, 2)
        push = tf.cond(condition, lambda: true_condition(E), false_condition)
        pull_list.append(pull)
        push_list.append(push)
    push_list = tf.convert_to_tensor(push_list)
    pull_list = tf.convert_to_tensor(pull_list)
    return tf.reduce_mean(pull_list)

def push_loss_vis(y_true, y_pred):
    def true_condition(E):
        push = push_loss(E)
        return push
    def false_condition():
        return 0.
    push_list = []
    for bi in range(4):
        pull, E, tam_ = pull_loss(y_true, y_pred, bi)
        condition = tf.greater_equal(tam_, 2)
        push = tf.cond(condition, lambda: true_condition(E), false_condition)
        push_list.append(push)
    push_list = tf.convert_to_tensor(push_list)
    return tf.reduce_mean(push_list)

def both_joint_loss_function(y_true, y_pred):
    detection_loss = weighted_mean_se(y_true, y_pred)
    pp_loss = push_pull_loss(y_true, y_pred)
    return detection_loss + pp_loss

def load_old_model(model_file):
    print("Loading pre-trained model")
    custom_objects = {'mse':mse, 'weighted_mean_se':weighted_mean_se, 'push_loss':push_loss, 'pull_aux_optimized': pull_aux_optimized,
        'pull_loss': pull_loss, 'push_pull_loss': push_pull_loss, 'both_joint_loss_function': both_joint_loss_function,
        'pull_loss_vis': pull_loss_vis, 'push_loss_vis': push_loss_vis}
    try:
        from keras_contrib.layers import InstanceNormalization
        custom_objects["InstanceNormalization"] = InstanceNormalization
    except ImportError:
        pass
    try:
        return load_model(model_file,custom_objects=custom_objects)
    except ValueError as error:
        if 'InstanceNormalization' in str(error):
            raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
                                          "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
        else:
            raise error

def normalization(mip_img):
    minval = np.percentile(mip_img, 0.1)
    maxval = np.percentile(mip_img, 99.9)
    mip_img = np.clip(mip_img, minval, maxval)
    mip_img = (((mip_img - minval) / (maxval - minval)) * 255).astype('uint8')
    return mip_img

def normalization2(mip_img):
    minval = np.percentile(mip_img, 0.05)
    maxval = np.percentile(mip_img, 99.95)
    mip_img = np.clip(mip_img, minval, maxval)
    mip_img = (((mip_img - minval) / (maxval - minval)) * 255).astype('uint8')
    return mip_img

def square_rooted(x):
    return round(np.sqrt(sum([a*a for a in x])),3)

def nms_vectors(n_centroids, g_centroids, n_probs, g_probs, thresh_, x_spacing, y_spacing):

    idxs = np.arange(0,len(n_centroids))
    #create an "idxs" list with the indexes of list vectors through which we need
    #to perform the "search"

    #go through each element (i) in that list and compute the distance to all the other
    #elements (j, where i!=j), if the distance to an element (j) is smaller than a threshold then:
    #if the probability of vector i is bigger than the probability of vector j then:
    #keep vector i as the "best" and "supress" element j (delete it from the idx list)
    #else keep the vector j as the "best" and "supress" element i and element j (delete
    #them from the idx list)

    pick = []
    while len(idxs)>0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]

        suppress = [i]
        for j in idxs:
            if j!=i:
                dist = distance_um(n_centroids[i], n_centroids[j], x_spacing, y_spacing)
                if dist <= thresh_:
                    prob_i_n = n_probs[i]
                    prob_j_n = n_probs[j]

                    prob_i_g = g_probs[i]
                    prob_j_g = g_probs[j]

                    if (prob_i_n + prob_i_g) >= (prob_j_n + prob_j_g):
                        suppress.append(j)
                    else:
                        suppress.append(j)
                        i = j
        #vector that was picked
        pick.append(i)
        idxs = np.setdiff1d(idxs, suppress)

    new_n_centroids_pred =  []
    new_g_centroids_pred = []
    new_n_probs = []
    new_g_probs = []

    for index in pick:
        new_n_centroids_pred.append(n_centroids[index])
        new_g_centroids_pred.append(g_centroids[index])
        new_g_probs.append(g_probs[index])
        new_n_probs.append(n_probs[index])

    return new_n_centroids_pred, new_g_centroids_pred, new_n_probs, new_g_probs

#Euclidean distance computed in um
def distance_um(p, q, dimx, dimy):
    dist_um = (((p[0]-q[0])*dimx)**2)+(((p[1]-q[1])*dimy)**2)
    return np.sqrt(dist_um)


def decoding_pairs(nctr, gctr, nemb, gemb, nprob, gprob, ii, jj, x_spacing, y_spacing):
    # Initialize new arrays to store paired centroids and probabilities
    nctr_new = []
    gctr_new = []
    nprob_new = []
    gprob_new = []
    gemb_new = []
    nemb_new = []

    nctr_new_final = []
    gctr_new_final = []

    # Copy the original arrays so that we can modify them without affecting the originals
    nctr_remaining = nctr.copy()
    gctr_remaining = gctr.copy()
    nprob_remaining = nprob.copy()
    gprob_remaining = gprob.copy()
    nemb_remaining = nemb.copy()
    gemb_remaining = gemb.copy()


    nctr_remaining = nctr_remaining.tolist()
    gctr_remaining = gctr_remaining.tolist()
    nprob_remaining = nprob_remaining.tolist()
    gprob_remaining = gprob_remaining.tolist()
    #nemb_remaining = nemb_remaining.tolist()
    #gemb_remaining = gemb_remaining.tolist()

    # Iterate until there are no more nuclei or Golgi left
    while len(nctr_remaining) > 0 and len(gctr_remaining) > 0:
        min_distance = float('inf')
        min_nucleus_index = None
        min_golgi_index = None

        found_ = False
        # Find the pair (nucleus, golgi) with the minimum embedding distance
        for ni, n_value in enumerate(nemb_remaining):
            for gi, g_value in enumerate(gemb_remaining):
                distance = abs(n_value - g_value) # L1-distance between nuclei embeddings and Golgi embeddings
                if distance < min_distance and distance_um(gctr_remaining[gi], nctr_remaining[ni], x_spacing, y_spacing)<12:
                    min_distance = distance
                    min_nucleus_index = ni
                    min_golgi_index = gi
                    found_ = True

        if found_:
          # Add the paired centroids and probabilities to the new arrays
          nctr_new.append(nctr_remaining[min_nucleus_index])
          gctr_new.append(gctr_remaining[min_golgi_index])
          nprob_new.append(nprob_remaining[min_nucleus_index])
          gprob_new.append(gprob_remaining[min_golgi_index])
          gemb_new.append(gemb_remaining[min_golgi_index])
          nemb_new.append(nemb_remaining[min_nucleus_index])

          nctr_new_final.append((nctr_remaining[min_nucleus_index][0]+ii, nctr_remaining[min_nucleus_index][1]+jj))
          gctr_new_final.append((gctr_remaining[min_golgi_index][0]+ii, gctr_remaining[min_golgi_index][1]+jj))

          # Remove the paired nuclei and Golgi from the remaining lists
          del nctr_remaining[min_nucleus_index]
          del gctr_remaining[min_golgi_index]
          del nemb_remaining[min_nucleus_index]
          del gemb_remaining[min_golgi_index]
          del nprob_remaining[min_nucleus_index]
          del gprob_remaining[min_golgi_index]

        else:
          break

    return nctr_new, gctr_new, nprob_new, gprob_new, nctr_new_final, gctr_new_final


def distance_euclidean(p,q):
    return np.sqrt((p-q)**2)

#Euclidean distance computed in um
def distance_um(p, q, dimx, dimy):
    dist_um = (((p[0]-q[0])*dimx)**2)+(((p[1]-q[1])*dimy)**2)
    return np.sqrt(dist_um) 

def test_3dcellpol(model_path, images_dir, _patch_size, _step, pred_dir, x_spacing, y_spacing):
    model = load_old_model(model_path)
    for img_name in os.listdir(images_dir):
        if '.npy' not in img_name:
            continue
            
        image = np.load(os.path.join(images_dir, img_name))
        uint16_max=np.max(image)
        image = image/uint16_max
        image = (image*255.0).astype('uint8')
        image = image[:,:,0:2]

        #image size
        size_y = np.shape(image)[0]
        size_x = np.shape(image)[1]
        aux_sizes_or = [size_y, size_x]

        output_final = np.zeros((np.shape(image)[0], np.shape(image)[1], 4)) #x,y,z,c

        ii=0
        while ii+_patch_size<=image.shape[0]:
            jj=0
            while jj+_patch_size<=image.shape[1]:

                n_centroids_aux = []
                g_centroids_aux = []

                _slice = image[ii:ii+_patch_size, jj:jj+_patch_size,0:2]
                _slice = _slice/255.0	

                tstimage = np.expand_dims(_slice, axis=0)
                preds_test = model.predict(tstimage)
                pred_patch = preds_test[0]
                output_final[ii:ii+_patch_size, jj:jj+_patch_size,:] = np.maximum(output_final[ii:ii+_patch_size, jj:jj+_patch_size,:], pred_patch)

                jj=jj+_step
            ii=ii+_step
        
        np.save(os.path.join(pred_dir, img_name.replace('.npy', '') + 'golgi.npy'), output_final[:,:,0])
        np.save(os.path.join(pred_dir, img_name.replace('.npy', '') + 'nuclei.npy'), output_final[:,:,1])