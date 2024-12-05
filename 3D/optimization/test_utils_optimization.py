#import the required packages
import tensorflow as tf
from tensorflow.python.ops import *
from keras.models import *
import cv2
import random
import numpy as np 
import os
from math import *
from tifffile import imread, imwrite

#custom loss functions:
#weighted mean squared error loss function
def mse(y_true, y_pred, sample_weight=None):
    squared  = math_ops.square(y_pred - y_true)
    if sample_weight==None:
        return tf.reduce_mean(squared)
    else:
        multiplication = tf.math.multiply(sample_weight, squared)
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
    quantity = tf.reduce_mean(tf.boolean_mask(max_values, tf.math.is_finite(max_values)))  
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
            condition =  tf.math.is_finite(paux)
            pull = tf.cond(condition, lambda: true_condition(pull, paux), lambda: false_condition(pull))
            E_tensor.append(mean_)                     
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
    for bi in range(2):
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
    for bi in range(2):
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
    for bi in range(2):
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

def square_rooted(x):
    return round(np.sqrt(sum([a*a for a in x])),3)

def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)

def test_3dcellpol(model_path, img_dir, _patch_size, _z_size, _step, _step_z, images_numbers, pred_dir, x_spacing, y_spacing, z_spacing):
    model = load_old_model(model_path)
    for image_nb in images_numbers:
        image = np.load(os.path.join(img_dir, image_nb))
        image = ((image/np.max(image))*255.0).astype('uint8')
        image = image[:,:,:,0:2]
        print('Image shape: {}'.format(image.shape))

        #image size
        size_y = np.shape(image)[0]
        size_x = np.shape(image)[1]
        size_z = np.shape(image)[2]
        aux_sizes_or = [size_y, size_x, size_z]

        output_final = np.zeros((np.shape(image)[0], np.shape(image)[1], np.shape(image)[2], 4)) #x,y,z,c

        ii=0
        while ii+_patch_size<=image.shape[0]:
            jj=0
            while jj+_patch_size<=image.shape[1]:
                k=0
                while k+_z_size<=image.shape[2]:

                    n_centroids_aux = []
                    g_centroids_aux = []

                    _slice = image[ii:ii+_patch_size, jj:jj+_patch_size,k:k+_z_size,0:2]
                    _slice = _slice/255.0	

                    tstimage = np.expand_dims(_slice, axis=0)
                    preds_test = model.predict(tstimage)
                    pred_patch = preds_test[0]
                    output_final[ii:ii+_patch_size, jj:jj+_patch_size,k:k+_z_size,:] = np.maximum(output_final[ii:ii+_patch_size, jj:jj+_patch_size,k:k+_z_size,:], pred_patch)
    
                    k=k+_step_z
                jj=jj+_step
            ii=ii+_step
        

        np.save(os.path.join(pred_dir, image_nb.replace('.npy', '') + 'golgi.npy'), output_final[:,:,:,0])
        np.save(os.path.join(pred_dir, image_nb.replace('.npy', '') + 'nuclei.npy'), output_final[:,:,:,1])