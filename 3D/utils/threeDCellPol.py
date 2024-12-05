#import the required packages
from keras import backend as K
import tensorflow as tf
from functools import partial
from keras.models import Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, Dropout, Conv3DTranspose
K.set_image_data_format("channels_last")
try:
        from keras.engine import merge
except ImportError:
        from keras.layers.merge import concatenate
from tensorflow.python.ops import *
from keras.models import *
from keras import layers as L
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.augmentations.spatial_transformations import *
import cv2
from scipy.ndimage.interpolation import rotate
import random
import numpy as np 
import os
import math
import keras

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

def threeDCellPol(n_classes=2, im_sz=128, depth=64, n_channels=2, n_filters_start=32, growth_factor=2, upconv=True):
        droprate=0.10
        n_filters = n_filters_start
        inputs = Input((im_sz, im_sz, depth, n_channels))
        #inputs = BatchNormalization(axis=-1)(inputs)
        conv1 = Conv3D(n_filters, (3,3,3), activation='relu', padding='same', data_format='channels_last')(inputs)
        conv1 = Conv3D(n_filters, (3,3,3), activation='relu', padding='same', data_format='channels_last')(conv1)
        pool1 = MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_last')(conv1)
        #pool1 = Dropout(droprate)(pool1)

        n_filters *= growth_factor
        pool1 = BatchNormalization(axis=-1)(pool1)
        conv2 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(pool1)
        conv2 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv2)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_last')(conv2)
        pool2 = Dropout(droprate)(pool2)

        n_filters *= growth_factor
        pool2 = BatchNormalization(axis=-1)(pool2)
        conv3 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(pool2)
        conv3 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv3)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_last')(conv3)
        pool3 = Dropout(droprate)(pool3)

        n_filters *= growth_factor
        pool3 = BatchNormalization(axis=-1)(pool3)
        conv4_0 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(pool3)
        conv4_0 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv4_0)
        pool4_1 = MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_last')(conv4_0)
        pool4_1 = Dropout(droprate)(pool4_1)

        n_filters *= growth_factor
        pool4_1 = BatchNormalization(axis=-1)(pool4_1)
        conv4_1 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(pool4_1)
        conv4_1 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv4_1)
        pool4_2 = MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_last')(conv4_1)
        pool4_2 = Dropout(droprate)(pool4_2)

        n_filters *= growth_factor
        conv5 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(pool4_2)
        conv5 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv5)

        n_filters //= growth_factor
        if upconv:
                up6_1 = concatenate([Conv3DTranspose(n_filters, (2, 2, 2), strides=(2, 2, 2), padding='same', data_format='channels_last')(conv5), conv4_1])
        else:
                up6_1 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv4_1])
        up6_1 = BatchNormalization(axis=-1)(up6_1)
        conv6_1 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(up6_1)
        conv6_1 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv6_1)
        conv6_1 = Dropout(droprate)(conv6_1)

        n_filters //= growth_factor
        if upconv:
                up6_2 = concatenate([Conv3DTranspose(n_filters, (2, 2, 2), strides=(2, 2, 2), padding='same', data_format='channels_last')(conv6_1), conv4_0])
        else:
                up6_2 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6_1), conv4_0])
        up6_2 = BatchNormalization(axis=-1)(up6_2)
        conv6_2 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(up6_2)
        conv6_2 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv6_2)
        conv6_2 = Dropout(droprate)(conv6_2)

        n_filters //= growth_factor
        if upconv:
                up7 = concatenate([Conv3DTranspose(n_filters, (2, 2, 2), strides=(2, 2, 2), padding='same', data_format='channels_last')(conv6_2), conv3])
        else:
                up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6_2), conv3])
        up7 = BatchNormalization(axis=-1)(up7)
        conv7 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(up7)
        conv7 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv7)
        conv7 = Dropout(droprate)(conv7)

        n_filters //= growth_factor
        if upconv:
                up8 = concatenate([Conv3DTranspose(n_filters, (2, 2, 2), strides=(2, 2, 2), padding='same', data_format='channels_last')(conv7), conv2])
        else:
                up8 = concatenate([UpSampling3D(size=(2, 2, 2))(conv7), conv2])
        up8 = BatchNormalization(axis=-1)(up8)
        conv8 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(up8)
        conv8 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv8)
        conv8 = Dropout(droprate)(conv8)

        n_filters //= growth_factor
        if upconv:
                up9 = concatenate([Conv3DTranspose(n_filters, (2, 2, 2), strides=(2, 2, 2), padding='same', data_format='channels_last')(conv8), conv1])
        else:
                up9 = concatenate([UpSampling3D(size=(2, 2, 2))(conv8), conv1])
        conv9 = Conv3D(n_filters, (3,3,3), activation='relu', padding='same', data_format='channels_last')(up9)
        conv9 = Conv3D(n_filters, (3,3,3), activation='relu', padding='same', data_format='channels_last')(conv9)

        conv9_n = Conv3D(n_filters, (3,3,3), activation='relu', padding='same', data_format='channels_last')(conv9)
        conv9_n = BatchNormalization(axis=-1)(conv9_n)

        conv9_g = Conv3D(n_filters, (3,3,3), activation='relu', padding='same', data_format='channels_last')(conv9)
        conv9_g = BatchNormalization(axis=-1)(conv9_g)

        #nuclei branch
        convnuclei_centroids = Conv3D(n_filters, (3,3,3), activation='relu',  padding='same', data_format='channels_last')(conv9_n)
        convnuclei_embeddings = Conv3D(n_filters, (3,3,3), activation='relu',  padding='same', data_format='channels_last')(conv9_n)
        convnuclei_centroids = Conv3D(1, (1, 1, 1), activation='sigmoid', data_format='channels_last')(convnuclei_centroids)
        convnuclei_embeddings = Conv3D(1, (1, 1, 1), activation='linear', data_format='channels_last')(convnuclei_embeddings)

        #golgi branch
        convgolgi_centroids = Conv3D(n_filters, (3,3,3), activation='relu',  padding='same', data_format='channels_last')(conv9_g)
        convgolgi_embeddings = Conv3D(n_filters, (3,3,3), activation='relu',  padding='same', data_format='channels_last')(conv9_g)
        convgolgi_centroids = Conv3D(1, (1, 1, 1), activation='sigmoid', data_format='channels_last')(convgolgi_centroids)
        convgolgi_embeddings = Conv3D(1, (1, 1, 1), activation='linear', data_format='channels_last')(convgolgi_embeddings)

        model = Model(inputs=inputs, outputs=concatenate([convgolgi_centroids, convnuclei_centroids, convgolgi_embeddings, convnuclei_embeddings], axis=4))
        opt = tf.keras.optimizers.Adam()
        model.compile(optimizer = opt, loss = both_joint_loss_function, metrics=[weighted_mean_se, pull_loss_vis, push_loss_vis])
        return model

#Learning rate schedule
def step_decay(epoch, initial_lrate, drop, epochs_drop):
        return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))

def get_callbacks(model_file, logging_file, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                                    learning_rate_patience=50, verbosity=1,
                                    early_stopping_patience=None):
        callbacks = list()
        callbacks.append(ModelCheckpoint(model_file, save_best_only=True))
        callbacks.append(CSVLogger(logging_file, append=True))
        if learning_rate_epochs:
                callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                                                                                             drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
        else:
                callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                                                                     verbose=verbosity))
        if early_stopping_patience:
                callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))
        return callbacks

def load_old_model(model_file):
        print("Loading pre-trained model")
        custom_objects = {'mse':mse, 'weighted_mean_se':weighted_mean_se, 'push_loss':push_loss, 'pull_aux_optimized': pull_aux_optimized,
        'pull_loss': pull_loss, 'push_pull_loss': push_pull_loss, 'pull_loss_vis': pull_loss_vis, 'push_loss_vis': push_loss_vis, 'both_joint_loss_function': both_joint_loss_function}
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

def train_model(model, model_file, logging_file, training_generator, validation_generator, steps_per_epoch, validation_steps,
                                initial_learning_rate=0.001, learning_rate_drop=0.5, learning_rate_epochs=None, n_epochs=400,
                                learning_rate_patience=20, early_stopping_patience=20):
        model.fit_generator(generator=training_generator,
                                                steps_per_epoch=steps_per_epoch,
                                                epochs=n_epochs,
                                                validation_data=validation_generator,
                                                validation_steps=validation_steps,
                                                callbacks=get_callbacks(model_file, logging_file,
                                                                                                initial_learning_rate=initial_learning_rate,
                                                                                                learning_rate_epochs=learning_rate_epochs,
                                                                                                early_stopping_patience=early_stopping_patience))
        return model 

# Generates data for Keras
class DataGenerator(keras.utils.all_utils.Sequence):

        def __init__(self, data_dir, partition, configs, data_aug):
                self.data_aug = data_aug
                self.partition = partition
                self.data_dir = data_dir
                self.list_IDs = sorted(os.listdir(self.data_dir+partition+'/images/'),key=self.order_dirs)
                self.dim = configs['dim']
                self.mask_dim = configs['mask_dim']
                self.batch_size = configs['batch_size']
                self.shuffle = configs['shuffle']
                self.on_epoch_end()

        def __len__(self):
                'Denotes the number of batches per epoch'
                return int(np.floor(len(self.list_IDs) / self.batch_size))

        def __getitem__(self, index):
                'Generate one batch of data'
                # Generate indexes of the batch
                indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
                # Find list of IDs
                list_IDs_temp = [self.list_IDs[k] for k in indexes]
                # Generate data
                X, mask = self.__data_generation(list_IDs_temp)
                X, mask = self.norm_(X,mask)
                X, mask = self.compute_weights(X,mask)    
                return X, mask

        def on_epoch_end(self):
                'Updates indexes after each epoch'
                self.indexes = np.arange(len(self.list_IDs))
                if self.shuffle == True:
                        np.random.shuffle(self.indexes)

        def __data_generation(self, list_IDs_temp):
                # Initialization
                X = np.empty((self.batch_size, *self.dim))
                mask = np.empty((self.batch_size, *self.mask_dim))
                # Generate data
                for i, ID_path in enumerate(list_IDs_temp):
                        img_aux = np.load(self.data_dir + self.partition +'/images/'+ ID_path)
                        msk_aux = np.load(self.data_dir + self.partition +'/outputs/'+ ID_path)
                        if self.data_aug:
                            img_aux = img_aux/255.0
                            #data augmentation
                            if random.uniform(0,1)<0.5:
                                img_aux, msk_aux = self.vertical_flip(img_aux, msk_aux)
                            if random.uniform(0,1)<0.5:
                                img_aux, msk_aux = self.horizontal_flip(img_aux, msk_aux)
                            if random.uniform(0,1)<0.5:
                                img_aux, msk_aux = self.intensity(img_aux, msk_aux)
                            if random.uniform(0,1)<0.5:
                                angle = np.random.choice(np.arange(0,360,90))
                                img_aux, msk_aux = self.rotation(img_aux, msk_aux, angle)
                            img_aux = img_aux*255.0
                        img_aux = img_aux[:,:,:,0:2]
                        X[i,] = img_aux
                        mask[i,] = msk_aux
                return X, mask

        def compute_weights(self,X,mask):
            mask_out = np.zeros((np.shape(mask)[0], np.shape(mask)[1], np.shape(mask)[2], np.shape(mask)[3], 6))
            for i in range(0, np.shape(mask)[0]): #batch size
                #aux_x = np.zeros((np.shape(mask)[1], np.shape(mask)[2], np.shape(mask)[3]))
                #aux_y = np.zeros((np.shape(mask)[1], np.shape(mask)[2], np.shape(mask)[3]))
                mask_out[i,:,:,:,0] = mask[i,:,:,:,0]/255.0 #golgi embeddings
                mask_out[i,:,:,:,1] = mask[i,:,:,:,1]/255.0 #nuclei embeddings              
                mask_out[i,:,:,:,2] = 5 + (800*(mask[i,:,:,:,0]/255.0)) #golgi weights
                mask_out[i,:,:,:,3] = 5 + (200*(mask[i,:,:,:,1]/255.0)) #nuclei weights
                mask_out[i,:,:,:,4] = mask[i,:,:,:,2] #golgi embeddings
                mask_out[i,:,:,:,5] = mask[i,:,:,:,3] #nuclei embeddings
            return X, mask_out

        def order_dirs(self, element):
            return element.replace('.npy','')

        #normalize image intensity values
        def norm_(self, X, mask):
            X = X/255.0
            return X, mask

        ##rotation
        def rotation(self, image, mask, angle):
            rot_image = np.zeros(np.shape(image))
            rot_mask = np.zeros(np.shape(mask))
            center = (128 / 2, 128 / 2)
            #print('here before {}'.format(np.unique(image)))
            # Rotate the image by angle degrees
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            for z in range(0, rot_image.shape[2]):
                rot_image[:,:,z,:] = cv2.warpAffine(image[:,:,z,:],  matrix, (128,128))
                rot_mask[:,:,z,:] = cv2.warpAffine(mask[:,:,z,:],  matrix, (128,128))
            return rot_image, rot_mask
        
        
        ##vertical flip
        def vertical_flip(self, image, mask):
            flippedimage = np.zeros(np.shape(image))
            flippedmask = np.zeros(np.shape(mask))
            for z in range(0, flippedimage.shape[2]):
                flippedimage[:,:,z,:] = cv2.flip(image[:,:,z,:], 0)
                flippedmask[:,:,z,:] = cv2.flip(mask[:,:,z,:], 0)
            return flippedimage, flippedmask

        ##horizontal flip
        def horizontal_flip(self, image, mask):
            flippedimage = np.zeros(np.shape(image))
            flippedmask = np.zeros(np.shape(mask))
            for z in range(0, flippedimage.shape[2]):
                flippedimage[:,:,z,:] = cv2.flip(image[:,:,z,:], 1)
                flippedmask[:,:,z,:] = cv2.flip(mask[:,:,z,:], 1)
            return flippedimage, flippedmask

        #brigtness and contrast variations
        def intensity(self, image, mask, alpha=None):
            image = image.astype('float64')
            image = image*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
            image = image.astype('float64')
            return image, mask