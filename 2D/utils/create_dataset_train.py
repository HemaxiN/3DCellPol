#import the packages
import numpy as np
from tifffile import imread
import csv
import os
from utils.DataUtils import centroids2Images

def check_dir(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        # Create the directory
        os.makedirs(directory_path)
        print("Directory {} created.".format(directory_path))
    else:
        print("Directory {} already exists.".format(directory_path))

def cellpol_data_creation(gt_dir, images_dir, nucleus_radius, golgi_radius, size_, step_, save_imgs_dir, save_outputs_dir):

    image_names = []
    for file_ in os.listdir(images_dir):
        img_dir = os.path.join(images_dir, file_)
        csv_dir = os.path.join(gt_dir, file_.replace('.tif','.csv'))
        if os.path.isfile(csv_dir) and os.path.isfile(img_dir):
            image_names.append(file_)
    
    numbers = np.arange(0,len(image_names))
    count_patches = 0

    check_dir(save_imgs_dir)
    check_dir(save_outputs_dir)
    
    save_gt_ctr_dir = save_outputs_dir.replace('outputs','vectors')
    check_dir(save_gt_ctr_dir)

    for image_nb in numbers:
        csv_dir = os.path.join(gt_dir, image_names[image_nb].replace('.tif','.csv'))
        img_dir = os.path.join(images_dir, image_names[image_nb]) 

        image = imread(img_dir)

        #image size
        size_y = np.shape(image)[0]
        size_x = np.shape(image)[1]
        aux_sizes_or = [size_y, size_x]

        #new image size multiple of the patch size
        new_size_y = int((size_y/size_) + 1) * size_
        new_size_x = int((size_x/size_) + 1) * size_
        aux_sizes = [new_size_y, new_size_x]
        
        ## zero padding
        aux_img = np.random.randint(1,10,(aux_sizes[0], aux_sizes[1], 2)).astype('uint8')
        aux_img[0:aux_sizes_or[0], 0:aux_sizes_or[1],:] = image[:,:,0:2]
        image = aux_img

        ## nuclei and golgi ground-truth centroids
        centroids_nuclei = []
        centroids_golgi = []

        #open the csv file and run through it
        file = open(csv_dir, "rU")
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            if row[0] != 'Xgreen,Ygreen,Xred,Yred':
                aux = row[0].split(",")
                YN = float(aux[0])
                XN = float(aux[1])
                YG = float(aux[2])
                XG = float(aux[3])
                centroids_nuclei.append((XN,YN))
                centroids_golgi.append((XG,YG))

        centroids_nuclei = np.asarray(centroids_nuclei)
        centroids_golgi = np.asarray(centroids_golgi)

        i = 0
        while i+size_<=image.shape[0]:
            j = 0
            while j+size_<=image.shape[1]:

                _slice = image[i:i+size_, j:j+size_,:]
                cnn = []
                cgg = []
                vecs_ = []

                for cn, cg in zip(centroids_nuclei, centroids_golgi):
                   cn2 = [cn[0]-i,cn[1]-j]
                   cg2 = [cg[0]-i,cg[1]-j]
                   if(0<=cn2[0]<size_ and 0<=cg2[0]<size_ and 0<=cn2[1]<size_ and 0<=cg2[1]<size_):
                       cnn.append([int(cn2[0]), int(cn2[1])])
                       cgg.append([int(cg2[0]), int(cg2[1])])
                       vecs_.append([int(cn2[1]), int(cn2[0]), int(cg2[1]), int(cg2[0])])
                
                if len(cnn)>1:

                    cnn = np.asarray(cnn)
                    cgg = np.asarray(cgg)
                    vecs_ = np.asarray(vecs_)

                    centroids_image = np.zeros((np.shape(_slice)[0], np.shape(_slice)[1], 4))
                    centroids_image[:,:,1], centroids_image[:,:,3] = centroids2Images(cnn, np.shape(_slice)[0], np.shape(_slice)[1], g_radius=nucleus_radius, th=0.5)
                    centroids_image[:,:,0], centroids_image[:,:,2] = centroids2Images(cgg, np.shape(_slice)[0], np.shape(_slice)[1], g_radius=golgi_radius, th=0.5)
                    _mask_slice = centroids_image

                    centroids_image[:,:,0] =  255.0*centroids_image[:,:,0]
                    centroids_image[:,:,1] =  255.0*centroids_image[:,:,1]

                    _slice = _slice.astype('uint8')
                    _mask_slice = _mask_slice.astype('uint8')
                    np.save(os.path.join(save_imgs_dir, str(count_patches) + '.npy'), _slice[:,:,0:2])
                    np.save(os.path.join(save_outputs_dir, str(count_patches) + '.npy'), _mask_slice)
                    np.save(os.path.join(save_gt_ctr_dir, str(count_patches) + 'nuclei.npy'), cnn)
                    np.save(os.path.join(save_gt_ctr_dir, str(count_patches) + 'golgi.npy'), cgg)
                    count_patches+=1

                j = j+step_
            i = i+step_