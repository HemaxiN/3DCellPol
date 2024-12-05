from utils.create_dataset_train import cellpol_data_creation
import os

split_ = 'train' #train or val
save_imgs_dir = r'/home/jovyan/discotreino/3DCellPol_Github/dataset/' + split_ + '/images' #directory where the image sub-volumes for training will be saved
save_outputs_dir = r'/home/jovyan/discotreino/3DCellPol_Github/dataset/' + split_ + '/outputs' #directory where the output sub-volumes for training will be saved
images_dir = r'/home/jovyan/discotreino/3DCellPol_Github/images' #directory containing the original .tif images, image shape = (x,y,z,channels)
gt_dir = r'/home/jovyan/discotreino/3DCellPol_Github/gt' #directory with the gt nucleus-Golgi vectors (.csv files)
size_ = 128 #size of the sub-volumes
step_ = 64 #overlap between sub-volumes
image_names = [f for f in os.listdir(images_dir) if f.endswith('.tif')] ## image names
n_radius = 9 #nucleus radius for the Gaussians in the Gaussian heatmap (in voxels)
g_radius = 5 #Golgi radius for the Gaussians in the Gaussian heatmap (in voxels)

cellpol_data_creation(image_names, gt_dir, images_dir, n_radius, g_radius, size_, step_, save_imgs_dir, save_outputs_dir)