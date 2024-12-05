pred_dir = r'/home/jovyan/discotreino/3DCellPol_Github/results_val' #directory to save the validation results
img_dir = r'/home/jovyan/discotreino/3DCellPol_Github/dataset/val/images' # directory containing the validation sub-volumes
model_path = r'/home/jovyan/discotreino/3DCellPol_Github/model/final_3dcellpol.hdf5' #path to the trained model

# test parameters
_patch_size = 128 #patch size along x and y directions
_z_size = 64 #patch size along z direction
_step = 32 #overlap along x and y directions between consecutive patches extracted from the image
_step_z = 64 #overlap along z directions between consecutive patches extracted from the image
x_spacing = 0.666 #resolution of the images across x
y_spacing = 0.666 #resolution of the images across y
z_spacing = 0.270 #resolution of the images across z

from test_utils_optimization import *
import os

## image names
imgs = [f for f in os.listdir(img_dir) if f.endswith('.npy')]
test_3dcellpol(model_path, img_dir, _patch_size, _z_size, _step, _step_z, imgs, pred_dir, x_spacing, y_spacing, z_spacing)
