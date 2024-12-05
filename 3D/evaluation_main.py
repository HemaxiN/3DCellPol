from utils.evaluation_utils import *
import os

img_TEST_dir = r'/home/jovyan/discotreino/3DCellPol_Github/images' # directory containing the original TEST .tif images, image shape = (x,y,z,channels)
pred_dir = r'/home/jovyan/discotreino/3DCellPol_Github/results' #prediction folder defined in predict_main.py
eval_dir = r'/home/jovyan/discotreino/3DCellPol_Github/eval' #plots of the evaluation metrics at different evaluation thresholds
gt_dir = r'/home/jovyan/discotreino/3DCellPol_Github/gt' #directory with the ground-truth vector files (.csv files) 
x_spacing = 0.666 #resolution of the images across x
y_spacing = 0.666 #resolution of the images across y
z_spacing = 0.270 #resolution of the images across z

evaluation(img_TEST_dir, gt_dir, pred_dir, eval_dir, x_spacing, y_spacing, z_spacing)